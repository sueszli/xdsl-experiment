# /// script
# requires-python = "==3.14"
# dependencies = [
#     "xdsl==0.56.0",
#     "unicorn==2.1.4",
#     "pyelftools==0.32",
# ]
# ///

import argparse
from contextlib import redirect_stdout
from functools import lru_cache
from io import StringIO
from pathlib import Path

from dialects import aziz
from frontend.ast_nodes import dump
from frontend.ir_gen import IRGen
from frontend.parser import AzizParser
from interpreter import AzizFunctions
from llvm_exec import execute_llvm
from qemu import emulate_riscv
from rewrites.lower import LowerAzizPass
from rewrites.lower_llvm import LowerPrintfToLLVMCallPass
from rewrites.lower_riscv import AddPrintRuntimePass, AddRecursionSupportPass, CustomLowerScfToRiscvPass, EmitDataSectionPass, LowerPrintfPass, LowerSelectPass, MapToPhysicalRegistersPass, RemoveUnprintableOpsPass, format_assembly
from rewrites.optimize import OptimizeAzizPass
from xdsl.backend.riscv.lowering.convert_arith_to_riscv import ConvertArithToRiscvPass
from xdsl.backend.riscv.lowering.convert_func_to_riscv_func import ConvertFuncToRiscvFuncPass
from xdsl.backend.riscv.lowering.convert_memref_to_riscv import ConvertMemRefToRiscvPass
from xdsl.backend.riscv.lowering.convert_riscv_scf_to_riscv_cf import ConvertRiscvScfToRiscvCfPass
from xdsl.context import Context
from xdsl.dialects import affine, arith, func, printf, riscv, riscv_func, riscv_scf, scf
from xdsl.dialects.builtin import Builtin, ModuleOp
from xdsl.interpreter import Interpreter
from xdsl.transforms.canonicalize import CanonicalizePass
from xdsl.transforms.dead_code_elimination import DeadCodeElimination
from xdsl.transforms.lower_affine import LowerAffinePass
from xdsl.transforms.lower_riscv_func import LowerRISCVFunc
from xdsl.transforms.reconcile_unrealized_casts import ReconcileUnrealizedCastsPass
from xdsl.transforms.riscv_allocate_registers import RISCVAllocateRegistersPass


def main():
    parser = argparse.ArgumentParser(description="aziz language")
    parser.add_argument("file", help="source file")
    # lowerings
    parser.add_argument("--emit-source", action="store_true", help="emit source code")
    parser.add_argument("--emit-ast", action="store_true", help="emit abstract syntax tree")
    parser.add_argument("--emit-mlir", action="store_true", help="emit mlir before and after optimization")
    parser.add_argument("--emit-riscv", action="store_true", help="emit RISC-V assembly")
    parser.add_argument("--emit-llvm", action="store_true", help="compile via LLVM pipeline")
    # output should be identical
    parser.add_argument("--interpret", action="store_true", help="interpret aziz mlir")
    parser.add_argument("--execute-llvm", action="store_true", help="execute LLVM executable")
    parser.add_argument("--execute-riscv", action="store_true", help="execute RISC-V assembly in qemu emulator")
    parser.add_argument("--all", action="store_true", help="emit all stages")
    args = parser.parse_args()

    if args.all:
        args.emit_source = args.emit_ast = args.emit_mlir = args.emit_llvm = args.emit_riscv = True
        args.interpret = args.execute_riscv = args.execute_llvm = True

    assert args.file.endswith(".aziz")
    src = Path(args.file).read_text()

    # source -> ast -> aziz dialect
    module_ast = AzizParser(None, src).parse_module()
    module_op = IRGen().ir_gen_module(module_ast)

    # interpret
    captured_output = StringIO()
    interpreter = Interpreter(module_op)
    interpreter.register_implementations(AzizFunctions())
    with redirect_stdout(captured_output):
        interpreter.call_op("main", ())
    interpreter_result = captured_output.getvalue()

    # a) aziz dialect -> lowered aziz mlir -> llvm dialect mlir (mlir-opt) -> llvm ir (mlir-translate) -> executable (llc + clang) -> execute
    module_op_llvm = module_op.clone()
    lower_llvm_mut(module_op_llvm)
    llvm_ir, llvm_exec_out, llvm_exec_err = execute_llvm(module_op_llvm)

    # b) aziz dialect -> lowered aziz mlir -> riscv dialect mlir -> riscv assembly codegen -> execute in qemu
    module_op_riscv = module_op.clone()
    lower_aziz_mut(module_op_riscv)
    lower_riscv_mut(module_op_riscv)
    io = StringIO()
    riscv.print_assembly(module_op_riscv, io)
    riscv_asm = format_assembly(io.getvalue())  # Rename riscv_ir -> riscv_asm for consistency
    emulator_result = emulate_riscv(riscv_asm, entry_symbol="main")

    # print
    w = 50
    print_block = lambda title, content: print(f"\033[90m╭{'─' * w}╮\033[0m\n\033[90m│{' ' * ((w - len(title) - 2) // 2)} {title} {' ' * (w - len(title) - 2 - (w - len(title) - 2) // 2)}│\033[0m\n\033[90m╰{'─' * w}╯\033[0m\n\n{content}\n")
    if args.emit_source:
        print_block("source", src)
    if args.emit_ast:
        print_block("ast", dump(module_ast))
    if args.emit_mlir:
        print_block("aziz dialect mlir", module_op)
    if args.emit_llvm:
        print_block("llvm ir", llvm_ir)
    if args.emit_riscv:
        print_block("riscv assembly", riscv_asm)
    if args.interpret:
        print_block("interpreter output", interpreter_result)
    if args.execute_riscv:
        print_block("riscv emulator output", emulator_result)
    if args.execute_llvm:
        print_block("llvm output", llvm_exec_out)
        assert not llvm_exec_err, f"llvm produced stderr: {llvm_exec_err}"


def lower_aziz_mut(module_op: ModuleOp):
    ctx = context()
    OptimizeAzizPass().apply(ctx, module_op)  # drop unused functions, inline one-liner functions
    LowerAzizPass().apply(ctx, module_op)  # lower to arith, func, scf, printf, llvm.global for strings
    LowerAffinePass().apply(ctx, module_op)
    CanonicalizePass().apply(ctx, module_op)  # automatically look up and apply canonicalization patterns for each op
    module_op.verify()


def lower_llvm_mut(module_op: ModuleOp):
    ctx = context()
    LowerAzizPass().apply(ctx, module_op)
    LowerAffinePass().apply(ctx, module_op)
    CanonicalizePass().apply(ctx, module_op)
    LowerPrintfToLLVMCallPass().apply(ctx, module_op)
    module_op.verify()


def lower_riscv_mut(module_op: ModuleOp):
    ctx = context()
    LowerSelectPass().apply(ctx, module_op)  # arith.select missing from xdsl lib
    RemoveUnprintableOpsPass().apply(ctx, module_op)  # handle llvm.global and llvm.address_of for strings
    EmitDataSectionPass().apply(ctx, module_op)
    AddPrintRuntimePass().apply(ctx, module_op)
    ConvertFuncToRiscvFuncPass().apply(ctx, module_op)  # func -> riscv_func
    AddRecursionSupportPass().apply(ctx, module_op)
    CustomLowerScfToRiscvPass().apply(ctx, module_op)  # replaces ConvertScfToRiscvPass
    ConvertMemRefToRiscvPass().apply(ctx, module_op)  # memref -> riscv load/store
    ConvertArithToRiscvPass().apply(ctx, module_op)  # arith -> riscv
    LowerPrintfPass().apply(ctx, module_op)  # printf -> print runtime calls (after type conversion)
    DeadCodeElimination().apply(ctx, module_op)  # dce
    ReconcileUnrealizedCastsPass().apply(ctx, module_op)  # cleanup casts
    RISCVAllocateRegistersPass(allow_infinite=True).apply(ctx, module_op)  # virtual -> physical registers
    MapToPhysicalRegistersPass().apply(ctx, module_op)
    LowerRISCVFunc(insert_exit_syscall=True).apply(ctx, module_op)  # riscv_func -> riscv labels and jumps
    ConvertRiscvScfToRiscvCfPass().apply(ctx, module_op)
    module_op.verify()


@lru_cache(None)
def context() -> Context:
    ctx = Context()
    ctx.load_dialect(aziz.Aziz)
    ctx.load_dialect(affine.Affine)
    ctx.load_dialect(arith.Arith)
    ctx.load_dialect(Builtin)
    ctx.load_dialect(func.Func)
    ctx.load_dialect(printf.Printf)
    ctx.load_dialect(riscv_func.RISCV_Func)
    ctx.load_dialect(riscv_scf.RISCV_Scf)
    ctx.load_dialect(riscv.RISCV)
    ctx.load_dialect(scf.Scf)
    return ctx


if __name__ == "__main__":
    main()
