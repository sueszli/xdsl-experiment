# /// script
# requires-python = "==3.14"
# dependencies = [
#     "xdsl==0.56.0",
#     "unicorn==2.1.4",
#     "pyelftools==0.32",
# ]
# ///

import argparse
import sys
from functools import lru_cache
from io import StringIO
from pathlib import Path

from dialects import aziz
from frontend.ast_nodes import dump
from frontend.ir_gen import IRGen
from frontend.parser import AzizParser
from interpreter import AzizFunctions
from qemu import emulate_riscv
from rewrites.lower import LowerAzizPass
from rewrites.lower_riscv import AddPrintRuntimePass, AddRecursionSupportPass, CustomLowerScfToRiscvPass, EmitDataSectionPass, LowerPrintfPass, LowerSelectPass, MapToPhysicalRegistersPass, RemoveUnprintableOpsPass
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


def lower_aziz_mut(module_op: ModuleOp):
    ctx = context()
    # drop unused functions, inline one-liner functions
    OptimizeAzizPass().apply(ctx, module_op)

    # lower to arith, func, scf, printf, llvm.global for strings
    LowerAzizPass().apply(ctx, module_op)
    LowerAffinePass().apply(ctx, module_op)

    # automatically look up and apply canonicalization patterns for each op
    CanonicalizePass().apply(ctx, module_op)
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


def main():
    parser = argparse.ArgumentParser(description="aziz language")
    parser.add_argument("file", help="source file")
    parser.add_argument("--source", action="store_true", help="emit source code")
    parser.add_argument("--ast", action="store_true", help="emit abstract syntax tree")
    parser.add_argument("--mlir", action="store_true", help="emit mlir before and after optimization")
    parser.add_argument("--interpret", action="store_true", help="interpret aziz mlir")
    parser.add_argument("--asm", action="store_true", help="emit RISC-V assembly")
    parser.add_argument("--execute", action="store_true", help="execute RISC-V assembly")
    parser.add_argument("--all", action="store_true", help="emit all stages")
    args = parser.parse_args()
    if args.all:
        args.source = args.ast = args.mlir = args.interpret = args.asm = args.execute = True

    assert args.file.endswith(".aziz")
    src = Path(args.file).read_text()

    # source -> ast -> aziz mlir -> lowered mlir
    module_ast = AzizParser(None, src).parse_module()
    module_op = IRGen().ir_gen_module(module_ast)
    orig1 = module_op.clone()
    lower_aziz_mut(module_op)

    # interpret
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()
    interpreter = Interpreter(orig1)
    interpreter.register_implementations(AzizFunctions())
    interpreter.call_op("main", ())
    sys.stdout = old_stdout
    interpreter_result = captured_output.getvalue()

    # lowered mlir -> riscv mlir
    orig2 = module_op.clone()
    lower_riscv_mut(module_op)

    # emulate
    io = StringIO()
    riscv.print_assembly(module_op, io)
    source = io.getvalue()
    result = emulate_riscv(source, entry_symbol="main")

    # print results
    print_block = lambda title, content: print(f"\033[90m{'-' * 100}\n{title}\n{'-' * 100}\033[0m\n\n{content}\n")
    if args.source:
        print_block("source", src)
    if args.ast:
        print_block("ast", dump(module_ast))
    if args.mlir:
        print_block("before optimization", orig1)
        print_block("after optimization", orig2)
    if args.interpret:
        print_block("interpreter output", interpreter_result)
    if args.asm:
        print_block("riscv assembly", source)
    if args.execute:
        print_block("emulator output", result)


if __name__ == "__main__":
    main()
