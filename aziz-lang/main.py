# /// script
# requires-python = "==3.14"
# dependencies = [
#     "xdsl==0.56.0",
#     "unicorn==2.1.4",
#     "pyelftools==0.32",
# ]
# ///

import argparse
from functools import lru_cache
from io import StringIO
from pathlib import Path

from dialects import aziz
from frontend.ast_nodes import dump
from frontend.ir_gen import IRGen
from frontend.parser import AzizParser
from interpreter import AzizFunctions
from qemu import run_riscv
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
    module_op.verify()

    ConvertFuncToRiscvFuncPass().apply(ctx, module_op)  # func -> riscv_func
    AddRecursionSupportPass().apply(ctx, module_op)
    CustomLowerScfToRiscvPass().apply(ctx, module_op)  # replaces ConvertScfToRiscvPass
    ConvertMemRefToRiscvPass().apply(ctx, module_op)  # memref -> riscv load/store
    ConvertArithToRiscvPass().apply(ctx, module_op)  # arith -> riscv
    LowerPrintfPass().apply(ctx, module_op)  # printf -> print runtime calls (after type conversion)
    DeadCodeElimination().apply(ctx, module_op)  # dce
    ReconcileUnrealizedCastsPass().apply(ctx, module_op)  # cleanup casts
    RISCVAllocateRegistersPass(allow_infinite=True).apply(ctx, module_op)  # virtual -> physical registers (no spilling check)
    MapToPhysicalRegistersPass().apply(ctx, module_op)
    LowerRISCVFunc(insert_exit_syscall=True).apply(ctx, module_op)  # riscv_func -> riscv labels and jumps
    ConvertRiscvScfToRiscvCfPass().apply(ctx, module_op)
    module_op.verify()


def main():
    parser = argparse.ArgumentParser(description="aziz language")
    parser.add_argument("file", help="source file")
    xor_group = parser.add_mutually_exclusive_group()
    xor_group.add_argument("--ast", action="store_true", help="print final ir")
    xor_group.add_argument("--mlir", action="store_true", help="print final mlir")
    xor_group.add_argument("--asm", action="store_true", help="emit RISC-V assembly")
    xor_group.add_argument("--interpret", action="store_true", help="interpret the code")
    args = parser.parse_args()
    assert args.file.endswith(".aziz")
    src = Path(args.file).read_text()

    module_ast = AzizParser(None, src).parse_module()  # source -> ast
    module_op = IRGen().ir_gen_module(module_ast)  # ast -> mlir

    gray = lambda s: f"\n\033[90m{'-' * 100}\n{s}\n{'-' * 100}\n\033[0m"

    if args.ast:
        print(gray("ast"))
        print(dump(module_ast))
        return

    if args.interpret:
        interpreter = Interpreter(module_op)
        interpreter.register_implementations(AzizFunctions())
        print(gray("mlir"))
        print(module_op)
        print(gray("interpretation result"))
        interpreter.call_op("main", ())
        return

    orig = module_op.clone()
    lower_aziz_mut(module_op)

    if args.mlir:
        print(gray("before transformation"))
        print(orig)
        print(gray("after transformation"))
        print(module_op)
        return

    lower_riscv_mut(module_op)

    if args.asm:
        io = StringIO()
        riscv.print_assembly(module_op, io)  # mlir riscv dialect -> riscv assembly
        source = io.getvalue()

        print(gray("riscv assembly"))
        print(source)
        print(gray("emulation result"))
        run_riscv(source, entry_symbol="main")
        return


if __name__ == "__main__":
    main()
