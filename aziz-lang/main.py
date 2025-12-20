# /// script
# requires-python = "==3.14"
# dependencies = [
#     "xdsl==0.55.4",
# ]
# ///

import argparse
from pathlib import Path

from dialects import aziz
from frontend.ast_nodes import dump
from frontend.ir_gen import IRGen
from frontend.parser import AzizParser
from interpreter import AzizFunctions
from rewrites.lower import LowerAzizPass
from rewrites.optimize import OptimizeAzizPass
from xdsl.backend.riscv.lowering.convert_arith_to_riscv import ConvertArithToRiscvPass
from xdsl.backend.riscv.lowering.convert_func_to_riscv_func import ConvertFuncToRiscvFuncPass
from xdsl.backend.riscv.lowering.convert_memref_to_riscv import ConvertMemRefToRiscvPass
from xdsl.backend.riscv.lowering.convert_print_format_to_riscv_debug import ConvertPrintFormatToRiscvDebugPass
from xdsl.backend.riscv.lowering.convert_riscv_scf_to_riscv_cf import ConvertRiscvScfToRiscvCfPass
from xdsl.backend.riscv.lowering.convert_scf_to_riscv_scf import ConvertScfToRiscvPass
from xdsl.context import Context
from xdsl.dialects import affine, arith, func, memref, printf, riscv, riscv_func, scf
from xdsl.dialects.builtin import Builtin, ModuleOp
from xdsl.interpreter import Interpreter
from xdsl.transforms.canonicalize import CanonicalizePass
from xdsl.transforms.dead_code_elimination import DeadCodeElimination
from xdsl.transforms.lower_affine import LowerAffinePass
from xdsl.transforms.lower_riscv_func import LowerRISCVFunc
from xdsl.transforms.reconcile_unrealized_casts import ReconcileUnrealizedCastsPass
from xdsl.transforms.riscv_allocate_registers import RISCVAllocateRegistersPass
from xdsl.transforms.riscv_scf_loop_range_folding import RiscvScfLoopRangeFoldingPass


def transform(module_op: ModuleOp, target: str):
    ctx = Context()
    ctx.load_dialect(affine.Affine)
    ctx.load_dialect(arith.Arith)
    ctx.load_dialect(Builtin)
    ctx.load_dialect(func.Func)
    ctx.load_dialect(memref.MemRef)
    ctx.load_dialect(printf.Printf)
    ctx.load_dialect(riscv_func.RISCV_Func)
    ctx.load_dialect(riscv.RISCV)
    ctx.load_dialect(scf.Scf)
    ctx.load_dialect(aziz.Aziz)

    OptimizeAzizPass().apply(ctx, module_op)
    LowerAzizPass().apply(ctx, module_op)  # aziz-dialect to generic mlir ops

    CanonicalizePass().apply(ctx, module_op)  # standard canonicalization
    module_op.verify()

    if target == "aziz-lowered":
        return

    # Lowering generic dialects to RISC-V specific dialects

    LowerAffinePass().apply(ctx, module_op)

    if target == "scf":
        return

    ConvertFuncToRiscvFuncPass().apply(ctx, module_op)
    ConvertMemRefToRiscvPass().apply(ctx, module_op)
    ConvertPrintFormatToRiscvDebugPass().apply(ctx, module_op)
    ConvertArithToRiscvPass().apply(ctx, module_op)
    ConvertScfToRiscvPass().apply(ctx, module_op)
    DeadCodeElimination().apply(ctx, module_op)
    ReconcileUnrealizedCastsPass().apply(ctx, module_op)

    module_op.verify()

    if target == "riscv":
        return

    # Perform optimizations that don't depend on register allocation
    # e.g. constant folding
    CanonicalizePass().apply(ctx, module_op)
    RiscvScfLoopRangeFoldingPass().apply(ctx, module_op)
    CanonicalizePass().apply(ctx, module_op)

    module_op.verify()

    if target == "riscv-opt":
        return

    RISCVAllocateRegistersPass(allow_infinite=True).apply(ctx, module_op)

    module_op.verify()

    if target == "riscv-regalloc":
        return

    # Perform optimizations that depend on register allocation
    # e.g. redundant moves
    CanonicalizePass().apply(ctx, module_op)

    module_op.verify()

    if target == "riscv-regalloc-opt":
        return

    LowerRISCVFunc(insert_exit_syscall=True).apply(ctx, module_op)
    ConvertRiscvScfToRiscvCfPass().apply(ctx, module_op)

    if target == "riscv-lowered":
        return

    raise ValueError(f"Unknown target option {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="aziz language")
    parser.add_argument("file", help="source file")
    parser.add_argument("--target", help="target dialect", default="aziz-lowered")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--ast", action="store_true", help="print final ir")
    group.add_argument("--mlir", action="store_true", help="print final mlir")
    group.add_argument("--interpret", action="store_true", help="interpret the code")
    args = parser.parse_args()
    assert args.file.endswith(".aziz")
    src = Path(args.file).read_text()

    module_ast = AzizParser(None, src).parse_module()  # source -> ast
    module_op = IRGen().ir_gen_module(module_ast)  # ast -> mlir

    if args.interpret:
        interpreter = Interpreter(module_op)
        interpreter.register_implementations(AzizFunctions())
        interpreter.call_op("main", ())
        exit(0)

    original_module_op = module_op.clone()
    transform(module_op, target=args.target)

    if args.ast:
        print(dump(module_ast))
        exit(0)

    if args.mlir:
        gray = lambda s: f"\033[90m{s}\033[0m"
        print(gray(f"{'-' * 100}\nbefore transformation\n{'-' * 100}"))
        print(original_module_op)
        print(gray(f"{'-' * 100}\nafter transformation\n{'-' * 100}"))
        print(module_op)
        exit(0)

    # io = StringIO()
    # riscv.print_assembly(module_op, io)
    # result = io.getvalue()
    # print(result)
