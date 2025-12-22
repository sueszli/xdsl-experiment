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
from rewrites.lower_riscv import LowerSelectPass, RemoveUnprintableOpsPass, emit_data_section, map_virtual_to_physical_registers
from rewrites.optimize import OptimizeAzizPass
from xdsl.backend.riscv.lowering.convert_arith_to_riscv import ConvertArithToRiscvPass
from xdsl.backend.riscv.lowering.convert_func_to_riscv_func import ConvertFuncToRiscvFuncPass
from xdsl.backend.riscv.lowering.convert_memref_to_riscv import ConvertMemRefToRiscvPass
from xdsl.backend.riscv.lowering.convert_riscv_scf_to_riscv_cf import ConvertRiscvScfToRiscvCfPass
from xdsl.backend.riscv.lowering.convert_scf_to_riscv_scf import ConvertScfToRiscvPass
from xdsl.context import Context
from xdsl.dialects import affine, arith, func, printf, riscv, riscv_func, scf
from xdsl.dialects.builtin import Builtin, ModuleOp
from xdsl.interpreter import Interpreter
from xdsl.transforms.canonicalize import CanonicalizePass
from xdsl.transforms.dead_code_elimination import DeadCodeElimination
from xdsl.transforms.lower_affine import LowerAffinePass
from xdsl.transforms.lower_riscv_func import LowerRISCVFunc
from xdsl.transforms.reconcile_unrealized_casts import ReconcileUnrealizedCastsPass
from xdsl.transforms.riscv_allocate_registers import RISCVAllocateRegistersPass
from xdsl.transforms.riscv_scf_loop_range_folding import RiscvScfLoopRangeFoldingPass


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
    ctx.load_dialect(riscv.RISCV)
    ctx.load_dialect(scf.Scf)
    return ctx


def lower_aziz_mut(module_op: ModuleOp):
    ctx = context()
    # optimize (drop unused, inline functions)
    OptimizeAzizPass().apply(ctx, module_op)

    # lower to arith, func, llvm, printf, scf
    LowerAzizPass().apply(ctx, module_op)
    LowerAffinePass().apply(ctx, module_op)

    # looks up the canonicalization patterns for each op
    CanonicalizePass().apply(ctx, module_op)
    module_op.verify()


def lower_riscv_mut(module_op: ModuleOp):
    ctx = context()

    # lower func, memref, printf, arith, scf to riscv dialects
    LowerSelectPass().apply(ctx, module_op)  # convert arith.select to riscv (not supported by xdsl)
    RemoveUnprintableOpsPass().apply(ctx, module_op)  # todo: bring back printf in assembly

    ConvertFuncToRiscvFuncPass().apply(ctx, module_op)
    ConvertMemRefToRiscvPass().apply(ctx, module_op)
    ConvertArithToRiscvPass().apply(ctx, module_op)
    ConvertScfToRiscvPass().apply(ctx, module_op)

    # remove unused ops and resolve temporary cast operations
    DeadCodeElimination().apply(ctx, module_op)
    ReconcileUnrealizedCastsPass().apply(ctx, module_op)
    module_op.verify()

    # optimizations that don't depend on register allocation
    CanonicalizePass().apply(ctx, module_op)
    RiscvScfLoopRangeFoldingPass().apply(ctx, module_op)  # fold scf loop ranges into riscv operations
    CanonicalizePass().apply(ctx, module_op)
    module_op.verify()

    # assign virtual registers to physical riscv registers (doesnt handle spilling for recursion)
    RISCVAllocateRegistersPass(allow_infinite=True).apply(ctx, module_op)
    module_op.verify()

    # optimizations that depend on register allocation (e.g. redundant moves)
    CanonicalizePass().apply(ctx, module_op)
    module_op.verify()

    # lower riscv_func to labels and convert structured control flow to branches
    LowerRISCVFunc(insert_exit_syscall=True).apply(ctx, module_op)
    ConvertRiscvScfToRiscvCfPass().apply(ctx, module_op)


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

    if args.interpret:
        interpreter = Interpreter(module_op)
        interpreter.register_implementations(AzizFunctions())
        print(gray("mlir"))
        print(module_op)
        print(gray("interpretation result"))
        interpreter.call_op("main", ())
        return

    original_module_op = module_op.clone()
    lower_aziz_mut(module_op)
    lower_riscv_mut(module_op)

    if args.ast:
        print(dump(module_ast))
        return

    if args.mlir:
        print(gray("before transformation"))
        print(original_module_op)
        print(gray("after transformation"))
        print(module_op)
        return

    if args.asm:
        io = StringIO()
        riscv.print_assembly(module_op, io)  # mlir riscv dialect -> riscv assembly
        text_section = io.getvalue()

        # todo: move to seperate RewritePatterns
        data_section = emit_data_section(module_op)
        text_section = map_virtual_to_physical_registers(text_section)
        source = data_section + text_section

        print(gray("riscv assembly"))
        print(source)
        print(gray("emulation result"))
        run_riscv(source, entry_symbol="main")
        return


if __name__ == "__main__":
    main()
