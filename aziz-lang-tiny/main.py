# /// script
# requires-python = "==3.14"
# dependencies = [
#     "xdsl==0.56.0",
#     "unicorn==2.1.4",
#     "pyelftools==0.32",
# ]
# ///

import argparse
from parser import AzizParser
from pathlib import Path

import aziz
from ir_gen import IRGen
from lower import LowerAzizPass, LowerPrintfToLLVMCallPass
from optimize import OptimizeAzizPass
from xdsl.context import Context
from xdsl.dialects import affine, arith, func, printf, scf
from xdsl.dialects.builtin import Builtin
from xdsl.transforms.canonicalize import CanonicalizePass
from xdsl.transforms.lower_affine import LowerAffinePass


def main():
    parser = argparse.ArgumentParser(description="aziz language")
    parser.add_argument("file", help="source file")
    parser.add_argument("--debug", action="store_true", help="emit intermediate mlir representations")
    args = parser.parse_args()
    assert args.file.endswith(".aziz")
    src = Path(args.file).read_text()

    module_ast = AzizParser(None, src).parse_module()
    module_op = IRGen().ir_gen_module(module_ast)

    if args.debug:
        print(f"\n\n{'-' * 80} MLIR before optimization\n\n")
        print(module_op)

    ctx = context()
    OptimizeAzizPass().apply(ctx, module_op)
    module_op.verify()

    if args.debug:
        print(f"\n\n{'-' * 80} MLIR after optimization\n\n")
        print(module_op)

    LowerAzizPass().apply(ctx, module_op)
    LowerAffinePass().apply(ctx, module_op)
    CanonicalizePass().apply(ctx, module_op)
    LowerPrintfToLLVMCallPass().apply(ctx, module_op)
    module_op.verify()

    if args.debug:
        print(f"\n\n{'-' * 80} MLIR after lowering to LLVM dialect\n\n")
    print(module_op)


def context() -> Context:
    ctx = Context()
    ctx.load_dialect(aziz.Aziz)
    ctx.load_dialect(affine.Affine)
    ctx.load_dialect(arith.Arith)
    ctx.load_dialect(Builtin)
    ctx.load_dialect(func.Func)
    ctx.load_dialect(printf.Printf)
    ctx.load_dialect(scf.Scf)
    return ctx


if __name__ == "__main__":
    main()
