# /// script
# requires-python = "==3.14"
# dependencies = [
#     "xdsl==0.55.4",
# ]
# ///

import argparse
from pathlib import Path

from compiler import compile, context, transform
from dialects import aziz
from frontend.ir_gen import IRGen
from frontend.parser import AzizParser
from interpreter import AzizFunctions
from xdsl.context import Context
from xdsl.dialects import affine, arith, func, memref, printf, riscv, riscv_func, scf
from xdsl.dialects.builtin import Builtin
from xdsl.interpreter import Interpreter


def context() -> Context:
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
    return ctx


def main():
    # group = parser.add_mutually_exclusive_group()
    # group.add_argument("--ast", action="store_true", help="print IR")
    # group.add_argument("--mlir", action="store_true", help="print MLIR")
    # parser.add_argument("--interpret", action="store_true", help="Interpret the code")

    parser = argparse.ArgumentParser(description="aziz language")
    parser.add_argument("file", help="source file")
    parser.add_argument("--target", help="target dialect", default="riscv-lowered")
    parser.add_argument("--interpret", action="store_true", help="interpret the code")
    parser.add_argument("--print-op", action="store_true", help="print the operation after transformation")
    args = parser.parse_args()
    assert args.file.endswith(".aziz")
    src = Path(args.file).read_text()

    ctx = context()  # not actually used for interpreting
    module_ast = AzizParser(ctx, src).parse_module()  # source -> ast
    module_op = IRGen().ir_gen_module(module_ast)  # ast -> mlir

    if args.interpret:
        interpreter = Interpreter(module_op)
        interpreter.register_implementations(AzizFunctions())
        interpreter.call_op("main", ())
        return

    if args.print_op:
        transform(ctx, module_op, target=args.target)
        print(module_op)
        return

    # code = compile(program)
    # emulate_riscv(code)

    print(compile(src))


if __name__ == "__main__":
    main()
