# /// script
# requires-python = "==3.14"
# dependencies = [
#     "xdsl==0.55.4",
# ]
# ///

import argparse
from pathlib import Path

from xdsl.interpreter import Interpreter
from xdsl.context import Context
from xdsl.dialects import affine, arith, func, memref, printf, riscv, riscv_func, scf
from xdsl.dialects.builtin import Builtin

from frontend.ast_nodes import dump
from frontend.ir_gen import IRGen
from frontend.parser import AzizParser
from interpreter import AzizFunctions


parser = argparse.ArgumentParser(description="aziz language")
parser.add_argument("file", help="source file")
group = parser.add_mutually_exclusive_group()
group.add_argument("--ast", action="store_true", help="print IR")
group.add_argument("--mlir", action="store_true", help="print MLIR")
parser.add_argument("--interpret", action="store_true", help="Interpret the code")
args = parser.parse_args()
assert args.file.endswith(".aziz")
src = Path(args.file).read_text()

module_ast = AzizParser("in_memory", src).parse_module()  # source -> ast
if args.ast:
    print(dump(module_ast), "\n")

module_op = IRGen().ir_gen_module(module_ast)  # ast -> mlir
if args.mlir:
    print(module_op, "\n")

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
    # ctx.load_dialect(toy.Toy)
    return ctx

ctx = context()


# code = compile(program)
# emulate_riscv(code)


if args.interpret:
    interpreter = Interpreter(module_op)
    interpreter.register_implementations(AzizFunctions())
    interpreter.call_op("main", ())
