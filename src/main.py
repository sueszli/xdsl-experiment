# /// script
# requires-python = "==3.14"
# dependencies = [
#     "xdsl==0.55.4",
# ]
# ///

import sys
from parser import Parser

from xdsl.interpreter import Interpreter

from compiler import IRGen
from interpreter import AzizFunctions

assert len(sys.argv) == 2, "requires .aziz file as argument"
filename = sys.argv[1]

with open(filename, "r") as f:
    prog = f.read()

module_op = IRGen().ir_gen_module(Parser(prog, filename).parse_module())
print(f"\033[90m{module_op}\033[00m")

interpreter = Interpreter(module_op)
interpreter.register_implementations(AzizFunctions())
interpreter.call_op("main", ())
