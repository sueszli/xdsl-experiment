# /// script
# requires-python = "==3.14"
# dependencies = [
#     "xdsl==0.55.4",
# ]
# ///

import sys
from parser import Parser
from pathlib import Path

from xdsl.interpreter import Interpreter

from ast_nodes import dump
from compiler import IRGen
from interpreter import AzizFunctions

assert len(sys.argv) == 2
filename = sys.argv[1]
assert filename.endswith(".aziz")
prog = Path(filename).read_text()

parsed = Parser(prog, filename).parse_module()
print(f"\033[90m{dump(parsed)}\033[00m")

module_op = IRGen().ir_gen_module(parsed)
print(f"\033[90m{module_op}\033[00m")

interpreter = Interpreter(module_op)
interpreter.register_implementations(AzizFunctions())
interpreter.call_op("main", ())
