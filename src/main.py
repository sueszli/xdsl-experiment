# /// script
# requires-python = "==3.14"
# dependencies = [
#     "xdsl==0.55.4",
# ]
# ///

import sys
from parser import AzizParser
from pathlib import Path

from ast_nodes import dump

# from xdsl.interpreter import Interpreter

# from compiler import IRGen
# from interpreter import AzizFunctions

assert len(sys.argv) == 2
filename = sys.argv[1]
assert filename.endswith(".aziz")
prog = Path(filename).read_text()

parsed = AzizParser(Path(filename), prog).parse_module()
print(f"\n\033[90m{dump(parsed)}\033[00m\n", "-" * 80)

# module_op = IRGen().ir_gen_module(parsed)
# print(f"\n\033[90m{module_op}\033[00m\n", "-" * 80)

# interpreter = Interpreter(module_op)
# interpreter.register_implementations(AzizFunctions())
# interpreter.call_op("main", ())
