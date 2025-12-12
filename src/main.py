# /// script
# requires-python = "==3.14"
# dependencies = [
#     "xdsl==0.55.4",
# ]
# ///

import argparse
import sys
import traceback

from xdsl.context import Context
from xdsl.dialects.builtin import Builtin
from xdsl.interpreter import Interpreter
from xdsl.parser import Parser as XDSLParser

from compiler import IRGen, Parser
from interpreter import AzizFunctions
from ops import Aziz


def main():
    parser = argparse.ArgumentParser(description="Aziz Language Compiler")
    parser.add_argument("file", help="Input file")
    parser.add_argument("--emit-mlir", action="store_true", help="Print the MLIR to stdout")
    parser.add_argument("--output", help="Output file for MLIR")
    args = parser.parse_args()

    try:
        with open(args.file, "r") as f:
            prog = f.read()
    except FileNotFoundError:
        print(f"Error: '{args.file}' not found.", file=sys.stderr)
        sys.exit(1)

    try:
        if args.file.endswith(".mlir"):
            ctx = Context()
            ctx.load_dialect(Builtin)
            ctx.load_dialect(Aziz)
            module_op = XDSLParser(ctx, prog).parse_module()
        else:
            module_op = IRGen().ir_gen_module(Parser(prog, args.file).parse_module())

        if args.output:
            with open(args.output, "w") as f:
                print(module_op, file=f)

        if args.emit_mlir:
            print(module_op)

        # Run by default unless we are just emitting mlir or outputting to file?
        # Actually user might want to run AND output.
        # Let's run by default.
        # To avoid running, one could add a --no-run flag, but for now I'll just run it.
        # If the user purely wants conversion, they use --output or --emit-mlir.
        # If I want to be safe, I can say: "If --emit-mlir or --output is present, DO NOT run unless --run is also present?"
        # But user wants "uv run aziz examples/test.aziz" -> run.

        # Simple default: always run?
        # Maybe check if --output or --emit-mlir IS NOT set?
        # But what if I want to "compile and run"?

        # I'll stick to: Always run.
        # If I strictly follow "interpret by default", then I should run.
        i = Interpreter(module_op)
        i.register_implementations(AzizFunctions())
        i.call_op("main", ())

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
