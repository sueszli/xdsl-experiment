from pathlib import Path

from aziz import BinaryExprAST, CallExprAST, ExprAST, FunctionAST, IfExprAST, ModuleAST, NumberExprAST, PrintExprAST, PrototypeAST, StringExprAST, VariableExprAST
from lark import Lark, Token, Transformer, v_args
from xdsl.utils.lexer import Location, Position, Span

grammar = r"""
    start: top_level*

    ?top_level: defun | expr

    defun: "(" "defun" IDENTIFIER "(" args? ")" expr* ")"
    args: IDENTIFIER+

    ?expr: print_expr
         | if_expr
         | binary_expr
         | call_expr
         | atom

    print_expr: "(" "print" expr ")"
    if_expr: "(" "if" expr expr expr ")"
    
    # explicitly list binary operators to prioritize them over generic call
    binary_expr: "(" BINARY_OP expr expr ")"
    
    call_expr: "(" IDENTIFIER expr* ")"

    atom: NUMBER -> number
        | STRING -> string
        | IDENTIFIER -> variable

    BINARY_OP: "+" | "-" | "*" | "/" | "%" | "<=" | ">=" | "==" | "!=" | "<" | ">"
    
    COMMENT: /;[^\n]*/

    # primitives
    %import common.SIGNED_NUMBER -> NUMBER
    %import common.ESCAPED_STRING -> STRING
    %import common.WS
    %ignore WS
    %ignore COMMENT

    IDENTIFIER: /[a-zA-Z0-9_+\-*\/%<>=!?]+/
"""


class AzizTransformer(Transformer):
    def __init__(self, file: Path, program: str):
        self.file = str(file)
        self.program = program

    def _get_location(self, meta) -> Location:
        return Span(
            meta.start_pos,
            meta.end_pos,
            self._input,
        ).get_location()

    @property
    def _input(self):
        from xdsl.utils.lexer import Input

        if not hasattr(self, "__input"):
            self.__input = Input(self.program, self.file)
        return self.__input

    def start(self, items):
        return ModuleAST(tuple(items))

    def top_level(self, items):
        return items[0]

    @v_args(meta=True)
    def args(self, meta, items):
        return [str(item) for item in items]

    @v_args(meta=True)
    def defun(self, meta, items):
        name = str(items[0])
        args: list[str] = []
        body_start_index = 1

        # Check if second item is the args list (from args rule)
        if len(items) > 1 and isinstance(items[1], list):
            args = items[1]
            body_start_index = 2

        body = tuple(items[body_start_index:])
        loc = self._get_location(meta)
        proto = PrototypeAST(loc, name, args)
        return FunctionAST(loc, proto, body)

    @v_args(meta=True)
    def print_expr(self, meta, items):
        return PrintExprAST(self._get_location(meta), items[0])

    @v_args(meta=True)
    def if_expr(self, meta, items):
        return IfExprAST(self._get_location(meta), items[0], items[1], items[2])

    @v_args(meta=True)
    def binary_expr(self, meta, items):
        op = str(items[0])
        return BinaryExprAST(self._get_location(meta), op, items[1], items[2])

    @v_args(meta=True)
    def call_expr(self, meta, items):
        name = str(items[0])
        args = list(items[1:])
        return CallExprAST(self._get_location(meta), name, args)

    @v_args(meta=True)
    def number(self, meta, items):
        val = items[0]
        try:
            val = int(val)
        except ValueError:
            val = float(val)
        return NumberExprAST(self._get_location(meta), val)

    @v_args(meta=True)
    def string(self, meta, items):
        # strip quotes
        return StringExprAST(self._get_location(meta), str(items[0])[1:-1])

    @v_args(meta=True)
    def variable(self, meta, items):
        return VariableExprAST(self._get_location(meta), str(items[0]))


class AzizParser:
    def __init__(self, file: Path, program: str):
        self.file = file
        self.program = program
        self.parser = Lark(grammar, start="start", propagate_positions=True)

    def parse_module(self) -> ModuleAST:
        tree = self.parser.parse(self.program)
        return AzizTransformer(self.file, self.program).transform(tree)
