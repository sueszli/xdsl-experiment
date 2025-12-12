import re
from typing import List, Tuple

from xdsl.utils.lexer import Location

from ast_nodes import BinaryExprAST, CallExprAST, ExprAST, FunctionAST, ModuleAST, NumberExprAST, PrintExprAST, PrototypeAST, VariableExprAST


class Parser:
    def __init__(self, program: str, filename: str = "<string>"):
        self.filename = filename
        self.tokens = self._tokenize(program)
        self.pos = 0

    def _tokenize(self, text: str) -> List[Tuple[str, int, int]]:
        token_re = re.compile(r";[^\n]*|([()])|([^\s()]+)|\n")
        tokens, line, line_start = [], 1, 0
        for match in token_re.finditer(text):
            text_val, start = match.group(), match.start()
            if text_val.startswith(";") or not text_val.strip():
                if text_val == "\n":
                    line, line_start = line + 1, match.end()
                continue
            tokens.append((text_val, line, start - line_start + 1))
        return tokens

    def _peek(self, offset: int = 0):
        idx = self.pos + offset
        return self.tokens[idx] if idx < len(self.tokens) else None

    def _consume(self):
        if self.pos >= len(self.tokens):
            raise Exception("Unexpected EOF")
        t = self.tokens[self.pos]
        self.pos += 1
        return t

    def _expect(self, expected: str):
        if (t := self._peek()) and t[0] == expected:
            self._consume()
            return
        t_str = t[0] if t else "EOF"
        loc = Location(self.filename, t[1], t[2]) if t else Location(self.filename, 0, 0)
        raise Exception(f"Expected '{expected}', got '{t_str}' at {loc}")

    def _loc(self, token):
        return Location(self.filename, token[1], token[2])

    def parse_module(self) -> ModuleAST:
        ops = []
        while self._peek():
            if (t0 := self._peek()) and t0[0] == "(" and (t1 := self._peek(1)) and t1[0] == "defun":
                ops.append(self.parse_definition())
                continue
            ops.append(self.parse_expr())
        return ModuleAST(tuple(ops))

    def parse_definition(self) -> FunctionAST:
        self._expect("(")
        self._expect("defun")
        name_token = self._consume()
        self._expect("(")
        args = []
        while (t := self._peek()) and t[0] != ")":
            args.append(self._consume()[0])
        self._expect(")")

        body = []
        while (t := self._peek()) and t[0] != ")":
            body.append(self.parse_expr())
        self._expect(")")

        return FunctionAST(self._loc(name_token), PrototypeAST(self._loc(name_token), name_token[0], args), tuple(body))

    def parse_expr(self) -> ExprAST:
        if not (t := self._peek()):
            raise Exception("Unexpected EOF in expr")

        if t[0] == "(":
            self._consume()
            if not (head := self._peek()):
                raise Exception("Unexpected EOF in list")

            if head[0] == "print":
                self._consume()
                arg = self.parse_expr()
                self._expect(")")
                return PrintExprAST(self._loc(head), arg)

            if head[0] in ("+", "*"):
                op = self._consume()
                lhs, rhs = self.parse_expr(), self.parse_expr()
                self._expect(")")
                return BinaryExprAST(self._loc(op), head[0], lhs, rhs)

            callee, args = self._consume(), []
            while (pt := self._peek()) and pt[0] != ")":
                args.append(self.parse_expr())
            self._expect(")")
            return CallExprAST(self._loc(callee), callee[0], args)

        self._consume()
        if t[0].isdigit():
            return NumberExprAST(self._loc(t), int(t[0]))
        return VariableExprAST(self._loc(t), t[0])
