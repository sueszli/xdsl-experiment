from pathlib import Path

from ast_nodes import BinaryExprAST, CallExprAST, ExprAST, FunctionAST, IfExprAST, ModuleAST, NumberExprAST, PrintExprAST, PrototypeAST, StringExprAST, VariableExprAST
from lexer import AzizLexer, AzizToken, AzizTokenKind
from xdsl.parser import GenericParser, ParserState
from xdsl.utils.lexer import Input

# parse with lark grammar instead? https://github.com/lark-parser/lark
class AzizParser(GenericParser[AzizTokenKind]):
    def __init__(self, file: Path, program: str):
        """
        (0) calling `AzizParser` constructor

            AzizParser(Path("./hello.aziz"), "(defun hello_world (...)")

        (1) first creates an input text wrapper

            Input("(defun hello_world (...)", "./hello.aziz")

        (2) then creates a lexer, inheriting from `Lexer`, that implements the `lex` method

            AzizLexer(Input(...))

        (3) then creates a parser state. calls `lex()` to get the first token

            ParserState(AzizLexer(...))
            └─> self.lexer = AzizLexer(...)
            └─> self.current_token = lexer.lex()
            └─> self.dialect_stack = ["builtin"]

        (4) finally calls GenericParser constructor

            super().__init__(ParserState(...))
            └─> self._parser_state = ParserState(...)

        (5) then we can call `.parse_module()` to parse the module
        """
        super().__init__(ParserState(AzizLexer(Input(program, str(file)))))

    def parse_module(self) -> ModuleAST:  # entry point
        # module ::= ( top_level* )
        ops = []
        while self._current_token.kind != AzizTokenKind.EOF:
            ops.append(self.parse_top_level())
        return ModuleAST(tuple(ops))

    def parse_top_level(self) -> FunctionAST | ExprAST:
        # top_level ::= definition | expression
        if self._current_token.kind == AzizTokenKind.PAREN_OPEN:  # don't `_pop()` yet
            return self.parse_list()
        return self.parse_atom()

    def parse_list(self) -> FunctionAST | ExprAST:
        # list ::= '(' ( definition | expression_list ) ')'
        # definition ::= 'defun' identifier '(' identifier* ')' expression*
        # expression_list ::= operator expression* | function_call
        paren_loc = self._pop(AzizTokenKind.PAREN_OPEN).span.get_location()
        if self._current_token.kind == AzizTokenKind.DEFUN:
            return self.parse_defun(paren_loc)
        return self.parse_expr_list_content(paren_loc)

    def parse_defun(self, loc) -> FunctionAST:
        # defun ::= '(' 'defun' name '(' args* ')' body* ')'
        self._pop(AzizTokenKind.DEFUN)
        name_token = self._pop(AzizTokenKind.IDENTIFIER)
        name = name_token.text

        self._pop(AzizTokenKind.PAREN_OPEN)

        args: list[str] = []
        while self._current_token.kind == AzizTokenKind.IDENTIFIER:
            args.append(self._pop(AzizTokenKind.IDENTIFIER).text)
        self._pop(AzizTokenKind.PAREN_CLOSE)

        body: list[ExprAST] = []
        while self._current_token.kind != AzizTokenKind.PAREN_CLOSE:
            body.append(self.parse_expression())

        self._pop(AzizTokenKind.PAREN_CLOSE)

        proto = PrototypeAST(loc, name, args)
        return FunctionAST(loc, proto, tuple(body))

    def parse_expression(self) -> ExprAST:
        # expression ::= list | atom
        if self._current_token.kind == AzizTokenKind.PAREN_OPEN:
            start_loc = self._pop(AzizTokenKind.PAREN_OPEN).span.get_location()
            return self.parse_expr_list_content(start_loc)
        return self.parse_atom()

    def parse_expr_list_content(self, start_loc) -> ExprAST:
        # expr_list_content ::= 'print' expression
        #                     | 'if' expression expression expression
        #                     | binary_op expression expression
        #                     | function_name expression*
        if self._current_token.kind == AzizTokenKind.PRINT:
            self._pop(AzizTokenKind.PRINT)
            arg = self.parse_expression()
            self._pop(AzizTokenKind.PAREN_CLOSE)
            return PrintExprAST(start_loc, arg)

        if self._current_token.kind == AzizTokenKind.IF:
            self._pop(AzizTokenKind.IF)
            cond = self.parse_expression()
            then_expr = self.parse_expression()
            else_expr = self.parse_expression()
            self._pop(AzizTokenKind.PAREN_CLOSE)
            return IfExprAST(start_loc, cond, then_expr, else_expr)

        if self._current_token.kind == AzizTokenKind.IDENTIFIER:
            head = self._pop(AzizTokenKind.IDENTIFIER)
            name = head.text

            # binary operators
            if name in ("+", "*", "-", "/", "%", "<", ">", "<=", ">=", "==", "!="):
                lhs = self.parse_expression()
                rhs = self.parse_expression()
                self._pop(AzizTokenKind.PAREN_CLOSE)
                return BinaryExprAST(start_loc, name, lhs, rhs)

            # generic function call
            args: list[ExprAST] = []
            while self._current_token.kind != AzizTokenKind.PAREN_CLOSE:
                args.append(self.parse_expression())
            self._pop(AzizTokenKind.PAREN_CLOSE)
            return CallExprAST(start_loc, name, args)

        # handle other cases or errors
        if self._current_token.kind == AzizTokenKind.PAREN_CLOSE:
            self.raise_error("empty lists (nil) not supported as expressions", self._current_token)

        self.raise_error(f"unexpected token in list: {self._current_token.kind}", self._current_token)

    def parse_atom(self) -> ExprAST:
        # atom ::= number | string | identifier
        token = self._current_token
        if token.kind == AzizTokenKind.NUMBER:
            self._consume_token()
            try:
                val = int(token.text)
            except ValueError:
                val = float(token.text)
            return NumberExprAST(token.span.get_location(), val)

        if token.kind == AzizTokenKind.STRING:
            self._consume_token()
            return StringExprAST(token.span.get_location(), token.text[1:-1])  # strip quotes

        if token.kind == AzizTokenKind.IDENTIFIER:
            self._consume_token()
            return VariableExprAST(token.span.get_location(), token.text)

        self.raise_error(f"unexpected token: {token.kind}", token)

    def _pop(self, kind: AzizTokenKind) -> AzizToken:  # verify type
        return self._parse_token(kind, f"expected {kind}")  # calls GenericParser._parse_token
