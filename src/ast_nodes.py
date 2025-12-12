from dataclasses import dataclass

from xdsl.utils.lexer import Location


@dataclass(slots=True)
class ExprAST:
    loc: Location


@dataclass(slots=True)
class NumberExprAST(ExprAST):
    val: int


@dataclass(slots=True)
class VariableExprAST(ExprAST):
    name: str


@dataclass(slots=True)
class BinaryExprAST(ExprAST):
    op: str
    lhs: ExprAST
    rhs: ExprAST


@dataclass(slots=True)
class CallExprAST(ExprAST):
    callee: str
    args: list[ExprAST]


@dataclass(slots=True)
class PrintExprAST(ExprAST):
    arg: ExprAST


@dataclass(slots=True)
class PrototypeAST:
    loc: Location
    name: str
    args: list[str]


@dataclass(slots=True)
class FunctionAST:
    loc: Location
    proto: PrototypeAST
    body: tuple[ExprAST, ...]


@dataclass(slots=True)
class ModuleAST:
    ops: tuple[FunctionAST | ExprAST, ...]
