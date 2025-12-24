from dataclasses import dataclass

from xdsl.utils.lexer import Location


@dataclass(slots=True)
class ExprAST:
    loc: Location


@dataclass(slots=True)
class NumberExprAST(ExprAST):
    val: int | float


@dataclass(slots=True)
class VariableExprAST(ExprAST):
    name: str


@dataclass(slots=True)
class StringExprAST(ExprAST):
    val: str


@dataclass(slots=True)
class BinaryExprAST(ExprAST):
    op: str
    lhs: ExprAST
    rhs: ExprAST


@dataclass(slots=True)
class CallExprAST(ExprAST):  # function call expression
    callee: str
    args: list[ExprAST]


@dataclass(slots=True)
class PrintExprAST(ExprAST):
    arg: ExprAST


@dataclass(slots=True)
class IfExprAST(ExprAST):
    cond: ExprAST
    then_expr: ExprAST
    else_expr: ExprAST


@dataclass(slots=True)
class PrototypeAST:  # function's signature without body
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


def dump(node: object, indent: int = 0) -> str:
    if not isinstance(node, (ExprAST, PrototypeAST, FunctionAST, ModuleAST)):
        return repr(node)

    ind = "  " * indent

    def fmt(v):
        if isinstance(v, list | tuple):
            return f"[\n{ind}    " + f",\n{ind}    ".join(dump(x, indent + 2) for x in v) + f"\n{ind}  ]" if v else "[]"
        return dump(v, indent + 1)

    fields = (f"{k}={fmt(getattr(node, k))}" for k in node.__dataclass_fields__ if k != "loc")
    return f"{type(node).__name__}(\n{ind}  " + f",\n{ind}  ".join(fields) + f"\n{ind})"
