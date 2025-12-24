# /// script
# requires-python = "==3.14"
# dependencies = [
#     "xdsl==0.56.0",
#     "unicorn==2.1.4",
#     "pyelftools==0.32",
#     "lark==1.3.1",
# ]
# ///

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from lark import Lark, Transformer, v_args
from xdsl.builder import Builder, InsertPoint
from xdsl.context import Context
from xdsl.dialects import affine, arith, builtin, func, llvm, printf, scf
from xdsl.dialects.builtin import AnyFloat, ArrayAttr, FloatAttr, FunctionType, IntegerAttr, IntegerType, ModuleOp, StringAttr, SymbolNameConstraint, SymbolRefAttr, f64, i8, i32
from xdsl.ir import Attribute, Block, Dialect, ParametrizedAttribute, Region, SSAValue
from xdsl.irdl import AnyOf, IRDLOperation, attr_def, irdl_attr_definition, irdl_op_definition, operand_def, opt_attr_def, opt_operand_def, region_def, result_def, traits_def, var_operand_def, var_result_def
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import GreedyRewritePatternApplier, PatternRewriter, PatternRewriteWalker, RewritePattern, op_type_rewrite_pattern
from xdsl.traits import CallableOpInterface, HasParent, IsTerminator, Pure, SymbolOpInterface, SymbolTable
from xdsl.transforms.canonicalize import CanonicalizePass
from xdsl.transforms.dead_code_elimination import dce
from xdsl.transforms.lower_affine import LowerAffinePass
from xdsl.utils.lexer import Input, Location, Span
from xdsl.utils.scoped_dict import ScopedDict

# ==============================================================================================================
# ast
# ==============================================================================================================


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
class CallExprAST(ExprAST):
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


# ==============================================================================================================
# parser
# ==============================================================================================================

GRAMMAR = r"""
start: top_level*
?top_level: defun | expr
defun: "(" "defun" IDENTIFIER "(" args? ")" expr* ")"
args: IDENTIFIER+
?expr: print_expr | if_expr | binary_expr | call_expr | atom
print_expr: "(" "print" expr ")"
if_expr: "(" "if" expr expr expr ")"
binary_expr: "(" BINARY_OP expr expr ")"
call_expr: "(" IDENTIFIER expr* ")"
atom: NUMBER -> number | STRING -> string | IDENTIFIER -> variable
BINARY_OP: "+" | "-" | "*" | "/" | "%" | "<=" | ">=" | "==" | "!=" | "<" | ">"
COMMENT: /;[^\n]*/
%import common.SIGNED_NUMBER -> NUMBER
%import common.ESCAPED_STRING -> STRING
%import common.WS
%ignore WS
%ignore COMMENT
IDENTIFIER.-1: /[a-zA-Z0-9_+\-*\/%<>=!?]+/
"""


class AzizTransformer(Transformer):
    def __init__(self, file: Path, program: str):
        self.file, self.program = str(file), program

    def _loc(self, meta) -> Location:
        return Span(meta.start_pos, meta.end_pos, self._input).get_location()

    @property
    def _input(self):
        if not hasattr(self, "__input"):
            self.__input = Input(self.program, self.file)
        return self.__input

    def start(self, items):
        return ModuleAST(tuple(items))

    def top_level(self, items):
        return items[0]

    @v_args(meta=True)
    def args(self, _, items):
        return [str(item) for item in items]

    @v_args(meta=True)
    def defun(self, meta, items):
        name = str(items[0])
        args, body = (items[1], items[2:]) if len(items) > 1 and isinstance(items[1], list) else ([], items[1:])
        return FunctionAST(self._loc(meta), PrototypeAST(self._loc(meta), name, args), tuple(body))

    @v_args(meta=True)
    def print_expr(self, meta, items):
        return PrintExprAST(self._loc(meta), items[0])

    @v_args(meta=True)
    def if_expr(self, meta, items):
        return IfExprAST(self._loc(meta), items[0], items[1], items[2])

    @v_args(meta=True)
    def binary_expr(self, meta, items):
        return BinaryExprAST(self._loc(meta), str(items[0]), items[1], items[2])

    @v_args(meta=True)
    def call_expr(self, meta, items):
        return CallExprAST(self._loc(meta), str(items[0]), list(items[1:]))

    @v_args(meta=True)
    def number(self, meta, items):
        return NumberExprAST(self._loc(meta), int(items[0]) if items[0].isdigit() or items[0].lstrip("-").isdigit() else float(items[0]))

    @v_args(meta=True)
    def string(self, meta, items):
        return StringExprAST(self._loc(meta), str(items[0])[1:-1])

    @v_args(meta=True)
    def variable(self, meta, items):
        return VariableExprAST(self._loc(meta), str(items[0]))


class AzizParser:
    def __init__(self, file: Path, program: str):
        self.tf = AzizTransformer(file, program)
        self.parser = Lark(GRAMMAR, start="start", propagate_positions=True)

    def parse_module(self) -> ModuleAST:
        return self.tf.transform(self.parser.parse(self.tf.program))


# ==============================================================================================================
# irgen
# ==============================================================================================================


@dataclass(init=False)
class IRGen:
    module: ModuleOp
    builder: Builder
    symbol_table: ScopedDict[str, SSAValue] | None = None

    def __init__(self):
        self.module = ModuleOp([])
        self.builder = Builder(InsertPoint.at_end(self.module.body.blocks[0]))

    def ir_gen_module(self, module_ast: ModuleAST) -> ModuleOp:
        functions = [op for op in module_ast.ops if isinstance(op, FunctionAST)]
        if main_body := [op for op in module_ast.ops if isinstance(op, ExprAST)]:
            functions.append(FunctionAST(main_body[0].loc, PrototypeAST(main_body[0].loc, "main", []), tuple(main_body)))

        signatures = self._collect_signatures(module_ast)
        inferred = self._resolve_signatures(signatures)
        for func in functions:
            self._declare_function(func, inferred)
        for func in functions:
            self._ir_gen_function(func)
        self.module.verify()
        return self.module

    def _collect_signatures(self, module_ast: ModuleAST) -> dict[str, list[list[Attribute]]]:
        sigs = {}

        def _type_of(n):
            return StringType() if isinstance(n, StringExprAST) else (f64 if isinstance(n.val, float) else i32) if isinstance(n, NumberExprAST) else None

        def visit(node):
            if isinstance(node, CallExprAST):
                arg_types = [_type_of(arg) for arg in node.args]
                if all(arg_types):
                    sigs.setdefault(node.callee, []).append(arg_types)
            if hasattr(node, "__dataclass_fields__"):
                for f in node.__dataclass_fields__:
                    visit(getattr(node, f))
            elif isinstance(node, (list, tuple)):
                for i in node:
                    visit(i)

        visit(module_ast)
        return sigs

    def _resolve_signatures(self, signatures):
        resolved = {}
        for name, sigs in signatures.items():
            if not sigs:
                continue
            final = list(sigs[0])
            for sig in sigs[1:]:
                assert len(sig) == len(final), f"arity mismatch for {name}"
                for i, (t1, t2) in enumerate(zip(final, sig)):
                    if t1 == t2:
                        continue
                    assert {t1, t2} == {i32, f64}, f"type mismatch arg {i} for {name}: {t1} vs {t2}"
                    final[i] = f64
            resolved[name] = final
        return resolved

    def _declare_function(self, func_ast, inferred):
        name, args = func_ast.proto.name, func_ast.proto.args
        arg_types = inferred.get(name, [i32] * len(args))
        if len(arg_types) != len(args):
            arg_types = [i32] * len(args)
        ftype = FunctionType.from_lists(arg_types, [i32])
        self.builder.insert(FuncOp(name, ftype, Region(Block(arg_types=arg_types)), private=name != "main"))

    def _ir_gen_function(self, func_ast):
        name = func_ast.proto.name
        func_op = next(op for op in self.module.body.blocks[0].ops if isinstance(op, FuncOp) and op.sym_name.data == name)
        block = func_op.body.blocks[0]
        parent_builder, self.builder, self.symbol_table = self.builder, Builder(InsertPoint.at_end(block)), ScopedDict()

        try:
            for n, v in zip(func_ast.proto.args, block.args):
                self.symbol_table[n] = v
            last_val = None
            for expr in func_ast.body:
                last_val = self._ir_gen_expr(expr)
            if not isinstance(block.last_op, ReturnOp):
                self.builder.insert(ReturnOp(last_val if last_val else self.builder.insert(ConstantOp(0)).res))

            # update return type if inferred differently
            if isinstance(block.last_op, ReturnOp) and (ret := block.last_op.input):
                if list(func_op.function_type.outputs.data) != [ret.type]:
                    func_op.function_type = FunctionType.from_lists(func_op.function_type.inputs.data, [ret.type])
        finally:
            self.builder, self.symbol_table = parent_builder, None

    def _ir_gen_expr(self, expr) -> SSAValue | None:
        match expr:
            case BinaryExprAST(op=op, lhs=lhs, rhs=rhs):
                l, r = self._ir_gen_expr(lhs), self._ir_gen_expr(rhs)
                cls = {"+": AddOp, "-": SubOp, "*": MulOp, "<=": LessThanEqualOp}.get(op)
                assert cls, f"unknown binary op {op}"
                return self.builder.insert(cls(l, r)).res
            case NumberExprAST(val=v):
                return self.builder.insert(ConstantOp(v)).res
            case VariableExprAST(name=name):
                assert self.symbol_table and name in self.symbol_table, f"undefined var {name}"
                return self.symbol_table[name]
            case PrintExprAST(arg=arg):
                self.builder.insert(PrintOp(self._ir_gen_expr(arg)))
                return None
            case StringExprAST(val=val):
                return self.builder.insert(StringConstantOp(val)).res
            case IfExprAST(cond=cond, then_expr=then_e, else_expr=else_e):
                c_val = self._ir_gen_expr(cond)
                orig = self.builder

                t_blk, e_blk = Block(), Block()
                self.builder = Builder(InsertPoint.at_end(t_blk))
                self.builder.insert(YieldOp(self._ir_gen_expr(then_e)))
                self.builder = Builder(InsertPoint.at_end(e_blk))
                self.builder.insert(YieldOp(self._ir_gen_expr(else_e)))

                self.builder = orig
                return self.builder.insert(IfOp(c_val, t_blk.last_op.input.type, [Region(t_blk), Region(e_blk)])).res

            case CallExprAST(callee=callee, args=args):
                func_op = next((op for op in self.module.body.blocks[0].ops if isinstance(op, FuncOp) and op.sym_name.data == callee), None)
                assert func_op, f"unknown function {callee}"

                arg_vals = [self._ir_gen_expr(a) for a in args]
                final_args = []
                for i, (arg, exp) in enumerate(zip(arg_vals, func_op.function_type.inputs.data)):
                    if arg.type == exp:
                        final_args.append(arg)
                    elif arg.type == i32 and exp == f64:
                        final_args.append(self.builder.insert(CastIntToFloatOp(arg)).res)
                    else:
                        assert False, f"type mismatch arg {i} for {callee}"

                call = CallOp(callee, final_args, func_op.function_type.outputs.data[0])
                return self.builder.insert(call).res[0]
            case _:
                assert False, f"unknown expr {expr}"


# ==============================================================================================================
# dialect
# ==============================================================================================================


@irdl_attr_definition
class StringType(ParametrizedAttribute):
    name = "aziz.string"


@irdl_op_definition
class ConstantOp(IRDLOperation):
    name, traits = "aziz.constant", traits_def(Pure())
    value = attr_def(Attribute)
    res = result_def(AnyOf([IntegerType, AnyFloat]))

    def __init__(self, value: int | float):
        attr = FloatAttr(value, f64) if isinstance(value, float) else IntegerAttr(value, i32)
        super().__init__(result_types=[attr.type], attributes={"value": attr})


@irdl_op_definition
class StringConstantOp(IRDLOperation):
    name, traits = "aziz.string_constant", traits_def(Pure())
    value = attr_def(StringAttr)
    res = result_def(StringType)

    def __init__(self, value: str):
        super().__init__(result_types=[StringType()], attributes={"value": StringAttr(value)})


class _BinaryOpMixin:
    def verify_(self):
        assert self.lhs.type == self.rhs.type, f"expected {self.name} args to have the same type"


@irdl_op_definition
class AddOp(IRDLOperation, _BinaryOpMixin):
    name, traits = "aziz.add", traits_def(Pure())
    lhs = operand_def(AnyOf([IntegerType, AnyFloat]))
    rhs = operand_def(AnyOf([IntegerType, AnyFloat]))
    res = result_def(AnyOf([IntegerType, AnyFloat]))

    def __init__(self, lhs: SSAValue, rhs: SSAValue):
        super().__init__(operands=[lhs, rhs], result_types=[lhs.type])


@irdl_op_definition
class SubOp(IRDLOperation, _BinaryOpMixin):
    name, traits = "aziz.sub", traits_def(Pure())
    lhs = operand_def(AnyOf([IntegerType, AnyFloat]))
    rhs = operand_def(AnyOf([IntegerType, AnyFloat]))
    res = result_def(AnyOf([IntegerType, AnyFloat]))

    def __init__(self, lhs: SSAValue, rhs: SSAValue):
        super().__init__(operands=[lhs, rhs], result_types=[lhs.type])


@irdl_op_definition
class MulOp(IRDLOperation, _BinaryOpMixin):
    name, traits = "aziz.mul", traits_def(Pure())
    lhs = operand_def(AnyOf([IntegerType, AnyFloat]))
    rhs = operand_def(AnyOf([IntegerType, AnyFloat]))
    res = result_def(AnyOf([IntegerType, AnyFloat]))

    def __init__(self, lhs: SSAValue, rhs: SSAValue):
        super().__init__(operands=[lhs, rhs], result_types=[lhs.type])


@irdl_op_definition
class LessThanEqualOp(IRDLOperation, _BinaryOpMixin):
    name, traits = "aziz.le", traits_def(Pure())
    lhs = operand_def(AnyOf([IntegerType, AnyFloat]))
    rhs = operand_def(AnyOf([IntegerType, AnyFloat]))
    res = result_def(IntegerType)

    def __init__(self, lhs: SSAValue, rhs: SSAValue):
        super().__init__(operands=[lhs, rhs], result_types=[i32])


@irdl_op_definition
class PrintOp(IRDLOperation):
    name = "aziz.print"
    input = operand_def(AnyOf([IntegerType, AnyFloat, StringType]))

    def __init__(self, input: SSAValue):
        super().__init__(operands=[input])


class FuncOpCallableInterface(CallableOpInterface):
    @classmethod
    def get_callable_region(cls, op: IRDLOperation) -> Region:
        return cast(FuncOp, op).body

    @classmethod
    def get_argument_types(cls, op: IRDLOperation) -> tuple[Attribute, ...]:
        return cast(FuncOp, op).function_type.inputs.data

    @classmethod
    def get_result_types(cls, op: IRDLOperation) -> tuple[Attribute, ...]:
        return cast(FuncOp, op).function_type.outputs.data


@irdl_op_definition
class FuncOp(IRDLOperation):
    name = "aziz.func"
    body = region_def("single_block")
    sym_name = attr_def(SymbolNameConstraint())
    function_type = attr_def(FunctionType)
    sym_visibility = opt_attr_def(StringAttr)
    traits = traits_def(SymbolOpInterface(), FuncOpCallableInterface())

    def __init__(self, name: str, ftype: FunctionType, region: Region | type[Region.DEFAULT] = Region.DEFAULT, /, private: bool = False):
        attrs = {"sym_name": StringAttr(name), "function_type": ftype}
        if not isinstance(region, Region):
            region = Region(Block(arg_types=ftype.inputs))
        if private:
            attrs["sym_visibility"] = StringAttr("private")
        super().__init__(attributes=attrs, regions=[region])

    def verify_(self):
        assert self.body.block.ops, "expected FuncOp to not be empty"
        assert isinstance(self.body.block.last_op, ReturnOp), "expected last op of FuncOp to be a ReturnOp"


@irdl_op_definition
class ReturnOp(IRDLOperation):
    name = "aziz.return"
    input = opt_operand_def(AnyOf([IntegerType, AnyFloat, StringType]))
    traits = traits_def(IsTerminator(), HasParent(FuncOp))

    def __init__(self, input: SSAValue | None = None):
        super().__init__(operands=[input] if input else [])

    def verify_(self) -> None:
        func_op = cast(FuncOp, self.parent_op())
        rets = func_op.function_type.outputs.data
        if self.input:
            assert len(rets) == 1 and self.input.type == rets[0], "expected 1 return value matching function type"
        elif len(rets) != 0:
            assert False, "expected 0 return values for void function"


@irdl_op_definition
class CallOp(IRDLOperation):
    name = "aziz.call"
    callee = attr_def(SymbolRefAttr)
    arguments = var_operand_def(AnyOf([IntegerType, AnyFloat, StringType]))
    res = var_result_def(AnyOf([IntegerType, AnyFloat, StringType]))

    def __init__(self, callee: str | SymbolRefAttr, operands: Sequence[SSAValue], return_types: Sequence[Attribute]):
        if isinstance(callee, str):
            callee = SymbolRefAttr(callee)
        super().__init__(operands=[operands], result_types=[return_types], attributes={"callee": callee})


@irdl_op_definition
class YieldOp(IRDLOperation):
    name, traits = "aziz.yield", traits_def(IsTerminator())
    input = operand_def(AnyOf([IntegerType, AnyFloat, StringType]))

    def __init__(self, input: SSAValue):
        super().__init__(operands=[input])


@irdl_op_definition
class IfOp(IRDLOperation):
    name = "aziz.if"
    cond = operand_def(IntegerType)
    res = result_def(AnyOf([IntegerType, AnyFloat, StringType]))
    then_region, else_region = region_def(), region_def()

    def __init__(self, cond: SSAValue, result_type: Attribute = i32, regions: list[Region] | None = None):
        super().__init__(operands=[cond], result_types=[result_type], regions=regions or [Region(Block()), Region(Block())])

    def verify_(self) -> None:
        assert isinstance(self.then_region.block.last_op, YieldOp) and isinstance(self.else_region.block.last_op, YieldOp), "expected last op of regions to be YieldOp"
        assert self.then_region.block.last_op.input.type == self.else_region.block.last_op.input.type, "if expression branches have different types"


@irdl_op_definition
class CastIntToFloatOp(IRDLOperation):
    name, traits = "aziz.cast_int_to_float", traits_def(Pure())
    input, res = operand_def(IntegerType), result_def(AnyFloat)

    def __init__(self, input: SSAValue):
        super().__init__(operands=[input], result_types=[f64])


Aziz = Dialect("aziz", [ConstantOp, StringConstantOp, AddOp, SubOp, MulOp, LessThanEqualOp, PrintOp, FuncOp, ReturnOp, CallOp, YieldOp, IfOp, CastIntToFloatOp], [StringType])


# ==============================================================================================================
# rewrites
# ==============================================================================================================


class InlineFunctions(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: CallOp, rewriter: PatternRewriter):
        if not (callee := SymbolTable.lookup_symbol(op, op.callee)):
            return
        if any(isinstance(c, CallOp) and c.callee.string_value() == callee.sym_name.data for c in callee.walk()):
            return  # recursive
        if not (iface := callee.get_trait(CallableOpInterface)) or len(iface.get_callable_region(callee).blocks) != 1:
            return

        blk = iface.get_callable_region(callee).clone().block
        for opr, arg in zip(op.operands, blk.args):
            arg.replace_by(opr)
        while blk.args:
            rewriter.erase_block_argument(blk.args[-1])
        rewriter.inline_block(blk, InsertPoint.before(op))

        ret = op.prev_op
        rewriter.replace_op(op, [], [ret.input] if ret.input else [])
        rewriter.erase_op(ret)


class RemoveUnusedPrivateFunctions(RewritePattern):
    _used: set[str] | None = None

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: FuncOp, rewriter: PatternRewriter):
        if op.sym_visibility != StringAttr("private"):
            return
        if self._used is None:
            self._used = {c.callee.string_value() for c in op.parent_op().walk() if isinstance(c, CallOp)}
        if op.sym_name.data not in self._used:
            rewriter.erase_op(op)


class OptimizeAzizPass(ModulePass):
    name = "optimize-aziz"

    def apply(self, _, op: ModuleOp):
        PatternRewriteWalker(InlineFunctions()).rewrite_module(op)
        PatternRewriteWalker(RemoveUnusedPrivateFunctions()).rewrite_module(op)
        dce(op)


class ArithOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: AddOp | SubOp | MulOp, rewriter: PatternRewriter):
        tbl = {AddOp: (arith.AddfOp, arith.AddiOp), SubOp: (arith.SubfOp, arith.SubiOp), MulOp: (arith.MulfOp, arith.MuliOp)}
        f_cls, i_cls = tbl[type(op)]
        rewriter.replace_op(op, f_cls(op.lhs, op.rhs) if isinstance(op.lhs.type, AnyFloat) else i_cls(op.lhs, op.rhs))


class LessThanEqualOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: LessThanEqualOp, rewriter: PatternRewriter):
        if isinstance(op.lhs.type, AnyFloat):
            return rewriter.replace_op(op, arith.CmpfOp(op.lhs, op.rhs, "ole"))
        lt = rewriter.insert_op(arith.CmpiOp(op.rhs, op.lhs, "slt"), InsertPoint.before(op))  # (b < a)
        z = rewriter.insert_op(arith.ConstantOp(IntegerAttr(0, IntegerType(1))), InsertPoint.before(op))
        rewriter.replace_op(op, arith.CmpiOp(lt.result, z.result, "eq"))


class CastIntToFloatOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: CastIntToFloatOp, rewriter: PatternRewriter):
        rewriter.replace_op(op, arith.SIToFPOp(op.input, op.res.type))


class ConstantOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ConstantOp, rewriter: PatternRewriter):
        rewriter.replace_op(op, arith.ConstantOp(op.value))


class IfOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: IfOp, rewriter: PatternRewriter):
        cond = op.cond
        if cond.type.width.data != 1:
            z = rewriter.insert_op(arith.ConstantOp(IntegerAttr(0, cond.type)), InsertPoint.before(op))
            cond = rewriter.insert_op(arith.CmpiOp(cond, z.result, "ne"), InsertPoint.before(op)).result
        rewriter.replace_op(op, scf.IfOp(cond, [r.type for r in op.results], rewriter.move_region_contents_to_new_regions(op.then_region), rewriter.move_region_contents_to_new_regions(op.else_region)))


class ReturnOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ReturnOp, rewriter: PatternRewriter):
        rewriter.replace_op(op, func.ReturnOp(op.input) if op.input else func.ReturnOp())


class FuncOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: FuncOp, rewriter: PatternRewriter):
        conv = lambda t: llvm.LLVMPointerType() if isinstance(t, StringType) else t
        inputs, outputs = [conv(t) for t in op.function_type.inputs], [conv(t) for t in op.function_type.outputs]
        new_body = rewriter.move_region_contents_to_new_regions(op.body)
        for block in new_body.blocks:
            new_args = [conv(a.type) for a in block.args]
            if new_args != [a.type for a in block.args]:
                nb = Block(arg_types=new_args)
                for oa, na in zip(block.args, nb.args):
                    oa.replace_by(na)
                for o in list(block.ops):
                    o.detach()
                    nb.add_op(o)
                new_body.detach_block(block)
                new_body.add_block(nb)
        rewriter.replace_op(op, func.FuncOp(op.sym_name.data, func.FunctionType.from_lists(inputs, outputs), new_body, visibility=op.sym_visibility))


class CallOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: CallOp, rewriter: PatternRewriter):
        conv = lambda t: llvm.LLVMPointerType() if isinstance(t, StringType) else t
        rewriter.replace_op(op, func.CallOp(op.callee, op.arguments, [conv(t) for t in op.res.types]))


class YieldOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: YieldOp, rewriter: PatternRewriter):
        rewriter.replace_op(op, scf.YieldOp(op.input))


class PrintOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: PrintOp, rewriter: PatternRewriter):
        rewriter.replace_op(op, printf.PrintFormatOp("{}", op.input))


class StringConstantOpLowering(RewritePattern):
    _cache, _cnt = {}, 0

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: StringConstantOp, rewriter: PatternRewriter):
        s = op.value.data
        if s not in self._cache:
            mod = op.parent_op()
            while not isinstance(mod, ModuleOp):
                mod = mod.parent_op()
            name = f".aziz.str.{self._cnt}"
            self._cnt += 1
            self._cache[s] = name
            data = s.encode("utf-8") + b"\0"
            g = llvm.GlobalOp(llvm.LLVMArrayType.from_size_and_type(len(data), i8), StringAttr(name), linkage=llvm.LinkageAttr("internal"), constant=True, value=ArrayAttr([IntegerAttr(b, i8) for b in data]))
            mod.body.blocks[0].insert_op_before(g, mod.body.blocks[0].first_op)
        rewriter.replace_op(op, [], [rewriter.insert_op(llvm.AddressOfOp(self._cache[s], llvm.LLVMPointerType()), InsertPoint.before(op)).result])


class LowerAzizPass(ModulePass):
    name = "lower-aziz"

    def apply(self, _, op: ModuleOp):
        PatternRewriteWalker(GreedyRewritePatternApplier([ArithOpLowering(), LessThanEqualOpLowering(), CastIntToFloatOpLowering(), ConstantOpLowering(), ReturnOpLowering(), FuncOpLowering(), CallOpLowering(), IfOpLowering(), YieldOpLowering(), StringConstantOpLowering(), PrintOpLowering()])).rewrite_module(op)


class PrintFormatOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: printf.PrintFormatOp, rewriter: PatternRewriter):
        if not op.operands:
            return rewriter.erase_op(op)
        val, mod = op.operands[0], op.get_toplevel_object()
        fmt, hint = ("%s\n", "s") if isinstance(val.type, llvm.LLVMPointerType) else ("%d\n", "d") if isinstance(val.type, (builtin.IntegerType, builtin.IndexType)) else ("%f\n", "f")

        # ensure global
        gname = f"__str_fmt_{hint}"
        if not (g := next((o for o in mod.body.blocks[0].ops if isinstance(o, llvm.GlobalOp) and o.sym_name.data == gname), None)):
            d = fmt.encode("utf-8") + b"\0"
            g = llvm.GlobalOp(llvm.LLVMArrayType.from_size_and_type(len(d), i8), StringAttr(gname), linkage=llvm.LinkageAttr("internal"), constant=True, value=ArrayAttr([IntegerAttr(b, i8) for b in d]))
            mod.body.blocks[0].insert_op_before(g, mod.body.blocks[0].first_op)

        # casts
        if isinstance(val.type, (builtin.Float32Type, builtin.Float64Type)) and not isinstance(val.type, builtin.Float64Type):
            val = rewriter.insert_op(arith.ExtFOp(val, builtin.f64), InsertPoint.before(op)).result
        elif isinstance(val.type, (builtin.IntegerType, builtin.IndexType)):
            if isinstance(val.type, builtin.IndexType):
                val = rewriter.insert_op(arith.index_cast(val, builtin.i32), InsertPoint.before(op)).result
            elif val.type.width.data != 32:
                val = rewriter.insert_op(arith.TruncIOp(val, builtin.i32) if val.type.width.data > 32 else arith.ExtSIOp(val, builtin.i32), InsertPoint.before(op)).result

        fmt_ptr = rewriter.insert_op(llvm.AddressOfOp(g.sym_name, llvm.LLVMPointerType()), InsertPoint.before(op)).results[0]
        call = llvm.CallOp(SymbolRefAttr("printf"), fmt_ptr, val, return_type=builtin.i32)
        call.attributes["var_callee_type"] = llvm.LLVMFunctionType([llvm.LLVMPointerType()], builtin.i32, is_variadic=True)
        rewriter.insert_op(call, InsertPoint.before(op))
        rewriter.erase_op(op)


class LowerPrintfToLLVMCallPass(ModulePass):
    name = "lower-printf-to-llvm-call"

    def apply(self, _, op: ModuleOp):
        if not any(o.sym_name.data == "printf" for o in op.body.blocks[0].ops if hasattr(o, "sym_name")):
            op.body.blocks[0].add_op(llvm.FuncOp("printf", llvm.LLVMFunctionType([llvm.LLVMPointerType()], builtin.i32, is_variadic=True), linkage=llvm.LinkageAttr("external")))
        PatternRewriteWalker(PrintFormatOpLowering()).rewrite_module(op)


# ==============================================================================================================
# main
# ==============================================================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="aziz language")
    parser.add_argument("file", help="source file")
    parser.add_argument("--debug", action="store_true", help="emit intermediate mlir representations")
    args = parser.parse_args()
    assert Path(args.file).is_file() and Path(args.file).suffix == ".aziz", "input file must be a .aziz file"

    module_ast = AzizParser(None, Path(args.file).read_text()).parse_module()
    module_op = IRGen().ir_gen_module(module_ast)

    if args.debug:
        print(f"\n\n{'-' * 80} MLIR before optimization\n\n{module_op}")

    ctx = Context()
    for d in [Aziz, affine.Affine, arith.Arith, builtin.Builtin, func.Func, printf.Printf, scf.Scf, llvm.LLVM]:
        ctx.load_dialect(d)

    OptimizeAzizPass().apply(ctx, module_op)
    module_op.verify()
    if args.debug:
        print(f"\n\n{'-' * 80} MLIR after optimization\n\n{module_op}")

    LowerAzizPass().apply(ctx, module_op)
    LowerAffinePass().apply(ctx, module_op)
    CanonicalizePass().apply(ctx, module_op)
    LowerPrintfToLLVMCallPass().apply(ctx, module_op)
    module_op.verify()

    if args.debug:
        print(f"\n\n{'-' * 80} MLIR after lowering to LLVM dialect\n\n")
    print(module_op)
