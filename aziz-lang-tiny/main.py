# /// script
# requires-python = "==3.14"
# dependencies = [
#     "xdsl==0.56.0",
#     "lark==1.3.1",
# ]
# ///

import argparse
from functools import lru_cache
from pathlib import Path

from lark import Lark, Tree
from xdsl.builder import Builder, InsertPoint
from xdsl.context import Context
from xdsl.dialects import arith, builtin, func, llvm, scf
from xdsl.dialects.builtin import ArrayAttr, FloatAttr, FunctionType, IntegerAttr, ModuleOp, StringAttr
from xdsl.dialects.builtin import SymbolRefAttr as SymbolAttr
from xdsl.dialects.builtin import f64, i8, i32
from xdsl.ir import Block, Region
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriteWalker, RewritePattern, op_type_rewrite_pattern
from xdsl.transforms.dead_code_elimination import dce

#
# parsing + ir gen
#

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


class IRGen:
    def __init__(self):
        self.module = ModuleOp([])
        self.builder = Builder(InsertPoint.at_end(self.module.body.blocks[0]))  # string builder
        self.symbol_table = {}  # arg name -> SSAValue in current scope (manually reset per function)
        self.str_cache = {}  # reference string consts on dup values
        self.str_cnt = 0  # unique id gen for const strings

        # declare printf signature from libc, so it can be called
        self.builder.insert(llvm.FuncOp("printf", llvm.LLVMFunctionType([llvm.LLVMPointerType()], builtin.i32, is_variadic=True), linkage=llvm.LinkageAttr("external")))

    def gen(self, tree: Lark) -> ModuleOp:
        # generate functions
        funcs = [n for n in tree.children if n.data == "defun"]
        inferred_arg_types = self._get_func_signatures()
        for f in funcs:
            func_name = str(f.children[0])
            nodes = f.children[1:]
            arg_nodes = next((n for n in nodes if n.data == "args"), None)
            arg_names = [str(t) for t in arg_nodes.children] if arg_nodes else []
            body_nodes = [n for n in nodes if n.data != "args"]

            arg_types = inferred_arg_types.get(func_name, [i32] * len(arg_names))  # default to i32
            ftype = FunctionType.from_lists(arg_types, [i32])  # assume 1x i32 return

            # create mlir func_op
            entry = Block(arg_types=arg_types)
            func_op = func.FuncOp(func_name, ftype, Region(entry))
            self.module.body.blocks[0].add_op(func_op)

            # enter function scope
            prev_builder = self.builder
            self.builder = Builder(InsertPoint.at_end(entry))
            self.symbol_table = {n: a for n, a in zip(arg_names, entry.args)}  # arg names -> ssa values

            # create mlir return
            last_val = self._gen_expr(body_nodes[-1]) if body_nodes else None
            const_0 = self.builder.insert(arith.ConstantOp(IntegerAttr(0, i32))).results[0]
            ret_val = last_val if last_val else const_0  # default return 0
            self.builder.insert(func.ReturnOp(ret_val))
            func_op.function_type = FunctionType.from_lists(arg_types, [ret_val.type])

            # exit function scope
            self.builder = prev_builder

        # create mlir main
        main_exprs = [n for n in tree.children if n.data != "defun"]
        if main_exprs:
            entry = Block()
            func_op = func.FuncOp("main", FunctionType.from_lists([], [i32]), Region(entry))
            self.module.body.blocks[0].add_op(func_op)
            prev_builder = self.builder
            self.builder = Builder(InsertPoint.at_end(entry))
            self.symbol_table = {}
            for expr in main_exprs:
                self._gen_expr(expr)
            self.builder.insert(func.ReturnOp(self.builder.insert(arith.ConstantOp(IntegerAttr(0, i32))).results[0]))
            self.builder = prev_builder

        return self.module

    def _get_func_signatures(self):
        # sigs: function name -> list of argument types from all call sites
        sigs = {}
        num_type = lambda n: f64 if "." in n.children[0] else i32
        type_of = lambda n: llvm.LLVMPointerType() if n.data == "string" else num_type(n) if n.data == "number" else None

        def _visit(n: Tree) -> None:
            if n.data == "call_expr":
                args = [type_of(a) for a in n.children[1:]]
                if all(args):
                    sigs.setdefault(str(n.children[0]), []).append(args)
            for c in n.children:
                if hasattr(c, "children"):
                    _visit(c)

        _visit(tree)

        # resolved: function name -> final argument types
        resolved = {}
        for name, call_sigs in sigs.items():
            if not call_sigs:
                continue
            final = list(call_sigs[0])
            for sig in call_sigs[1:]:
                for i, (t1, t2) in enumerate(zip(final, sig)):
                    if t1 != t2:
                        final[i] = f64  # promote to float if mismatch
            resolved[name] = final
        return resolved

    def _gen_expr(self, node):
        if node.data == "number":
            val = node.children[0]
            is_float = "." in val
            return self.builder.insert(arith.ConstantOp(FloatAttr(float(val), f64) if is_float else IntegerAttr(int(val), i32))).results[0]

        if node.data == "string":
            return self._get_str_global(str(node.children[0])[1:-1])

        if node.data == "variable":
            return self.symbol_table[str(node.children[0])]

        if node.data == "binary_expr":
            op = str(node.children[0])
            lhs_node, rhs_node = node.children[1], node.children[2]
            lhs, rhs = self._gen_expr(lhs_node), self._gen_expr(rhs_node)
            is_float = isinstance(lhs.type, builtin.Float64Type)
            match op:
                case "+":
                    cls = arith.AddfOp if is_float else arith.AddiOp
                case "-":
                    cls = arith.SubfOp if is_float else arith.SubiOp
                case "*":
                    cls = arith.MulfOp if is_float else arith.MuliOp
                case "<=":
                    if is_float:
                        return self.builder.insert(arith.CmpfOp(lhs, rhs, "ole")).results[0]
                    # lhs <= rhs equivalent !(rhs < lhs)
                    lt = self.builder.insert(arith.CmpiOp(rhs, lhs, "slt")).results[0]
                    z = self.builder.insert(arith.ConstantOp(IntegerAttr(0, i32) if isinstance(lt.type, builtin.IntegerType) and lt.type.width.data > 1 else IntegerAttr(0, builtin.i1))).results[0]
                    return self.builder.insert(arith.CmpiOp(lt, z, "eq")).results[0]
            return self.builder.insert(cls(lhs, rhs)).results[0]

        if node.data == "call_expr":
            name = str(node.children[0])
            call_args = [self._gen_expr(c) for c in node.children[1:]]
            func_op = next((o for o in self.module.body.blocks[0].ops if isinstance(o, func.FuncOp) and o.sym_name.data == name), None)
            # cast args on type mismatch with signature
            final_args = []
            for call_arg, actual_type in zip(call_args, func_op.function_type.inputs.data):
                if call_arg.type != actual_type:
                    final_args.append(self.builder.insert(arith.SIToFPOp(call_arg, f64)).results[0])
                else:
                    final_args.append(call_arg)
            return self.builder.insert(func.CallOp(name, final_args, func_op.function_type.outputs.data)).results[0]

        if node.data == "if_expr":
            cond = self._gen_expr(node.children[0])
            is_bool = cond.type == builtin.i1
            if not is_bool:
                z = self.builder.insert(arith.ConstantOp(IntegerAttr(0, cond.type))).results[0]
                cond = self.builder.insert(arith.CmpiOp(cond, z, "ne")).results[0]

            orig_builder = self.builder

            then_block = Block()
            self.builder = Builder(InsertPoint.at_end(then_block))
            then_val = self._gen_expr(node.children[1])
            self.builder.insert(scf.YieldOp(then_val))

            else_block = Block()
            self.builder = Builder(InsertPoint.at_end(else_block))
            else_val = self._gen_expr(node.children[2])
            self.builder.insert(scf.YieldOp(else_val))

            self.builder = orig_builder
            return self.builder.insert(scf.IfOp(cond, [then_val.type], Region(then_block), Region(else_block))).results[0]

        if node.data == "print_expr":
            val = self._gen_expr(node.children[0])
            fmt = "%s\n" if isinstance(val.type, llvm.LLVMPointerType) else "%f\n" if isinstance(val.type, builtin.Float64Type) else "%d\n"

            # casts for printf
            if fmt == "%f\n" and val.type != f64:
                val = self.builder.insert(arith.ExtFOp(val, f64)).results[0]
            if fmt == "%d\n":  # doesn't handle i64, defaults to i32
                if val.type.width.data < 32:
                    val = self.builder.insert(arith.ExtSIOp(val, i32)).results[0]
                elif val.type.width.data > 32:
                    val = self.builder.insert(arith.TruncIOp(val, i32)).results[0]

            fmt_ptr = self._get_str_global(fmt)
            call = llvm.CallOp(SymbolAttr("printf"), fmt_ptr, val, return_type=i32)
            call.attributes["var_callee_type"] = llvm.LLVMFunctionType([llvm.LLVMPointerType()], builtin.i32, is_variadic=True)
            self.builder.insert(call)
            return None

    def _get_str_global(self, val: str) -> builtin.SSAValue:
        if val not in self.str_cache:
            name = f".str.{self.str_cnt}"
            self.str_cnt += 1
            self.str_cache[val] = name
            data = val.encode("utf-8") + b"\0"
            g = llvm.GlobalOp(llvm.LLVMArrayType.from_size_and_type(len(data), i8), StringAttr(name), linkage=llvm.LinkageAttr("internal"), constant=True, value=ArrayAttr([IntegerAttr(b, i8) for b in data]))
            self.module.body.blocks[0].insert_op_before(g, self.module.body.blocks[0].first_op)

        return self.builder.insert(llvm.AddressOfOp(self.str_cache[val], llvm.LLVMPointerType())).results[0]


#
# rewrites
#


class InlineFunctions(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.CallOp, rewriter: PatternRewriteWalker):
        if not self._callee(op):
            return
        if any(isinstance(c, func.CallOp) and c.callee.string_value() == self._callee(op).sym_name.data for c in self._callee(op).walk()):
            return

        # clone block and inline
        blk = self._callee(op).clone().body.blocks[0]
        for opr, arg in zip(op.operands, blk.args):
            arg.replace_by(opr)

        ops = list(blk.ops)
        ret = ops[-1]
        for o in ops[:-1]:
            o.detach()
            rewriter.insert_op(o, InsertPoint.before(op))

        rewriter.replace_op(op, [], ret.operands)

    @lru_cache(None)
    def _callee(self, op):
        mod = op
        while not isinstance(mod, ModuleOp):
            mod = mod.parent_op()
        return next((o for o in mod.body.blocks[0].ops if isinstance(o, func.FuncOp) and o.sym_name.data == op.callee.string_value()), None)


class RemoveUnusedPrivateFunctions(RewritePattern):
    _used = None

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.FuncOp, rewriter: PatternRewriteWalker):
        if op.sym_name.data == "main" or op.sym_visibility != StringAttr("private"):
            return
        if self._used is None:
            self._used = {c.callee.string_value() for c in op.parent_op().walk() if isinstance(c, func.CallOp)}
        if op.sym_name.data not in self._used:
            rewriter.erase_op(op)


class OptimizePass(ModulePass):
    name = "optimize"

    def apply(self, _, op: ModuleOp):
        PatternRewriteWalker(InlineFunctions()).rewrite_module(op)
        PatternRewriteWalker(RemoveUnusedPrivateFunctions()).rewrite_module(op)
        dce(op)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    tree = Lark(GRAMMAR, start="start").parse(Path(args.file).read_text())
    module_op = IRGen().gen(tree)

    if args.debug:
        print(f"\n{'-'*80} source\n{Path(args.file).read_text()}")
        print(f"\n{'-'*80} mlir before optimization\n{module_op}")

    ctx = Context()
    for d in [arith.Arith, builtin.Builtin, func.Func, scf.Scf, llvm.LLVM]:
        ctx.load_dialect(d)

    OptimizePass().apply(ctx, module_op)
    module_op.verify()

    if args.debug:
        print(f"\n{'-'*80} mlir after optimization\n")
    print(module_op)
