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
# parsing + irgen
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
        self.symbol_table = {}  # var name -> SSAValue in current scope (ScopedDict could be used for nested scopes)
        self.str_cache = {}  # reference string consts on dup values
        self.str_cnt = 0  # string name gen

        # declare printf signature from libc, so it can be called
        self.builder.insert(llvm.FuncOp("printf", llvm.LLVMFunctionType([llvm.LLVMPointerType()], builtin.i32, is_variadic=True), linkage=llvm.LinkageAttr("external")))

    def gen(self, tree: Lark) -> ModuleOp:
        funcs = [n for n in tree.children if n.data == "defun"]
        main_exprs = [n for n in tree.children if n.data != "defun"]

        # sigs: function name -> list of type signatures from all call sites in the code
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

        # Generate Functions
        for f in funcs:
            name, nodes = str(f.children[0]), f.children[1:]
            args_node = next((n for n in nodes if n.data == "args"), None)
            args = [str(t) for t in args_node.children] if args_node else []
            body = [n for n in nodes if n.data != "args"]

            arg_types = resolved.get(name, [i32] * len(args))
            ftype = FunctionType.from_lists(arg_types, [i32])  # Default return i32, updated later

            entry = Block(arg_types=arg_types)
            func_op = func.FuncOp(name, ftype, Region(entry))
            self.module.body.blocks[0].add_op(func_op)

            prev_builder, self.builder = self.builder, Builder(InsertPoint.at_end(entry))
            self.symbol_table = {n: a for n, a in zip(args, entry.args)}

            last_val = None
            for expr in body:
                last_val = self._gen_expr(expr)

            ret_val = last_val if last_val else self.builder.insert(arith.ConstantOp(IntegerAttr(0, i32))).results[0]
            self.builder.insert(func.ReturnOp(ret_val))

            # Update return type based on actual return
            func_op.function_type = FunctionType.from_lists(arg_types, [ret_val.type])
            self.builder = prev_builder

        # Generate Main
        if main_exprs:
            entry = Block()
            func_op = func.FuncOp("main", FunctionType.from_lists([], [i32]), Region(entry))
            self.module.body.blocks[0].add_op(func_op)
            prev_builder, self.builder = self.builder, Builder(InsertPoint.at_end(entry))
            self.symbol_table = {}
            for expr in main_exprs:
                self._gen_expr(expr)
            self.builder.insert(func.ReturnOp(self.builder.insert(arith.ConstantOp(IntegerAttr(0, i32))).results[0]))
            self.builder = prev_builder

        return self.module

    def _get_str_global(self, val: str) -> builtin.SSAValue:
        if val not in self.str_cache:
            name = f".str.{self.str_cnt}"
            self.str_cnt += 1
            self.str_cache[val] = name
            data = val.encode("utf-8") + b"\0"
            g = llvm.GlobalOp(llvm.LLVMArrayType.from_size_and_type(len(data), i8), StringAttr(name), linkage=llvm.LinkageAttr("internal"), constant=True, value=ArrayAttr([IntegerAttr(b, i8) for b in data]))
            self.module.body.blocks[0].insert_op_before(g, self.module.body.blocks[0].first_op)

        return self.builder.insert(llvm.AddressOfOp(self.str_cache[val], llvm.LLVMPointerType())).results[0]

    def _gen_expr(self, node):
        if node.data == "number":
            val = node.children[0]
            return self.builder.insert(arith.ConstantOp(FloatAttr(float(val), f64) if "." in val else IntegerAttr(int(val), i32))).results[0]

        if node.data == "string":
            return self._get_str_global(str(node.children[0])[1:-1])

        if node.data == "variable":
            return self.symbol_table[str(node.children[0])]

        if node.data == "binary_expr":
            op, lhs_node, rhs_node = str(node.children[0]), node.children[1], node.children[2]
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
                    # (b < a) == 0  <-- Simplified SLE logic
                    lt = self.builder.insert(arith.CmpiOp(rhs, lhs, "slt")).results[0]
                    z = self.builder.insert(arith.ConstantOp(IntegerAttr(0, i32) if isinstance(lt.type, builtin.IntegerType) and lt.type.width.data > 1 else IntegerAttr(0, builtin.i1))).results[0]
                    return self.builder.insert(arith.CmpiOp(lt, z, "eq")).results[0]
            return self.builder.insert(cls(lhs, rhs)).results[0]

        if node.data == "call_expr":
            name, args = str(node.children[0]), [self._gen_expr(c) for c in node.children[1:]]
            func_op = next((o for o in self.module.body.blocks[0].ops if isinstance(o, func.FuncOp) and o.sym_name.data == name), None)

            # Cast arguments
            final_args = []
            for arg, exp_type in zip(args, func_op.function_type.inputs.data):
                if arg.type != exp_type:
                    final_args.append(self.builder.insert(arith.SIToFPOp(arg, f64)).results[0])
                else:
                    final_args.append(arg)

            return self.builder.insert(func.CallOp(name, final_args, func_op.function_type.outputs.data)).results[0]

        if node.data == "if_expr":
            cond, then_n, else_n = self._gen_expr(node.children[0]), node.children[1], node.children[2]

            # Ensure boolean condition
            if cond.type != builtin.i1:
                z = self.builder.insert(arith.ConstantOp(IntegerAttr(0, cond.type))).results[0]
                cond = self.builder.insert(arith.CmpiOp(cond, z, "ne")).results[0]

            orig_builder = self.builder

            # Generate blocks first to determine result type
            t_blk = Block()
            self.builder = Builder(InsertPoint.at_end(t_blk))
            then_val = self._gen_expr(node.children[1])
            self.builder.insert(scf.YieldOp(then_val))

            e_blk = Block()
            self.builder = Builder(InsertPoint.at_end(e_blk))
            else_val = self._gen_expr(node.children[2])
            self.builder.insert(scf.YieldOp(else_val))

            self.builder = orig_builder
            return self.builder.insert(scf.IfOp(cond, [then_val.type], Region(t_blk), Region(e_blk))).results[0]

        if node.data == "print_expr":
            val = self._gen_expr(node.children[0])
            fmt = "%s\n" if isinstance(val.type, llvm.LLVMPointerType) else "%f\n" if isinstance(val.type, builtin.Float64Type) else "%d\n"

            # Casts for printf
            if fmt == "%f\n" and val.type != f64:
                val = self.builder.insert(arith.ExtFOp(val, f64)).results[0]
            if fmt == "%d\n":
                if val.type.width.data < 32:
                    val = self.builder.insert(arith.ExtSIOp(val, i32)).results[0]
                elif val.type.width.data > 32:
                    val = self.builder.insert(arith.TruncIOp(val, i32)).results[0]
            # (Note: ExtSIOp/TruncIOp conditional logic simplified for brevity, assuming mostly i32)

            fmt_ptr = self._get_str_global(fmt)
            call = llvm.CallOp(SymbolAttr("printf"), fmt_ptr, val, return_type=i32)
            call.attributes["var_callee_type"] = llvm.LLVMFunctionType([llvm.LLVMPointerType()], builtin.i32, is_variadic=True)
            self.builder.insert(call)
            return None


#
# rewrites
#


class InlineFunctions(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.CallOp, rewriter: PatternRewriteWalker):  # Check types
        if not (callee := self.get_callee(op)):
            return
        if any(isinstance(c, func.CallOp) and c.callee.string_value() == callee.sym_name.data for c in callee.walk()):
            return

        # Clone block and inline
        blk = callee.clone().body.blocks[0]
        for opr, arg in zip(op.operands, blk.args):
            arg.replace_by(opr)

        ops = list(blk.ops)
        ret = ops[-1]
        for o in ops[:-1]:
            o.detach()
            rewriter.insert_op(o, InsertPoint.before(op))

        rewriter.replace_op(op, [], ret.operands)

    def get_callee(self, op):
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
        print(f"\n{'-'*80} MLIR before optimization\n{module_op}")

    ctx = Context()
    for d in [arith.Arith, builtin.Builtin, func.Func, scf.Scf, llvm.LLVM]:
        ctx.load_dialect(d)

    OptimizePass().apply(ctx, module_op)
    module_op.verify()

    if args.debug:
        print(f"\n{'-'*80} MLIR after optimization\n")
    print(module_op)
