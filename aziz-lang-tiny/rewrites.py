import aziz
from xdsl.context import Context
from xdsl.dialects import arith, builtin, func, llvm, printf, scf
from xdsl.dialects.builtin import AnyFloat, ArrayAttr, FloatAttr, IntegerAttr, IntegerType, ModuleOp, StringAttr, SymbolRefAttr, i8
from xdsl.ir import Block, Operation
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import GreedyRewritePatternApplier, PatternRewriter, PatternRewriteWalker, RewritePattern, op_type_rewrite_pattern
from xdsl.rewriter import InsertPoint
from xdsl.traits import CallableOpInterface, SymbolTable
from xdsl.transforms.dead_code_elimination import dce

#
# lower
#


class InlineFunctions(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: aziz.CallOp, rewriter: PatternRewriter):
        callee = SymbolTable.lookup_symbol(op, op.callee)

        is_unknown = callee is None
        if is_unknown:
            return

        is_self_call = lambda op: isinstance(op, aziz.CallOp) and op.callee.string_value() == callee.sym_name.data
        is_recursive = any(is_self_call(child_op) for child_op in callee.walk())
        if is_recursive:
            return

        callable_interface = callee.get_trait(CallableOpInterface)
        not_callable = callable_interface is None
        if not_callable:
            return

        # just inline one-liner functions for simplicity
        impl_body = callable_interface.get_callable_region(callee)
        is_one_liner = len(impl_body.blocks) == 1  # just inline on
        if not is_one_liner:
            return

        impl_block = impl_body.clone().block
        assert len(op.operands) == len(impl_block.args), f"arity mismatch in inlining {op} and {callee}"

        for operand, arg in zip(op.operands, impl_block.args):
            arg.replace_by(operand)

        while len(impl_block.args):
            rewriter.erase_block_argument(impl_block.args[-1])

        rewriter.inline_block(impl_block, InsertPoint.before(op))

        return_op = op.prev_op
        assert return_op is not None
        assert isinstance(return_op, aziz.ReturnOp)
        if return_op.input:
            rewriter.replace_op(op, [], [return_op.input])
        else:
            rewriter.replace_op(op, [], [])
        rewriter.erase_op(return_op)


class RemoveUnusedPrivateFunctions(RewritePattern):
    _used_funcs: set[str] | None = None

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: aziz.FuncOp, rewriter: PatternRewriter):
        if self._is_unused(op):
            rewriter.erase_op(op)

    def _is_unused(self, op: aziz.FuncOp) -> bool:
        if op.sym_visibility != StringAttr("private"):
            return False

        if self._used_funcs is None:
            module = op.parent_op()
            assert isinstance(module, ModuleOp)
            self._used_funcs = {op.callee.string_value() for op in module.walk() if isinstance(op, aziz.CallOp)}

        return op.sym_name.data not in self._used_funcs


class OptimizeAzizPass(ModulePass):
    name = "optimize-aziz"

    def apply(self, _: Context, op: ModuleOp) -> None:
        # no GreedyRewritePatternApplier here because we want to control the order
        PatternRewriteWalker(InlineFunctions()).rewrite_module(op)
        PatternRewriteWalker(RemoveUnusedPrivateFunctions()).rewrite_module(op)
        dce(op)


#
# arith
#


class AddOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: aziz.AddOp, rewriter: PatternRewriter):
        if isinstance(op.lhs.type, AnyFloat):
            rewriter.replace_op(op, arith.AddfOp(op.lhs, op.rhs))
        else:
            rewriter.replace_op(op, arith.AddiOp(op.lhs, op.rhs))


class SubOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: aziz.SubOp, rewriter: PatternRewriter):
        if isinstance(op.lhs.type, AnyFloat):
            rewriter.replace_op(op, arith.SubfOp(op.lhs, op.rhs))
        else:
            rewriter.replace_op(op, arith.SubiOp(op.lhs, op.rhs))


class MulOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: aziz.MulOp, rewriter: PatternRewriter):
        if isinstance(op.lhs.type, AnyFloat):
            rewriter.replace_op(op, arith.MulfOp(op.lhs, op.rhs))
        else:
            rewriter.replace_op(op, arith.MuliOp(op.lhs, op.rhs))


class LessThanEqualOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: aziz.LessThanEqualOp, rewriter: PatternRewriter):
        if isinstance(op.lhs.type, AnyFloat):
            rewriter.replace_op(op, arith.CmpfOp(op.lhs, op.rhs, "ole"))  # ordered less equal
        else:
            # xDSL's sle lowering to RISC-V is buggy.
            # we implement a <= b as !(b < a)
            # e.g. (b < a) == 0

            # lt = (rhs < lhs)
            lt = arith.CmpiOp(op.rhs, op.lhs, "slt")
            rewriter.insert_op(lt, InsertPoint.before(op))

            # res = (lt == 0)
            zero = arith.ConstantOp(IntegerAttr(0, IntegerType(1)))
            rewriter.insert_op(zero, InsertPoint.before(op))

            rewriter.replace_op(op, arith.CmpiOp(lt.result, zero.result, "eq"))


class CastIntToFloatOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: aziz.CastIntToFloatOp, rewriter: PatternRewriter):
        rewriter.replace_op(op, arith.SIToFPOp(op.input, op.res.type))


class ConstantOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: aziz.ConstantOp, rewriter: PatternRewriter):
        val = op.value
        if isinstance(val, IntegerAttr):
            rewriter.replace_op(op, arith.ConstantOp(val))
        elif isinstance(val, FloatAttr):
            rewriter.replace_op(op, arith.ConstantOp(val))


class IfOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: aziz.IfOp, rewriter: PatternRewriter):
        # get the condition - convert to i1 if needed
        cond = op.cond
        wider_than_bool = isinstance(cond.type, IntegerType) and cond.type.width.data != 1
        if wider_than_bool:
            zero = arith.ConstantOp(IntegerAttr(0, cond.type))
            rewriter.insert_op(zero, InsertPoint.before(op))
            cmp = arith.CmpiOp(cond, zero.result, "ne")  # condition != 0
            rewriter.insert_op(cmp, InsertPoint.before(op))
            cond = cmp.result

        # create scf.if which preserves control flow (only executing the taken branch)
        then_region = rewriter.move_region_contents_to_new_regions(op.then_region)
        else_region = rewriter.move_region_contents_to_new_regions(op.else_region)

        new_op = scf.IfOp(cond, [r.type for r in op.results], then_region, else_region)
        rewriter.replace_op(op, new_op)


#
# func
#


class ReturnOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: aziz.ReturnOp, rewriter: PatternRewriter):
        rewriter.replace_op(op, func.ReturnOp(op.input) if op.input else func.ReturnOp())


class FuncOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: aziz.FuncOp, rewriter: PatternRewriter):  # strings become pointers
        # convert inputs/outputs
        convert_type = lambda t: llvm.LLVMPointerType() if isinstance(t, aziz.StringType) else t
        inputs = [convert_type(t) for t in op.function_type.inputs]
        outputs = [convert_type(t) for t in op.function_type.outputs]
        new_type = func.FunctionType.from_lists(inputs, outputs)

        # move region
        new_body = rewriter.move_region_contents_to_new_regions(op.body)

        # update block arguments in new body
        for block in list(new_body.blocks):
            new_arg_types = [convert_type(arg.type) for arg in block.args]
            if new_arg_types == [arg.type for arg in block.args]:
                continue

            # replace args uses
            new_block = Block(arg_types=new_arg_types)
            for old_arg, new_arg in zip(block.args, new_block.args):
                old_arg.replace_by(new_arg)

            # move ops
            ops = list(block.ops)
            for o in ops:
                o.detach()
                new_block.add_op(o)

            # replace block
            idx = next(i for i, b in enumerate(new_body.blocks) if b is block)
            new_body.detach_block(block)
            new_body.insert_block(new_block, idx)

        new_op = func.FuncOp(op.sym_name.data, new_type, new_body, visibility=op.sym_visibility)
        rewriter.replace_op(op, new_op)


class CallOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: aziz.CallOp, rewriter: PatternRewriter):
        convert_type = lambda t: llvm.LLVMPointerType() if isinstance(t, aziz.StringType) else t
        res_types = [convert_type(t) for t in op.res.types]
        rewriter.replace_op(op, func.CallOp(op.callee, op.arguments, res_types))


#
# scf
#


class YieldOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: aziz.YieldOp, rewriter: PatternRewriter):
        rewriter.replace_op(op, scf.YieldOp(op.input))


#
# printf
#


class PrintOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: aziz.PrintOp, rewriter: PatternRewriter):
        if isinstance(op.input.type, llvm.LLVMPointerType):
            # constant string (llvm global pointer)
            rewriter.replace_op(op, printf.PrintFormatOp("{}", op.input))
        else:
            # integers, floats
            rewriter.replace_op(op, printf.PrintFormatOp("{}", op.input))


#
# llvm
#


class StringConstantOpLowering(RewritePattern):
    def __init__(self):
        super().__init__()
        self._string_cache: dict[str, str] = {}  # string content -> global name
        self._counter: int = 0

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: aziz.StringConstantOp, rewriter: PatternRewriter):
        val = op.value
        assert isinstance(val, StringAttr)
        string_content = val.data
        encoded_val = string_content.encode("utf-8") + b"\0"

        # get or create llvm.global for the string
        if string_content not in self._string_cache:
            module = self._get_module(op)
            self._create_global_for_string(string_content, encoded_val, module, rewriter)

        global_name = self._string_cache[string_content]

        # create addressof and replace
        addr = llvm.AddressOfOp(global_name, llvm.LLVMPointerType())
        rewriter.insert_op(addr, InsertPoint.before(op))
        rewriter.replace_op(op, [], [addr.result])

    def _get_module(self, op: aziz.StringConstantOp) -> ModuleOp:
        module = op.parent_op()
        while module and not isinstance(module, ModuleOp):
            module = module.parent_op()
        assert isinstance(module, ModuleOp)
        return module

    def _create_global_for_string(self, string_content: str, encoded_val: bytes, module: ModuleOp, rewriter: PatternRewriter) -> None:
        global_name = f".aziz.str.{self._counter}"
        self._counter += 1
        self._string_cache[string_content] = global_name

        # Use ArrayAttr for initializer to match llvm.array type in mlir-translate
        array_type = llvm.LLVMArrayType.from_size_and_type(len(encoded_val), i8)
        initial_value = ArrayAttr([IntegerAttr(b, i8) for b in encoded_val])

        global_op = llvm.GlobalOp(array_type, global_name, linkage=llvm.LinkageAttr("internal"), constant=True, value=initial_value)
        rewriter.insert_op(global_op, InsertPoint.at_start(module.body.blocks[0]))


class LowerAzizPass(ModulePass):
    name = "lower-aziz"

    def apply(self, _: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    AddOpLowering(),
                    SubOpLowering(),
                    MulOpLowering(),
                    LessThanEqualOpLowering(),
                    CastIntToFloatOpLowering(),
                    ConstantOpLowering(),
                    ReturnOpLowering(),
                    FuncOpLowering(),
                    CallOpLowering(),
                    IfOpLowering(),
                    YieldOpLowering(),
                    StringConstantOpLowering(),
                    PrintOpLowering(),
                ]
            )
        ).rewrite_module(op)


#
# lower to llvm
#


class PrintFormatOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: printf.PrintFormatOp, rewriter: PatternRewriter):
        if not op.operands:
            rewriter.erase_op(op)
            return

        arg = op.operands[0]
        arg_type = arg.type
        module_op = op.get_toplevel_object()

        # delegate to specific handlers based on type
        if isinstance(arg_type, llvm.LLVMPointerType):
            self._lower_string(op, rewriter, module_op, arg)
        elif isinstance(arg_type, (builtin.IntegerType, builtin.IndexType)):
            self._lower_integer(op, rewriter, module_op, arg)
        elif isinstance(arg_type, (builtin.Float32Type, builtin.Float64Type)):
            self._lower_float(op, rewriter, module_op, arg)

    def _lower_string(self, op: printf.PrintFormatOp, rewriter: PatternRewriter, module_op: ModuleOp, val: Operation):
        # printf(str)
        fmt_str_global = self._ensure_global_string_constant(module_op, "%s\n", "fmt_s")
        self._create_printf_call(op, rewriter, fmt_str_global, [val])

    def _lower_integer(self, op: printf.PrintFormatOp, rewriter: PatternRewriter, module_op: ModuleOp, val: Operation):
        # printf("%d\n", val)
        fmt_str_global = self._ensure_global_string_constant(module_op, "%d\n", "fmt_d")

        # printf expects i32 for %d format
        val = self._cast_integer_to_i32(val, rewriter, op)
        self._create_printf_call(op, rewriter, fmt_str_global, [val])

    def _lower_float(self, op: printf.PrintFormatOp, rewriter: PatternRewriter, module_op: ModuleOp, val: Operation):
        # printf("%f\n", val)
        fmt_str_global = self._ensure_global_string_constant(module_op, "%f\n", "fmt_f")

        # float varargs in C execution must be promoted to double (f64)
        if not isinstance(val.type, builtin.Float64Type):
            cast = arith.ExtFOp(val, builtin.f64)
            rewriter.insert_op(cast, InsertPoint.before(op))
            val = cast.result

        self._create_printf_call(op, rewriter, fmt_str_global, [val])

    def _cast_integer_to_i32(self, val: Operation, rewriter: PatternRewriter, insertion_point_op: Operation):
        target_width = 32
        val_type = val.type

        # if it's index type, we need index_cast
        if isinstance(val_type, builtin.IndexType):
            cast = arith.index_cast(val, builtin.i32)
            rewriter.insert_op(cast, InsertPoint.before(insertion_point_op))
            return cast.result

        # if it's integer type
        current_width = val_type.width.data
        if current_width == target_width:
            return val

        if current_width > target_width:
            cast = arith.TruncIOp(val, builtin.i32)
        else:
            cast = arith.ExtSIOp(val, builtin.i32)

        rewriter.insert_op(cast, InsertPoint.before(insertion_point_op))
        return cast.result

    def _create_printf_call(self, op: Operation, rewriter: PatternRewriter, fmt_global: llvm.GlobalOp, args: list[Operation]):
        # get address of format string globally
        fmt_dev_ptr = llvm.AddressOfOp(fmt_global.sym_name, llvm.LLVMPointerType())
        rewriter.insert_op(fmt_dev_ptr, InsertPoint.before(op))

        # prepare call arguments (format string + values)
        call_args = [fmt_dev_ptr.results[0]] + args

        # create CallOp
        # printf signature: (i8*, ...) -> i32
        callee_name = SymbolRefAttr("printf")
        printf_type = llvm.LLVMFunctionType([llvm.LLVMPointerType()], builtin.i32, is_variadic=True)

        call = llvm.CallOp(callee_name, *call_args, return_type=builtin.i32)
        call.attributes["var_callee_type"] = printf_type

        rewriter.insert_op(call, InsertPoint.before(op))
        rewriter.erase_op(op)

    def _ensure_global_string_constant(self, module: ModuleOp, content: str, name_hint: str) -> llvm.GlobalOp:
        sym_name = f"__str_{name_hint}"
        module_block = module.body.blocks[0]

        for op in module_block.ops:
            if isinstance(op, llvm.GlobalOp) and op.sym_name.data == sym_name:
                return op

        # doesn't exist, create new global string constant
        data_bytes = content.encode("utf-8") + b"\0"
        arr_type = llvm.LLVMArrayType.from_size_and_type(len(data_bytes), builtin.i8)
        val_attr = builtin.ArrayAttr([builtin.IntegerAttr(b, builtin.i8) for b in data_bytes])
        global_op = llvm.GlobalOp(arr_type, StringAttr(sym_name), linkage=llvm.LinkageAttr("internal"), constant=True, value=val_attr)

        # insert at top of module
        if module_block.ops:
            module_block.insert_op_before(global_op, module_block.first_op)
        else:
            module_block.add_op(global_op)

        return global_op


class LowerPrintfToLLVMCallPass(ModulePass):
    name = "lower-printf-to-llvm-call"

    def apply(self, ctx: Context, op: ModuleOp):
        self._ensure_printf_decl(op)
        PatternRewriteWalker(PrintFormatOpLowering()).rewrite_module(op)

    def _ensure_printf_decl(self, module: ModuleOp):
        # we need to declare `printf` so we can call it.
        # declare external i32 @printf(i8*, ...)

        module_block = module.body.blocks[0]

        # check if already exists
        for op in module_block.ops:
            if hasattr(op, "sym_name") and op.sym_name.data == "printf":
                return

        ptr_type = llvm.LLVMPointerType()
        i32_type = builtin.i32

        printf_sig = llvm.LLVMFunctionType([ptr_type], i32_type, is_variadic=True)

        printf_decl = llvm.FuncOp("printf", printf_sig, linkage=llvm.LinkageAttr("external"))

        module_block.add_op(printf_decl)
