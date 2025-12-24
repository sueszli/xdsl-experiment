from xdsl.context import Context
from xdsl.dialects import arith, builtin, llvm, printf
from xdsl.dialects.builtin import ModuleOp, StringAttr, SymbolRefAttr
from xdsl.ir import Operation
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriter, PatternRewriteWalker, RewritePattern, op_type_rewrite_pattern
from xdsl.rewriter import InsertPoint


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
