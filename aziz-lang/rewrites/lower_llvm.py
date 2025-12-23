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

        if isinstance(arg_type, llvm.LLVMPointerType):
            # assume direct string printing: printf(str)
            fmt_str_ptr = self._get_or_create_global_string(module_op, "%s\n", "fmt_s")
            self._create_printf_call(op, rewriter, fmt_str_ptr, [arg])

        elif isinstance(arg_type, (builtin.IntegerType, builtin.IndexType)):
            # print integer: printf("%d\n", val)
            fmt_str_ptr = self._get_or_create_global_string(module_op, "%d\n", "fmt_d")

            # Cast to i32 for %d
            val = arg
            if isinstance(arg_type, builtin.IndexType) or arg_type.width.data != 32:
                cast = arith.index_cast(val, builtin.i32) if isinstance(arg_type, builtin.IndexType) else (arith.TruncIOp(val, builtin.i32) if arg_type.width.data > 32 else arith.ExtSIOp(val, builtin.i32))
                rewriter.insert_op(cast, InsertPoint.before(op))
                val = cast.result

            self._create_printf_call(op, rewriter, fmt_str_ptr, [val])

        elif isinstance(arg_type, (builtin.Float32Type, builtin.Float64Type)):
            # print float: printf("%f\n", val)
            fmt_str_ptr = self._get_or_create_global_string(module_op, "%f\n", "fmt_f")

            # float varargs must be promoted to double (f64)
            val = arg
            if not isinstance(arg_type, (builtin.Float64Type)):
                cast = arith.ExtFOp(val, builtin.f64)
                rewriter.insert_op(cast, InsertPoint.before(op))
                val = cast.result

            self._create_printf_call(op, rewriter, fmt_str_ptr, [val])

    def _create_printf_call(self, op: Operation, rewriter: PatternRewriter, fmt_ptr, args):
        # We need to construct the call.
        # First argument is format string pointer.

        # We need `llvm.mlir.addressof` for `fmt_ptr` global?
        # _get_or_create_global_string returns the GlobalOp. We need address of it.

        # Wait, if we are in `lower_llvm`, maybe we should assume `llvm` dialect is available/primary target.
        # But `LowerAzizPass` runs before us.

        # Get address of format string
        # We need to insert `llvm.mlir.addressof` here.
        fline = llvm.AddressOfOp(fmt_ptr.sym_name, llvm.LLVMPointerType())
        rewriter.insert_op(fline, InsertPoint.before(op))

        callee = SymbolRefAttr("printf")
        call_ops = [fline.results[0]] + args

        # result of printf is i32, we discard it
        # var_callee_type required for variadic calls
        printf_type = llvm.LLVMFunctionType([llvm.LLVMPointerType()], builtin.i32, is_variadic=True)
        call = llvm.CallOp(callee, *call_ops, return_type=builtin.i32)
        call.attributes["var_callee_type"] = printf_type
        rewriter.insert_op(call, InsertPoint.before(op))
        rewriter.erase_op(op)

    def _get_or_create_global_string(self, module: ModuleOp, content: str, name_hint: str) -> llvm.GlobalOp:
        # Check if already exists
        # We'll suffix name with sanitized content hash or something?
        # For simple cases like "%d\n", let's use fixed names.

        sym_name = f"__str_{name_hint}"

        # check existence
        for op in module.body.blocks[0].ops:
            if isinstance(op, llvm.GlobalOp) and op.sym_name.data == sym_name:
                return op

        # create if not
        data_bytes = content.encode("utf-8") + b"\0"

        arr_type = llvm.LLVMArrayType.from_size_and_type(len(data_bytes), builtin.i8)

        val_attr = builtin.ArrayAttr([builtin.IntegerAttr(b, builtin.i8) for b in data_bytes])

        glob = llvm.GlobalOp(arr_type, StringAttr(sym_name), linkage=llvm.LinkageAttr("internal"), constant=True, value=val_attr)

        if module.body.blocks[0].ops:
            module.body.blocks[0].insert_op_before(glob, module.body.blocks[0].first_op)
        else:
            module.body.blocks[0].add_op(glob)

        return glob


class LowerPrintfToLLVMCallPass(ModulePass):
    name = "lower-printf-to-llvm-call"

    def apply(self, ctx: Context, op: ModuleOp):
        self._ensure_printf_decl(op)
        PatternRewriteWalker(PrintFormatOpLowering()).rewrite_module(op)

    def _ensure_printf_decl(self, module: ModuleOp):
        # declares printf signature if not already present, so we can call it
        if "printf" in [getattr(o, "sym_name", None) and o.sym_name.data for o in module.body.blocks[0].ops]:
            return

        i32 = builtin.i32
        ptr = llvm.LLVMPointerType()

        f = llvm.FuncOp("printf", llvm.LLVMFunctionType([ptr], i32, is_variadic=True), linkage=llvm.LinkageAttr("external"))
        module.body.blocks[0].add_op(f)
