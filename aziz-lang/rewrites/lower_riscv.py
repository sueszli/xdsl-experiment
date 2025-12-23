from qemu import STDOUT_ADDR
from xdsl.context import Context
from xdsl.dialects import arith, func, llvm, printf, riscv, riscv_func, scf
from xdsl.dialects.builtin import IntegerAttr, ModuleOp, StringAttr, SymbolRefAttr, UnrealizedConversionCastOp
from xdsl.ir import Attribute, Block, Region
from xdsl.irdl import attr_def, base, irdl_op_definition, result_def
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriter, PatternRewriteWalker, RewritePattern, op_type_rewrite_pattern
from xdsl.rewriter import InsertPoint


def extract_register_name(reg_type) -> str | None:
    """extract register name from register type, returns None if not a named register"""
    is_valid = isinstance(reg_type, (riscv.IntRegisterType, riscv.FloatRegisterType)) and hasattr(reg_type, "register_name") and reg_type.register_name
    if not is_valid:
        return None
    name = reg_type.register_name
    return name.data if isinstance(name, StringAttr) else name


def is_virtual_int_register(name: str) -> bool:
    """check if register name is a virtual integer register (j_*)"""
    return name.startswith("j_")


def is_virtual_float_register(name: str) -> bool:
    """check if register name is a virtual float register (fj_*)"""
    return name.startswith("fj_")


def unwrap_conversion_cast(value):
    """unwrap UnrealizedConversionCastOp if present, returns (actual_value, cast_op)"""
    if isinstance(value.owner, UnrealizedConversionCastOp) and value.owner.operands:
        return value.owner.operands[0], value.owner
    return value, None


def is_string_label_op(owner) -> bool:
    """check if operation produces a string label (for printf)"""
    is_string_label_op = isinstance(owner, riscv.LiOp) and isinstance(owner.immediate, riscv.LabelAttr) or isinstance(owner, RISCVLaOp) and isinstance(owner.label, riscv.LabelAttr)
    return is_string_label_op


#
# custom riscv operations
#


@irdl_op_definition
class RISCVGlobalOp(riscv.RISCVAsmOperation):
    """represents a global data symbol in the .data section"""

    name = "riscv.global"
    sym_name = attr_def(StringAttr)
    value = attr_def(base(Attribute))
    is_constant = attr_def(base(Attribute))

    def __init__(self, sym_name: str | StringAttr, value: Attribute, is_constant: bool = True):
        if isinstance(sym_name, str):
            sym_name = StringAttr(sym_name)
        constant_attr = IntegerAttr(1 if is_constant else 0, 1)
        super().__init__(attributes={"sym_name": sym_name, "value": value, "is_constant": constant_attr})

    def assembly_line(self) -> str | None:
        return None  # handled by emit_data_section


@irdl_op_definition
class RISCVLaOp(riscv.RISCVAsmOperation):
    """load address pseudo-instruction"""

    name = "riscv.la"
    rd = result_def(riscv.IntRegisterType)
    label = attr_def(riscv.LabelAttr)

    def __init__(self, label, rd_type):
        if isinstance(label, str):
            label = riscv.LabelAttr(label)
        super().__init__(result_types=[rd_type], attributes={"label": label})

    def assembly_line(self) -> str | None:
        reg_name = self.rd.type.register_name.data
        return f"la {reg_name}, {self.label.data}"


@irdl_op_definition
class RISCVDirectiveOp(riscv.RISCVAsmOperation):
    """raw assembly directive"""

    name = "riscv.directive"
    directive = attr_def(StringAttr)
    value = attr_def(StringAttr)

    def __init__(self, directive: str, value: str = ""):
        super().__init__(attributes={"directive": StringAttr(directive), "value": StringAttr(value)})

    def assembly_line(self) -> str | None:
        if self.value.data:
            return f"{self.directive.data} {self.value.data}"
        return self.directive.data


@irdl_op_definition
class RISCVLabelOp(riscv.RISCVAsmOperation):
    """assembly label"""

    name = "riscv.label"
    label = attr_def(riscv.LabelAttr)

    def __init__(self, label: str | riscv.LabelAttr):
        if isinstance(label, str):
            label = riscv.LabelAttr(label)
        super().__init__(attributes={"label": label})

    def assembly_line(self) -> str | None:
        return f"{self.label.data}:"


@irdl_op_definition
class RISCVAddSpOp(riscv.RISCVAsmOperation):
    """adjust stack pointer by immediate offset"""

    name = "riscv.add_sp"
    immediate = attr_def(IntegerAttr)

    def __init__(self, immediate: int):
        super().__init__(attributes={"immediate": IntegerAttr(immediate, 32)})

    def assembly_line(self) -> str | None:
        return f"addi sp, sp, {self.immediate.value.data}"


@irdl_op_definition
class RISCVSaveRaOp(riscv.RISCVAsmOperation):
    """save return address to stack"""

    name = "riscv.save_ra"

    def assembly_line(self) -> str | None:
        return "sd ra, 0(sp)"


@irdl_op_definition
class RISCVRestoreRaOp(riscv.RISCVAsmOperation):
    """restore return address from stack"""

    name = "riscv.restore_ra"

    def assembly_line(self) -> str | None:
        return "ld ra, 0(sp)"


#
# branching lowering
#


class SelectOpLowering(RewritePattern):
    """lower arith.select using bitwise operations to avoid branches"""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.SelectOp, rewriter: PatternRewriter):
        # branchless select: mask = 0b1111 if cond else 0b0000
        # result = (true_val & mask) | (false_val & ~mask)
        reg_type = riscv.IntRegisterType.unallocated()

        # cast inputs to riscv registers
        cond_cast = UnrealizedConversionCastOp.create(operands=[op.cond], result_types=[reg_type])
        rewriter.insert_op(cond_cast, InsertPoint.before(op))

        true_cast = UnrealizedConversionCastOp.create(operands=[op.lhs], result_types=[reg_type])
        rewriter.insert_op(true_cast, InsertPoint.before(op))

        false_cast = UnrealizedConversionCastOp.create(operands=[op.rhs], result_types=[reg_type])
        rewriter.insert_op(false_cast, InsertPoint.before(op))

        # create mask: 0 - cond produces 0x0000 or 0xFFFF
        zero = riscv.GetRegisterOp(riscv.Registers.ZERO)
        rewriter.insert_op(zero, InsertPoint.before(op))
        mask = riscv.SubOp(zero.res, cond_cast.results[0], rd=reg_type)
        rewriter.insert_op(mask, InsertPoint.before(op))

        # true_val & mask
        t1 = riscv.AndOp(true_cast.results[0], mask.rd, rd=reg_type)
        rewriter.insert_op(t1, InsertPoint.before(op))

        # false_val & ~mask
        not_mask = riscv.XoriOp(mask.rd, -1, rd=reg_type)
        rewriter.insert_op(not_mask, InsertPoint.before(op))
        t2 = riscv.AndOp(false_cast.results[0], not_mask.rd, rd=reg_type)
        rewriter.insert_op(t2, InsertPoint.before(op))

        # result = t1 | t2
        result = riscv.OrOp(t1.rd, t2.rd, rd=reg_type)
        rewriter.insert_op(result, InsertPoint.before(op))

        # cast back to original type
        result_cast = UnrealizedConversionCastOp.create(operands=[result.rd], result_types=[op.result.type])
        rewriter.insert_op(result_cast, InsertPoint.before(op))

        rewriter.replace_op(op, [], [result_cast.results[0]])


class LowerSelectPass(ModulePass):
    name = "lower-select"

    def apply(self, _: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(SelectOpLowering()).rewrite_module(op)


#
# global data lowering
#


class RemoveUnprintableOpsPass(ModulePass):
    """convert llvm.GlobalOp and llvm.AddressOfOp to riscv equivalents"""

    name = "remove-unprintable-ops"

    def apply(self, _: Context, op: ModuleOp) -> None:
        # convert llvm.GlobalOp to RISCVGlobalOp
        global_ops = [o for o in op.walk() if isinstance(o, llvm.GlobalOp)]
        for g_op in global_ops:
            new_global = RISCVGlobalOp(g_op.sym_name.data, g_op.value, g_op.constant is not None)
            g_op.parent_block().insert_op_before(new_global, g_op)
            g_op.detach()

        # convert llvm.AddressOfOp to RISCVLaOp
        addrof_ops = [o for o in op.walk() if isinstance(o, llvm.AddressOfOp)]
        for addr_op in addrof_ops:
            label = riscv.LabelAttr(addr_op.global_name.root_reference.data)
            la_op = RISCVLaOp(label, riscv.IntRegisterType.unallocated())
            addr_op.parent_block().insert_op_before(la_op, addr_op)
            addr_op.results[0].replace_by(la_op.rd)
            addr_op.detach()


class LowerRISCVGlobalOp(RewritePattern):
    """convert RISCVGlobalOp to assembly directives"""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: RISCVGlobalOp, rewriter: PatternRewriter):
        sym_name = op.sym_name.data

        # emit .data section directives
        rewriter.insert_op(RISCVDirectiveOp(".data"), InsertPoint.before(op))
        rewriter.insert_op(RISCVDirectiveOp(".globl", sym_name), InsertPoint.before(op))
        rewriter.insert_op(RISCVLabelOp(sym_name), InsertPoint.before(op))

        # extract and escape string content
        assert hasattr(op.value, "data") and hasattr(op.value.data, "data")
        string_bytes = bytes(op.value.data.data)
        try:
            null_index = string_bytes.index(0)
            string_content = string_bytes[:null_index].decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            string_content = string_bytes.decode("utf-8", errors="replace").rstrip("\x00")

        escaped = string_content.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\t", "\\t")
        rewriter.insert_op(RISCVDirectiveOp(".string", f'"{escaped}"'), InsertPoint.before(op))

        rewriter.erase_op(op)


class EmitDataSectionPass(ModulePass):
    """emit .data section for globals and .text section for functions"""

    name = "emit-data-section"

    def apply(self, _: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(LowerRISCVGlobalOp()).rewrite_module(op)

        # insert .text directive before first function
        for oper in op.body.blocks[0].ops:
            if isinstance(oper, (riscv_func.FuncOp, func.FuncOp)):
                op.body.blocks[0].insert_op_before(RISCVDirectiveOp(".text"), oper)
                break


#
# printf lowering
#


class LowerPrintfPass(ModulePass):
    """lower printf operations to runtime print calls"""

    name = "lower-printf"

    def apply(self, _: Context, op: ModuleOp) -> None:
        printf_ops = [o for o in op.walk() if isinstance(o, printf.PrintFormatOp)]
        for print_op in printf_ops:
            self._lower_printf_op(print_op)

    def _lower_printf_op(self, print_op):
        """lower single printf operation"""
        # guard: empty operands
        if not print_op.operands:
            print_op.detach()
            return

        arg = print_op.operands[0]
        block = print_op.parent_block()

        # unwrap conversion cast if present
        actual_arg, cast_op = unwrap_conversion_cast(arg)

        # handle string labels (must check before int register type)
        if is_string_label_op(actual_arg.owner):
            self._emit_print_call(block, print_op, actual_arg, "_print_string", is_float=False)
            self._cleanup_ops(print_op, cast_op)
            return

        # handle integer
        if isinstance(actual_arg.type, riscv.IntRegisterType):
            self._emit_print_call(block, print_op, actual_arg, "_print_int", is_float=False)
            self._cleanup_ops(print_op, cast_op)
            return

        # handle float
        if isinstance(actual_arg.type, riscv.FloatRegisterType):
            self._emit_print_call(block, print_op, actual_arg, "_print_float", is_float=True)
            self._cleanup_ops(print_op, cast_op)
            return

        # fallback: remove op
        self._cleanup_ops(print_op, cast_op)

    def _emit_print_call(self, block, print_op, arg, func_name: str, is_float: bool):
        """emit move to a0/fa0 and call to print function"""
        if is_float:
            fa0_type = riscv.FloatRegisterType.from_name("fa0")
            mv_op = riscv.FMvDOp(arg, rd=fa0_type)
        else:
            a0_type = riscv.IntRegisterType.from_name("a0")
            mv_op = riscv.AddiOp(arg, 0, rd=a0_type)

        block.insert_op_before(mv_op, print_op)
        call = riscv_func.CallOp(SymbolRefAttr(func_name), [mv_op.rd], [])
        block.insert_op_before(call, print_op)

    def _cleanup_ops(self, print_op, cast_op):
        """detach printf op and optional cast op"""
        print_op.detach()
        if cast_op:
            cast_op.detach()


#
# print runtime generation
#


class AddPrintRuntimePass(ModulePass):
    """add runtime functions for printing strings, integers, and floats"""

    name = "add-print-runtime"

    def apply(self, _: Context, op: ModuleOp) -> None:
        a0_type = riscv.IntRegisterType.from_name("a0")
        fa0_type = riscv.FloatRegisterType.from_name("fa0")

        # create function signatures
        func_type_int = func.FunctionType.from_lists([a0_type], [])
        func_type_float = func.FunctionType.from_lists([fa0_type], [])

        # add _print_string
        func_str = riscv_func.FuncOp(name="_print_string", region=Region(Block(arg_types=[a0_type])), function_type=func_type_int)
        self._add_print_string_body(func_str.body.blocks[0])
        op.body.blocks[0].add_op(func_str)

        # add _print_int
        func_int = riscv_func.FuncOp(name="_print_int", region=Region(Block(arg_types=[a0_type])), function_type=func_type_int)
        self._add_print_int_body(func_int.body.blocks[0])
        op.body.blocks[0].add_op(func_int)

        # add _print_float
        func_float = riscv_func.FuncOp(name="_print_float", region=Region(Block(arg_types=[fa0_type])), function_type=func_type_float)
        self._add_print_float_body(func_float.body.blocks[0])
        op.body.blocks[0].add_op(func_float)

    def _emit_newline(self) -> list:
        """emit newline character to mmio"""
        return [RISCVDirectiveOp("li", "t1, 10"), RISCVDirectiveOp("sb", "t1, 0(t2)")]

    def _emit_digit_extraction_loop(self, label_prefix: str) -> list:
        """emit loop to extract decimal digits to stack"""
        return [
            RISCVDirectiveOp("li", "t3, 10"),
            RISCVDirectiveOp("li", "t5, 0"),  # digit count
            RISCVLabelOp(f"{label_prefix}_extract_loop"),
            RISCVDirectiveOp("beqz", f"t0, {label_prefix}_print_loop"),
            RISCVDirectiveOp("rem", "t1, t0, t3"),
            RISCVDirectiveOp("div", "t0, t0, t3"),
            RISCVDirectiveOp("addi", "t1, t1, 48"),  # to ascii
            RISCVDirectiveOp("addi", "sp, sp, -16"),
            RISCVDirectiveOp("sd", "t1, 0(sp)"),
            RISCVDirectiveOp("addi", "t5, t5, 1"),
            RISCVDirectiveOp("j", f"{label_prefix}_extract_loop"),
        ]

    def _emit_digit_print_loop(self, label_prefix: str, end_label: str) -> list:
        """emit loop to print digits from stack"""
        return [
            RISCVLabelOp(f"{label_prefix}_print_loop"),
            RISCVDirectiveOp("beqz", f"t5, {end_label}"),
            RISCVDirectiveOp("ld", "t1, 0(sp)"),
            RISCVDirectiveOp("addi", "sp, sp, 16"),
            RISCVDirectiveOp("sb", "t1, 0(t2)"),
            RISCVDirectiveOp("addi", "t5, t5, -1"),
            RISCVDirectiveOp("j", f"{label_prefix}_print_loop"),
        ]

    def _add_print_string_body(self, block):
        """generate string printing loop: iterate chars until null terminator"""
        ops = [
            RISCVDirectiveOp("mv", "t0, a0"),
            RISCVDirectiveOp("li", f"t2, {STDOUT_ADDR}"),
            RISCVLabelOp("_print_string_loop"),
            RISCVDirectiveOp("lbu", "t1, 0(t0)"),
            RISCVDirectiveOp("beqz", "t1, _print_string_end"),
            RISCVDirectiveOp("sb", "t1, 0(t2)"),
            RISCVDirectiveOp("addi", "t0, t0, 1"),
            RISCVDirectiveOp("j", "_print_string_loop"),
            RISCVLabelOp("_print_string_end"),
            *self._emit_newline(),
            riscv_func.ReturnOp(),
        ]
        for o in ops:
            block.add_op(o)

    def _add_print_int_body(self, block):
        """generate integer printing: handle zero/negative, extract digits to stack, print in reverse"""
        ops = [
            RISCVDirectiveOp("li", f"t2, {STDOUT_ADDR}"),
            # zero check
            RISCVDirectiveOp("bnez", "a0, _p_int_nonzero"),
            RISCVDirectiveOp("li", "t1, 48"),  # '0'
            RISCVDirectiveOp("sb", "t1, 0(t2)"),
            RISCVDirectiveOp("j", "_p_int_nl"),
            # negative check
            RISCVLabelOp("_p_int_nonzero"),
            RISCVDirectiveOp("bgez", "a0, _p_int_pos"),
            RISCVDirectiveOp("li", "t1, 45"),  # '-'
            RISCVDirectiveOp("sb", "t1, 0(t2)"),
            RISCVDirectiveOp("neg", "a0, a0"),
            # extract and print digits
            RISCVLabelOp("_p_int_pos"),
            RISCVDirectiveOp("mv", "t0, a0"),
            *self._emit_digit_extraction_loop("_p_int"),
            *self._emit_digit_print_loop("_p_int", "_p_int_nl"),
            # newline
            RISCVLabelOp("_p_int_nl"),
            *self._emit_newline(),
            riscv_func.ReturnOp(),
        ]
        for o in ops:
            block.add_op(o)

    def _add_print_float_body(self, block):
        """generate float printing: print integer part, dot, then 6 decimal digits"""
        ops = [
            RISCVDirectiveOp("li", f"t2, {STDOUT_ADDR}"),
            # convert to int and check sign
            RISCVDirectiveOp("fcvt.w.d", "t0, fa0, rtz"),
            RISCVDirectiveOp("bgez", "t0, _pf_int_pos"),
            RISCVDirectiveOp("li", "t1, 45"),  # '-'
            RISCVDirectiveOp("sb", "t1, 0(t2)"),
            RISCVDirectiveOp("neg", "t0, t0"),
            # print integer part
            RISCVLabelOp("_pf_int_pos"),
            RISCVDirectiveOp("bnez", "t0, _pf_int_digits"),
            RISCVDirectiveOp("li", "t1, 48"),  # '0'
            RISCVDirectiveOp("sb", "t1, 0(t2)"),
            RISCVDirectiveOp("j", "_pf_dot"),
            # extract and print integer digits
            RISCVLabelOp("_pf_int_digits"),
            *self._emit_digit_extraction_loop("_pf_int"),
            *self._emit_digit_print_loop("_pf_int", "_pf_dot"),
            # print decimal point
            RISCVLabelOp("_pf_dot"),
            RISCVDirectiveOp("li", "t1, 46"),  # '.'
            RISCVDirectiveOp("sb", "t1, 0(t2)"),
            # extract fractional part: multiply by 1000000 and convert to int
            RISCVDirectiveOp("fcvt.w.d", "t0, fa0, rtz"),
            RISCVDirectiveOp("fcvt.d.w", "ft0, t0"),
            RISCVDirectiveOp("fsub.d", "fa0, fa0, ft0"),
            RISCVDirectiveOp("fabs.d", "fa0, fa0"),
            RISCVDirectiveOp("li", "t0, 1000000"),
            RISCVDirectiveOp("fcvt.d.w", "ft1, t0"),
            RISCVDirectiveOp("fmul.d", "fa0, fa0, ft1"),
            RISCVDirectiveOp("fcvt.w.d", "t0, fa0, rtz"),
            # extract 6 fractional digits (fixed count loop)
            RISCVDirectiveOp("li", "t3, 10"),
            RISCVDirectiveOp("li", "t5, 0"),
            RISCVDirectiveOp("li", "t6, 6"),
            RISCVLabelOp("_pf_frac_loop"),
            RISCVDirectiveOp("rem", "t1, t0, t3"),
            RISCVDirectiveOp("div", "t0, t0, t3"),
            RISCVDirectiveOp("addi", "t1, t1, 48"),
            RISCVDirectiveOp("addi", "sp, sp, -16"),
            RISCVDirectiveOp("sd", "t1, 0(sp)"),
            RISCVDirectiveOp("addi", "t5, t5, 1"),
            RISCVDirectiveOp("addi", "t6, t6, -1"),
            RISCVDirectiveOp("bnez", "t6, _pf_frac_loop"),
            # print fractional digits
            *self._emit_digit_print_loop("_pf_frac", "_pf_nl"),
            # newline
            RISCVLabelOp("_pf_nl"),
            *self._emit_newline(),
            riscv_func.ReturnOp(),
        ]
        for o in ops:
            block.add_op(o)


#
# recursion support
#


class AddRecursionSupportPass(ModulePass):
    """add prologue/epilogue to save/restore ra and callee-saved registers"""

    stack_frame_size_bytes = 104  # = 13 * 8 = ra + 12 s-registers (for recursion support)

    name = "add-recursion-support"

    def apply(self, _: Context, op: ModuleOp) -> None:
        for func_op in op.walk():
            if not isinstance(func_op, riscv_func.FuncOp):
                continue
            # skip special functions that don't need recursion support
            if func_op.sym_name.data in ["main", "_start", "_print_string", "_print_int", "_print_float"]:
                continue

            self._add_prologue(func_op)
            self._add_epilogue(func_op)

    def _add_prologue(self, func_op):
        """add stack frame setup: allocate space and save ra + s-registers"""
        block = func_op.body.blocks[0]
        if not block.ops:
            return

        first_op = list(block.ops)[0]

        # allocate stack: ra + 12 s-registers = 104 bytes
        block.insert_op_before(RISCVAddSpOp(-self.stack_frame_size_bytes), first_op)
        block.insert_op_before(RISCVSaveRaOp(), first_op)

        # save s0-s11 at offsets 8, 16, 24, ..., 96
        for i in range(12):
            block.insert_op_before(RISCVDirectiveOp("sd", f"s{i}, {8 + i*8}(sp)"), first_op)

    def _add_epilogue(self, func_op):
        """add stack frame teardown before each return: restore registers and ra"""
        for block in func_op.body.blocks:
            for ret in list(block.ops):
                if not isinstance(ret, riscv_func.ReturnOp):
                    continue

                # restore s11-s0 in reverse order
                for i in range(11, -1, -1):
                    block.insert_op_before(RISCVDirectiveOp("ld", f"s{i}, {8 + i*8}(sp)"), ret)

                block.insert_op_before(RISCVRestoreRaOp(), ret)
                block.insert_op_before(RISCVAddSpOp(self.stack_frame_size_bytes), ret)


#
# scf.if lowering
#


class CustomScfIfToRiscvLowering(RewritePattern):
    """lower scf.if to riscv branches with stack-based result passing"""

    def __init__(self):
        super().__init__()
        self.label_counter = 0

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: scf.IfOp, rewriter: PatternRewriter):
        # generate unique labels for this if statement
        self.label_counter += 1
        suffix = f"{self.label_counter}"
        else_label = f"else_{suffix}"
        cont_label = f"cont_{suffix}"

        reg_type = riscv.IntRegisterType.unallocated()

        # cast condition to riscv register
        cond_cast = UnrealizedConversionCastOp.create(operands=[op.cond], result_types=[reg_type])
        rewriter.insert_op(cond_cast, InsertPoint.before(op))

        zero = riscv.GetRegisterOp(riscv.Registers.ZERO)
        rewriter.insert_op(zero, InsertPoint.before(op))

        # allocate stack slots for results before branching
        res_ptrs = []
        if op.results:
            for _ in op.results:
                rewriter.insert_op(RISCVAddSpOp(-8), InsertPoint.before(op))
                sp = riscv.GetRegisterOp(riscv.Registers.SP)
                rewriter.insert_op(sp, InsertPoint.before(op))
                res_ptrs.append(sp.res)

        # branch to else if condition is zero
        beq = riscv.BeqOp(cond_cast.results[0], zero.res, offset=riscv.LabelAttr(else_label))
        rewriter.insert_op(beq, InsertPoint.before(op))

        # emit true branch
        self._emit_branch_ops(op.true_region.block, rewriter, op, res_ptrs, reg_type)
        rewriter.insert_op(riscv.JOp(riscv.LabelAttr(cont_label)), InsertPoint.before(op))

        # emit else label and false branch
        rewriter.insert_op(RISCVLabelOp(else_label), InsertPoint.before(op))
        if op.false_region.block:
            self._emit_branch_ops(op.false_region.block, rewriter, op, res_ptrs, reg_type)

        # emit continuation label
        rewriter.insert_op(RISCVLabelOp(cont_label), InsertPoint.before(op))

        # load results from stack
        final_results = []
        for i, ptr in enumerate(res_ptrs):
            load = riscv.LwOp(ptr, 0, rd=reg_type)
            rewriter.insert_op(load, InsertPoint.before(op))

            res_cast = UnrealizedConversionCastOp.create(operands=[load.rd], result_types=[op.results[i].type])
            rewriter.insert_op(res_cast, InsertPoint.before(op))
            final_results.append(res_cast.results[0])

            rewriter.insert_op(RISCVAddSpOp(8), InsertPoint.before(op))

        rewriter.replace_op(op, [], final_results)

    def _emit_branch_ops(self, block, rewriter, insert_before_op, res_ptrs, reg_type):
        """emit operations from branch block, handling yield specially"""
        for bop in list(block.ops):
            if isinstance(bop, scf.YieldOp):
                # store yielded values to stack slots
                for i, val in enumerate(bop.operands):
                    val_cast = UnrealizedConversionCastOp.create(operands=[val], result_types=[reg_type])
                    rewriter.insert_op(val_cast, InsertPoint.before(insert_before_op))
                    store = riscv.SwOp(res_ptrs[i], val_cast.results[0], 0)
                    rewriter.insert_op(store, InsertPoint.before(insert_before_op))
            else:
                bop.detach()
                rewriter.insert_op(bop, InsertPoint.before(insert_before_op))


class CustomLowerScfToRiscvPass(ModulePass):
    name = "custom-lower-scf-to-riscv"

    def apply(self, _: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(CustomScfIfToRiscvLowering()).rewrite_module(op)


#
# register allocation
#


def collect_virtual_registers(func_op):
    """collect all virtual registers from function, returns (int_set, float_set)"""
    virtual_int_regs = set()
    virtual_float_regs = set()

    # collect from operation results
    for oper in func_op.walk():
        for result in oper.results:
            if name := extract_register_name(result.type):
                if is_virtual_int_register(name):
                    virtual_int_regs.add(name)
                elif is_virtual_float_register(name):
                    virtual_float_regs.add(name)

    # collect from block arguments
    for block in func_op.body.blocks:
        for arg in block.args:
            if name := extract_register_name(arg.type):
                if is_virtual_int_register(name):
                    virtual_int_regs.add(name)
                elif is_virtual_float_register(name):
                    virtual_float_regs.add(name)

    return virtual_int_regs, virtual_float_regs


def remap_operation_registers(oper, int_map, float_map):
    """remap virtual registers to physical in operation, returns True if changed"""
    # guard: skip if no result types
    if not hasattr(oper, "result_types"):
        return False

    new_types = []
    changed = False

    for typ in oper.result_types:
        if name := extract_register_name(typ):
            # try int mapping
            if name in int_map:
                new_types.append(riscv.IntRegisterType.from_name(int_map[name]))
                changed = True
                continue
            # try float mapping
            if name in float_map:
                new_types.append(riscv.FloatRegisterType.from_name(float_map[name]))
                changed = True
                continue
        new_types.append(typ)

    # guard: skip if nothing changed
    if not changed:
        return False

    # reconstruct operation with new register types
    regions = [r for r in oper.regions]
    for r in regions:
        r.detach()

    new_op = oper.__class__.create(operands=oper.operands, result_types=new_types, attributes=oper.attributes, successors=oper.successors, regions=regions)

    # replace usages and update block
    for i, old_res in enumerate(oper.results):
        old_res.replace_by(new_op.results[i])

    if block := oper.parent_block():
        block.insert_op_before(new_op, oper)
        block.detach_op(oper)

    return True


def remap_block_arg_registers(block, int_map, float_map):
    """remap virtual registers in block arguments"""
    for arg in block.args:
        if name := extract_register_name(arg.type):
            if name in int_map:
                arg.type = riscv.IntRegisterType.from_name(int_map[name])
            elif name in float_map:
                arg.type = riscv.FloatRegisterType.from_name(float_map[name])


class MapToPhysicalRegistersPass(ModulePass):
    """map virtual registers (j_*, fj_*) to physical registers (s*, t*, fs*, ft*)"""

    name = "map-to-physical-registers"

    def apply(self, _: Context, op: ModuleOp) -> None:
        for func_op in op.walk():
            if not isinstance(func_op, riscv_func.FuncOp):
                continue

            # collect virtual registers
            virtual_int_regs, virtual_float_regs = collect_virtual_registers(func_op)

            # guard: skip if no virtual registers
            if not virtual_int_regs and not virtual_float_regs:
                continue

            # create mapping to physical registers
            # prioritize callee-saved (s*, fs*) to reduce spilling across calls
            physical_int_regs = [f"s{i}" for i in range(12)] + [f"t{i}" for i in range(7)]
            physical_float_regs = [f"fs{i}" for i in range(12)] + [f"ft{i}" for i in range(12)]

            # guard: check register limits
            if len(virtual_int_regs) > len(physical_int_regs):
                raise RuntimeError(f"too many virtual int registers: {len(virtual_int_regs)} > {len(physical_int_regs)}")
            if len(virtual_float_regs) > len(physical_float_regs):
                raise RuntimeError(f"too many virtual float registers: {len(virtual_float_regs)} > {len(physical_float_regs)}")

            # sort virtual registers by index for deterministic mapping
            virtual_int_list = sorted(virtual_int_regs, key=lambda x: int(x.split("_")[1]))
            virtual_float_list = sorted(virtual_float_regs, key=lambda x: int(x.split("_")[1]))

            int_reg_map = {v: physical_int_regs[i] for i, v in enumerate(virtual_int_list)}
            float_reg_map = {v: physical_float_regs[i] for i, v in enumerate(virtual_float_list)}

            # apply mapping to function and all children
            remap_operation_registers(func_op, int_reg_map, float_reg_map)
            for child in func_op.walk():
                remap_operation_registers(child, int_reg_map, float_reg_map)
                # remap block arguments in nested regions
                for region in child.regions:
                    for block in region.blocks:
                        remap_block_arg_registers(block, int_reg_map, float_reg_map)
