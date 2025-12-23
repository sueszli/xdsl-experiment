from typing import Callable

from qemu import STDOUT_ADDR
from xdsl.context import Context
from xdsl.dialects import arith, func, llvm, printf, riscv, riscv_func, scf
from xdsl.dialects.builtin import IntegerAttr, ModuleOp, StringAttr, SymbolRefAttr, UnrealizedConversionCastOp
from xdsl.ir import Attribute, Block, Region, SSAValue
from xdsl.irdl import attr_def, base, irdl_op_definition, result_def
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriter, PatternRewriteWalker, RewritePattern, op_type_rewrite_pattern
from xdsl.rewriter import InsertPoint

#
# custom risc-v assembly ops (used later in lowering)
#


@irdl_op_definition
class RISCVGlobalOp(riscv.RISCVAsmOperation):
    # global symbol in .data section:
    #
    #   .data
    #   .globl str_hello
    #   str_hello:
    #   .string "hello!"

    name = "riscv.global"
    sym_name = attr_def(StringAttr)  # variable name, e.g. str0
    value = attr_def(base(Attribute))  # data value, e.g. "hello!"
    is_constant = attr_def(base(Attribute))  # 1 = constant, 0 = mutable

    def __init__(self, sym: str | StringAttr, val: Attribute, const: bool = True):
        super().__init__(attributes={"sym_name": StringAttr(sym) if isinstance(sym, str) else sym, "value": val, "is_constant": IntegerAttr(1 if const else 0, 1)})

    def assembly_line(self):
        # doesn't emit, gets transformed later by LowerRISCVGlobalOp
        return None


@irdl_op_definition
class RISCVLaOp(riscv.RISCVAsmOperation):
    name = "riscv.la"  # load address
    rd = result_def(riscv.IntRegisterType)  # register destination, e.g. a0
    label = attr_def(riscv.LabelAttr)  # label to load, e.g. str_hello

    def __init__(self, label: str | riscv.LabelAttr, rd_type):
        super().__init__(result_types=[rd_type], attributes={"label": riscv.LabelAttr(label) if isinstance(label, str) else label})

    def assembly_line(self):
        return f"la {self.rd.type.register_name.data}, {self.label.data}"


@irdl_op_definition
class RISCVDirectiveOp(riscv.RISCVAsmOperation):
    # directives are program metadata like .data, .text, ...
    name = "riscv.directive"
    directive = attr_def(StringAttr)
    value = attr_def(StringAttr)

    def __init__(self, directive: str, value: str = ""):
        super().__init__(attributes={"directive": StringAttr(directive), "value": StringAttr(value)})

    def assembly_line(self):
        return f"{self.directive.data} {self.value.data}" if self.value.data else self.directive.data


@irdl_op_definition
class RISCVLabelOp(riscv.RISCVAsmOperation):
    # label definition, so that jumps can target it, like "main:"

    name = "riscv.label"
    label = attr_def(riscv.LabelAttr)

    def __init__(self, label: str):
        super().__init__(attributes={"label": riscv.LabelAttr(label)})

    def assembly_line(self):
        return f"{self.label.data}:"


#
# arith.SelectOp lowering
#


class SelectOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.SelectOp, rewriter: PatternRewriter):
        # branchless select:
        # result = (true_val & mask) | (false_val & ~mask)
        # mask = 0 - cond (produces 0x00.. or 0xFF..)
        rtype = riscv.IntRegisterType.unallocated()

        def cast(val):
            c = UnrealizedConversionCastOp.create(operands=[val], result_types=[rtype])
            rewriter.insert_op(c, InsertPoint.before(op))
            return c.results[0]

        cond, true_v, false_v = cast(op.cond), cast(op.lhs), cast(op.rhs)
        zero = riscv.GetRegisterOp(riscv.Registers.ZERO)
        rewriter.insert_op(zero, InsertPoint.before(op))

        # create mask
        mask = riscv.SubOp(zero.res, cond, rd=rtype)
        rewriter.insert_op(mask, InsertPoint.before(op))

        # t1 = true_val & mask
        t1 = riscv.AndOp(true_v, mask.rd, rd=rtype)
        rewriter.insert_op(t1, InsertPoint.before(op))

        # t2 = false_val & ~mask
        not_mask = riscv.XoriOp(mask.rd, -1, rd=rtype)
        rewriter.insert_op(not_mask, InsertPoint.before(op))
        t2 = riscv.AndOp(false_v, not_mask.rd, rd=rtype)
        rewriter.insert_op(t2, InsertPoint.before(op))

        # result = t1 | t2
        res = riscv.OrOp(t1.rd, t2.rd, rd=rtype)
        rewriter.insert_op(res, InsertPoint.before(op))

        # cast back
        final = UnrealizedConversionCastOp.create(operands=[res.rd], result_types=[op.result.type])
        rewriter.insert_op(final, InsertPoint.before(op))
        rewriter.replace_op(op, [], [final.results[0]])


class LowerSelectPass(ModulePass):
    name = "lower-select"

    def apply(self, ctx: Context, op: ModuleOp):
        PatternRewriteWalker(SelectOpLowering()).rewrite_module(op)


#
# global data & data section
#


class RemoveUnprintableOpsPass(ModulePass):
    name = "remove-unprintable-ops"

    def apply(self, ctx: Context, op: ModuleOp):
        # llvm.global ops -> riscv.global ops
        for g in [o for o in op.walk() if isinstance(o, llvm.GlobalOp)]:
            g.parent_block().insert_op_before(RISCVGlobalOp(g.sym_name.data, g.value, g.constant is not None), g)
            g.detach()

        # llvm.addressof -> riscv.la opss
        for a in [o for o in op.walk() if isinstance(o, llvm.AddressOfOp)]:
            la = RISCVLaOp(a.global_name.root_reference.data, riscv.IntRegisterType.unallocated())
            a.parent_block().insert_op_before(la, a)
            a.results[0].replace_by(la.rd)
            a.detach()


class LowerRISCVGlobalOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: RISCVGlobalOp, rewriter: PatternRewriter):
        data = bytes(op.value.data.data)
        content = data.split(b"\0", 1)[0].decode("utf-8", errors="replace")
        escaped = content.translate(str.maketrans({"\n": "\\n", "\t": "\\t", '"': '\\"', "\\": "\\\\"}))

        # .data               # switch to data section
        # .globl str_hello    # make symbol visible to linker
        # str_hello:          # define the label
        # .string "hello!"    # emit the string data
        rewriter.insert_op(RISCVDirectiveOp(".data"), InsertPoint.before(op))
        rewriter.insert_op(RISCVDirectiveOp(".globl", op.sym_name.data), InsertPoint.before(op))
        rewriter.insert_op(RISCVLabelOp(op.sym_name.data), InsertPoint.before(op))
        rewriter.insert_op(RISCVDirectiveOp(".string", f'"{escaped}"'), InsertPoint.before(op))
        rewriter.erase_op(op)


class EmitDataSectionPass(ModulePass):
    name = "emit-data-section"

    def apply(self, ctx: Context, op: ModuleOp):
        PatternRewriteWalker(LowerRISCVGlobalOp()).rewrite_module(op)
        # prepend .text to first function (which is "main" in our case)
        if f_op := next((o for o in op.body.blocks[0].ops if isinstance(o, (riscv_func.FuncOp, func.FuncOp))), None):
            op.body.blocks[0].insert_op_before(RISCVDirectiveOp(".text"), f_op)


#
# printf lowering
#


class LowerPrintfPass(ModulePass):
    name = "lower-printf"

    def apply(self, ctx: Context, op: ModuleOp):
        unwrap_cast = lambda val: (val.owner.operands[0], val.owner) if isinstance(val.owner, UnrealizedConversionCastOp) else (val, None)

        # printf.PrintFormatOp -> calls to one of the _print_* runtime functions
        for p in [o for o in op.walk() if isinstance(o, printf.PrintFormatOp) if o.operands]:
            val, cast = unwrap_cast(p.operands[0])
            is_lbl = isinstance(getattr(val.owner, "label", None), riscv.LabelAttr) or isinstance(getattr(val.owner, "immediate", None), riscv.LabelAttr)
            is_int = isinstance(val.type, riscv.IntRegisterType)
            is_flt = isinstance(val.type, riscv.FloatRegisterType)

            if is_lbl:
                self._emit(p, val, "_print_string", False)
            elif is_int:
                self._emit(p, val, "_print_int", False)
            elif is_flt:
                self._emit(p, val, "_print_float", True)

            p.detach()
            if cast:
                cast.detach()

    def _emit(self, op: printf.PrintFormatOp, arg: SSAValue, fn: str, is_float: bool) -> None:
        # before:
        #   printf.print_format_op(%result)
        #
        # after:
        #   addi a0, t0, 0     # copy value from source register (e.g. t0) to a0 (by adding 0)
        #   call _print_int    # call the _print_int function (from runtime)
        reg_type = riscv.FloatRegisterType.from_name("fa0") if is_float else riscv.IntRegisterType.from_name("a0")
        mv = riscv.FMvDOp(arg, rd=reg_type) if is_float else riscv.AddiOp(arg, 0, rd=reg_type)
        op.parent_block().insert_op_before(mv, op)
        op.parent_block().insert_op_before(riscv_func.CallOp(SymbolRefAttr(fn), [mv.rd], []), op)


#
# runtime library
#


class AddPrintRuntimePass(ModulePass):
    name = "add-print-runtime"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        self._add_func(op, "_print_string", "a0", self._print_string_asm)
        self._add_func(op, "_print_int", "a0", self._print_int_asm)
        self._add_func(op, "_print_float", "fa0", self._print_float_asm)

    def _add_func(self, module: ModuleOp, name: str, arg_reg: str, asm_gen_fn: Callable[[], list[str]]) -> None:
        arg_t = riscv.FloatRegisterType.from_name(arg_reg) if arg_reg.startswith("f") else riscv.IntRegisterType.from_name(arg_reg)

        # function with signature: fn(arg_t) -> void
        f = riscv_func.FuncOp(name=name, region=Region(Block(arg_types=[arg_t])), function_type=func.FunctionType.from_lists([arg_t], []))

        asm_instructions = asm_gen_fn()
        self._emit_block(f.body.blocks[0], asm_instructions)

        module.body.blocks[0].add_op(f)

    def _emit_block(self, block: Block, asm_instructions: list[str]) -> None:
        # convert assembly instruction strings to IR operations
        for line in asm_instructions:
            if line == "ret":
                # jump back to caller (= `jalr x0, x1, 0` where x1 is return address register)
                block.add_op(riscv_func.ReturnOp())
                continue
            if line.endswith(":"):
                # label definition: "_ps_loop:" -> RISCVLabelOp("_ps_loop")
                block.add_op(RISCVLabelOp(line[:-1]))
            else:
                # regular instruction: "mv t0, a0" -> RISCVDirectiveOp("mv", "t0, a0")
                parts = line.split(" ", 1)
                block.add_op(RISCVDirectiveOp(parts[0], parts[1] if len(parts) > 1 else ""))

    def _print_string_asm(self) -> list[str]:
        # while (*str != '\0') { *STDOUT_ADDR = *str; str++; }
        return [
            f"mv t0, a0",
            f"li t2, {STDOUT_ADDR}",
            "_ps_loop:",
            "lbu t1, 0(t0)",
            "beqz t1, _ps_done",
            "sb t1, 0(t2)",
            "addi t0, t0, 1",
            "j _ps_loop",
            "_ps_done:",
        ] + self._print_newline()

    def _print_int_asm(self) -> list[str]:
        return (
            [
                f"li t2, {STDOUT_ADDR}",
                "bnez a0, _pi_nz",
                "li t1, 48",  # '0'
                "sb t1, 0(t2)",
                "j _pi_done",
                "_pi_nz:",
                "bgez a0, _pi_pos",
                "li t1, 45",  # '-'
                "sb t1, 0(t2)",
                "neg a0, a0",
                "_pi_pos:",
                "mv t0, a0",
            ]
            + self._emit_digits("_pi_body")
            + ["_pi_done:"]
            + self._print_newline()
        )

    def _print_float_asm(self) -> list[str]:
        return (
            # print integer part
            [
                f"li t2, {STDOUT_ADDR}",
                "fcvt.w.d t0, fa0, rtz",  # round towards zero
                "bgez t0, _pf_pos",
                "li t1, 45",  # '-'
                "sb t1, 0(t2)",
                "neg t0, t0",
                "_pf_pos:",
                "bnez t0, _pf_int",
                "li t1, 48",  # '0'
                "sb t1, 0(t2)",
                "j _pf_dot",
                "_pf_int:",
            ]
            + self._emit_digits("_pf_body")
            # print fractional part
            + [
                "_pf_dot:",
                "li t1, 46",  # '.'
                "sb t1, 0(t2)",
                "fcvt.w.d t0, fa0, rtz",  # float to int
                "fcvt.d.w ft0, t0",  # int to float
                "fsub.d fa0, fa0, ft0",  # fa0 = fa0 - intpart
                "fabs.d fa0, fa0",  # get fractional part
                "li t0, 1000000",
                "fcvt.d.w ft1, t0",
                "fmul.d fa0, fa0, ft1",  # * 1_000_000 to shift decimal places
                "fcvt.w.d t0, fa0, rtz",
                # print 6 frac digits
                "li t3, 10",
                "li t5, 0",
                "li t6, 6",
                "_pf_frac_loop:",  # can't call `_emit_digits` here due to leading zeros
                "rem t1, t0, t3",  # t1 = t0 % 10
                "div t0, t0, t3",  # t0 = t0 / 10
                "addi t1, t1, 48",  # convert to ASCII, where '0' = 48
                "addi sp, sp, -16",
                "sd t1, 0(sp)",
                "addi t5, t5, 1",
                "addi t6, t6, -1",
                "bnez t6, _pf_frac_loop",
                "_pf_frac_print:",
                "beqz t5, _pf_done",
                "ld t1, 0(sp)",
                "addi sp, sp, 16",
                "sb t1, 0(t2)",
                "addi t5, t5, -1",
                "j _pf_frac_print",
                "_pf_done:",
            ]
            + self._print_newline()
        )

    def _emit_digits(self, lbl: str) -> list[str]:
        return [
            f"li t3, 10",
            f"li t5, 0",
            # push digits to stack
            f"{lbl}_loop:",
            f"beqz t0, {lbl}_print",  # while t0 != 0
            f"rem t1, t0, t3",  # t1 = t0 % 10
            f"div t0, t0, t3",  # t0 = t0 / 10
            f"addi t1, t1, 48",  # convert to ASCII, where '0' = 48
            f"addi sp, sp, -16",
            f"sd t1, 0(sp)",
            f"addi t5, t5, 1",
            f"j {lbl}_loop",
            # pop and print digits
            f"{lbl}_print:",
            f"beqz t5, {lbl}_end",
            f"ld t1, 0(sp)",
            f"addi sp, sp, 16",
            f"sb t1, 0(t2)",  # print char
            f"addi t5, t5, -1",
            f"j {lbl}_print",
            f"{lbl}_end:",
        ]

    def _print_newline(self) -> list[str]:
        return [
            f"li t1, 10",  # 10 is newline character
            f"sb t1, 0(t2)",  # t2 stores STDOUT_ADDR
            "ret",
        ]


#
# recursion support
#


class AddRecursionSupportPass(ModulePass):
    # adds code to all user-defined functions to support recursion and function calls.
    # - pushes current frame (return address and callee-saved registers) onto the stack at function entry (prologue).
    # - restores frame from stack before returning (epilogue).
    name = "add-recursion-support"
    FRAME_SZ = 104  # ra (8 bytes) + 12 s-regs (s0-s11, 8 bytes each) = 8 + 96 = 104 bytes

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        skip = {"main", "_start", "_print_string", "_print_int", "_print_float"}

        user_functions = [f for f in op.walk() if isinstance(f, riscv_func.FuncOp) and f.sym_name.data not in skip]
        for f in user_functions:
            self._prologue(f.body.blocks[0])

            # epilogues, per return statement (could be multiple per function)
            for ret in [r for b in f.body.blocks for r in b.ops if isinstance(r, riscv_func.ReturnOp)]:
                self._epilogue(ret)

    def _prologue(self, blk: Block) -> None:
        if not blk.ops:
            return

        ops: list[RISCVDirectiveOp] = [
            RISCVDirectiveOp("addi", f"sp, sp, -{self.FRAME_SZ}"),  # allocate stack space, move stack pointer
            RISCVDirectiveOp("sd", "ra, 0(sp)"),  # store return address (ra) at offset 0
        ]
        ops += [RISCVDirectiveOp("sd", f"s{i}, {8 + i*8}(sp)") for i in range(12)]  # store callee-saved registers (s0-s11) at offsets 8 to 104

        self._emit(blk, list(blk.ops)[0], ops)  # insert before first op in block

    def _epilogue(self, ret: riscv_func.ReturnOp) -> None:
        ops: list[RISCVDirectiveOp] = [RISCVDirectiveOp("ld", f"s{i}, {8 + i*8}(sp)") for i in range(12)]

        ops += [
            RISCVDirectiveOp("ld", "ra, 0(sp)"),  # restore return address from offset 0
            RISCVDirectiveOp("addi", f"sp, sp, {self.FRAME_SZ}"),  # deallocate stack frame
        ]

        self._emit(ret.parent_block(), ret, ops)  # insert before return operation

    def _emit(self, block: Block, point: riscv.RISCVAsmOperation, ops: list[RISCVDirectiveOp]) -> None:
        for o in ops:
            block.insert_op_before(o, point)


#
# scf.IfOp lowering
#


class CustomScfIfToRiscvLowering(RewritePattern):
    label_cnt = 0

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: scf.IfOp, rewriter: PatternRewriter):
        # unique labels for else-branch and continuation-point
        self.label_cnt += 1
        lbl_else = f"else_{self.label_cnt}"
        lbl_cont = f"cont_{self.label_cnt}"
        rtype = riscv.IntRegisterType.unallocated()

        # pass condition to register
        reg_cond = UnrealizedConversionCastOp.create(operands=[op.cond], result_types=[rtype])
        rewriter.insert_op(reg_cond, InsertPoint.before(op))
        zero = riscv.GetRegisterOp(riscv.Registers.ZERO)
        rewriter.insert_op(zero, InsertPoint.before(op))

        # scf.IfOp is expressed as a function with scf.YieldOps for results / return values.
        # for each result, allocate 8 bytes on stack to store it.
        res_ptrs = []
        for _ in op.results:
            rewriter.insert_op(RISCVDirectiveOp("addi", "sp, sp, -8"), InsertPoint.before(op))
            res_ptrs.append(riscv.GetRegisterOp(riscv.Registers.SP))
            rewriter.insert_op(res_ptrs[-1], InsertPoint.before(op))

        # reg_cond != zero -> jump to else label
        # reg_cond == zero -> fall through to true branch
        rewriter.insert_op(riscv.BeqOp(reg_cond.results[0], zero.res, offset=riscv.LabelAttr(lbl_else)), InsertPoint.before(op))

        def emit_region(region, dest_lbl=None):
            """
            Emit operations from a region (true or false branch).
            - Replaces scf.YieldOp with stack stores (sw) to save results
            - Moves all other operations inline
            - Optionally jumps to dest_lbl at end
            """
            if not region.block:
                return
            for o in list(region.block.ops):
                if isinstance(o, scf.YieldOp):
                    # YieldOp returns values from the branch - store them to stack
                    for i, val in enumerate(o.operands):
                        vc = UnrealizedConversionCastOp.create(operands=[val], result_types=[rtype])
                        rewriter.insert_op(vc, InsertPoint.before(op))
                        rewriter.insert_op(riscv.SwOp(res_ptrs[i].res, vc.results[0], 0), InsertPoint.before(op))  # sw: store word
                else:
                    # regular operation - move it inline
                    o.detach()
                    rewriter.insert_op(o, InsertPoint.before(op))
            if dest_lbl:
                # jump to continuation label (skips the else branch)
                rewriter.insert_op(RISCVDirectiveOp("j", dest_lbl), InsertPoint.before(op))

        # emit true branch and jump to continuation
        emit_region(op.true_region, lbl_cont)
        # emit else label (target for failed condition)
        rewriter.insert_op(RISCVLabelOp(lbl_else), InsertPoint.before(op))
        # emit false branch (no jump needed, falls through to continuation)
        emit_region(op.false_region)
        # emit continuation label where both branches meet
        rewriter.insert_op(RISCVLabelOp(lbl_cont), InsertPoint.before(op))

        # load results from stack and convert back to original types
        finals = []
        for i, ptr in enumerate(res_ptrs):
            load = riscv.LwOp(ptr.res, 0, rd=rtype)  # lw: load word from stack
            rewriter.insert_op(load, InsertPoint.before(op))
            fn = UnrealizedConversionCastOp.create(operands=[load.rd], result_types=[op.results[i].type])
            rewriter.insert_op(fn, InsertPoint.before(op))
            finals.append(fn.results[0])
            rewriter.insert_op(RISCVDirectiveOp("addi", "sp, sp, 8"), InsertPoint.before(op))  # deallocate stack slot

        # replace the original if-op with the loaded results
        rewriter.replace_op(op, [], finals)


class CustomLowerScfToRiscvPass(ModulePass):
    name = "custom-lower-scf-to-riscv"

    def apply(self, ctx: Context, op: ModuleOp):
        PatternRewriteWalker(CustomScfIfToRiscvLowering()).rewrite_module(op)


#
# register allocation
#


class MapToPhysicalRegistersPass(ModulePass):
    name = "map-to-physical-registers"

    def apply(self, ctx: Context, op: ModuleOp):
        for f in [o for o in op.walk() if isinstance(o, riscv_func.FuncOp)]:
            self._process_func(f)

    def _process_func(self, f):
        v_int, v_flt = set(), set()

        def get_reg_name(reg_type) -> str | None:
            if not isinstance(reg_type, (riscv.IntRegisterType, riscv.FloatRegisterType)):
                return None
            name = getattr(reg_type, "register_name", None)
            return name.data if isinstance(name, StringAttr) else name

        # Collect virtuals
        is_virtual_reg = lambda name, prefix: name and name.startswith(prefix)
        for o in f.walk():
            for r in list(o.results) + [a for b in o.regions for bb in b.blocks for a in bb.args]:
                if n := get_reg_name(r.type):
                    if is_virtual_reg(n, "j_"):
                        v_int.add(n)
                    elif is_virtual_reg(n, "fj_"):
                        v_flt.add(n)

        if not v_int and not v_flt:
            return

        # Map to physical
        p_int = [f"s{i}" for i in range(12)] + [f"t{i}" for i in range(7)]
        p_flt = [f"fs{i}" for i in range(12)] + [f"ft{i}" for i in range(12)]

        if len(v_int) > len(p_int):
            raise RuntimeError(f"too many int regs: {len(v_int)}")
        if len(v_flt) > len(p_flt):
            raise RuntimeError(f"too many float regs: {len(v_flt)}")

        imap = {v: p_int[i] for i, v in enumerate(sorted(v_int, key=lambda x: int(x.split("_")[1])))}
        fmap = {v: p_flt[i] for i, v in enumerate(sorted(v_flt, key=lambda x: int(x.split("_")[1])))}

        # Apply mapping
        def remap_type(t):
            n = get_reg_name(t)
            if n in imap:
                return riscv.IntRegisterType.from_name(imap[n])
            if n in fmap:
                return riscv.FloatRegisterType.from_name(fmap[n])
            return t

        all_ops = list(f.walk())
        all_ops.append(f)
        for o in all_ops:
            # Remap results
            target_op = o
            if hasattr(o, "result_types"):
                new_t = [remap_type(t) for t in o.result_types]
                if new_t != list(o.result_types):
                    new_op = o.__class__.create(operands=o.operands, result_types=new_t, attributes=o.attributes, successors=o.successors)
                    for i, r in enumerate(o.regions):
                        r.move_blocks(new_op.regions[i])

                    if o.parent_block():
                        o.parent_block().insert_op_before(new_op, o)
                        for i, res in enumerate(o.results):
                            res.replace_by(new_op.results[i])
                        o.detach()
                    target_op = new_op

            # Remap region args
            for r in target_op.regions:
                for b in list(r.blocks):
                    new_types = [remap_type(a.type) for a in b.args]
                    if any(n != a.type for n, a in zip(new_types, b.args)):
                        new_b = Block(arg_types=new_types)
                        for old_a, new_a in zip(b.args, new_b.args):
                            old_a.replace_by(new_a)
                        while b.ops:
                            op = b.first_op
                            op.detach()
                            new_b.add_op(op)

                        # Replace block
                        idx = next(i for i, blk in enumerate(r.blocks) if blk is b)
                        r.detach_block(b)
                        r.insert_block(new_b, idx)

            # Update FuncOp signature
            if isinstance(target_op, (riscv_func.FuncOp, func.FuncOp)):
                if target_op.body.blocks:
                    in_t = [a.type for a in target_op.body.blocks[0].args]
                    out_t = [remap_type(t) for t in target_op.function_type.outputs]
                    target_op.function_type = func.FunctionType.from_lists(in_t, out_t)
