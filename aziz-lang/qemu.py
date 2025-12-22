import shutil
import subprocess
import tempfile
from pathlib import Path

from elftools.elf.elffile import ELFFile
from unicorn import UC_ARCH_RISCV, UC_HOOK_CODE, UC_HOOK_MEM_WRITE, UC_MODE_RISCV64, Uc
from unicorn.riscv_const import UC_RISCV_REG_A0, UC_RISCV_REG_A1, UC_RISCV_REG_A7, UC_RISCV_REG_MSTATUS, UC_RISCV_REG_SP, UC_RISCV_REG_T0, UC_RISCV_REG_T1, UC_RISCV_REG_T2

MEMORY_BASE_ADDR = 0x10000  # offset for unicorn engine's address space
MEMORY_SIZE_BYTES = 0x11000000  # 272 MB total virtual memory region in emulated program
STDOUT_ADDR = 0x10000000
HALT_ADDR = 0x100000
HALT_MAGIC_VALUE = 0x5555
SYSCALL_EXIT = 93  # id for exit syscall in RISC-V linux ABI
ECALL_INSTRUCTION = b"\x73\x00\x00\x00"
MAX_INSTRUCTION_COUNT = 1_000_000  # livelock prevention


def _assemble_and_link(asm_code: str, tmp: Path) -> Path:
    # assembly -> linking
    (tmp / "code.s").write_text(asm_code)
    result = subprocess.run(["riscv64-unknown-elf-as", "-o", tmp / "code.o", tmp / "code.s"], capture_output=True, text=True)
    assert result.returncode == 0, f"assembly error:\n{result.returncode=}\n{result.args=}\n{result.stdout=}\n{result.stderr=}"

    # linking -> ELF executable
    linker_script = f"""
    SECTIONS {{
        . = {MEMORY_BASE_ADDR:#x};
        .text : {{ *(.text*) }}
        .data : {{ *(.data*) }}
        .rodata : {{ *(.rodata*) }}
    }}
    """
    (tmp / "link.ld").write_text(linker_script)
    result = subprocess.run(["riscv64-unknown-elf-ld", "-T", tmp / "link.ld", "-o", tmp / "code.elf", tmp / "code.o"], capture_output=True, text=True)
    assert result.returncode == 0, f"linking error:\n{result.returncode=}\n{result.args=}\n{result.stdout=}\n{result.stderr=}"
    return tmp / "code.elf"


def _find_symbol_address(elf_path: Path, symbol_name: str) -> int:
    # locate mem address of a symbol in ELF (e.g. "main") executable
    nm_output = subprocess.run(["riscv64-unknown-elf-nm", elf_path], capture_output=True, text=True, check=True).stdout
    assert nm_output
    for line in nm_output.splitlines():
        if symbol_name in line:
            return int(line.split()[0], 16)
    assert False, f"symbol '{symbol_name}' not found"


def _load_elf_segments(elf: ELFFile, emulator: Uc) -> None:
    # load ELF segments into emulator memory

    for segment in elf.iter_segments():
        if segment["p_type"] != "PT_LOAD":
            continue
        if segment_data := segment.data():
            emulator.mem_write(segment["p_vaddr"], segment_data)


def _create_execution_hooks(emulator: Uc, output_buffer: list[str]) -> None:
    # sets up syscall and memory I/O interception

    def code_hook(uc: Uc, addr: int, size_bytes: int, _) -> None:
        # exit syscall
        if uc.mem_read(addr, 4) != ECALL_INSTRUCTION:
            return
        if uc.reg_read(UC_RISCV_REG_A7) == SYSCALL_EXIT:
            uc.emu_stop()

    def mem_write_hook(uc: Uc, _, addr: int, size_bytes: int, value: int, __) -> None:
        if addr == STDOUT_ADDR:
            # map writes to STDOUT_ADDR to console output
            output_buffer.append(chr(value & 0xFF))
        elif addr == HALT_ADDR and value == HALT_MAGIC_VALUE:
            # map writes of HALT_MAGIC_VALUE to HALT_ADDR to emulator stop
            uc.emu_stop()

    emulator.hook_add(UC_HOOK_CODE, code_hook)
    emulator.hook_add(UC_HOOK_MEM_WRITE, mem_write_hook)


def emulate_riscv(asm_code: str, entry_symbol: str = "main") -> str:
    assert shutil.which("riscv64-unknown-elf-as") and shutil.which("riscv64-unknown-elf-ld"), "compiler not found"
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        elf_path = _assemble_and_link(asm_code, tmp_dir)
        entry_addr = _find_symbol_address(elf_path, entry_symbol)

        emulator = Uc(UC_ARCH_RISCV, UC_MODE_RISCV64)

        # enable fpu: set mstatus.fs (bits 13-14) to 0b11="dirty" state
        # without this, floating-point instructions like fld/fsd cause cpu exceptions
        mstatus = emulator.reg_read(UC_RISCV_REG_MSTATUS)
        mstatus |= 0b11 << 13
        emulator.reg_write(UC_RISCV_REG_MSTATUS, mstatus)

        emulator.mem_map(MEMORY_BASE_ADDR, MEMORY_SIZE_BYTES)
        _load_elf_segments(ELFFile(open(elf_path, "rb")), emulator)

        output_buffer = []
        _create_execution_hooks(emulator, output_buffer)

        # set stack pointer to top of memory with 16-byte alignment (required by risc-v abi)
        # subtract 16 to leave room for stack growth, then mask lower 4 bits to align
        stack_top = (MEMORY_BASE_ADDR + MEMORY_SIZE_BYTES - 16) & ~0xF
        emulator.reg_write(UC_RISCV_REG_SP, stack_top)

        # start emulation from entry point until hooks stop it or max instruction count reached
        # end_addr=0 means run indefinitely (until stopped by hooks or count limit)
        try:
            emulator.emu_start(entry_addr, 0, count=MAX_INSTRUCTION_COUNT)
        except Exception as e:
            print(f"[ERROR] Emulation exception: {e}")
            print(f"[ERROR] SP = 0x{emulator.reg_read(UC_RISCV_REG_SP):x}")
            print(f"[ERROR] Entry addr = 0x{entry_addr:x}")
            raise

        reg_map = [("t0", UC_RISCV_REG_T0), ("t1", UC_RISCV_REG_T1), ("t2", UC_RISCV_REG_T2), ("a0", UC_RISCV_REG_A0), ("a1", UC_RISCV_REG_A1), ("a7", UC_RISCV_REG_A7)]
        result = {"output": "".join(output_buffer), "regs": {name: emulator.reg_read(reg_id) for name, reg_id in reg_map}}
        return result["output"]
