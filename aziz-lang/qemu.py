# /// script
# requires-python = "==3.14"
# dependencies = [
#     "unicorn==2.1.4",
#     "pyelftools==0.32",
# ]
# ///

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from elftools.elf.elffile import ELFFile
from unicorn import UC_ARCH_RISCV, UC_HOOK_CODE, UC_HOOK_MEM_WRITE, UC_MODE_RISCV64, Uc
from unicorn.riscv_const import UC_RISCV_REG_A0, UC_RISCV_REG_A1, UC_RISCV_REG_A7, UC_RISCV_REG_T0, UC_RISCV_REG_T1, UC_RISCV_REG_T2

MEMORY_BASE_ADDR = 0x10000
MEMORY_SIZE_BYTES = 0x11000000
STDOUT_ADDR = 0x10000000
HALT_ADDR = 0x100000
HALT_MAGIC_VALUE = 0x5555
SYSCALL_EXIT = 93
ECALL_INSTRUCTION = b"\x73\x00\x00\x00"
MAX_INSTRUCTION_COUNT = 100000
LINKER_SCRIPT = """
SECTIONS {
    . = 0x10000;
    .text : { *(.text*) }
    .data : { *(.data*) }
    .rodata : { *(.rodata*) }
}
"""


def _assemble_and_link(asm_code: str, tmp: Path) -> Path:
    # assembly -> executable elf binary
    (tmp / "link.ld").write_text(LINKER_SCRIPT)
    (tmp / "code.s").write_text(asm_code)
    result = subprocess.run(["riscv64-unknown-elf-as", "-o", tmp / "code.o", tmp / "code.s"], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"assembly error:\n{result.stderr}")
        raise subprocess.CalledProcessError(result.returncode, result.args, result.stdout, result.stderr)
    result = subprocess.run(["riscv64-unknown-elf-ld", "-T", tmp / "link.ld", "-o", tmp / "code.elf", tmp / "code.o"], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"linking error:\n{result.stderr}")
        raise subprocess.CalledProcessError(result.returncode, result.args, result.stdout, result.stderr)
    return tmp / "code.elf"


def _find_symbol_address(elf_path: Path, symbol_name: str) -> int:
    # find memory address of symbol in ELF like "main"
    nm_output = subprocess.run(["riscv64-unknown-elf-nm", elf_path], capture_output=True, text=True, check=True).stdout
    for line in nm_output.splitlines():
        if symbol_name in line:
            return int(line.split()[0], 16)
    assert False, f"symbol '{symbol_name}' not found"


def _load_elf_segments(elf: ELFFile, emulator: Uc) -> None:
    # copy executable code and data from ELF into emulator memory
    for segment in elf.iter_segments():
        if segment["p_type"] != "PT_LOAD":
            continue
        if segment_data := segment.data():
            emulator.mem_write(segment["p_vaddr"], segment_data)


def _create_execution_hooks(emulator: Uc, output_buffer: list[str]) -> None:
    # intercept syscalls and memory-mapped io during execution
    def code_hook(uc: Uc, addr: int, size_bytes: int, _) -> None:
        # stop emulation when exit syscall is called
        if uc.mem_read(addr, 4) != ECALL_INSTRUCTION:
            return
        if uc.reg_read(UC_RISCV_REG_A7) == SYSCALL_EXIT:
            uc.emu_stop()

    def mem_write_hook(uc: Uc, _, addr: int, size_bytes: int, value: int, __) -> None:
        # capture writes to stdout and halt addresses
        if addr == STDOUT_ADDR:
            output_buffer.append(chr(value & 0xFF))
        elif addr == HALT_ADDR and value == HALT_MAGIC_VALUE:
            uc.emu_stop()

    emulator.hook_add(UC_HOOK_CODE, code_hook)
    emulator.hook_add(UC_HOOK_MEM_WRITE, mem_write_hook)


def run_riscv(asm_code: str, entry_symbol: str = "main") -> dict[str, any]:
    assert shutil.which("riscv64-unknown-elf-as") and shutil.which("riscv64-unknown-elf-ld"), "compiler not found"
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        elf_path = _assemble_and_link(asm_code, tmp_dir)
        entry_addr = _find_symbol_address(elf_path, entry_symbol)

        emulator = Uc(UC_ARCH_RISCV, UC_MODE_RISCV64)
        emulator.mem_map(MEMORY_BASE_ADDR, MEMORY_SIZE_BYTES)
        _load_elf_segments(ELFFile(open(elf_path, "rb")), emulator)

        output_buffer = []
        _create_execution_hooks(emulator, output_buffer)
        try:
            emulator.emu_start(entry_addr, 0xFFFFFFFF, count=MAX_INSTRUCTION_COUNT)
        except:
            pass

        reg_map = [("t0", UC_RISCV_REG_T0), ("t1", UC_RISCV_REG_T1), ("t2", UC_RISCV_REG_T2), ("a0", UC_RISCV_REG_A0), ("a1", UC_RISCV_REG_A1), ("a7", UC_RISCV_REG_A7)]
        return {"output": "".join(output_buffer), "regs": {name: emulator.reg_read(reg_id) for name, reg_id in reg_map}}


def map_virtual_to_physical_registers(asm: str) -> str:
    # risc-v has t0-t6 (7 temps) and s0-s11 (12 saved regs) = 19 total physical registers

    virtual_regs = set(re.findall(r"\bj_\d+\b", asm))
    physical_regs = [f"t{i}" for i in range(7)] + [f"s{i}" for i in range(12)]

    if len(virtual_regs) > len(physical_regs):
        raise RuntimeError(f"too many virtual registers ({len(virtual_regs)}) for available physical registers ({len(physical_regs)})")

    virtual_list = sorted(virtual_regs, key=lambda x: int(x.split("_")[1]))
    reg_map = {v: physical_regs[i] for i, v in enumerate(virtual_list)}

    for virt, phys in reg_map.items():
        asm = re.sub(rf"\b{virt}\b", phys, asm)

    return asm
