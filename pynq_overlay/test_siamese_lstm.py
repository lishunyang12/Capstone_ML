"""
Siamese LSTM FPGA Test — Direct hardware access (no PYNQ Overlay)
Bypasses PYNQ 3.0.1 device detection issue (XRT not installed)
Uses: /dev/mem for MMIO registers, /dev/xlnk for CMA DMA buffers
"""

import numpy as np
import os
import sys
import subprocess
import mmap
import struct
import ctypes
import time

# ─── HLS IP Register Map (from csynth report) ───
ADDR_AP_CTRL    = 0x00
ADDR_SEQ1_LO    = 0x10
ADDR_SEQ1_HI    = 0x14
ADDR_SEQ2_LO    = 0x1c
ADDR_SEQ2_HI    = 0x20
ADDR_RESULT_LO  = 0x28
ADDR_RESULT_HI  = 0x2c
ADDR_SEQ1_LEN   = 0x34
ADDR_SEQ2_LEN   = 0x3c

INPUT_DIM = 7
HLS_BASE_ADDR = 0xA0000000  # M_AXI_HPM0_FPD base
HLS_ADDR_RANGE = 0x10000
BIT_FILE = "/home/xilinx/siamese_lstm.bit"
PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")


# ─── FPGA Programming ───
def program_fpga(bitfile):
    """Program FPGA via sysfs FPGA manager"""
    print(f"  Programming FPGA with {os.path.basename(bitfile)}...")
    try:
        fw_path = "/lib/firmware/siamese_lstm.bin"
        subprocess.run(["cp", bitfile, fw_path], check=True)
        with open("/sys/class/fpga_manager/fpga0/flags", "w") as f:
            f.write("0")
        with open("/sys/class/fpga_manager/fpga0/firmware", "w") as f:
            f.write("siamese_lstm.bin")
        time.sleep(0.5)
        with open("/sys/class/fpga_manager/fpga0/state", "r") as f:
            state = f.read().strip()
        print(f"  FPGA state: {state}")
        return state == "operating"
    except Exception as e:
        print(f"  Programming failed: {e}")
        return False


# ─── MMIO: Register access via /dev/mem ───
class MMIO:
    def __init__(self, base_addr, length=0x10000):
        self.base_addr = base_addr
        self.length = length
        self.fd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
        self.mm = mmap.mmap(self.fd, length, mmap.MAP_SHARED,
                            mmap.PROT_READ | mmap.PROT_WRITE,
                            offset=base_addr)

    def read(self, offset):
        self.mm.seek(offset)
        return struct.unpack("<I", self.mm.read(4))[0]

    def write(self, offset, value):
        self.mm.seek(offset)
        self.mm.write(struct.pack("<I", value & 0xFFFFFFFF))

    def close(self):
        self.mm.close()
        os.close(self.fd)


# ─── CMA Buffer: Contiguous DMA memory via /dev/xlnk ───
class CMABuffer:
    """Allocate physically contiguous memory using /dev/xlnk mmap"""

    def __init__(self, size_bytes):
        self.size = size_bytes
        # Open xlnk device for CMA allocation
        self.xlnk_fd = os.open("/dev/xlnk", os.O_RDWR | os.O_SYNC)
        # mmap allocates CMA memory through the xlnk driver
        self.mm = mmap.mmap(self.xlnk_fd, size_bytes, mmap.MAP_SHARED,
                            mmap.PROT_READ | mmap.PROT_WRITE)
        # Get physical address from /proc/self/pagemap
        self.virt_addr = ctypes.addressof(ctypes.c_char.from_buffer(self.mm))
        self.phys_addr = self._get_phys_addr(self.virt_addr)

    def _get_phys_addr(self, virt_addr):
        """Look up physical address via /proc/self/pagemap"""
        page_num = virt_addr // PAGE_SIZE
        try:
            with open("/proc/self/pagemap", "rb") as pm:
                pm.seek(page_num * 8)
                entry = struct.unpack("Q", pm.read(8))[0]
                if entry & (1 << 63):  # page present
                    pfn = entry & ((1 << 55) - 1)
                    return pfn * PAGE_SIZE + (virt_addr % PAGE_SIZE)
        except:
            pass
        # Fallback: use /dev/mem to find the mapping
        # The xlnk mmap maps CMA at a known offset
        return None

    def write_floats(self, data):
        """Write float32 array to buffer"""
        raw = data.astype(np.float32).tobytes()
        self.mm.seek(0)
        self.mm.write(raw)

    def read_floats(self, count):
        """Read float32 array from buffer"""
        self.mm.seek(0)
        raw = self.mm.read(count * 4)
        return np.frombuffer(raw, dtype=np.float32).copy()

    def close(self):
        self.mm.close()
        os.close(self.xlnk_fd)


# ─── Alternative: Use /dev/mem at fixed CMA address ───
class FixedCMABuffer:
    """Map a fixed physical address region for DMA via /dev/mem.
    Uses the CMA region (top of DDR). Safe if region is in CMA pool.
    """

    # CMA base: on 2GB system with 128MB CMA, typically near end of DDR
    # We'll use a conservative address in the CMA pool
    _next_offset = 0

    def __init__(self, size_bytes, base_phys=0x70000000):
        self.size = ((size_bytes + PAGE_SIZE - 1) // PAGE_SIZE) * PAGE_SIZE
        self.phys_addr = base_phys + FixedCMABuffer._next_offset
        FixedCMABuffer._next_offset += self.size

        self.fd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
        self.mm = mmap.mmap(self.fd, self.size, mmap.MAP_SHARED,
                            mmap.PROT_READ | mmap.PROT_WRITE,
                            offset=self.phys_addr)

    def write_floats(self, data):
        raw = data.astype(np.float32).tobytes()
        self.mm.seek(0)
        self.mm.write(raw)

    def read_floats(self, count):
        self.mm.seek(0)
        raw = self.mm.read(count * 4)
        return np.frombuffer(raw, dtype=np.float32).copy()

    def close(self):
        self.mm.close()
        os.close(self.fd)


def alloc_buffer(size_bytes):
    """Try CMA allocation, fallback to fixed address"""
    try:
        buf = CMABuffer(size_bytes)
        if buf.phys_addr is not None:
            return buf
        buf.close()
    except Exception as e:
        pass

    # Fallback: use fixed physical address in CMA region
    return FixedCMABuffer(size_bytes)


def run_inference(mmio, seq1, seq2):
    """Run Siamese LSTM inference on FPGA"""
    seq1_len, seq2_len = seq1.shape[0], seq2.shape[0]
    seq1_flat = seq1.flatten().astype(np.float32)
    seq2_flat = seq2.flatten().astype(np.float32)

    # Allocate DMA buffers
    FixedCMABuffer._next_offset = 0  # Reset allocation offset
    buf_seq1 = alloc_buffer(seq1_flat.nbytes)
    buf_seq2 = alloc_buffer(seq2_flat.nbytes)
    buf_result = alloc_buffer(4)

    # Write input data
    buf_seq1.write_floats(seq1_flat)
    buf_seq2.write_floats(seq2_flat)
    buf_result.write_floats(np.array([0.0], dtype=np.float32))

    # Write physical addresses to IP registers
    mmio.write(ADDR_SEQ1_LO, buf_seq1.phys_addr & 0xFFFFFFFF)
    mmio.write(ADDR_SEQ1_HI, (buf_seq1.phys_addr >> 32) & 0xFFFFFFFF)
    mmio.write(ADDR_SEQ2_LO, buf_seq2.phys_addr & 0xFFFFFFFF)
    mmio.write(ADDR_SEQ2_HI, (buf_seq2.phys_addr >> 32) & 0xFFFFFFFF)
    mmio.write(ADDR_RESULT_LO, buf_result.phys_addr & 0xFFFFFFFF)
    mmio.write(ADDR_RESULT_HI, (buf_result.phys_addr >> 32) & 0xFFFFFFFF)
    mmio.write(ADDR_SEQ1_LEN, seq1_len)
    mmio.write(ADDR_SEQ2_LEN, seq2_len)

    # Start accelerator
    start = time.time()
    mmio.write(ADDR_AP_CTRL, 0x01)

    # Wait for completion: sleep based on sequence length, then verify
    # HLS latency: ~0.18ms per timestep at 100MHz
    max_steps = max(seq1_len, seq2_len)
    est_time = max_steps * 0.4 / 1000.0 + 0.05  # generous estimate
    time.sleep(est_time)

    # Check AP_IDLE (bit 2) - more reliable than AP_DONE on ARM
    ctrl = mmio.read(ADDR_AP_CTRL)
    if not (ctrl & 0x04):
        # Not idle yet, wait more
        time.sleep(1.0)

    elapsed = time.time() - start

    # Read result
    result = buf_result.read_floats(1)
    score = float(result[0])

    # Cleanup
    buf_seq1.close()
    buf_seq2.close()
    buf_result.close()

    return score, elapsed


def main():
    print("=" * 60)
    print("Siamese LSTM FPGA Accelerator — Direct HW Test")
    print("=" * 60)

    # Check FPGA state
    try:
        with open("/sys/class/fpga_manager/fpga0/state", "r") as f:
            state = f.read().strip()
        print(f"\nFPGA state: {state}")
        if state != "operating":
            if not program_fpga(BIT_FILE):
                print("ERROR: Cannot program FPGA")
                sys.exit(1)
    except:
        if not program_fpga(BIT_FILE):
            print("ERROR: Cannot program FPGA")
            sys.exit(1)

    # Open MMIO
    print(f"Opening MMIO at 0x{HLS_BASE_ADDR:08X}...")
    mmio = MMIO(HLS_BASE_ADDR, HLS_ADDR_RANGE)
    ctrl = mmio.read(ADDR_AP_CTRL)
    print(f"AP_CTRL = 0x{ctrl:08X} (expect 0x04 = idle)")

    if ctrl != 0x04:
        print("WARNING: IP not in idle state, re-programming FPGA...")
        program_fpga(BIT_FILE)
        time.sleep(1)
        ctrl = mmio.read(ADDR_AP_CTRL)
        print(f"AP_CTRL = 0x{ctrl:08X}")

    # Run tests
    tests = [
        ("Identical sequences (len=50)",  42, 50, 42, 50, True),
        ("Random different (len=50)",     42, 50, 99, 50, False),
        ("Short sequences (len=10)",      42, 10, 99, 10, False),
        ("Different lengths (30 vs 20)",  42, 30, 99, 20, False),
    ]

    for name, seed1, len1, seed2, len2, identical in tests:
        print(f"\n--- {name} ---")
        np.random.seed(seed1)
        seq1 = np.random.randn(len1, INPUT_DIM).astype(np.float32) * 0.5
        if identical:
            seq2 = seq1.copy()
        else:
            np.random.seed(seed2)
            seq2 = np.random.randn(len2, INPUT_DIM).astype(np.float32) * 0.5

        try:
            score, t = run_inference(mmio, seq1, seq2)
            print(f"  Score: {score:.6f}")
            print(f"  Time:  {t*1000:.1f} ms")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    mmio.close()
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
