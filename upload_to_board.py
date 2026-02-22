"""Upload overlay files to PYNQ board via SFTP"""
import paramiko
import os

HOST = "makerslab-fpga-43.ddns.comp.nus.edu.sg"
USER = "xilinx"
PASS = "xilinx"
REMOTE_DIR = "/home/xilinx/"

FILES = [
    "pynq_overlay/siamese_lstm.bit",
    "pynq_overlay/siamese_lstm.hwh",
    "pynq_overlay/test_siamese_lstm.py",
]

base = os.path.dirname(os.path.abspath(__file__))

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
print(f"Connecting to {HOST}...")
ssh.connect(HOST, username=USER, password=PASS)
sftp = ssh.open_sftp()

for f in FILES:
    local = os.path.join(base, f)
    remote = REMOTE_DIR + os.path.basename(f)
    size = os.path.getsize(local)
    print(f"Uploading {os.path.basename(f)} ({size:,} bytes)...")
    sftp.put(local, remote)
    print(f"  -> {remote} OK")

sftp.close()
ssh.close()
print("\nAll files uploaded successfully!")
