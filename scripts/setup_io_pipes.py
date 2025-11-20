#!/usr/bin/env python3
import sys
import subprocess

if len(sys.argv) != 3:
    print(f"Usage: {sys.argv[0]} <n_fpgas> <topology>")
    print("Where <topology> is one of [\"ring\"]")
    exit(1)

n_fpgas = int(sys.argv[1])
topology = sys.argv[2]

if topology not in ["ring"]:
    print(f"Unknown topology \"{topology}\"")
    exit(1)

command = ["changeFPGAlinks"]
for i_fpga in range(0, n_fpgas-1):
    i_node = i_fpga // 2
    i_acl = i_fpga % 2
    i_next_node = (i_fpga + 1) // 2
    i_next_acl = (i_fpga + 1) % 2

    command.append(f"--fpgalink=n{i_node:02}:acl{i_acl}:ch2-n{i_next_node:02}:acl{i_next_acl}:ch0")
    command.append(f"--fpgalink=n{i_node:02}:acl{i_acl}:ch3-n{i_next_node:02}:acl{i_next_acl}:ch1")

i_last_node = (n_fpgas - 1) // 2
i_last_acl = (n_fpgas - 1) % 2
command.append(f"--fpgalink=n{i_last_node:02}:acl{i_last_acl}:ch2-n00:acl0:ch0")
command.append(f"--fpgalink=n{i_last_node:02}:acl{i_last_acl}:ch3-n00:acl0:ch1")
print(command)
subprocess.run(command)
