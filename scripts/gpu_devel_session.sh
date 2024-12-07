#!/usr/bin/env bash

srun -A hpc-lco-kenter --qos=devel --partition=dgx --gres=gpu:a100:1 -t 01:00:00 --mem 32G --pty bash
