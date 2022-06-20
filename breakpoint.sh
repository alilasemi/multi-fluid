#!/bin/bash
gdb ~/software/anaconda3/bin/python -ex 'set breakpoint pending on' -ex "b $1" -ex 'r main.py'
