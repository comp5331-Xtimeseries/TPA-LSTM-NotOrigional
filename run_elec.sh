#!/bin/bash

python3 main.py --data electricity --save electricity2.pt --cuda False -e 2 -spe 1 -nl 1 -not 168 -sl 24 -sp -max 2>&1 | tee elecLog.txt

