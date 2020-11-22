!#/bin/bash

python main.py --data electricity --save electricity.pt --cuda False -e 1000 -spe 1 -nl 1 -not 168 -sl 24 -sp -max 2>&1 | tee elecLog.txt

