#!/bin/bash

python3 main.py --data traffic --save traffic2.pt --cuda False -e 100 -spe 1 -nl 1 -not 168 -sl 12 -sp -max 2>&1 | tee trafficLog.txt

