#!/bin/bash

python3 main.py --data exchange_rate --save exchangeRate2.pt --cuda False -e 10 -spe 1 -nl 1 -not 168 -sl 12 -sp -max 2>&1 | tee exchangeLog.txt

