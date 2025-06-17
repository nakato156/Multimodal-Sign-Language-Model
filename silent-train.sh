#!/bin/bash

LOGFILE="./train.log"

> "$LOGFILE" 

nohup poe train --batch_size 8 > "$LOGFILE" 2>&1 &