#!/bin/bash

SESSION_NAME="poe_training"
LOGFILE="./train.log"

tmux new-session -d -s "$SESSION_NAME" "poe train | tee $LOGFILE"
