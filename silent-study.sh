#!/bin/bash

SESSION_NAME="poe_study"
LOGFILE="./study.log"

tmux new-session -d -s "$SESSION_NAME" "poe study | tee $LOGFILE"
