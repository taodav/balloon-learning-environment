#!/usr/bin/env bash

# This script takes a folder as input, and moves all replay related files
# to a separate folder called "buffer"
# file structure:
# $PASSED_IN_DIR/
# - quantile
# --- 1
# ----- checkpoints
# ----- metrics
# ----- buffer
#
# we'll be moving all quantile/1/checkpoints/$store$* and quantile/1/checkpoints/sum_tree_*
# files to buffer.

if [ $# -eq 0 ]
  then
    echo "No path given to parse."
    exit 0
fi

SEEDS_LIVE_HERE="$1/quantile"

SEEDS_FILES=($(ls -d $SEEDS_LIVE_HERE/*))
for i in "${SEEDS_FILES[@]}"
do
  STORE_FILES=$i/checkpoints/\$store\$*
  SUM_TREE_FILES=$i/checkpoints/sum_tree*
  BUFFER_DIR=$i/buffer
  mkdir -p $BUFFER_DIR

#  mv $STORE_FILES $BUFFER_DIR
  mv $SUM_TREE_FILES $BUFFER_DIR

done
