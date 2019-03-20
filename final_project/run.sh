#!/bin/bash

filename=
sched_policy=0  # 0 for naive, 1 for optimal
draw=1

while [ -n "$1" ]; do
    case $1 in 
        -i  | --file)               shift
                                    filename=$1
                                    ;;
        -s  | --sched_policy)       shift
                                    sched_policy=$1
                                    ;;
        -h  | --help)               echo "# Usage: -i [file] -s [scheduling policy: 0 for naive, 1 for optimal]"
                                    exit
                                    ;;
        * )                         echo "Option $1 not recognized!"
    esac
    shift
done

./exe ${filename} ${sched_policy} ${draw}$> log.txt

python3 draw_blocks_timeline.py log.txt