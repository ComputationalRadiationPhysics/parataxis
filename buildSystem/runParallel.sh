#!/usr/bin/env bash

#
runParallel()
{
    numParallel=$1
    descriptions=$2
    cmds=$3
    outputFiles=$4
    
    running=0
    for ((i=0;i<${#descriptions[@]};i++)); do
        echo "Processing ${descriptions[$i]}"

        if [ "$numParallel" -gt "1" ]; then
            eval ${cmds[$i]} &> ${outputFiles[$i]} &

            running="`jobs -p | wc -l`"

            while [ "$running" -ge "$numParallel" ]; do
                sleep 5
                running="`jobs -p | wc -l`"
            done
        else
            eval ${cmds[$i]} | tee ${outputFiles[$i]}
        fi
    done
}

