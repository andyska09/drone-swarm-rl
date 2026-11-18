#!/bin/bash
set -u

if [ $# -ne 1 ]; then
    echo "usage: tools/rci/local_wait.sh <job numbers, with commas>"
    echo "Run it on the Mac. It reads the state of the jobs every 10 minutes."
    echo "It ends when every job has an end state, and prints one row for each job."
    exit 1
fi

JOBS=$1
COUNT=$(echo "$JOBS" | tr ',' '\n' | grep -c .)

while true; do
    if rows=$(ssh -o LogLevel=ERROR rci "bash -l -c 'sacct -j $JOBS -X -n -o JobID,State,Elapsed'"); then
        echo "$(date +%H:%M) $(echo "$rows" | awk '{print $2}' | sort | uniq -c | tr '\n' ' ')"
        listed=$(echo "$rows" | grep -c .)
        live=$(echo "$rows" | grep -c -E 'PENDING|RUNNING|REQUEUED|RESIZING|SUSPENDED|COMPLETING|CONFIGURING')
        if [ "$listed" -ge "$COUNT" ] && [ "$live" -eq 0 ]; then
            break
        fi
    else
        echo "$(date +%H:%M) ssh failed, the next try is in 10 minutes"
    fi
    sleep 600
done

echo "$rows"
