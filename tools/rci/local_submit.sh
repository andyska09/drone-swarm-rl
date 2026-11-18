#!/bin/bash
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "usage: tools/rci/local_submit.sh '<submit command>'"
    echo "Run it on the Mac. On the cluster it stops if a job waits in the queue, pulls,"
    echo "stops if the commit is not the pushed one, and runs the submit command."
    echo "The last line, 'jobs:', holds the job numbers."
    exit 1
fi

cd "$(dirname "$0")/../.."
PUSHED=$(git rev-parse '@{u}')

ssh -o LogLevel=ERROR rci 'bash -l -s' <<EOF | awk '{print} /^Submitted batch job/ {jobs = jobs sep $4; sep = ","} END {if (jobs) print "jobs: " jobs}'
set -euo pipefail
cd /mnt/personal/\$USER/drone-swarm-rl
if squeue -u \$USER -h -t PENDING | grep -q .; then
    echo "STOP: a job waits in the queue, and a git pull changes the code that it runs."
    exit 1
fi
git pull
if [ "\$(git rev-parse HEAD)" != "$PUSHED" ]; then
    echo "STOP: the cluster is at \$(git rev-parse --short HEAD), and the pushed commit is ${PUSHED:0:7}."
    exit 1
fi
$1
squeue -u \$USER
EOF
