#!/bin/bash
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "usage: tools/rci/local_copy.sh <experiment name>"
    echo "Run it on the Mac. It copies runs/<name> from the cluster, with no checkpoints/,"
    echo "and counts the runs, the latest.pkl files and the eval.json files."
    exit 1
fi

cd "$(dirname "$0")/../.."
NAME=$1
WHO=$(ssh -G rci 2>/dev/null | awk '$1 == "user" {print $2}')

rsync -a -e 'ssh -o LogLevel=ERROR' --exclude 'checkpoints/' "rci:/mnt/personal/$WHO/drone-swarm-rl/runs/$NAME" runs/

count() {
    find "runs/$NAME" "$@" | wc -l | tr -d ' '
}

echo "runs:            $(count -mindepth 2 -maxdepth 2 -type d)"
echo "with latest.pkl: $(count -name latest.pkl)"
echo "eval.json files: $(count -name eval.json)"
