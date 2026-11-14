#!/bin/bash
# submit.sh <job.py> [timeout_s]: copy job to queue, wait for completion, print log tail
job=$1; to=${2:-600}; name=$(basename $job .py)
rm -f /workspace/work/logs/$name.done
cp $job /workspace/work/queue/$name.py
for i in $(seq 1 $to); do [ -f /workspace/work/logs/$name.done ] && break; sleep 1; done
tail -c ${3:-6000} /workspace/work/logs/$name.log
