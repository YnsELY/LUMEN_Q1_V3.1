#!/bin/bash
# submit_ink.sh <job.py> [timeout_s] : copie le job dans la file, attend la fin, affiche la fin du journal
job=$1; to=${2:-600}; name=$(basename $job .py)
rm -f /workspace/work_ink/logs/$name.done
cp $job /workspace/work_ink/queue/$name.py
for i in $(seq 1 $to); do [ -f /workspace/work_ink/logs/$name.done ] && break; sleep 1; done
tail -c ${3:-6000} /workspace/work_ink/logs/$name.log
