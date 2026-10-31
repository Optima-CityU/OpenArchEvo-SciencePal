Lab session: supervise GPU jobs for the OpenArchEvo experiment. Work unattended; never ask questions; no web search. Keep your messages short (one line per check) to save tokens.

1. Record whether the previous worker is still alive: `cat /workspace/lab/worker.pid; ps -p $(cat /workspace/lab/worker.pid) -o pid,etime,cmd || echo DEAD`. Record the result.
2. Build the tokenized data cache once: `cd /workspace/lab && time python3 prep_data.py` (record wall time and output).
3. Start the queue worker detached (the file /workspace/lab/lab_worker.py was updated by the controller; start this version):
   `cd /workspace/lab && rm -f STOP && LAB_MAX_PARALLEL=2 nohup python3 lab_worker.py > worker.log 2>&1 & echo $! > worker.pid`
4. Supervise. Every 4–5 minutes (use `sleep 240` between checks) print one line: `cat /workspace/lab/heartbeat.json; ls /workspace/lab/results | wc -l`. The controller adds new job files to /workspace/lab/queue/ while you are supervising; that is expected. If the worker process dies, print the tail of worker.log, restart it once with the same command, and note it.
   Stop supervising when ANY of these is true: (a) the file /workspace/lab/SESSION_END exists; (b) the queue is empty, nothing is running, and no new result appeared for 20 minutes; (c) 170 minutes have passed since step 3.
5. Before finishing, write `/workspace/lab/supervise_report.json` with: step 1 result, prep time, start time, end time, number of results, any restarts or errors (tails of failed job logs from /workspace/lab/logs/ are welcome). Do not stop the worker yourself. Print the report path as the last line.
