import sys,json,time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'harness'))
from sp_client import Client, record
from run_task import find_files
from pathlib import Path
c=Client(); thread=sys.argv[1]; name=sys.argv[2]; expect=set(sys.argv[3:])
out=ROOT / 'runs' / 'sp' / name; (out/'files').mkdir(parents=True,exist_ok=True)
run=c.thread_runs(thread)['agent_runs'][-1]
st=c.wait(run['id'], deadline_s=3600)
run=c.thread_runs(thread)['agent_runs'][-1]
(out/'run.json').write_text(json.dumps(run,indent=1)); (out/'status.json').write_text(json.dumps(st,indent=1))
record('run_finished', thread_id=thread, run_id=run['id'], status=run.get('status'), tokens_used=run.get('tokens_used'), model=run.get('model_name'))
sid=c.thread_sandbox(thread)['sandbox_id']
found=find_files(c,sid,expect); (out/'found.json').write_text(json.dumps(found,indent=1))
for nm,p in found.items(): (out/'files'/nm).write_bytes(c.read_file(sid,p))
print(name, run.get('status'), run.get('tokens_used'), run.get('model_name'), sorted(found))
