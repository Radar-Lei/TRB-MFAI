import sys,subprocess,concurrent.futures,json,time
from pathlib import Path
root=Path(__file__).resolve().parent;out=root/'results/e110_v101';out.mkdir(exist_ok=True);script=root/'run_e110_v101_calibrate.py';st=time.time()
def run(args):
 with (out/('_'.join(args)+'.log')).open('w') as log:subprocess.run([sys.executable,str(script)]+args,stdout=log,stderr=subprocess.STDOUT,check=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(run,[['candidate',str(w)] for w in [.001,.01,.1]]))
run(['select']);selection=json.loads((out/'selection.json').read_text());print('selected',selection['selected_weight'],flush=True)
if selection['selected_weight'] is not None:
 run(['train','0'])
 with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:list(pool.map(run,[['train',str(s)] for s in range(1,5)]+[['paths']]))
 run(['eval'])
(out/'wall_seconds.json').write_text(json.dumps({'total_wall_seconds':time.time()-st},indent=2));print('done',time.time()-st,flush=True)
