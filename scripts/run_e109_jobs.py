"""Five seed workers; methods sequential per seed, equal fixed budget."""
import concurrent.futures,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parent
py=sys.executable
out=root/'results/e109_v101'
def seed_job(seed):
 for method in ['flow','response','structure']:
  with (out/f'{method}_{seed}.log').open('w') as log:
   subprocess.run([py,str(root/'run_e109_v101_structured.py'),'train','--seed',str(seed),'--method',method],stdout=log,stderr=subprocess.STDOUT,check=True)
 print('seed complete',seed,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:list(pool.map(seed_job,range(5)))
subprocess.run([py,str(root/'run_e109_v101_structured.py'),'eval'],check=True)
