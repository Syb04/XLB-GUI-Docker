import argparse, json, resource, time
from pathlib import Path
from workbench.schema import default_project, validate_project
from workbench.geometry import build_mesh
from workbench.solver import simulate

parser=argparse.ArgumentParser()
parser.add_argument('--cells',type=int,default=50)
parser.add_argument('--output',required=True)
args=parser.parse_args()
p=default_project()
p['name']='GPU+RAM capacity check'
p['mesh']={'cells':[3*args.cells,args.cells,args.cells]}
p['study'].update(device='cuda:0-ram',gpu_batch_cells=65536,steps=3,output_interval=1,snapshot_interval=0,dt=0.05/args.cells)
p['physics']['turbulence']={'model':'smagorinsky','smagorinsky_constant':.17,'turbulent_prandtl':.9}
p=validate_project(p)
root=Path(args.output)
root.mkdir(parents=True,exist_ok=True)
(root/'input.json').write_text(json.dumps(p,indent=2))
mesh=build_mesh(p,root/'assets')
start=time.perf_counter()
samples=[]
def progress(row):
    samples.append(dict(row,elapsed_seconds=time.perf_counter()-start))
    print(json.dumps(samples[-1]),flush=True)
result=simulate(p,mesh,root,on_progress=progress)
assert result['status']=='completed'
assert result['diagnostics']['state_device'].lower().find('cpu')>=0
assert result['diagnostics']['collision_device'].lower().find('cuda')>=0
record={'total_cells':int(mesh['fluid_mask'].size),'elapsed_seconds':time.perf_counter()-start,'peak_host_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'samples':samples,'result':result}
(root/'benchmark.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record,indent=2),flush=True)
