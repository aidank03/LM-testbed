"""Execute only LM-testbed course lessons 05–14 in fresh Jupyter kernels."""
from pathlib import Path
import argparse,json,time,os
import nbformat
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parents[1]

def course_notebooks(root):
    result=[]
    for path in sorted((Path(root)/'notebooks').glob('*.ipynb')):
        if path.name[:2] not in {f'{i:02d}' for i in range(5,15)}: continue
        notebook=nbformat.read(path,as_version=4)
        if notebook.metadata.get('lm_testbed',{}).get('lesson')==path.name[:2]: result.append(path)
    return result

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--lessons',nargs='*')
    parser.add_argument('--kernel',default='factor-ai'); args=parser.parse_args()
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'runs'/'.mplconfig'))
    report_path=ROOT/'reports'/'notebook_execution.json'
    previous=json.loads(report_path.read_text()) if report_path.exists() and args.lessons else []
    results=[r for r in previous if r['notebook'][:2] in {f'{i:02d}' for i in range(5,15)} and r['notebook'][:2] not in (args.lessons or [])]
    selected=[p for p in course_notebooks(ROOT) if not args.lessons or p.name[:2] in args.lessons]
    if not selected: raise SystemExit('No matching LM-testbed lessons 05–14')
    for path in selected:
        start=time.monotonic(); notebook=nbformat.read(path,as_version=4)
        print('START',path.name,flush=True)
        try:
            NotebookClient(notebook,timeout=240,kernel_name=args.kernel,
                resources={'metadata':{'path':str(ROOT/'notebooks')}}).execute()
            nbformat.write(notebook,path)
            row={'notebook':path.name,'status':'passed','seconds':round(time.monotonic()-start,2),
                 'code_cells':sum(c.cell_type=='code' for c in notebook.cells)}
        except Exception as exc:
            # Preserve the existing notebook if execution fails.
            failed=ROOT/'reports'/(path.stem+'.failed.ipynb'); nbformat.write(notebook,failed)
            row={'notebook':path.name,'status':'failed','seconds':round(time.monotonic()-start,2),'error':str(exc)}
        results.append(row); print(json.dumps(row),flush=True)
        report_path.write_text(json.dumps(results,indent=2))
    print('COMPLETE',sum(r['status']=='passed' for r in results),'/',len(results),flush=True)
    if any(r['status']!='passed' for r in results): raise SystemExit(1)

if __name__=='__main__': main()
