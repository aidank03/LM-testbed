from pathlib import Path
import json,base64,io
from PIL import Image,ImageOps,ImageDraw
root=Path(__file__).resolve().parents[1]; out=root/'reports'/'visuals-v1';out.mkdir(exist_ok=True)
figures=[]; total=0; animations=[]
for p in sorted((root/'notebooks').glob('*.ipynb')):
 if p.name[:2] not in {f'{i:02}' for i in range(5,15)}:continue
 n=json.loads(p.read_text()); k=0
 for c in n['cells']:
  for output in c.get('outputs',[]):
   data=output.get('data',{}); png=data.get('image/png'); html=data.get('text/html','')
   if isinstance(html,list):html=''.join(html)
   if 'function Animation' in html: animations.append(p.name)
   if not png:continue
   total+=1
   if c.get('metadata',{}).get('world_visual_pack')!='world-visuals-v1':continue
   k+=1; raw=base64.b64decode(png); path=out/f'{p.name[:2]}-{k:02}.png';path.write_bytes(raw)
   im=Image.open(io.BytesIO(raw)).convert('RGB');figures.append((path.name,im))
for page,start in enumerate(range(0,len(figures),9)):
 canvas=Image.new('RGB',(1800,1260),'#eeeeee');d=ImageDraw.Draw(canvas)
 for slot,(label,im) in enumerate(figures[start:start+9]):
  x=(slot%3)*600;y=(slot//3)*420;tile=ImageOps.contain(im,(585,380))
  canvas.paste(tile,(x+(600-tile.width)//2,y+30));d.text((x+10,y+6),label,fill='black')
 canvas.save(out/f'contact-{page+1}.jpg')
report={'total_static_figures':total,'added_static_figures':len(figures),'animations':animations}
(out/'inventory.json').write_text(json.dumps(report,indent=2));print(report)
