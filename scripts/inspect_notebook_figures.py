from pathlib import Path
import json,base64,io
from PIL import Image,ImageOps,ImageDraw
root=Path(__file__).resolve().parents[1]
figures=[]
for p in sorted((root/'notebooks').glob('*.ipynb')):
 if p.name[:2] not in {f'{i:02d}' for i in range(5,15)}: continue
 n=json.loads(p.read_text()); k=0
 for c in n['cells']:
  for out in c.get('outputs',[]):
   data=out.get('data',{}).get('image/png')
   if data:
    k+=1; im=Image.open(io.BytesIO(base64.b64decode(data))).convert('RGB')
    figures.append((f'{p.name[:2]} figure {k}',im))
for page,start in enumerate(range(0,len(figures),9)):
 canvas=Image.new('RGB',(1500,1050),'#eeeeee'); d=ImageDraw.Draw(canvas)
 for slot,(label,im) in enumerate(figures[start:start+9]):
  x=(slot%3)*500; y=(slot//3)*350
  tile=ImageOps.contain(im,(485,315)); canvas.paste(tile,(x+(500-tile.width)//2,y+28))
  d.text((x+10,y+6),label,fill='black')
 canvas.save(root/'reports'/f'figures_contact_{page+1}.jpg')
print('Checked image count',len(figures))
