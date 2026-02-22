import os
import torch
from torchvision import transforms
from PIL import Image

device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
print('Device:', device)

ckpt_dir = 'checkpoints'
all_pths = [os.path.join(ckpt_dir,f) for f in os.listdir(ckpt_dir) if f.endswith('.pth')] if os.path.isdir(ckpt_dir) else []
if not all_pths:
    raise SystemExit('No checkpoints found')

pix_files = [p for p in all_pths if 'ckpt_epoch' in p or 'pix2pix' in os.path.basename(p).lower()]
if pix_files:
    pix_ckpt = sorted(pix_files, key=os.path.getmtime)[-1]
else:
    non_mapper = [p for p in all_pths if 'mapper' not in os.path.basename(p).lower()]
    pix_ckpt = sorted(non_mapper if non_mapper else all_pths, key=os.path.getmtime)[-1]
print('Using pix2pix checkpoint:', pix_ckpt)

from pix2pix_generator import Generator
G = Generator().to(device)
ck = torch.load(pix_ckpt, map_location=device)
if isinstance(ck, dict) and 'G_state' in ck:
    G.load_state_dict(ck['G_state'])
else:
    try:
        G.load_state_dict(ck)
    except Exception:
        for v in ck.values() if isinstance(ck, dict) else []:
            if isinstance(v, dict):
                try:
                    G.load_state_dict(v)
                    break
                except Exception:
                    pass
G.eval()

from mapper import SimpleMapper
mapper_ckpt = os.path.join('checkpoints','mapper.pth')
mapper = SimpleMapper().to(device)
if os.path.exists(mapper_ckpt):
    mapper.load_state_dict(torch.load(mapper_ckpt, map_location=device))
    mapper.eval()
    print('Loaded mapper checkpoint:', mapper_ckpt)
else:
    print('Mapper checkpoint not found; exiting')
    raise SystemExit(1)

import torch as th
attrs = {'gender':'male','hair_length':'short','hair_color':'black','beard':'no','glasses':'no','face_shape':'oval'}
gender = 0.0 if attrs.get('gender','male')=='male' else 1.0
hair_length = 0.0 if attrs.get('hair_length','short')=='short' else 1.0
hair_color = {'black':0.0,'brown':1.0,'blonde':2.0}.get(attrs.get('hair_color','black'),0.0)
vec = th.tensor([gender, hair_length, hair_color/2.0], dtype=th.float32, device=device)
attr_map = (vec*2.0-1.0).unsqueeze(-1).unsqueeze(-1).repeat(1,256,256).unsqueeze(0)
print('attr_map', 'shape',tuple(attr_map.shape), 'min',attr_map.min().item(),'max',attr_map.max().item(),'mean',attr_map.mean().item())

with torch.no_grad():
    mapper_out = mapper(attr_map)
print('mapper_out raw', 'shape',tuple(mapper_out.shape), 'min',mapper_out.min().item(),'max',mapper_out.max().item(),'mean',mapper_out.mean().item())

mapper_out_clamped = mapper_out.clamp(-1.0,1.0)
print('mapper_out clamped', 'min',mapper_out_clamped.min().item(),'max',mapper_out_clamped.max().item(),'mean',mapper_out_clamped.mean().item())

trans = transforms.ToPILImage()
trans(((mapper_out_clamped.squeeze(0).cpu()+1.0)/2.0).clamp(0,1)).save('debug_mapper.png')
print('Saved debug_mapper.png')

with torch.no_grad():
    gen_out = G(mapper_out_clamped)
print('generator raw', 'shape',tuple(gen_out.shape), 'min',gen_out.min().item(),'max',gen_out.max().item(),'mean',gen_out.mean().item())

gen_img = ((gen_out + 1.0)/2.0)
print('generator image pre-clamp min',gen_img.min().item(),'max',gen_img.max().item(),'mean',gen_img.mean().item())
trans(gen_img.squeeze(0).cpu().clamp(0,1)).save('debug_generator.png')
print('Saved debug_generator.png')

scale=0.5
mapper_scaled = mapper_out_clamped * scale
print('mapper_scaled', 'min',mapper_scaled.min().item(),'max',mapper_scaled.max().item(),'mean',mapper_scaled.mean().item())
with torch.no_grad():
    gen_out_scaled = G(mapper_scaled)
print('generator raw scaled', 'min',gen_out_scaled.min().item(),'max',gen_out_scaled.max().item(),'mean',gen_out_scaled.mean().item())
gen_img_scaled = ((gen_out_scaled + 1.0)/2.0)
trans(gen_img_scaled.squeeze(0).cpu().clamp(0,1)).save('debug_generator_scaled.png')
print('Saved debug_generator_scaled.png')
print('generator image scaled pre-clamp min',gen_img_scaled.min().item(),'max',gen_img_scaled.max().item(),'mean',gen_img_scaled.mean().item())
