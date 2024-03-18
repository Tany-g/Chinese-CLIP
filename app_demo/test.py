
import json
import sys
sys.path.append("/home/ubuntu/GITHUG/Chinese-CLIP")
from cn_clip.clip.model import CLIP,convert_weights
import torch
from cn_clip.clip import load_from_name, available_models

model, preprocess = load_from_name("ViT-B-16", device="cuda", download_root='./')
checkpoint = torch.load("/home/ubuntu/DataSet/BS/experiments/phonePicture/checkpoints/epoch85.pt", map_location='cuda')
sd = checkpoint["state_dict"]
if next(iter(sd.items()))[0].startswith('module'):
    sd = {k[len('module.'):]: v for k, v in sd.items() if "bert.pooler" not in k}
model.load_state_dict(sd)

# vision_model_config_file = "/home/ubuntu/GITHUG/Chinese-CLIP/cn_clip/clip/model_configs/ViT-B-16.json"
    
# text_model_config_file = "/home/ubuntu/GITHUG/Chinese-CLIP/cn_clip/clip/model_configs/RoBERTa-wwm-ext-base-chinese.json"
    
# with open(vision_model_config_file, 'r') as fv, open(text_model_config_file, 'r') as ft:
#     model_info = json.load(fv)
#     if isinstance(model_info['vision_layers'], str):
#         model_info['vision_layers'] = eval(model_info['vision_layers'])        
#     for k, v in json.load(ft).items():
#         model_info[k] = v

# model = CLIP(**model_info)
# convert_weights(model)
# model.cuda(0)


# model.load_state_dict(sd)
# model.eval()

# model.encode_image