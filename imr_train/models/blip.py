'''
 * Adapted from BLIP (https://github.com/salesforce/BLIP)
'''

import warnings
warnings.filterwarnings("ignore")

from models.vit import interpolate_pos_embed
from models.vit_timecond import VisionTransformer as ViT_timecond
from models.vit import VisionTransformer as ViT_normal
from models.vit_adaln import VisionTransformer as ViT_adaln
from transformers import BertTokenizer

import torch
import os
from urllib.parse import urlparse
from timm.models.hub import download_cached_file


def init_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_special_tokens({'bos_token':'[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]']})       
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]  
    return tokenizer


def create_vit(vit, image_size, use_grad_checkpointing=False, 
               ckpt_layer=0, drop_path_rate=0, vit_type='vit_normal'):
        
    assert vit in ['base', 'large'], "vit parameter must be base or large"
    # //// ViT with time condition AdaLn or without
    if vit_type == 'vit_normal':
        ViT = ViT_normal
    elif vit_type == 'vit_timecond':
        ViT = ViT_timecond
    elif vit_type == 'vit_adaln':
        ViT = ViT_adaln
    else:
        raise ValueError(f"vit_type must be 'vit_normal' or 'vit_timecond' or 'vit_adaln', but got {vit_type}")
    print(f"vit_type: {vit_type}")
    # ////
    if vit=='base':
        vision_width = 768
        visual_encoder = ViT(img_size=image_size, patch_size=16, embed_dim=vision_width, depth=12, 
                                           num_heads=12, use_grad_checkpointing=use_grad_checkpointing, ckpt_layer=ckpt_layer,
                                           drop_path_rate=0 or drop_path_rate
                                          )   
    elif vit=='large':
        vision_width = 1024
        visual_encoder = ViT(img_size=image_size, patch_size=16, embed_dim=vision_width, depth=24, 
                                           num_heads=16, use_grad_checkpointing=use_grad_checkpointing, ckpt_layer=ckpt_layer,
                                           drop_path_rate=0.1 or drop_path_rate
                                          )   
    return visual_encoder, vision_width


def is_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")

def load_checkpoint(model,url_or_filename):
    if is_url(url_or_filename):
        cached_file = download_cached_file(url_or_filename, check_hash=False, progress=True)
        print('cached_file: ', cached_file)
        assert 0 
        checkpoint = torch.load(cached_file, map_location='cpu') 
    elif os.path.isfile(url_or_filename):       
        checkpoint = torch.load(url_or_filename, map_location='cpu') 
    else:
        raise RuntimeError('checkpoint url or path is invalid')
        
    state_dict = checkpoint['model']
    
    state_dict['visual_encoder.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder) 
    if 'visual_encoder_m.pos_embed' in model.state_dict().keys():
        state_dict['visual_encoder_m.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],
                                                                         model.visual_encoder_m)    
    for key in model.state_dict().keys():
        if key in state_dict.keys():
            if state_dict[key].shape!=model.state_dict()[key].shape:
                print(key, ": ", state_dict[key].shape, ', ', model.state_dict()[key].shape)
                del state_dict[key]
    
    msg = model.load_state_dict(state_dict,strict=False)
    print('load checkpoint from %s'%url_or_filename)  
    assert 0 
    return model,msg
    