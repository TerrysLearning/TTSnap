import imp
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from models.blip_pretrain import BLIP_Pretrain
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import yaml
from utils import _transform, init_tokenizer



class MLP(nn.Module):
# From the ImageRward open-source code, use only linear projection no non-linearity
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            #nn.ReLU(),
            nn.Linear(16, 1)
        )
        # initial MLP param
        for name, param in self.layers.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=1.0/(self.input_size+1))
            if 'bias' in name:
                nn.init.constant_(param, val=0.0)
        
    def forward(self, input):
        return self.layers(input)



class ImageReward_Model(nn.Module):
    def __init__(self, config_path, vit_type='vit_normal', device='cpu'):
        super().__init__()
        # I think this part can be better organized /////
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.device = device
        
        self.blip = BLIP_Pretrain(image_size=224, vit='large', med_config='setup/med_config.json', vit_type=vit_type)
        self.preprocess = _transform(config['BLIP']['image_size'])
        self.mlp = MLP(config['ImageReward']['mlp_dim'])
        # use the same MLP to predict the uncertainty of the reward
        # self.mean = 0.16717362830052426
        # self.std = 1.0333394966054072

    def encode_single(self, text_ids, text_mask, image, noise_level):
        text_ids = text_ids.view(text_ids.shape[0], -1).to(self.device) # [batch_size, seq_len]
        text_mask = text_mask.view(text_mask.shape[0], -1).to(self.device) # [batch_size, seq_len]
        image = image.to(self.device) # [batch_size, C, H, W]
        noise_level = noise_level.to(self.device)
        
        # encode image
        image_embeds = self.blip.visual_encoder(image, noise_level)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(self.device)
        # encode text
        emb_text = self.blip.text_encoder(text_ids,
                                            attention_mask = text_mask,
                                            encoder_hidden_states = image_embeds,
                                            encoder_attention_mask = image_atts,
                                            return_dict = True) 
        last_emb = emb_text.last_hidden_state
        last_emb = last_emb[:, 0, :].float()
        return {
            'last_emb': last_emb,
        }
    
 
    def forward(self, batch_data):
        # encode pair
        model_output = self.encode_single(batch_data['text_ids'], batch_data['text_mask'], batch_data['image'], batch_data['t_id'])
        # forward
        last_emb = model_output['last_emb']
        reward = self.mlp(last_emb)
        batch_data = {
            'reward_pred': reward,
        }
        return batch_data
    

        
    