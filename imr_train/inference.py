import imp
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from models.blip_pretrain_oft import BLIP_Pretrain
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


class MLP_Uncertainty(nn.Module):
    # Make it save for the uncertainty prediction
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1),
            # nn.ReLU(),
            # nn.Dropout(0.2),
            # nn.Linear(1024, 1),
        )
        # initial MLP param
        for name, param in self.layers.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=0.05)
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
        self.uncertainty_head = MLP_Uncertainty(config['ImageReward']['mlp_dim'])
        # self.mean = 0.16717362830052426
        # self.std = 1.0333394966054072

    def encode_single(self, text_ids, text_mask, image, noise_level, output_feature=False):
        text_ids = text_ids.view(text_ids.shape[0], -1).to(self.device) # [batch_size, seq_len]
        text_mask = text_mask.view(text_mask.shape[0], -1).to(self.device) # [batch_size, seq_len]
        image = image.to(self.device) # [batch_size, C, H, W]
        noise_level = noise_level.to(self.device)
        
        # encode image
        image_embeds, visual_feature = self.blip.visual_encoder(image, noise_level, output_feature=output_feature)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(self.device)
        # encode text
        emb_text, text_feature = self.blip.text_encoder(text_ids,
                                            attention_mask = text_mask,
                                            encoder_hidden_states = image_embeds,
                                            encoder_attention_mask = image_atts,
                                            return_dict = True,
                                            output_feature=output_feature,
                                           ) 
        last_emb = emb_text.last_hidden_state
        last_emb = last_emb[:, 0, :].float()
        if output_feature:
            # visual_feature = visual_feature[:, 0, :] # [B*N, 1024]
            # text_feature = text_feature[:, 0, :] # [B*N, seq_len, 768]
            # only take the last hidden layer
            visual_feature = visual_feature.reshape(-1, 24, 197, 1024) # [B*N, 1024]
            visual_feature = visual_feature[:,-1,0,:]
            text_feature = text_feature.reshape(-1, 12, 35, 768) # [B*N, seq_len, 768]
            text_feature = text_feature[:,-1,0,:]
        else:
            visual_feature = None
            text_feature = None
        return {
            'last_emb': last_emb,
            'visual_feature': visual_feature, #  [B*N=24, 1, 1024]  [B*N=24, 197, 1024]
            'text_feature': text_feature # [B*N=24, 1, 1024] [B*N=12, seq_len, 768] 
        }
    
 
    def forward(self, batch_data, output_feature=False):
        # encode pair
        out_features = self.encode_single(batch_data['text_ids'], batch_data['text_mask'], batch_data['image'], batch_data['t_id'], output_feature=output_feature)
        # forward
        last_emb = out_features['last_emb']
        reward = self.mlp(last_emb)
        uncertainty = self.uncertainty_head(last_emb)
        # reward = (reward - self.mean) / self.std
        # if output_feature:
        #     visual_feature = self.repa_visual_proj_head(out_features['visual_feature'].reshape(-1, 1024)) # [B*N, 1024]
        #     text_feature = self.repa_text_proj_head(out_features['text_feature'].reshape(-1, 768)) # [B*N, 768]
        #     # visual_feature = out_features['visual_feature'].reshape(-1, 1024)
        #     # text_feature = out_features['text_feature'].reshape(-1, 768)
        # else:
        visual_feature = None
        text_feature = None
        batch_data = {
            'reward_pred': reward,
            'uncertainty_pred': uncertainty,
            'visual_feature': visual_feature,
            'text_feature': text_feature
        }
        return batch_data
    

        
    