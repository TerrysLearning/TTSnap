import os
import torch
from PIL import Image
from HPSv2.hpsv2.src.open_clip import create_model_and_transforms

class HPS_Model(torch.nn.Module):
    def __init__(self, device: str = 'cpu'):
        super().__init__()
        self.model, _, _ = create_model_and_transforms(
            'ViT-H-14',
            'laion2B-s32B-b79k',
            precision='amp',
            device=device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False
        )

       
    def forward(self, batch_data):

        images = batch_data['image']
        text_input = self.processor(batch_data['prompt']).to(images.device)

        outputs = self.model(images, text_input)
        image_features, text_features = outputs['image_features'], outputs['text_features']
        
        # compute similarity logits
        logits_per_image = image_features @ text_features.T
        hps_scores = torch.diagonal(logits_per_image) # [batch_size]
        
        batch_data = {
            'reward_pred': hps_scores
        }
        
        return batch_data


