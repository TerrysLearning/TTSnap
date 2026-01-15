import torch
from transformers import CLIPModel

class PickScore_Model(CLIPModel):

    def forward(self, batch_data):
        # get base model features
        text_input = self.processor(text=batch_data['prompt'], padding=True, 
                truncation=True, max_length=77, return_tensors="pt")
        text_ids = text_input['input_ids'].to(self.device)
        text_mask = text_input['attention_mask'].to(self.device)
        images = batch_data['image'].to(self.device)

        image_embs = self.get_image_features(**{'pixel_values': images})
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

        text_embs = self.get_text_features(**{'input_ids': text_ids, 'attention_mask': text_mask})
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

        # score
        scores = self.logit_scale.exp() * (text_embs @ image_embs.T) 
        pickscore_scores = torch.diagonal(scores)  # [batch_size]

        batch_data = {
            'reward_pred': pickscore_scores,
        }

        return batch_data
