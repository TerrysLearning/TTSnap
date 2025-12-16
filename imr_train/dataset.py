from torch.utils.data import Dataset
import os
import yaml
from utils import _transform, init_tokenizer
from PIL import Image
import random 
import torch 
from concurrent.futures import ThreadPoolExecutor
from utils import FLUX_SIGMAS_30
import json

preprocess_image = _transform(224)

class TerryDataset(Dataset):
    def __init__(self,
                data_type, # 'train', 'valid', 'test'
                data_config # from the yaml config file
                ):
        self.data_type = data_type
        self.data_config = data_config
        self.tokenizer = init_tokenizer()
        
        with open(self.data_config[f'{data_type}_prompts_file'], 'r') as f:
            prompts_list = json.load(f)
        with open(self.data_config[f'{data_type}_gt_file'], 'r') as f:
            self.reward_gt_dict = yaml.safe_load(f)

        self.prompts_info = {}
        self.data_tuples = [] # [(prompt_id, image_id, image_score_gt)]
        for p_id, prompt in enumerate(prompts_list):
            if p_id < self.data_config[f'{data_type}_prompt_range']:
                self.prompts_info[p_id] = prompt
                for img_id in range(self.data_config[f'{data_type}_num_per_prompt']):
                    self.data_tuples.append((p_id, img_id))

    def __len__(self):
        return len(self.data_tuples)
    
    def get_image_path(self, img_id, p_id, t_id):   
        return os.path.join(self.data_config[f'{self.data_type}_data_folder'], f"p{p_id:04d}", 
                f"id{img_id:03d}", f"s{t_id:02d}.png")
    
    def load_(self, p_id, img_id):
        reward_gt = self.reward_gt_dict[f"p{p_id:04d}"][f"id{img_id:03d}"]
        prompt = self.prompts_info[p_id]
        text_input = self.tokenizer(prompt, padding='max_length', 
                        truncation=True, max_length=35, return_tensors="pt")
        text_ids = text_input.input_ids
        text_mask = text_input.attention_mask
        return reward_gt, text_ids, text_mask

        
    def __getitem__(self, idx):
        p_id, img_id = self.data_tuples[idx]
        reward_gt = self.reward_gt_dict[f"p{p_id:04d}"][f"id{img_id:03d}"]
        prompt = self.prompts_info[p_id]
        text_input = self.tokenizer(prompt, padding='max_length', 
                        truncation=True, max_length=35, return_tensors="pt")
        text_ids = text_input.input_ids
        text_mask = text_input.attention_mask
        # Maybe same batch should have the same time step, but may do that later
        return {
            'text_ids': text_ids, #[1, 35]
            'text_mask': text_mask, #[1, 35]
            'reward_gt': torch.tensor([reward_gt]), 
            'p_id': torch.tensor([p_id]), # these two ids maybe can be used for REPA later
            'img_id': torch.tensor([img_id]),
        }


class TerryDataset_Pair(TerryDataset):

    def __getitem__(self, idx):
        p_id, img_id = self.data_tuples[idx]
        reward_gt = self.reward_gt_dict[f"p{p_id:04d}"][f"id{img_id:03d}"]
        prompt = self.prompts_info[p_id]
        text_input = self.tokenizer(prompt, padding='max_length', 
                        truncation=True, max_length=35, return_tensors="pt")
        text_ids = text_input.input_ids
        text_mask = text_input.attention_mask
        
        other_img_ids = [x for x in range(self.data_config[f'{self.data_type}_num_per_prompt']) if x != img_id]
        img_id_ = random.choice(other_img_ids)
        reward_gt_ = self.reward_gt_dict[f"p{p_id:04d}"][f"id{img_id_:03d}"]
        if reward_gt > reward_gt_:
            return {
                'text_ids': torch.cat([text_ids, text_ids], dim=0),
                'text_mask': torch.cat([text_mask, text_mask], dim=0),
                'reward_gt': torch.tensor([reward_gt, reward_gt_], dtype=torch.float32),
                'p_id': torch.tensor([p_id, p_id], dtype=torch.long),
                'img_id': torch.tensor([img_id, img_id_], dtype=torch.long),
            }
        else:
            return {
                'text_ids': torch.cat([text_ids, text_ids], dim=0),
                'text_mask': torch.cat([text_mask, text_mask], dim=0),
                'reward_gt': torch.tensor([reward_gt_, reward_gt], dtype=torch.float32),
                'p_id': torch.tensor([p_id, p_id], dtype=torch.long),
                'img_id': torch.tensor([img_id_, img_id], dtype=torch.long),
            }
        


class Time_Sampler:
    # The sampler for time steps of diffusion
    def __init__(self, time_config, curr_interval):
        self.step_targets = torch.tensor(time_config['step_targets']).long()
        # self.step_targets_eval = torch.tensor(time_config['step_targets_eval']).long()
        # Curriculum interval 
        self.curr_interval = curr_interval

    def random_one_time_step(self, batch_size):
        t_id = self.step_targets[torch.randint(0, len(self.step_targets), (1,))]
        return t_id.repeat(batch_size)

    def random_time_step(self, batch_size):
        t_ids = self.step_targets[torch.randint(0, len(self.step_targets), (batch_size,))]
        return t_ids

    def get_curr_time_step(self, epoch, batch_size):
        index = epoch // self.curr_interval
        return self.step_targets[index].repeat(batch_size)
    
    def get_one_time_step(self, batch_size):
        t_id = self.step_targets[0].repeat(batch_size)
        return t_id


def collate_fn(batch):
    text_ids = torch.cat([item['text_ids'] for item in batch], dim=0)
    text_mask = torch.cat([item['text_mask'] for item in batch], dim=0)
    reward_gt = torch.cat([item['reward_gt'] for item in batch])
    p_id = torch.cat([item['p_id'] for item in batch])
    img_id = torch.cat([item['img_id'] for item in batch])
    return {
        'text_ids': text_ids,
        'text_mask': text_mask,
        'reward_gt': reward_gt,
        'p_id': p_id,
        'img_id': img_id
    }


def load_image_batch(dataset: TerryDataset, img_id, p_id, t_id, use_pair=False):
    # load image batch in parallel given time_step
    # they are torch tensors of length b
    if use_pair:
        t_id = t_id.repeat_interleave(2)
    def load_fn(i):
        img_path = dataset.get_image_path(img_id[i].item(), p_id[i].item(), t_id[i].item())
        img = Image.open(img_path)
        return preprocess_image(img).unsqueeze(0)
    with ThreadPoolExecutor(max_workers=10) as ex:
        results = list(ex.map(load_fn, range(len(img_id))))
        images = torch.cat(results, dim=0)
    return images
    
def load_feature_batch(img_id, p_id, feature_type='visual_feature'):
    # load image batch in parallel given time_step
    # they are torch tensors of length b
    features = []
    for i in range(len(img_id)):
        feature_path = os.path.join("gt_features", f"p{p_id[i].item():04d}_id{img_id[i].item():03d}.pt")
        feature = torch.load(feature_path)[feature_type][-1,0,:]
        feature = feature.reshape(-1, feature.shape[-1]) # [N, 1024] or [N, 768]
        features.append(feature) # [N, 1024]
    return torch.cat(features) # [B*N, 1024]
    



# class TerryDataset(Dataset):
#     def __init__(self,
#                 data_type, # 'train', 'valid', 'test'
#                 data_config # from the yaml config file
#                 ):
#         self.data_type = data_type
#         self.data_config = data_config
#         self.tokenizer = init_tokenizer()
#         assert len(self.data_config[f'{data_type}_prompts_file']) == len(self.data_config[f'{data_type}_gt_file'])
#         for i in range(len(self.data_config[f'{data_type}_prompts_file'])):
#             self.load_data(i, self.data_config[f'{data_type}_prompts_file'][i], self.data_config[f'{data_type}_gt_file'][i])

#         self.prompts_info = {}

#     def load_data(self, subfolder_id, prompt_file, gt_file):
#         with open(prompt_file, 'r') as f:
#             prompts_list = json.load(f)
#         with open(gt_file, 'r') as f:
#             reward_gt_dict = yaml.safe_load(f)
#         self.image_prompt_score_tuples = [] # [(prompt_id, image_id, image_score_gt)]
#         for p_id, prompt in enumerate(prompts_list):
#             if p_id < self.data_config[f'{self.data_type}_prompt_range']:
#                 text_input = self.tokenizer(prompt, padding='max_length', 
#                             truncation=True, max_length=35, return_tensors="pt")
#                 text_id = text_input.input_ids
#                 text_mask = text_input.attention_mask
#                 for j in range(len(reward_gt_dict[f"p{p_id:04d}"])):
#                     gt = reward_gt_dict[f"p{p_id:04d}"][f"id{j:03d}"]
#                     self.image_prompt_score_tuples.append((subfolder_id, p_id, j, gt))

#     def __len__(self):
#         return len(self.image_prompt_score_tuples)
    
#     def get_image_path(self, img_id, p_id, t_id):   
#         return os.path.join(self.data_config[f'{self.data_type}_data_folder'], f"p{p_id:04d}", 
#                 f"id{img_id:03d}", f"s{t_id:02d}.png")
        
#     def __getitem__(self, idx):
#         subfolder_id, p_id, img_id, reward_gt = self.image_prompt_score_tuples[idx]
#         return {
#             'text_ids': text_id, #[1, 35]
#             'text_mask': text_mask, #[1, 35]
#             'reward_gt': reward_gt, 
#             'p_id': p_id, # these two ids maybe can be used for REPA later
#             'img_id': img_id,
#             'subfolder_id': subfolder_id,
#         }




