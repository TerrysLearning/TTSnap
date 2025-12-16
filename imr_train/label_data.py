import os 
import json 
import numpy as np 
import yaml
import ImageReward as RM 

model = RM.load("ImageReward-v1.0")

model.mean = 0.0
model.std = 1.0

def label_gt():
    data_folder = "data_sdxl_val"
    with open("prompt_json/dataset_prompts_validation.json", "r") as f:
        prompt_list = json.load(f)
    
    start_p_id = 0 
    end_p_id = 200
    reward_gt_dict = {}
    end_img_id = 200

    for p_id in range(len(prompt_list)):
        if p_id < start_p_id:
            continue
        if p_id >= end_p_id:
            break
        p_folder = f"p{p_id:04d}"
        reward_gt_dict[p_folder] = {}
        for img_id, img_subfolder in enumerate(sorted(os.listdir(f"{data_folder}/{p_folder}"))):
            if img_id >= end_img_id:
                break
            img_path = os.path.join(f"{data_folder}/{p_folder}/{img_subfolder}", f"s19.png")
            reward = model.score(prompt_list[p_id], [img_path])
            reward_gt_dict[p_folder][img_subfolder] = reward
            print(f"p_id: {p_id}, {img_subfolder}: {reward}")
        
    with open("reward_gt_v_4k.yaml", "w") as f:
        yaml.dump(reward_gt_dict, f)
    print("done")

if __name__ == "__main__":
    # get_step_mean_std()
    label_gt()
