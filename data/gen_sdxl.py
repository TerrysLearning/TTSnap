from time import time
import torch
import json 
import os 
from sdxl_pipeline_ntm_modify import SDXLNegToMePipeline
from sdxl_scheduler_modify import MyCustomSDXLScheduler
import time


# SDXL align with the NegToMe used model
scheduler = MyCustomSDXLScheduler.from_pretrained("SG161222/RealVisXL_V4.0", 
                                                  subfolder="scheduler",
                                                  torch_dtype=torch.bfloat16)

pipe = SDXLNegToMePipeline.from_pretrained("SG161222/RealVisXL_V4.0", 
                    scheduler=scheduler,
                    torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")
# print(pipe.scheduler.config.stochastic_sampling)  # Disable stochastic sampling for deterministic output

negtome_args = {
    'use_negtome': False,
    'merging_alpha': 2.5, #controls diversity: higher alpha pushes images further apart
    'merging_threshold': 0.7, #controls which features are pushed apart: higher threshold preserves original features more
    'merging_dropout': 0.,
    'merging_t_start': 950,
    'merging_t_end': 800,
}
print(negtome_args)
# input prompt 
with open('dataset_prompts_train.json', 'r') as f:
    prompts = json.load(f)

# hyperparameters
num_inference_steps = 20
num_images_per_prompt = 8 # generate 8 images across the batch
height = width = 512 
out_folder = "data_sdxl_train"
start_p_id = 1533
end_p_id = 1600
print(f"Generating prompts from {start_p_id} to {end_p_id}")
os.makedirs(out_folder, exist_ok=True)
for p_id, prompt in enumerate(prompts[start_p_id:end_p_id]):
    print(prompt)
    # generator = torch.Generator(pipe.device).manual_seed(0)    
    # Measure time
    start_time = time.time()
    x0_out_list = pipe(
        prompt=prompt,
        guidance_scale=5.0,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        generator=None,
        num_images_per_prompt=num_images_per_prompt,
        use_negtome=True,
        negtome_args=negtome_args,
        return_dict=False
    )

    p_id_save = p_id + start_p_id
    for i in range(num_images_per_prompt):
        os.makedirs(f"{out_folder}/p{p_id_save:04d}/id{i:03d}", exist_ok=True)

    for i, x0_pred in enumerate(x0_out_list):
        for j, x0_img in enumerate(x0_pred):
            x0_img.save(f"{out_folder}/p{p_id_save:04d}/id{j:03d}/s{i:02d}.png")
    print(f"p_id {p_id_save} done")
    end_time = time.time()
    print(f"Time taken for prompt {p_id_save}: {end_time - start_time} seconds")
print(f"all done: generate {start_p_id}-{end_p_id} training images")