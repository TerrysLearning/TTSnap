from time import time
import torch
import json 
import os 
from flux_pipeline_ntm_modify import FluxNegToMePipeline
from flux_scheduler_modify import MyCustomFluxScheduler
import time

scheduler = MyCustomFluxScheduler.from_pretrained("black-forest-labs/FLUX.1-dev", 
                                                  subfolder="scheduler",
                                                  torch_dtype=torch.bfloat16)
pipe = FluxNegToMePipeline.from_pretrained("black-forest-labs/FLUX.1-dev", 
                    scheduler=scheduler,
                    torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")
print(pipe.scheduler.config.stochastic_sampling)  # Disable stochastic sampling for deterministic output

negtome_args = {
    'use_negtome': False,
    'merging_alpha': 1.2, # 0.9
    'merging_threshold': 0.65, 
    'merging_t_start': 1000, 
    'merging_t_end': 900,
    'num_joint_blocks': -1, # number of joint transformer blocks (flux) to apply negtome
    'num_single_blocks': -1, # number of single transformer blocks (flux) to apply negtome
}

# input prompt 
with open('dataset_prompts_train.json', 'r') as f:
    prompts = json.load(f)

# hyperparameters
num_inference_steps = 20
num_images_per_prompt = 8 # generate 8 images across the batch
height = width = 512  # image resolution
out_folder = "data_flux_train" # name of the output folder
start_p_id = 3600 # starting prompt id
end_p_id = 4000 # ending prompt id

os.makedirs(out_folder, exist_ok=True)
for p_id, prompt in enumerate(prompts[start_p_id:end_p_id]):
    print(prompt)
    # generator = torch.Generator(pipe.device).manual_seed(0)    
    # Measure time
    start_time = time.time()
    x0_out_list = pipe(
        prompt=prompt,
        guidance_scale=3.5,
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