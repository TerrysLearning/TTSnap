# TTSnap
The official implementation of TTSnap.
More and more will be updated soon. 

## Introduction
We increase the efficiency of global search method for Test Time Scaling on Text to Image diffusion models, e.g. best-of-n. 

## Generate Data

Given a set of prompts saved in json file, we generate images per-prompt using NegToMe. 

NegToMe increases the diversity of the images.

The code is modified from  https://github.com/1jsingh/negtome

The generated datafile structure is like: 
```
# p represents the prompt 
# id represents each image 
# s represents the tweedie/ruler estimation decoded image at each timestep 
<data_folder_name>
    p0000
        id000
            s00
            s01 
            ...
            s19
        id001
        id002
        ...
        idxxx
    p0001
    p0002
    pxxxx
```

The code for generate the flux and sdxl data are in 'data/gen_flux.py' and 'data/gen_sdxl.py'.

**Environment Setup:**
```
pip install diffusers
```
**Run the code:** 
```
python gen_flux.py
```

## Train the Reward Model

We use our self-distillation strategy to finetune the reward model.

Given the datafolder of the generated images, their prompts and their reward values, we train the noise-aware reward model for each inference step.



**Environment Setup:**
```
conda create --n reward python=3.10.18
pip install image-reward
pip install peft
pip install transformers==4.53.1
pip install matplotlib
pip install scipy
pip install accelerate
pip install clint ftfy
```

**Checkpoints download**
Original reward model checkpoints before finetuning: [link](https://drive.google.com/drive/folders/1Vzlba2rBCAEi9rUG_wrmttIfKGaLE1mk?usp=drive_link)

Reward checkpoints after finetuning: 

**Run the code:** 
Modify the location of the datafolder and checkpoint path in the config file, e.g. c_flux.yaml. 
You can also use only part of data to train by modifying: *train_prompt_range*,  *train_num_per_prompt*. 

To train the image-reward model:
```
cd imr_train
accelerate launch --num_processes=<Number of your GPUs> train_cur.py \
    --c configs/c_flux.yaml \
    --name <name of your choice, for saving the outputs> \
    --save_checkpoint  \ 
```
Other arguments can be specified in the config file or add in the command lines

## Run the TTSnap Simulation
This part is to test and compare the performance TTSnap under the same set of generate trajectories.  

## Run TTSnap

**TODO:** 
- [ ] release the code for training HPS and Training 
- [ ] release the code for TTSnap simulation
- [ ] release the code for TTSnap run
- [ ] add the base model for pickscore
- [ ] link for trained checkpoints
- [ ] link for data


