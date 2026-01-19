# TTSnap
The official implementation of TTSnap.
More and more will be updated soon. 

We increase the efficiency of global search method for Test Time Scaling on Text to Image diffusion models, e.g. best-of-n. 

## 1. Generate Data

Given a set of text prompts, we generate training data by sampling from the diffusion model and storing the intermediate Tweedie-estimated images.

We use [NegToMe](https://github.com/1jsingh/negtome) increases the diversity of the images.

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

## 2. Train the Reward Model

We use our self-distillation strategy to finetune the reward model.

To train the noise-aware reward models, we introduce a curriculum self-distillation strategy that gradually shifts the training domain from clean images to increasingly noisy ones.
After one epoch at each noise level, we save the model weights and proceed to the next, ensuring small domain gaps and stable, efficient training.
![Noise-Aware Finetuning](doc/figure-1.png)

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

**Training guidance:** 
Modify the location of the datafolder and checkpoint path in the config file, e.g. c_flux.yaml. 

You can also use only part of data to train by modifying: *train_prompt_range*,  *train_num_per_prompt*. 

The prompts and the ground truth rewards of training and validation data is listed in reward_train/setup. 
Prompts are from imagereward training set. 

Commands for training: 
```
cd reward_train
accelerate launch --num_processes=<Number of your GPUs> train_cur.py \
    --c configs/imr_flux.yaml \
    --name <name of your choice, for saving the outputs> \
    --<other commands you need> 
```
Other arguments can be specified in the config file or add in the command lines

## 3. Run the TTSnap Simulation

To avoid the prohibitive computational cost of real-time image generation during testing, we adopt an **offline simulation protocol**. This approach allows for **rapid** and **statistically robust** evaluation of various Test-Time Scaling (TTS) strategies by utilizing a pre-computed image pool.

### Reward Matrix Pre-computation
We first generate an extensive image pool on the validation set. For each prompt, we record multiple generation trajectories and compute their corresponding reward values. This results in a reward tensor with the following dimensions: (*prompts_number*, *image_number_per_prompt*, *timestep_number*)


Given a pool of images generated on the validation set, we can compute rewards values of shape 
Then we can run the test-time scaling simulation on these reward values.  

In the simulation: 
- Given these fix set of generation trajectories.
- image_number_per_prompt is a large number e.g. 200
- random take some as trajectories and do TTS

The objective of simulation: 
- Fair comparision performance on the same set of generate trajectories.
- So we can find better configritions and fair performance with other settings
- To mitigate randomness, need run generation many times, but this is very costive. each time one generation 
 
The reward computed for each prompt, each image and each timestep with/without NAFT are saved in simulation/values.
The prompts are from the imagereward validation set. 
In our validation process, we uses 200 prompts and 200 images each prompt with 20 timesteps. 

## 4. Run TTSnap

**TODO:** 
- [ ] release the code for TTSnap simulation
- [ ] release the code for TTSnap run
- [ ] link for trained checkpoints
- [ ] link for data

If you find our work useful, please cite: 
```
@article{yu2025ttsnap,
  title={TTSnap: Test-Time Scaling of Diffusion Models via Noise-Aware Pruning},
  author={Yu, Qingtao and Song, Changlin and Sun, Minghao and Yu, Zhengyang and Verma, Vinay Kumar and Roy, Soumya and Negi, Sumit and Li, Hongdong and Campbell, Dylan},
  journal={arXiv preprint arXiv:2511.22242},
  year={2025}
}
```

