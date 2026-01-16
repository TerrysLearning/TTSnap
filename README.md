# TTSnap
The official implementation of TTSnap.
More and more will be updated soon. 

We increase the efficiency of global search method for Test Time Scaling on Text to Image diffusion models, e.g. best-of-n. 

## 1. Generate Data

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

## 2. Train the Reward Model

We use our self-distillation strategy to finetune the reward model.

Given the datafolder of the generated images, their prompts and their reward values, we train the noise-aware reward model for each inference step.

![Figure description](figure-1.png)


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

The ground truth reward of training and validation data is listed in reward_train/setup.py 

To train the image-reward model:
```
cd reward_train
accelerate launch --num_processes=<Number of your GPUs> train_cur.py \
    --c configs/imr_flux.yaml \
    --name <name of your choice, for saving the outputs> \
    --<other commands you need> 
```
Other arguments can be specified in the config file or add in the command lines

## 3. Run the TTSnap Simulation
This part is to test and compare the performance TTSnap under the same set of generate trajectories.  

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

