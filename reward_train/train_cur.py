'''
This script is used to train the model in multiple GPUs.
'''
import numpy as np
import torch 
import yaml 
from types import SimpleNamespace
import random
import wandb
import argparse
from lr_scheduler import get_scheduler
from peft import get_peft_model, LoraConfig, TaskType
from dataset import *
from torch.utils.data import DataLoader
from imagereward import ImageReward_Model
from pickscore import PickScore_Model
from hps import HPS_Model
from utils import * 
from eval import evaluate, save_eval, stat
from loss import compute_loss
import os 
from accelerate import Accelerator
import shutil
from accelerate.utils import DistributedDataParallelKwargs
from time import time
from set_args import set_args
import json
from HPSv2.hpsv2.src.open_clip import get_tokenizer

if __name__ == "__main__":
    config_dict, args_data, args_train, args_log = set_args()
    device = 'cuda'
    
    seed = config_dict['seed']
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    if args_log['use_wandb']:
        wandb_log = wandb.init(
            project='TTSnap',
            name=args_log['name'],
            config=config_dict,
        )
    # set up the output directory for the plots
    history = {}
    plot_dir = os.path.join(args_log['plot_dir'], args_log['name'])
    os.makedirs(plot_dir, exist_ok=True)

    # setup the dataset and dataloader
    def setup_dataset(data_type, batch_size, use_pair=False):
        if use_pair and data_type == 'train':
            Dataset_Class = TerryDataset_Pair
        else:
            Dataset_Class = TerryDataset
        dataset = Dataset_Class(
            reward_name=config_dict['reward_name'],
            data_type=data_type,
            data_config=args_data)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True if data_type == 'train' else False,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn)
        return dataset, dataloader

    if args_train['loss_preference_scale'] > 0:
        use_pair = True
        batch_size_train = args_train['batch_size']// 2
    else:
        use_pair = False
        batch_size_train = args_train['batch_size']
        
    train_dataset, train_loader = setup_dataset('train', batch_size=batch_size_train, use_pair=use_pair)
    valid_dataset, valid_loader = setup_dataset('valid', batch_size=args_train['batch_size'], use_pair=use_pair)
   
    # setup the model
    if config_dict['reward_name'] == 'imr':
        model = ImageReward_Model("setup/config.yaml", args_train['vit_type'])
        state_dict = torch.load(args_train['base_model_path'], map_location="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        model.to(device)
        print("missing & unexpected keys:", missing, unexpected)

    elif config_dict['reward_name'] == 'pick':
        model = PickScore_Model.from_pretrained("yuvalkirstain/PickScore_v1")
        if args_train['base_model_path'] != "yuvalkirstain/PickScore_v1":
            state_dict = torch.load(args_train['base_model_path'], map_location="cpu")
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print("missing & unexpected keys:", missing, unexpected)
        model.to(device)    
        model.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    
    elif config_dict['reward_name'] == 'hps':
        model = HPS_Model(device) # hps model is warped inside HPS_Model
        model.processor = get_tokenizer('ViT-H-14')
        state_dict = torch.load(args_train["base_model_path"], map_location='cpu')
        missing, unexpected = model.model.load_state_dict(state_dict, strict=False)
        model.model.to(device)
        print("missing & unexpected keys:", missing, unexpected)
    
    else: 
        raise NotImplementedError(f"Reward model {config_dict['reward_name']} not implemented.")

    model.requires_grad_(True)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(gradient_accumulation_steps=args_train['accum_steps'], kwargs_handlers=[ddp_kwargs])    

    model, train_loader, valid_loader = accelerator.prepare(
        model, train_loader, valid_loader
    )

    def validate(dataloader, history, t_target, message="valid", epoch=0):
        model.eval()
        eval_t, eval_p, eval_pred, eval_gt = evaluate(model, dataloader, t_target, args_train, 
                            message=message)

        eval_t = accelerator.gather_for_metrics(eval_t)
        eval_p = accelerator.gather_for_metrics(eval_p)
        eval_pred = accelerator.gather_for_metrics(eval_pred)
        eval_gt = accelerator.gather_for_metrics(eval_gt)

        if accelerator.is_main_process:
            out = stat(eval_t, eval_p, eval_pred, eval_gt)
            history = save_eval(history, out, epoch, message, plot_dir)
        accelerator.wait_for_everyone()
        model.train()
        return history

    valid_every_step = args_log['valid_every_epoch'] * len(train_loader)

    last_time_lr_scale = 1.0 # control the decay of initial lr for each time step in curriculum learning
    step_targets = torch.tensor(config_dict['time_config']['step_targets']).long()
    step_targets, _ = torch.sort(step_targets, descending=True)

    # Output the parameters number of the model and save the model structure
    # if accelerator.is_main_process:
    #     params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #     print(f"params: { params}")  
    #     write_down_modules(model)
    #     print_dict(args_data)
    #     print_dict(args_log)
    #     print_dict(args_train)


    if args_log['eval_start']:
        if args_log['do_valid']:
            for s in step_targets.tolist():
                time_id = torch.tensor([s]).long()
                history = validate(valid_loader, history, t_target= time_id, message="valid", epoch=0)

    epochs_pass = 0 
    for i, time_id in enumerate(step_targets):

        time_id = torch.tensor([time_id]).long()
        batch_time_id = time_id.repeat(batch_size_train)
        total_steps = accelerator.num_processes * len(train_loader) * args_train['curr_interval'] / (args_train['accum_steps'])

        start_lr_ratio = -(1.0 - last_time_lr_scale)*i / len(step_targets) + 1.0
        lr_start = float(args_train['lr']) * start_lr_ratio

        if accelerator.is_main_process:
            print("#################################################")
            print("Training for time step:", time_id.item())
            print("Initial learning rate for this time step:", lr_start)
        
        # Free the memory, otherwise OOM may happen
        if i > 0: 
            del optimizer, scheduler
            accelerator.free_memory()

        optimizer = torch.optim.AdamW(model.parameters(), 
                    lr=lr_start, 
                    betas=(float(args_train['adam_beta1']), float(args_train['adam_beta2'])), 
                    eps= float(args_train['adam_eps']), 
                    weight_decay=float(args_train['weight_decay']))

        scheduler = get_scheduler(optimizer, 
                            total_steps, 
                            warmup_ratio=float(args_train['warmup_ratio']),
                            warmup_type=args_train['warmup_type'],
                            scheduler_type=args_train['scheduler_type'], 
                            min_lr_scale=float(args_train['min_lr_scale']))

        optimizer, scheduler = accelerator.prepare(optimizer, scheduler) #//////
        step_count = 0
        model.train()
        for epoch in range(args_train['curr_interval']):
            for step, batch_data in enumerate(train_loader):
                batch_data['t_id'] = batch_time_id.to(device)
                batch_data['image'] = load_image_batch(train_dataset, batch_data['img_id'], 
                            batch_data['p_id'],  batch_data['t_id'], use_pair=use_pair).to(device)
                
                with accelerator.accumulate(model):
                    batch_pred = model(batch_data)
                    loss, loss_list = compute_loss(batch_pred, batch_data, args_train)
                    accelerator.backward(loss)
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), args_train['max_grad_norm'])
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()

                step_count += 1
           
                if accelerator.is_main_process:
                    if step_count % args_log['print_loss_every_iter'] == 0:
                        # gather losses from all GPUs, then average
                        print(f"epoch: {epochs_pass}, step {step+1}: loss {loss.item():.4f}")

                    if args_log['use_wandb']:
                        for key, value in loss_list.items():
                            wandb_log.log({
                                f"loss_{key}": value.item(),
                            })
                        wandb_log.log({
                            f"loss_total": loss.item(),
                        })
                        wandb_log.log({
                            f"learning rate": scheduler.get_last_lr()[0],
                        })
                    # print(f'epoch: {epochs_pass}, step {step+1}: lr', scheduler.get_last_lr()[0])

            epochs_pass += 1
            if args_log['do_valid'] and (epochs_pass) % args_log['valid_every_epoch'] == 0:
                history = validate(valid_loader, history, t_target= time_id, message="valid", epoch=epoch)

            # if accelerator.is_main_process and args_log['save_checkpoint']:
            #     state_dict = accelerator.get_state_dict(model)
            #     save_ckpt_dir = os.path.join(args_log['checkpoint_dir']+'_'+ config_dict['reward_name'], args_log['name'])
            #     os.makedirs(save_ckpt_dir, exist_ok=True)
            #     torch.save(state_dict, f'{save_ckpt_dir}/Ct{time_id.item()}.pt')
            #     print("saved the checkpoint to", f'{save_ckpt_dir}/Ct{time_id.item()}.pt')
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
