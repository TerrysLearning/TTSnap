import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from math import inf


# ------ Loss Functions ------
def mse_uncertainty_loss_fn(pred, gt, uncertainty_pred):
    mse = F.mse_loss(pred, gt, reduction='none')
    loss = 0.5*torch.exp(-uncertainty_pred)*mse + 0.5*uncertainty_pred
    loss = torch.mean(loss)
    return loss

def mse_loss_fn(pred, gt):
    loss = F.mse_loss(pred, gt, reduction='mean')
    return loss

def conf_loss_fn(uncertainty_pred, conf_target):
    loss = F.mse_loss(uncertainty_pred, conf_target, reduction='mean')
    return loss


def preference_loss_fn(pred, tau=0.1):
    # make batch_data score gt into pair [B/2, 2]
    # gt_pairs = batch_data['reward_gt'].reshape(-1, 2)
    pred_pairs = pred.reshape(-1, 2)
    # make sure first column is higher than corresponding second column 
    # indices = torch.argsort(gt_pairs, dim=1, descending=True)
    # pred_pairs_ranked = torch.gather(pred_pairs, dim=1, index=indices)
    # Compute the B Terry loss
    reward_pred_scaled = pred_pairs / tau
    target = torch.zeros(pred_pairs.shape[0], dtype=torch.long).to(pred_pairs.device)
    loss = F.cross_entropy(reward_pred_scaled, target, reduction='mean')
    return loss

def compute_loss(batch_pred, batch_data, args_train, **kwargs):
    device = batch_pred['reward_pred'].device # should be cuda
    loss = torch.tensor(0.0, device=device)
    batch_data['reward_gt'] = batch_data['reward_gt'].to(device).unsqueeze(1)
    reward_pred = batch_pred['reward_pred']  # [B, 1]
    reward_gt = batch_data['reward_gt'] # [B, 1]

    if args_train['bn_before_loss']:
        reward_pred = reward_pred - torch.mean(reward_pred)
        reward_gt = reward_gt - torch.mean(reward_gt)
        reward_pred = reward_pred / (torch.std(reward_pred) + 1e-10)
        reward_gt = reward_gt / (torch.std(reward_gt) + 1e-10)

    loss_list = {} # each element is a list of one type of loss
    if args_train['loss_mse_scale'] > 0:
        loss_mse = mse_loss_fn(reward_pred, reward_gt) #//////
        loss += args_train['loss_mse_scale'] * loss_mse
        loss_list['mse']= loss_mse

    if args_train['loss_preference_scale'] > 0:
        loss_pref = preference_loss_fn(reward_pred, tau=args_train['loss_preference_tau'])
        loss += args_train['loss_preference_scale'] * loss_pref
        loss_list['pref'] = loss_pref

    return loss, loss_list

