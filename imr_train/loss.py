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

def repa_loss_fn(pred_feature, gt_feature):
    loss = -F.cosine_similarity(pred_feature, gt_feature, dim=1)
    # loss = F.mse_loss(pred_feature, gt_feature, reduction='none').mean(dim=1)
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

    if args_train['shift_mean']:
        reward_gt -= kwargs['step_mean'][args_train['base_model_time']].to(reward_gt.device)
        reward_pred -= kwargs['step_mean'][batch_data['t_id']].unsqueeze(1).to(reward_pred.device)
    
    if args_train['scale_std']:
        reward_gt /= (kwargs['step_std'][args_train['base_model_time']].to(reward_gt.device) + 1e-10)
        reward_pred /= (kwargs['step_std'][batch_data['t_id']].unsqueeze(1).to(reward_pred.device) + 1e-10)

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
        if args_train['loss_conf_scale'] > 0:
            conf_target = loss_mse.detach()
            loss_conf = conf_loss_fn(batch_pred['uncertainty_pred'], conf_target) #////
            loss += args_train['loss_conf_scale'] * loss_conf
            loss_list['conf'] = loss_conf

    if args_train['loss_mse_uncertainty_scale'] > 0:
        loss_mse_uncertainty = mse_uncertainty_loss_fn(reward_pred, reward_gt, batch_pred['uncertainty_pred']) #////
        loss += args_train['loss_mse_uncertainty_scale'] * loss_mse_uncertainty
        loss_list['mse_uncertainty'] = loss_mse_uncertainty

    if args_train['loss_preference_scale'] > 0:
        loss_pref = preference_loss_fn(reward_pred, tau=args_train['loss_preference_tau'])
        loss += args_train['loss_preference_scale'] * loss_pref
        loss_list['pref'] = loss_pref

    if args_train['loss_repa_scale'] > 0:
        loss_repa_v = repa_loss_fn(batch_pred['visual_feature'], batch_data['visual_feature'])
        loss_repa_t = repa_loss_fn(batch_pred['text_feature'], batch_data['text_feature'])
        loss_repa = torch.mean(torch.cat([loss_repa_v, loss_repa_t]))
        loss += args_train['loss_repa_scale'] * loss_repa
        loss_list['repa'] = loss_repa

    return loss, loss_list


# def mse_uncertainty_loss_fn_beta(batch_pred, batch_data, beta=0.5):
#     mse = F.mse_loss(batch_pred['img_score_pred'], batch_data['img_score_gt'], reduction='none')
#     s = batch_pred['img_uncertain_pred']
#     loss = 0.5*torch.exp(-s)*mse + 0.5*s
#     if beta > 0:
#         loss = beta*torch.exp(s.detach()) * loss
#     loss = torch.mean(loss)
#     return loss


# def mse_weighted_loss_fn(batch_pred, batch_data):
#     mse = F.mse_loss(batch_pred['img_score_pred'], batch_data['img_score_gt'], reduction='none')
#     w = torch.square(1-batch_data['img_time_step'].float()/1000).to(mse.device)
#     loss = torch.mean(w*mse)
#     return loss



# def repa_cosine_loss_fn(batch_pred, batch_data):
#     # cosine similarity between two [b*N, D] features, D should be 768 for Bert 
#     cosin_sim = F.cosine_similarity(batch_pred['feature'], batch_data['feature'], dim=1) 
#     loss = -torch.mean(cosin_sim) # shape (B,)
#     return loss

# def repa_mse_loss_fn(batch_pred, batch_data):
#     loss = F.mse_loss(batch_pred['feature'], batch_data['feature'], reduction='mean')
#     return loss



# This it is the same as the preference loss function above
# def loss_preference2_fn(reward_pred):
#     def sigmoid(x):
#         return 1/(1+torch.exp(-x))
#     reward_pred_scaled = reward_pred / 1
#     loss_list = -torch.log(sigmoid(reward_pred_scaled[:,0]-reward_pred_scaled[:,1]))
#     loss = torch.mean(loss_list)
#     return loss

    
# def calculate_acc(reward):
#     reward_diff = reward[:, 0] - reward[:, 1]
#     acc = torch.mean((reward_diff > 0).clone().detach().float())
#     return acc