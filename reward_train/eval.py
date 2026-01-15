import numpy as np 
import torch
from tqdm import tqdm
from loss import mse_loss_fn
from dataset import load_image_batch
import wandb
import matplotlib.pyplot as plt
import json
from scipy.stats import kendalltau

# ------ Evaluation Functions ------
def top_select_eval(pred, gt, alpha):
    # pred and gt are arrays of two scores 
    assert len(pred) == len(gt), "pred and gt must have the same length"
    n = len(pred)
    k = int(alpha*n)
    k_th_pred = np.sort(pred)[-k]
    topk_idx_pred = np.where(pred >= k_th_pred)[0]
    gap = np.max(gt) - np.max(gt[topk_idx_pred])
    if gap < 1e-10:
        return 1.0, gap
    else:
        return 0.0, gap
    

def top_select_eval_M(pred, gt):
    alpha_list = np.arange(0.05, 1.0, 0.05)
    top_accs = []
    top_gaps = []
    for alpha in alpha_list:
        a, b = top_select_eval(pred, gt, alpha)
        top_accs.append(a)
        top_gaps.append(b)
    return np.mean(top_accs), np.mean(top_gaps)


def evaluate(model, dataloader, t_target, args_train, message="Evaluating", **kwargs):
    model.eval()

    eval_times = []
    eval_prompts = []
    eval_preds = []
    eval_gts = []
    eval_stds = []
    with torch.no_grad():
        for t_id in t_target.tolist():
            for batch_data in tqdm(dataloader, desc=message+' time_id = '+str(t_id)):
                t_id_batch = torch.tensor([t_id]*len(batch_data['img_id']))
                batch_data['image'] = load_image_batch(dataloader.dataset, batch_data['img_id'], batch_data['p_id'], t_id_batch).to(model.device)
                batch_data['t_id'] = t_id_batch
                batch_pred = model(batch_data)
                reward_pred = batch_pred['reward_pred']
                reward_gt = batch_data['reward_gt']
                eval_prompts.extend(batch_data['p_id'].tolist())
                eval_times.extend(t_id_batch.tolist())
                eval_preds.extend(reward_pred.tolist())
                eval_gts.extend(reward_gt.tolist())
    return eval_times, eval_prompts, eval_preds, eval_gts


def stat(eval_times, eval_prompts, eval_preds, eval_gts):

    n = len(eval_times)
    out_pred = {}
    out_gt = {}
    for i in range(n):
        t_id = eval_times[i]
        p_id = eval_prompts[i]
        pred = eval_preds[i]
        gt = eval_gts[i]
        if t_id not in out_pred.keys():
            out_pred[t_id] = {}
        if p_id not in out_pred[t_id].keys():
            out_pred[t_id][p_id] = []
        out_pred[t_id][p_id].append(pred)
        if t_id not in out_gt.keys():
            out_gt[t_id] = {}
        if p_id not in out_gt[t_id].keys():
            out_gt[t_id][p_id] = []
        out_gt[t_id][p_id].append(gt)

    output_dict = {}
    outputs_kendall = {}
    outputs_mse = {}
    outputs_top_select_acc = {}
    outputs_top_select_gap = {}
    outputs_top_select_acc_given = {}
    outputs_top_select_gap_given = {}
    

    for t_id in out_pred.keys():
        kendall_per_p = []
        mse_per_p = []
        top_select_acc_per_p = []
        top_select_gap_per_p = []
        top_select_acc_given_per_p = []
        top_select_gap_given_per_p = []
        for p_id in out_pred[t_id].keys():
            pred_array = np.array(out_pred[t_id][p_id])
            gt_array = np.array(out_gt[t_id][p_id])
            kendall_per_p.append(kendalltau(pred_array, gt_array)[0])
            mse_per_p.append(np.mean((pred_array - gt_array)**2))
            top_select_acc, top_select_gap = top_select_eval_M(pred_array, gt_array)
            top_select_acc_per_p.append(top_select_acc)
            top_select_gap_per_p.append(top_select_gap)
            top_select_acc_given, top_select_gap_given = top_select_eval(pred_array, gt_array, 0.25)
            top_select_acc_given_per_p.append(top_select_acc_given)
            top_select_gap_given_per_p.append(top_select_gap_given)

        outputs_kendall[t_id] = np.array(kendall_per_p).mean()
        outputs_mse[t_id] = np.array(mse_per_p).mean()
        outputs_top_select_acc[t_id] = np.array(top_select_acc_per_p).mean()
        outputs_top_select_gap[t_id] = np.array(top_select_gap_per_p).mean()
        outputs_top_select_acc_given[t_id] = np.array(top_select_acc_given_per_p).mean()
        outputs_top_select_gap_given[t_id] = np.array(top_select_gap_given_per_p).mean()

        print(f"time_id = {t_id:2d} |" +
            f"Tacc: {outputs_top_select_acc[t_id]:.4f} Tgap: {outputs_top_select_gap[t_id]:.6f} | " +
            f"Ken: {outputs_kendall[t_id]:.4f} | " +
            # f"| MSE: {outputs_mse[t_id]:.6f} |" +
            f"Tacc_25: {outputs_top_select_acc_given[t_id]:.4f} | Tgap_25: {outputs_top_select_gap_given[t_id]:.6f}")

    output_dict = {
        'kendall': outputs_kendall,
        'mse': outputs_mse,
        'Tacc': outputs_top_select_acc,
        'Tgap': outputs_top_select_gap,
    }
    return output_dict


def save_eval(history, outputs, epoch, eval_name, output_dir):
    # name is either valid or valid2
    if eval_name not in history:
        history[eval_name] = {}
    history[eval_name][epoch] = outputs.copy()

    # plot_eval(history, eval_name, output_dir)
    save_eval_json(history, output_dir)
    save_eval_txt(history, output_dir)
    return history


def plot_eval(history, name, output_dir):
    epoch_list = list(history[name].keys())  # x-axis is epoch,
    values_epoch_t = {} # y-axis is the metric value for each time_id
    for epoch, outputs_epoch in history[name].items():
        for metric_name, time_dict in outputs_epoch.items():
            if metric_name not in values_epoch_t.keys():
                values_epoch_t[metric_name] = {}
            for time_id, value in time_dict.items():
                if time_id not in values_epoch_t[metric_name].keys():
                    values_epoch_t[metric_name][time_id] = []
                values_epoch_t[metric_name][time_id].append(value)

    makers_list = ['.', 'x', 'v', '+', 's', '^','o', 'd', ',', '1', '*', '', '>', '<']
    for metric_name, time_values in values_epoch_t.items():
        # plot one chart for each metric_name
        i = 0
        for time_id, values in time_values.items():
            print(epoch_list, values)
            plt.plot(epoch_list, values, label=f'time={time_id}', marker=makers_list[i])
            i += 1
        plt.legend()
        plt.title(name+' - '+metric_name)
        plt.xlabel('epoch')
        if metric_name == 'mse':
            plt.yscale('log')
            plt.ylabel('log(mse)')
        else:   
            plt.ylabel(metric_name)
        plt.savefig(f'{output_dir}/{name}_{metric_name}.png')
        plt.close()
            

def save_eval_json(history, output_dir):
    for name, his in history.items():
        for epoch, outputs in his.items():
            for metric_name, time_dict in outputs.items():
                for time_id, value_ in time_dict.items():
                    history[name][epoch][metric_name][time_id] = round(value_, 6)

        with open(f'{output_dir}/outputs_{name}.json', 'w') as f:
            json.dump(history[name], f, indent=4)


def save_eval_txt(history, output_dir):
    for name, his in history.items():
        with open(f'{output_dir}/outputs_{name}.txt', 'w') as f:
            f.write(name+'\n')
        for epoch, outputs in his.items():
            lines = {} 
            for metric_name, time_dict in outputs.items():
                for time_id, value in time_dict.items():
                    if time_id not in lines.keys():
                        lines[time_id] = {}
                    lines[time_id][metric_name] = round(value, 6)
            for time_id, metric_dict in lines.items():
                line = f'epoch={epoch+1} time_id={time_id}: '
                for metric_name, value in metric_dict.items():
                    line += f'{metric_name}={value:.6f} '
                line += '\n'
            with open(f'{output_dir}/outputs_{name}.txt', 'a') as f:
                f.write(line)