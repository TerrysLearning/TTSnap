import argparse
import yaml
import os

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", type=str, default="configs/a.yaml")
    args, _ = parser.parse_known_args()

    with open(args.c, "r") as f:
        config_dict = yaml.safe_load(f)
    
    args_data = config_dict['data']
    args_train = config_dict['train']
    args_log = config_dict['log']
    
    parser.add_argument("--plot_dir", type=str, default=args_log['plot_dir'])
    parser.add_argument("--save_checkpoint", action="store_true", default=args_log['save_checkpoint'])
    parser.add_argument("--checkpoint_dir", type=str, default=args_log['checkpoint_dir'])
    parser.add_argument("--eval_start", action="store_true", default=args_log['eval_start'])
    parser.add_argument("--do_valid", action='store_true', default=args_log['do_valid'])
    parser.add_argument("--name", type=str, default=args_log['name'])
    parser.add_argument("--step_targets", type=str, default=str(config_dict['time_config']['step_targets']))

    parser.add_argument("--base_model_path", type=str, default=args_train['base_model_path'])
    parser.add_argument("--lr", type=float, default=args_train['lr'])
    parser.add_argument("--batch_size", type=int, default=args_train['batch_size'])
    parser.add_argument("--accum_steps", type=int, default=args_train['accum_steps'])
    parser.add_argument("--max_grad_norm", type=float, default=args_train['max_grad_norm'])
    parser.add_argument("--warmup_ratio", type=float, default=args_train['warmup_ratio'])
    parser.add_argument("--bn_before_loss", action="store_true", default=args_train['bn_before_loss'])
    parser.add_argument("--loss_mse_scale", type=float, default=args_train['loss_mse_scale'])
    parser.add_argument("--loss_preference_scale", type=float, default=args_train['loss_preference_scale'])
    parser.add_argument("--loss_preference_tau", type=float, default=args_train['loss_preference_tau'])
    parser.add_argument("--scheduler_type", type=str, default=args_train['scheduler_type'])
    parser.add_argument("--train_prompt_range", type=int, default=args_data['train_prompt_range'])
    parser.add_argument("--curr_interval", type=int, default=args_train['curr_interval'])
    parser.add_argument("--weight_decay", type=float, default=args_train['weight_decay'])
    parser.add_argument("--min_lr_scale", type=float, default=args_train['min_lr_scale'])
    
    args = parser.parse_args()

    os.makedirs(args.plot_dir, exist_ok=True)
    args_log['plot_dir'] = args.plot_dir
    args_log['eval_start'] = args.eval_start
    args_data['train_prompt_range'] = args.train_prompt_range
    args_log['save_checkpoint'] = args.save_checkpoint
    args_log['checkpoint_dir'] = args.checkpoint_dir
    args_log['do_valid'] = args.do_valid
    args_log['name'] = args.name
    if args.save_checkpoint:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    config_dict['time_config']['step_targets'] = [int(i) for i in args.step_targets.strip('[]').split(',')]

    args_train['base_model_path'] = args.base_model_path
    args_train['lr'] = args.lr
    args_train['batch_size'] = args.batch_size
    args_train['accum_steps'] = args.accum_steps
    args_train['max_grad_norm'] = args.max_grad_norm
    args_train['warmup_ratio'] = args.warmup_ratio
    args_train['bn_before_loss'] = args.bn_before_loss
    args_train['loss_mse_scale'] = args.loss_mse_scale
    args_train['loss_preference_scale'] = args.loss_preference_scale
    args_train['loss_preference_tau'] = args.loss_preference_tau
    args_train['scheduler_type'] = args.scheduler_type
    args_train['curr_interval'] = args.curr_interval
    args_train['weight_decay'] = args.weight_decay
    args_train['min_lr_scale'] = args.min_lr_scale
    
    os.makedirs(os.path.join(args.plot_dir, args.name), exist_ok=True)
    with open(os.path.join(args.plot_dir, args.name, 'args.yaml'), 'w') as f:
        config_dict["train"] = args_train
        config_dict["data"] = args_data
        config_dict["log"] = args_log
        yaml.dump(config_dict, f)
        print(f"Config saved to {os.path.join(args.plot_dir, args.name, 'args.yaml')}")

    return config_dict, args_data, args_train, args_log