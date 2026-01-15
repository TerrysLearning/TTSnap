import math
from torch.optim.lr_scheduler import LambdaLR

def get_scheduler(optimizer, 
                total_steps, 
                warmup_ratio=0,  
                warmup_type="linear",
                scheduler_type="cosine", 
                min_lr_scale=0.0):
    """
    Custom scheduler with warmup + multiple decay options.

    Args:
        optimizer: torch optimizer
        total_steps: total training steps (epochs or iterations)
        warmup_steps: number of warmup steps
        scheduler_type: 'cosine' | 'linear' | 'exp'
        min_lr_scale: minimum learning rate scale factor (e.g. 0.0 means decay to zero)
        exp_gamma: for exp decay, gamma factor (e.g. 0.95)

    Returns:
        scheduler: torch LambdaLR
    """
    warmup_steps = int(total_steps * warmup_ratio)
    max_lr_scale = 1.0
    
    def lr_lambda(current_step):
        # Warmup phase
        if current_step < warmup_steps:
            # use constant warmup 
            if warmup_type == "linear":
                return float(current_step) / float(max(1, warmup_steps))
            elif warmup_type == "constant":
                return 1.0
            else:
                raise ValueError(f"Unknown warmup type: {warmup_type}")
        # Decay phase
        progress = (current_step - warmup_steps) / float(max(max_lr_scale, total_steps - warmup_steps))
        progress = min(progress, max_lr_scale)  # Clamp to 1
        scale_diff = max_lr_scale - min_lr_scale
        
        if scheduler_type == "cosine":
            decay = min_lr_scale + 0.5 * scale_diff * (1 + math.cos(math.pi * progress))
        elif scheduler_type == "linear":
            decay = max_lr_scale - scale_diff * progress
        elif scheduler_type == "exp":
            target_ratio = min_lr_scale / max_lr_scale
            decay_factor = target_ratio ** progress
            decay = max_lr_scale * decay_factor
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

        return decay

    return LambdaLR(optimizer, lr_lambda)
