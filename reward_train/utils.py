import os
import torch
import yaml 
import torch.distributed as dist
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from transformers import BertTokenizer
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC
import numpy as np


def print_dict(dict_input):
    for key, value in dict_input.items():
        print(key, " ", value)

FLUX_SIGMAS_30 = np.array([0.9820, 0.9634, 0.9441, 0.9243, 
0.9037, 0.8825, 0.8605, 0.8378, 0.8142,
0.7897, 0.7643, 0.7380, 0.7106, 0.6821, 
0.6525, 0.6216, 0.5895, 0.5559, 0.5209, 
0.4842, 0.4459, 0.4057, 0.3636, 0.3195, 
0.2730, 0.2241, 0.1726, 0.1183, 0.0608, 0.0000])


def get_trainable_modules(model):
       trainable = set()
       for name, param in model.named_parameters():
           if param.requires_grad:
               module_name = ".".join(name.split("."))  # 去掉参数名（weight/bias）
               trainable.add(module_name)
       return sorted(trainable)

def write_down_trainable_modules(model):
    trainable_modules = get_trainable_modules(model)
    with open('trainable_modules.txt', 'w') as f:
        for module in trainable_modules:
            f.write(module + '\n')



def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def init_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_special_tokens({'bos_token':'[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]']})       
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]  
    return tokenizer

def save_model(model, name):
    save_path = os.path.join('checkpoints', name)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


def load_model(model, name):
    load_path = os.path.join('checkpoints', name)
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Checkpoint {load_path} does not exist.")
    print(f"Loading model from {load_path}")
    state_dict = torch.load(load_path, map_location='cpu')
    msg = model.load_state_dict(state_dict, strict=False)
    print("Missing keys:", msg.missing_keys)
    return model



# Distributed training, not sure if it necessary
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


_MODELS = {
    "ImageReward-v1.0": "https://huggingface.co/THUDM/ImageReward/blob/main/ImageReward.pt",
}


preprocess_image = _transform(224)


# # Image reward download
# def ImageReward_download(url: str, root: str):
#     os.makedirs(root, exist_ok=True)
#     filename = os.path.basename(url)
#     download_target = os.path.join(root, filename)
#     hf_hub_download(repo_id="THUDM/ImageReward", filename=filename, local_dir=root)
#     return download_target


# def load(name: str = "ImageReward-v1.0", device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", download_root: str = None, med_config: str = None):
#     """Load a ImageReward model

#     Parameters
#     ----------
#     name : str
#         A model name listed by `ImageReward.available_models()`, or the path to a model checkpoint containing the state_dict

#     device : Union[str, torch.device]
#         The device to put the loaded model

#     download_root: str
#         path to download the model files; by default, it uses "~/.cache/ImageReward"

#     Returns
#     -------
#     model : torch.nn.Module
#         The ImageReward model
#     """
#     if name in _MODELS:
#         model_path = ImageReward_download(_MODELS[name], download_root or os.path.expanduser("~/.cache/ImageReward"))
#     elif os.path.isfile(name):
#         model_path = name
#     else:
#         raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

#     print('load checkpoint from %s'%model_path)
#     state_dict = torch.load(model_path, map_location='cpu')
    
#     # med_config
#     if med_config is None:
#         med_config = ImageReward_download("https://huggingface.co/THUDM/ImageReward/blob/main/med_config.json", download_root or os.path.expanduser("~/.cache/ImageReward"))
    
#     model = ImageReward(device=device, med_config=med_config).to(device)
#     msg = model.load_state_dict(state_dict,strict=False)
#     print("checkpoint loaded")
#     model.eval()

#     return model






    # print('load checkpoint from %s'%model_name)
    # checkpoint = torch.load(model_name, map_location='cpu') 
    # state_dict = checkpoint
    # msg = model.load_state_dict(state_dict,strict=False)
    # print("missing keys:", msg.missing_keys)

    # return model 
