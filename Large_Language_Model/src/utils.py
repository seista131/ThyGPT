import gc
import torch

def clear_torch_cache():
    gc.collect()
    if torch.cuda.device_count() > 0:
        torch.cuda.empty_cache()

def reset_user_input():
    return gr.update(value='')

def reset_state():
    return []

def generate_prompt(instruction, my_input):
    return f"已知：{my_input}\n请问：{instruction}"
