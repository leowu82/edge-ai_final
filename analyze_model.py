# Analyze model architecture
import torch
from transformers import AutoModelForCausalLM

device = 'cuda:0'
    
model_name = "meta-llama/Llama-3.2-3B-Instruct"   
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map=device,
)

print(model)