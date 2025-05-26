import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import os

# Disable distributed training
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Restrict to GPU 0
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # Reduce fragmentation
torch.distributed.destroy_process_group() if torch.distributed.is_initialized() else None

# Setup
device = 'cuda:0'
teacher_model_name = "meta-llama/Llama-3.2-1B-Instruct"
student_model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(student_model_name)
tokenizer.pad_token = tokenizer.eos_token

torch.cuda.empty_cache()

# Load teacher model with 4-bit quantization
teacher_model = AutoModelForCausalLM.from_pretrained(
    teacher_model_name,
    torch_dtype=torch.float16,
    device_map={'': 0},
    output_hidden_states=True,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
)
teacher_model.eval()

# Load student model
student_model = AutoModelForCausalLM.from_pretrained(
    student_model_name,
    torch_dtype=torch.float16,
    device_map={'': 0},
    output_hidden_states=True
)

# Apply LoRA adapters
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
student_model = get_peft_model(student_model, lora_config)

# Verify trainable parameters
trainable_params = [name for name, param in student_model.named_parameters() if param.requires_grad]
print("Trainable parameters:", trainable_params)
assert len(trainable_params) > 0, "No trainable parameters found!"

# Load and preprocess dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
dataset = dataset.filter(lambda example: len(example["text"].strip()) > 10 and len(example["text"]) < 1000)

def tokenize_function(examples, tokenizer=tokenizer):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=128,
        padding="max_length",
        return_tensors=None
    )
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": tokenized["input_ids"]
    }

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names, num_proc=4)

def filter_valid_samples(example):
    non_pad_count = sum(1 for id in example["input_ids"] if id != 128009)
    return non_pad_count >= 10

tokenized_dataset = tokenized_dataset.filter(filter_valid_samples, num_proc=4)

# Define distillation loss
class DistillationLoss(nn.Module):
    def __init__(self, student_hidden_size=3072, teacher_hidden_size=2048):
        super().__init__()
        self.projection = nn.Linear(student_hidden_size, teacher_hidden_size, dtype=torch.float16).to(device)
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.mse_loss = nn.MSELoss()

    def forward(self, student_outputs, teacher_outputs, temperature=5.0, alpha=0.5, chunk_size=2048):
        student_logits = torch.clamp(student_outputs.logits / temperature, min=-100, max=100)
        teacher_logits = torch.clamp(teacher_outputs.logits / temperature, min=-100, max=100)

        kl_div = 0
        batch_size, seq_len, vocab_size = student_logits.shape
        for i in range(0, vocab_size, chunk_size):
            s_chunk = student_logits[:, :, i:i+chunk_size]
            t_chunk = teacher_logits[:, :, i:i+chunk_size]
            kl_div += self.kl_loss(
                F.log_softmax(s_chunk.float(), dim=-1),
                F.softmax(t_chunk.float(), dim=-1)
            )
        kl_div = kl_div * (temperature ** 2) / (vocab_size // chunk_size)

        hidden_loss = 0
        for s_hidden, t_hidden in zip(student_outputs.hidden_states, teacher_outputs.hidden_states):
            s_hidden_projected = self.projection(s_hidden.to(torch.float16))
            hidden_loss += self.mse_loss(s_hidden_projected, t_hidden.to(torch.float16))

        total_loss = alpha * kl_div + (1 - alpha) * hidden_loss
        return total_loss
from torch.amp import autocast, GradScaler

class DistillationTrainer(Trainer):
    def __init__(self, teacher_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model.eval()
        self.distillation_loss = DistillationLoss().to(device)
        self.use_amp = self.args.fp16 or self.args.bf16
        self.scaler = GradScaler(device='cuda', enabled=self.use_amp)

    def compute_loss(self, model, inputs, return_outputs=False):
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = inputs.pop("labels", None)
        with autocast(device_type='cuda', enabled=self.use_amp):
            student_outputs = model(**inputs)
            with torch.no_grad():
                teacher_outputs = self.teacher_model(**inputs)
            loss = self.distillation_loss(student_outputs, teacher_outputs)

        if not torch.isfinite(loss):
            print(f"Warning: Non-finite loss detected, inputs: {inputs['input_ids']}")
            return (torch.tensor(0.0, device=device, requires_grad=True), student_outputs) if return_outputs else torch.tensor(0.0, device=device, requires_grad=True)
        return (loss, student_outputs) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if loss is None or not torch.isfinite(loss):
            return None

        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        return loss.detach()

# Updated training arguments using AdamW
training_args = TrainingArguments(
    output_dir="./distillation_output",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,
    save_steps=500,
    logging_steps=100,
    fp16=True,
    gradient_checkpointing=True,
    optim="adamw_torch",  # ✅ Switched optimizer here
    report_to="none",
    label_names=["labels"],
    max_grad_norm=1.0,
    remove_unused_columns=False
)

# Launch training
trainer = DistillationTrainer(
    teacher_model=teacher_model,
    model=student_model,
    args=training_args,
    train_dataset=tokenized_dataset
)

trainer.train()

# Save distilled model
student_model.save_pretrained("./distilled_llama_3b")
tokenizer.save_pretrained("./distilled_llama_3b")
