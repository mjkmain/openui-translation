import os

import bitsandbytes as bnb
import torch
import datasets
from datasets import concatenate_datasets, load_from_disk
from torch import nn
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_pt_utils import get_parameter_names
from transformers import AutoModelForCausalLM
from peft import LoraConfig, PeftModel, TaskType, get_peft_model

from translation.arguments import DataArguments, ModelArguments, MyTrainingArguments
from translation.data_utils import build_dataset, DataCollatorForSFT
hf_read_token = os.environ.get("HF_TOKEN")

parser = HfArgumentParser((ModelArguments, DataArguments, MyTrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()

set_seed(training_args.seed)

PAD_TOKEN_ID = 128011
tokenizer = AutoTokenizer.from_pretrained(
    model_args.tokenizer_name_or_path,
    token=hf_read_token,
)
tokenizer.pad_token_id = PAD_TOKEN_ID

data_collator = DataCollatorForSFT(tokenizer)

raw_ds = datasets.load_dataset(
    data_args.law_data_path,
    token=hf_read_token,
)
train_dataset = build_dataset(
    raw_ds,
    tokenizer,
    data_args.language,
    dataset_dir=data_args.dataset_dir,
    split='train'
)


torch_dtype = getattr(torch, model_args.torch_dtype)
compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

device_map = {"":int(os.environ.get("LOCAL_RANK") or 0)}

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_args.model_name_or_path,
    torch_dtype=torch_dtype,
    device_map=device_map,
    pad_token_id=PAD_TOKEN_ID,
    attn_implementation="flash_attention_2",
    token=hf_read_token,
)

if training_args.peft_path is not None:
    model = PeftModel.from_pretrained(model, training_args.peft_path, device_map=device_map)
else:
    target_modules = training_args.trainable.split(",")
    modules_to_save = training_args.modules_to_save.split(",")
    
    lora_rank = training_args.lora_rank
    lora_alpha = training_args.lora_alpha
    lora_dropout = training_args.lora_dropout
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
        inference_mode=False,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        modules_to_save=modules_to_save
    )
    model = get_peft_model(model, peft_config)

for name, param in model.named_parameters():
    if param.requires_grad:
        print("Trainable : ", name)

training_args.gradient_checkpointing_kwargs={'use_reentrant':False}
training_args.torch_compile=True

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

if __name__ == '__main__':
    trainer.train()
    trainer.save_model()
    trainer.save_state()