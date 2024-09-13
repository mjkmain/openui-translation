from dataclasses import dataclass, field
from typing import Optional, Dict
from transformers import TrainingArguments

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    tokenizer_name_or_path: Optional[str] = field(default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    use_fast_tokenizer: bool = field(default=True)
    model_revision: str = field(default="main")
    torch_dtype: Optional[str] = field(default="bfloat16")
    encoder_path: Optional[str] = field(default=None)

@dataclass
class DataArguments:
    law_data_path: Optional[str] = field(default="/home/maverick/openui-trans/data/raw/raw_dataset")
    language: Optional[str] = field(default=None)

@dataclass
class MyTrainingArguments(TrainingArguments):
    trainable : Optional[str] = field(default="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj")
    lora_rank : Optional[int] = field(default=32)
    lora_dropout : Optional[float] = field(default=0.05)
    modules_to_save : Optional[str] = field(default="lm_head")
    peft_path : Optional[str] = field(default=None)
    use_flash_attention_2 : Optional[bool] = field(default=True)
    do_train : Optional[bool] = field(default=True)
    do_eval : Optional[bool] = field(default=False)
    bf16 : Optional[bool] = field(default=True)
    learning_rate : Optional[float] = field(default=5e-5)
    weight_decay : Optional[float] = field(default=0.01)
    num_train_epochs : Optional[int] = field(default=1)
    eval_strategy : Optional[str] = field(default="no")
    eval_steps : Optional[int] = field(default=1000)
    optim = "adamw_bnb_8bit"
    per_device_train_batch_size : Optional[int] = field(default=1)
    per_device_eval_batch_size : Optional[int] = field(default=1)
    gradient_accumulation_steps : Optional[int] = field(default=8)
    
    def __post_init__(self):
        self.lora_alpha = self.lora_rank * 2
        return super().__post_init__()
        