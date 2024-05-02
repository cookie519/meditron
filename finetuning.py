from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from datasets import load_dataset
#from random import randrange
import pandas as pd


dataset = load_dataset("/scratch/gpfs/jx0800/databricks-dolly-15k", split="train")
print(dataset)

def format_instruction(sample):
	return f"""### Instruction:
Use the Input below to create an instruction, which could have been used to generate the input using an LLM.
 
### Input:
{sample['response']}
 
### Response:
{sample['instruction']}
"""
  
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)
 
# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "/scratch/gpfs/jx0800/meditron-7b",
    #quantization_config=bnb_config,
    torch_dtype=torch.float16,
    use_cache=False,
    use_flash_attention_2=True,
    device_map="auto",
)
model.config.pretraining_tp = 1
model.train()
print("model loaded")
 
tokenizer = AutoTokenizer.from_pretrained("/scratch/gpfs/jx0800/meditron-7b")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
print("tokenizer loaded")

from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
 
# LoRA config based on QLoRA paper
peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
)

# prepare model for training
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)
print("model prepared")

from transformers import TrainingArguments
 
args = TrainingArguments(
    output_dir="/scratch/gpfs/jx0800/finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=6,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=2e-4,
    bf16=True,
    tf32=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    disable_tqdm=True, # disable tqdm since with packing values are in correct
    report_to = "none" #"tensorboard"
)


from trl import SFTTrainer
 
max_seq_length = 2048 # max sequence length for model and packing of the dataset
 
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    packing=True,
    formatting_func=format_instruction,
    args=args,
)
print("trainer set up")

# train
print("training start")
trainer.train() # there will not be a progress bar since tqdm is disabled
 
# save model
trainer.save_model()
print("model saved")

