import importlib
import transformers
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from random import randrange

print("Reloading llama model, unpatching flash attention")
importlib.reload(transformers.models.llama.modeling_llama)
 
# load base LLM model and tokenizer
model_dir = "/scratch/gpfs/jx0800/meditron-7b"
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    load_in_4bit=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_dir, max_length=2048)
if tokenizer.pad_token is None:
    print("tokenizer.pad_token = tokenizer.eos_token")
    tokenizer.pad_token = tokenizer.eos_token


from datasets import load_dataset
from random import randrange
 
# Load dataset from the hub and get a sample
dataset = load_dataset("csv", data_files="/scratch/gpfs/jx0800/data/test_out.csv")
dataset = dataset['train']
dataset = dataset.shuffle(seed=42)
print(dataset)

def get_prompt(sample):
    if sample['category'] == 'RAREDISEASE':
        prompt = '''### Task: 
Extract the exact name or names of rare diseases from the input text and output them in a list.
### Definition:
Rare diseases are defined as diseases that affect a small number of people compared to the general population.
### Input Text: '''
    elif sample['category'] == 'DISEASE':
        prompt = '''### Task: 
Extract the exact name or names of diseases from the input text and output them in a list.
### Definition:
Diseases are defined as abnormal conditions resulting from various causes, such as infection, inflammation, environmental factors, or genetic defect, and characterized by an identifiable group of signs, symptoms, or both.
### Input Text: '''
    elif sample['category'] == 'SYMPTOM':
        prompt = '''### Task: 
Extract the exact name or names of symptoms from the input text and output them in a list.
### Definition:
Symptoms are defined as physical or mental problems that cannot be measured from tests or observed by a doctor.
### Input Text: '''
    elif sample['category'] == 'SIGN':
        prompt = '''### Task: 
Extract the exact name or names of signs from the input text and output them in a list.
### Definition:
Signs are defined as physical or mental problems that can be measured from tests or observed by a doctor.
### Input Text: '''
    return f"""{prompt}
{sample['context']}
### Output:"""


sample = dataset[randrange(len(dataset))]

prompt = get_prompt(sample)
input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
outputs = model.generate(input_ids=input_ids, pad_token_id=tokenizer.pad_token_id, max_new_tokens=200, do_sample=True, top_p=0.9, temperature=0.9)
generated = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]

print(f"Prompt:\n{prompt}\n")
print(f"Generated:\n{generated}")
print(f"Ground truth:\n{sample['response']}")
