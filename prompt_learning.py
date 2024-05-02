from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset
#from random import randrange
import pandas as pd
#import time
#import spacy
#import random
#import re

# Ensure CUDA (GPU support) is available and specify the device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained("/scratch/gpfs/jx0800/meditron-7b")
tokenizer = AutoTokenizer.from_pretrained("/scratch/gpfs/jx0800/meditron-7b")

##################################################################################
#                             Zero-shot Structured                               #
##################################################################################

# ChatGPT Zero-shot API function
def zero_shot_structured(note, prompt):

    input_ids = tokenizer(f"{prompt}\n {note}\n ### Output:", return_tensors="pt", truncation=True).input_ids.cuda()
    outputs = model.generate(input_ids=input_ids, max_new_tokens=50, do_sample=True, top_p=0.9, temperature=0.9)
    out = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]
    
    return out

# Read in test data
input_file_test = '/home/jx0800/meditron/data/test_dat_category_all.xlsx'
complete_df_test = pd.read_excel(input_file_test, keep_default_na = False, na_values = '')

# Zero-shot learning
for idx, row in list(complete_df_test.iterrows()):
    note = row['note']
    category = row['category']
    if category == 'RAREDISEASE':
        prompt = '''### Task: 
                    Extract the exact name or names of rare diseases from the input text and output them in a list.
                    ### Definition:
                    Rare diseases are defined as diseases that affect a small number of people compared to the general population.
                    ### Input Text: 
                    '''
    elif category == 'DISEASE':
        prompt = '''### Task: 
                    Extract the exact name or names of diseases from the input text and output them in a list.
                    ### Definition:
                    Diseases are defined as abnormal conditions resulting from various causes, such as infection, inflammation, environmental factors, or genetic defect, and characterized by an identifiable group of signs, symptoms, or both.
                    ### Input Text: 
                    '''
    elif category == 'SYMPTOM':
        prompt = '''### Task: 
                    Extract the exact name or names of symptoms from the input text and output them in a list.
                    ### Definition:
                    Symptoms are defined as physical or mental problems that cannot be measured from tests or observed by a doctor.
                    ### Input Text: 
                    '''
    elif category == 'SIGN':
        prompt = '''### Task: 
                    Extract the exact name or names of signs from the input text and output them in a list.
                    ### Definition:
                    Signs are defined as physical or mental problems that can be measured from tests or observed by a doctor.
                    ### Input Text: 
                    '''
    complete_df_test.at[idx, 'prompt'] = ("").join(prompt)

    out = zero_shot_structured(note, prompt)
    
    print(idx)
    print(f"Prompt:\n{prompt} \n {note} \n ### Output: \n")
    print(f"Generated instruction:\n{out}")
    print(f"Ground truth:\n{row['gold_All']}")

    complete_df_test.at[idx, 'output'] = ("").join(out)
    
complete_df_test.to_excel('/home/jx0800/meditron/data/zero_shot_structured_output.xlsx', sheet_name = 'Sheet1')

