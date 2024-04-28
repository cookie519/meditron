from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch

# Ensure CUDA (GPU support) is available and specify the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained("/scratch/gpfs/jx0800/meditron-7b")
tokenizer = AutoTokenizer.from_pretrained("/scratch/gpfs/jx0800/meditron-7b")

model.to(device)

#recognizer = pipeline("question-answering", model=model, tokenizer=tokenizer) #text-generation

#generated_text = recognizer(prompt, max_length=100, temperature=1, top_k=0, top_p=0)
#print(generated_text[0]["generated_text"])

prompt = """
###Task:
Extract the exact names of rare diseases from the input text and output them in a list.

### Definition:
Rare diseases are defined as diseases that affect a small number of individuals.

### Input text: 
"Ahumada-Del Castillo is a rare endocrine disorder affecting adult females, which is characterized by impairment in the function of the pituitary and hypothalamus glands. Symptoms may include the production of breast milk (lactation) not associated with nursing and the absence of menstrual periods (amenorrhea) due to the lack of monthly ovulation (anovulation). The symptoms of Ahumada-Del Castillo syndrome include the abnormal production of breast milk (galactorrhea) without childbirth and nursing, and the lack of regular menstrual periods (amenorrhea).  Women with this disorder have breasts and nipples of normal size and appearance.  Secondary female sexual characteristics, such as hair distribution and voice, are also normal.  Since the ovaries do not produce eggs, affected females cannot become pregnant. The exact cause of Ahumada-Del Castillo syndrome is not known, although some research suggests that small tumors in the pituitary or hypothalamus glands may be responsible for some cases.  These tumors are frequently microscopic and extremely difficult to detect.  Rarer causes of Ahumada-Del Castillo syndrome may be associated with low levels of thyroid hormone (hypothyroidism), chronic use of drugs that inhibit dopamine (antagonistics) (e.g., chlorpromazine or thorazine), and discontinuation of oral contraceptives (birth control pills).  In all cases, an over-secretion of the milk-producing hormone prolactin (hyperprolactinemia) results in the symptoms of Ahumada-Del Castillo. Ahumada-Del Castillo affects only females. The symptoms usually begin during adulthood."

### Output: 
"""
inputs = tokenizer(prompt, return_tensors="pt")
print(inputs)
inputs = {k: v.to(device) for k, v in inputs.items()}
print(inputs)
# Generate
generate_ids = model.generate(inputs["input_ids"], max_new_tokens=50) #max_length=200 inputs.input_ids
out = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(out)
