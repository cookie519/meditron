from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer


model = AutoModelForCausalLM.from_pretrained("/scratch/gpfs/jx0800/meditron-7b")
tokenizer = AutoTokenizer.from_pretrained("/scratch/gpfs/jx0800/meditron-7b")
recognizer = pipeline("question-answering", model=model, tokenizer=tokenizer) #text-generation

prompt = """what is diabete"""

generated_text = recognizer(prompt, max_length=100, temperature=1, top_k=0, top_p=0)
print(generated_text[0]["generated_text"])
