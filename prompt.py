from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer


model = AutoModelForCausalLM.from_pretrained("/scratch/gpfs/jx0800/meditron-7b")
tokenizer = AutoTokenizer.from_pretrained("/scratch/gpfs/jx0800/meditron-7b")
recognizer = pipeline("text-generation", model=model, tokenizer=tokenizer)

prompt = """what is diabete"""

generated_text = generator(prompt, max_length=100, temperature=1, top_k=0, top_p=0)
print(generated_text[0]["generated_text"])
