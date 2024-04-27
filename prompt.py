from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer


model = AutoModelForCausalLM.from_pretrained("/scratch/gpfs/jx0800/meditron-7b")
tokenizer = AutoTokenizer.from_pretrained("/scratch/gpfs/jx0800/meditron-7b")
#recognizer = pipeline("question-answering", model=model, tokenizer=tokenizer) #text-generation

#generated_text = recognizer(prompt, max_length=100, temperature=1, top_k=0, top_p=0)
#print(generated_text[0]["generated_text"])

prompt = "what is diabetes?"
inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

# Generate
generate_ids = model.generate(inputs.input_ids, max_length=200)
out = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(out)
