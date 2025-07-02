from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Setup
tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoder2-3b")
model = AutoModelForCausalLM.from_pretrained("bigcode/starcoder2-3b")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Input
prompt = "def print_hello_world():"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Fix pad_token_id and attention_mask
inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

output = model.generate(
    **inputs,
    max_new_tokens=50,
    do_sample=False,             # Greedy decoding for determinism
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id
)

print(tokenizer.decode(output[0], skip_special_tokens=True))
