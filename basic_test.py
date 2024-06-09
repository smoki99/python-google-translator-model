# pip install transformers
# pip install sentencepiece
from transformers import T5Tokenizer, T5Model, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")

input = "Mein Name ist Christian Müller und ich lebe in Erding bei München."

input_enc = tokenizer.encode("translate German to English: "+input, return_tensors="pt")

output_enc = model.generate(input_enc, max_new_tokens=1000)

decoded = tokenizer.decode(output_enc[0], skip_special_tokens=True)

print(decoded)