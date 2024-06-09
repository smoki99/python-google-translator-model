# Install the required libraries
# pip install transformers
# pip install sentencepiece

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Set the device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Print whether CPU or GPU is being used
if device.type == 'cuda':
    print("Using GPU (CUDA)")
else:
    print("Using CPU")

# Load the tokenizer and model, and move the model to the GPU if available
tokenizer = T5Tokenizer.from_pretrained("t5-base", legacy=False)
model = T5ForConditionalGeneration.from_pretrained("t5-base").to(device)

# Define the input text
input_text = "Mein Name ist Christian Müller und ich lebe in Erding bei München."

# Encode the input text and move the tensor to the GPU if available
input_enc = tokenizer.encode("translate German to English: " + input_text, return_tensors="pt").to(device)

# Generate the output sequence
output_enc = model.generate(input_enc, max_new_tokens=1000)

# Decode the output sequence
decoded = tokenizer.decode(output_enc[0], skip_special_tokens=True)

# Print the decoded output
print(decoded)
