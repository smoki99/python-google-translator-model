#########################################
# 
# This translates an input files and put it out on screen
#
#########################################
#

# Define the file path for the input text file
input_file_path = "input/example1.txt"


import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Set the device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Print whether CPU or GPU is being used
if device.type == 'cuda':
    print("Using GPU (CUDA)")
else:
    print("Using CPU")

# Load the tokenizer with legacy=False to suppress the warning
tokenizer = T5Tokenizer.from_pretrained("t5-base", legacy=False)
model = T5ForConditionalGeneration.from_pretrained("t5-base").to(device)

# Read input sentences from the file
with open(input_file_path, "r", encoding="utf-8") as file:
    sentences = file.readlines()

# Loop over each sentence, translate it, and print the result
for sentence in sentences:
    # Remove any leading/trailing whitespace from the sentence
    sentence = sentence.strip()
    if not sentence:
        continue  # Skip empty lines

    # Encode the input sentence and move the tensor to the GPU if available
    input_enc = tokenizer.encode("translate English to German: " + sentence, return_tensors="pt").to(device)

    # Generate the output sequence
    output_enc = model.generate(input_enc, max_new_tokens=4000)

    # Decode the output sequence
    decoded = tokenizer.decode(output_enc[0], skip_special_tokens=True)

    # Print the decoded output
    print(f"Original: {sentence}")
    print(f"Translated: {decoded}")
    print()
