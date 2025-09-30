import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText, BitsAndBytesConfig

from trl import SFTConfig

from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from tqdm.auto import tqdm

from utils.dataset import load_dataset

import re
import json


dataset_id = "philschmid/gretel-synthetic-text-to-sql" 
dataset = load_dataset(dataset_id)

# Hugging Face model id
model_id = "google/gemma-3-27b-pt" # or `google/gemma-3-4b-pt`, `google/gemma-3-12b-pt`, `google/gemma-3-27b-pt`

# Select model class based on id
if model_id == "google/gemma-3-1b-pt": # 1B is text only
    model_class = AutoModelForCausalLM
else:
    model_class = AutoModelForImageTextToText

# Check if GPU benefits from bfloat16
if torch.cuda.get_device_capability()[0] >= 8:
    torch_dtype = torch.bfloat16
else:
    torch_dtype = torch.float16

print(torch_dtype)
# Define model init arguments
model_kwargs = dict(
    #attn_implementation="eager", # Use "flash_attention_2" when running on Ampere or newer GPU
    attn_implementation="flash_attention_2", # Use "flash_attention_2" when running on Ampere or newer GPU
    torch_dtype=torch_dtype, # What torch dtype to use, defaults to auto
    device_map="auto", # Let torch decide how to load the model
)

# # BitsAndBytesConfig: Enables 4-bit quantization to reduce model size/memory usage
# model_kwargs["quantization_config"] = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type='nf4',
#     bnb_4bit_compute_dtype=model_kwargs['torch_dtype'],
#     bnb_4bit_quant_storage=model_kwargs['torch_dtype'],
# )

#tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it") # Load the Instruction Tokenizer to use the official Gemma template
# Load tokenizer
print(f"loading tokenizer from ./model/{model_id} ...")
tokenizer = AutoTokenizer.from_pretrained(f"./model/{model_id}") # Load the Instruction Tokenizer to use the official Gemma template

# Convert as test example into a prompt with the Gemma template
stop_token_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<end_of_turn>")]

#print(f"applying chat template to {dataset.}...")
dataset = dataset.map(lambda x: {"formatted_chat": tokenizer.apply_chat_template(x["messages"][:2], tokenize=False, add_generation_prompt=True)})

# print(dataset["train"][0]["formatted_chat"])

# load the model
print(f"loading model from ./model/{model_id} ...")
model = model_class.from_pretrained(f"./model/{model_id}", **model_kwargs)

# load the model and tokenizer into the pipeline
pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Generate our SQL query.
print(f"processing {dataset['test'].num_rows} records")
results = pipeline(KeyDataset(dataset['test'], "formatted_chat"), 
                    max_new_tokens=256, 
                    do_sample=True, 
                    temperature=0.1, 
                    top_k=50, 
                    top_p=0.1, 
                    batch_size=32,
                    use_cache=True,
                    eos_token_id=stop_token_ids, 
                    disable_compile=True)

with open('output.jsonl', 'w') as outfile:
    #for idx, output in enumerate(dataset):
    idx = 0
    for output in tqdm(results):
      test_sample = dataset['test'][idx]
      idx = idx + 1
      prompt = tokenizer.apply_chat_template(test_sample["messages"][:2], tokenize=False, add_generation_prompt=True)

      # Extract the user query and original answer
      answer = {
        "sql_context": re.search(r'<SCHEMA>\n(.*?)\n</SCHEMA>', test_sample['messages'][0]['content'], re.DOTALL).group(1).strip(),
        "user_query": re.search(r'<USER_QUERY>\n(.*?)\n</USER_QUERY>', test_sample['messages'][0]['content'], re.DOTALL).group(1).strip(),
        "ground_truth": test_sample['messages'][1]['content'],
        "generated_answer": output[0]['generated_text'][len(prompt):].strip(),
      }

      # write to file
      outfile.write(f"{json.dumps(answer)}\n")
      
# print(outputs)
  
  # break
