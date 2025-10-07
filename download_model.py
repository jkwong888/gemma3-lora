import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText

import os
import sys
import asyncio
from utils.model import download_from_hf, upload_to_gcs

GCS_BUCKET_NAME = "jkwng-model-data"  
GCS_DESTINATION_PATH = "models" # The folder path inside your GCS bucket

HF_TOKEN = os.getenv("HF_TOKEN")

# Hugging Face model id
#model_id = "google/gemma-3-27b-pt" # or `google/gemma-3-4b-pt`, `google/gemma-3-12b-pt`, `google/gemma-3-27b-pt`
#model_id = "unsloth/gemma-3-12b-it-unsloth-bnb-4bit" # or `google/gemma-3-4b-pt`, `google/gemma-3-12b-pt`, `google/gemma-3-27b-pt`
model_id = os.getenv("MODEL_ID") or "unsloth/gemma-3-12b-it-unsloth-bnb-4bit"

if __name__ == "__main__":
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

    # Define model init arguments
    model_kwargs = dict(
        #attn_implementation="eager", # Use "flash_attention_2" when running on Ampere or newer GPU
        #attn_implementation="flash_attention_2", # Use "flash_attention_2" when running on Ampere or newer GPU
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


    local_dir = f"./model/{model_id}"
    download_from_hf(model_id, local_dir)

    # Load model and tokenizer from downloaded directory
    # model = model_class.from_pretrained(local_dir, token=HF_TOKEN, **model_kwargs)

    #tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it") # Load the Instruction Tokenizer to use the official Gemma template
    # tokenizer = AutoTokenizer.from_pretrained(local_dir, token=HF_TOKEN) # Load the Instruction Tokenizer to use the official Gemma template

    # save the full precision model and the tokenizer
    # model.save_pretrained(local_dir, safe_serialization=False)
    # tokenizer.save_pretrained(local_dir)

    asyncio.run(upload_to_gcs(GCS_BUCKET_NAME, GCS_DESTINATION_PATH, model_id, local_dir))
