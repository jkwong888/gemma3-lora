import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText

import os
import sys
import asyncio
import argparse
from utils.model import download_from_hf, upload_to_gcs

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Download model from Hugging Face and upload to GCS")
parser.add_argument("--gcs-bucket-name", type=str, default="jkwng-model-data",
                    help="GCS bucket name (default: jkwng-model-data)")
parser.add_argument("--gcs-destination-path", type=str, default="models",
                    help="GCS destination path inside bucket (default: models)")
parser.add_argument("--model-id", type=str, default="unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
                    help="Hugging Face model ID (default: unsloth/gemma-3-12b-it-unsloth-bnb-4bit)")
args = parser.parse_args()

GCS_BUCKET_NAME = args.gcs_bucket_name
GCS_DESTINATION_PATH = args.gcs_destination_path
model_id = args.model_id

HF_TOKEN = os.getenv("HF_TOKEN")

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
