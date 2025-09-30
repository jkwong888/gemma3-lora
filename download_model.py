import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText, BitsAndBytesConfig
from google.cloud import storage
from google.api_core.exceptions import NotFound
from huggingface_hub import snapshot_download

import os
import sys
import asyncio

GCS_BUCKET_NAME = "jkwng-model-data"  
GCS_DESTINATION_PATH = "models" # The folder path inside your GCS bucket

HF_TOKEN = os.getenv("HF_TOKEN")

# Hugging Face model id
#model_id = "google/gemma-3-27b-pt" # or `google/gemma-3-4b-pt`, `google/gemma-3-12b-pt`, `google/gemma-3-27b-pt`
model_id = "google/gemma-3-27b-it" # or `google/gemma-3-4b-pt`, `google/gemma-3-12b-pt`, `google/gemma-3-27b-pt`

async def upload_blob_async(bucket, semaphore, local_path, gcs_path):
    """
    Asynchronously uploads a single file to GCS, managed by a semaphore.
    
    Args:
        bucket (storage.Bucket): The GCS bucket object.
        semaphore (asyncio.Semaphore): The semaphore to control concurrency.
        local_path (str): The path to the local file to upload.
        gcs_path (str): The destination path in the GCS bucket.
    """
    # The 'async with' statement waits until a slot is available in the semaphore.
    # If the semaphore counter is at its max, this line will pause until
    # another task releases the semaphore.
    async with semaphore:
        try:
            # blob.upload_from_filename is a blocking I/O call.
            # asyncio.to_thread runs this blocking function in a separate thread,
            # allowing the main event loop to stay responsive.
            await asyncio.to_thread(
                bucket.blob(gcs_path).upload_from_filename, local_path
            )
            print(f"  Uploaded '{local_path}'")
        except Exception as e:
            print(f"  FAILED to upload '{local_path}'. Reason: {e}")


async def main():
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
    print(f"downloading model {model_id} to folder {local_dir} ...")
    snapshot_download(repo_id=model_id, local_dir=local_dir)

    # Load model and tokenizer from downloaded directory
    model = model_class.from_pretrained(local_dir, token=HF_TOKEN, **model_kwargs)

    #tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it") # Load the Instruction Tokenizer to use the official Gemma template
    tokenizer = AutoTokenizer.from_pretrained(local_dir, token=HF_TOKEN) # Load the Instruction Tokenizer to use the official Gemma template

    try:
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(GCS_BUCKET_NAME)
        print(f"Successfully accessed bucket: '{GCS_BUCKET_NAME}'")
    except NotFound:
        print(f"Error: Bucket '{GCS_BUCKET_NAME}' not found.")
        print("Please ensure the bucket exists and you have entered the correct name.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while accessing the bucket: {e}")
        print("Please ensure you have authenticated correctly ('gcloud auth application-default login')")
        print("and have the 'Storage Object Admin' or 'Storage Admin' role on the bucket.")
        sys.exit(1)

    gcs_uri = f"gs://{GCS_BUCKET_NAME}/{GCS_DESTINATION_PATH}/{model_id}"
    print(f"Target GCS URI: {gcs_uri}")

    # save the full precision model and the tokenizer
    model.save_pretrained(local_dir, safe_serialization=False)
    tokenizer.save_pretrained(local_dir)


    print(f"Uploading files from {local_dir} to {GCS_BUCKET_NAME}...")
    # 1. Create a semaphore to limit concurrent operations.
    semaphore = asyncio.Semaphore(8)

    tasks = [] 
    # os.walk() generates the file names in a directory tree by walking it.
    for root, _, files in os.walk(local_dir):
        for filename in files:
            # Construct the full local path of the file.
            local_path = os.path.join(root, filename)

            # Construct the full GCS path for the blob.
            # os.path.relpath gets the path of the file relative to the source_directory.
            relative_path = os.path.relpath(local_path, local_dir)
            # Ensure the GCS path uses forward slashes, which is the standard for object storage.
            gcs_path = os.path.join(GCS_DESTINATION_PATH, relative_path).replace(os.path.sep, '/')

            # Create a blob object and upload the file.
            task = upload_blob_async(bucket, semaphore, local_path, gcs_path)
            tasks.append(task)
            # blob = bucket.blob(gcs_path)
            # blob.upload_from_filename(local_path)
            
            # print(f"  Uploaded '{local_path}' to '{gcs_path}'")
            # files_uploaded += 1

    if not tasks:
        return

    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
