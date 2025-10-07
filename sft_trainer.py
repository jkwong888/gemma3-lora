import os
import torch
import asyncio
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText, BitsAndBytesConfig

from peft import LoraConfig
from trl import SFTConfig

from utils.dataset import load_dataset_from_gcs, create_conversation
from utils.model import load_model_from_gcs
from prompt import system_message, user_prompt

GCS_MODEL_BUCKET_NAME = "jkwng-model-data"  
GCS_MODEL_PATH = "models" # The folder path inside your GCS bucket

GCS_DATASET_BUCKET_NAME = "jkwng-hf-datasets"  
GCS_DATASET_PATH = "datasets" # The folder path inside your GCS bucket


# Load dataset from the hub
# dataset = load_dataset("philschmid/gretel-synthetic-text-to-sql", split="train")
dataset_id = "philschmid/gretel-synthetic-text-to-sql" 
dataset = load_dataset_from_gcs(f"gs://{GCS_DATASET_BUCKET_NAME}/{GCS_DATASET_PATH}/{dataset_id}", split="train")

dataset = dataset.map(create_conversation, batched=False, fn_kwargs={"system_message": system_message, "user_prompt": user_prompt})

# Print formatted user prompt
#print(dataset["train"][345]["messages"][1]["content"])

# Hugging Face model id
#model_id = "google/gemma-3-27b-pt" # or `google/gemma-3-4b-pt`, `google/gemma-3-12b-pt`, `google/gemma-3-27b-pt`
model_id = os.getenv("MODEL_ID") or "google/gemma-3-12b-it" # or `google/gemma-3-4b-pt`, `google/gemma-3-12b-pt`, `google/gemma-3-27b-pt`
#model_id = "unsloth/gemma-3-12b-it-unsloth-bnb-4bit" # or `google/gemma-3-4b-pt`, `google/gemma-3-12b-pt`, `google/gemma-3-27b-pt`

local_dir = f"./model/{model_id}"
asyncio.run(load_model_from_gcs(f"gs://{GCS_MODEL_BUCKET_NAME}/{GCS_MODEL_PATH}/{model_id}", local_dir))

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

# BitsAndBytesConfig: Enables 4-bit quantization to reduce model size/memory usage
model_kwargs["quantization_config"] = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=model_kwargs['torch_dtype'],
    bnb_4bit_quant_storage=model_kwargs['torch_dtype'],
)

# Load model and tokenizer
print(f"loading model from {local_dir} ...")
model = model_class.from_pretrained(local_dir, **model_kwargs)

# Load tokenizer
print(f"loading tokenizer from {local_dir} ...")
#tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it") # Load the Instruction Tokenizer to use the official Gemma template
tokenizer = AutoTokenizer.from_pretrained(f"{local_dir}") # Load the Instruction Tokenizer to use the official Gemma template


peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=16,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
    modules_to_save=["lm_head", "embed_tokens"] # make sure to save the lm_head and embed_tokens as you train the special tokens
)

args = SFTConfig(
    output_dir="gemma-text-to-sql",         # directory to save and repository id
    max_seq_length=256,                   # max sequence length for model and packing of the dataset - set to longest sequence 
    packing=True,                           # Groups multiple samples in the dataset into a single sequence
    num_train_epochs=3,                     # number of training epochs
    per_device_train_batch_size=1,          # batch size per device during training
    gradient_accumulation_steps=8,          # number of steps before performing a backward/update pass
    gradient_checkpointing=True,            # use gradient checkpointing to save memory
    optim="adamw_torch_fused",              # use fused adamw optimizer
    logging_steps=10,                       # log every 10 steps
    save_strategy="epoch",                  # save checkpoint every epoch
    learning_rate=2e-4,                     # learning rate, based on QLoRA paper
    fp16=True if torch_dtype == torch.float16 else False,   # use float16 precision
    bf16=True if torch_dtype == torch.bfloat16 else False,   # use bfloat16 precision
    max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
    warmup_ratio=0.03,                      # warmup ratio based on QLoRA paper
    lr_scheduler_type="constant",           # use constant learning rate scheduler
    push_to_hub=False,                       # push model to hub
    report_to="tensorboard",                # report metrics to tensorboard
    dataset_kwargs={
        "add_special_tokens": False, # We template with special tokens
        "append_concat_token": True, # Add EOS token as separator token between examples
    }
)

from trl import SFTTrainer

# Create Trainer object
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    peft_config=peft_config,
    processing_class=tokenizer
)

# Start training, the model will be automatically saved to the Hub and the output directory
trainer.train()

# Save the final model again to the Hugging Face Hub
trainer.save_model()

# free the memory again
del model
del trainer
torch.cuda.empty_cache()

from peft import PeftModel

# Load Model base model
model = model_class.from_pretrained(model_id, low_cpu_mem_usage=True)

# Merge LoRA and base model and save
peft_model = PeftModel.from_pretrained(model, args.output_dir)
merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained("merged_model", safe_serialization=True, max_shard_size="2GB")

processor = AutoTokenizer.from_pretrained(args.output_dir)
processor.save_pretrained("merged_model")