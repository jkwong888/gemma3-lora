from transformers import AutoTokenizer
from tqdm import tqdm

from vllm import LLM, SamplingParams

from utils.dataset import load_dataset_from_gcs, create_conversation
from utils.model import load_model_from_gcs
from prompt import system_message, user_prompt

import json
import asyncio
import os

GCS_BUCKET_NAME = "jkwng-hf-datasets"  
GCS_DESTINATION_PATH = "datasets" # The folder path inside your GCS bucket

def main():
    # Hugging Face model id
    model_id = "unsloth/gemma-3-12b-it-unsloth-bnb-4bit" # or `google/gemma-3-4b-pt`, `google/gemma-3-12b-pt`, `google/gemma-3-27b-pt`
    #model_id = "google/gemma-3-27b-pt" # or `google/gemma-3-4b-pt`, `google/gemma-3-12b-pt`, `google/gemma-3-27b-pt`

    dataset_id = "philschmid/gretel-synthetic-text-to-sql" 

    dataset = load_dataset_from_gcs(f"gs://{GCS_BUCKET_NAME}/{GCS_DESTINATION_PATH}/{dataset_id}")

    # Generate our SQL query.
    print(f"processing {dataset['test'].num_rows} records")

    print(f"applying chat template to dataset...")
    # create OAI style conversation (?)
    dataset = dataset.map(create_conversation, batched=False, fn_kwargs={"system_message": system_message, "user_prompt": user_prompt})

    # print(dataset["train"][0]["formatted_chat"])
    #print(dataset["test"]["messages"])

    local_dir = f"./model/{model_id}"
    if not os.path.exists(local_dir):
        asyncio.run(load_model_from_gcs(f"gs://jkwng-model-data/models/{model_id}", local_dir))

    tokenizer = AutoTokenizer.from_pretrained(local_dir) # Load the Instruction Tokenizer to use the official Gemma template

    # Convert as test example into a prompt with the Gemma template
    stop_token_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<end_of_turn>")]
    dataset = dataset.map(lambda x: {"formatted_chat": tokenizer.apply_chat_template(x["messages"][:2], tokenize=False, add_generation_prompt=True)})

    # load the model
    print(f"loading model from {local_dir} ...")
    sampling_params = SamplingParams(
       temperature=0.1,
       top_k=50,
       top_p=0.1,
       max_tokens=256,
       #stop_token_ids=stop_token_ids,
    )

    vllm_model = LLM(
       model=local_dir,
       tokenizer=local_dir,
       gpu_memory_utilization=0.95,
       max_num_seqs=8,
       max_model_len=10000,
    )

    outputs = vllm_model.generate([example["formatted_chat"] for example in dataset['test']], sampling_params)
    with open('output.jsonl', 'w') as outfile:
        for idx, output in enumerate(tqdm(outputs)):
          # Extract the user query and original answer
          answer = {
            "original_prompt": output.prompt,
            "sql_context": dataset['test'][idx]['sql_context'],
            "user_query": dataset['test'][idx]['sql_prompt'],
            "ground_truth": dataset['test'][idx]['sql'],
            "generated_answer": output.outputs[0].text,
          }

          # write to file
          outfile.write(f"{json.dumps(answer)}\n")
          
    # print(outputs)
      
    # break


if __name__ == '__main__':
    main()
