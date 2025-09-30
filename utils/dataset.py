
import gcsfs
from datasets import load_from_disk


# --- Configuration ---
# PLEASE REPLACE THESE VALUES WITH YOUR OWN
GCS_BUCKET_NAME = "jkwng-hf-datasets"  

# The path to your dataset within the GCS bucket.
# This should point to the directory where your dataset was saved
# (e.g., using dataset.save_to_disk()). This directory contains
# files like 'dataset_info.json', 'state.json', and your data files
# (e.g., in .arrow or .parquet format).
GCS_DATASET_PATH = "my-gemma-model-parallel-upload" # <-- 🚫 CHANGE THIS to the path of your dataset

# System message for the assistant 
system_message = """You are a text to SQL query translator. Users will ask you questions in English and you will generate a SQL query based on the provided SCHEMA."""

# User prompt that combines the user query and the schema
user_prompt = """Given the <USER_QUERY> and the <SCHEMA>, generate the corresponding SQL command to retrieve the desired data, considering the query's syntax, semantics, and schema constraints.

<SCHEMA>
{context}
</SCHEMA>

<USER_QUERY>
{question}
</USER_QUERY>
"""

def create_conversation(sample):
    return {
        "messages": [
        # {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt.format(question=sample["sql_prompt"], context=sample["sql_context"])},
        {"role": "assistant", "content": sample["sql"]}
        ]
    }  

def load_dataset_from_gcs(gcs_uri, split='test'):
    gcs = gcsfs.GCSFileSystem(project="jkwng-kueue-dev")

    # Load dataset from GCS - this step needs gcsfs
    dataset = load_from_disk(gcs_uri)

    print(f"dataset at {gcs_uri} has {len(dataset[split])} records")
    #dataset = dataset[split].shuffle().select(range(12500))

    # Convert dataset to OAI messages
    dataset = dataset[split].map(create_conversation, batched=False)

    # split dataset into 10,000 training samples and 2,500 test samples
    dataset = dataset.train_test_split(test_size=2500/12500)



    return dataset

def load_dataset(dataset_id):
    # Load dataset from the hub
    #dataset_id = "philschmid/gretel-synthetic-text-to-sql" 
    dataset = load_from_disk(f"dataset/{dataset_id}/train")
    dataset = dataset.shuffle().select(range(12500))

    # Convert dataset to OAI messages
    dataset = dataset.map(create_conversation, remove_columns=dataset.features,batched=False)

    # split dataset into 10,000 training samples and 2,500 test samples
    dataset = dataset.train_test_split(test_size=2500/12500)

    # Print formatted user prompt -- test
    # print(dataset["train"][345]["messages"][0]["content"])
    # print(dataset["train"][345]["messages"][1]["content"])

    return dataset

