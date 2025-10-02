
import gcsfs
from utils.prompt import create_conversation
from datasets import load_from_disk

def create_conversation(sample,system_message, user_prompt):
    return {
        "messages": [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt.format(question=sample["sql_prompt"], context=sample["sql_context"])},
        {"role": "assistant", "content": sample["sql"]}
        ]
    }  

def load_dataset_from_gcs(gcs_uri, split='test'):
    gcs = gcsfs.GCSFileSystem(project="jkwng-kueue-dev")

    # Load dataset from GCS - this step needs gcsfs
    dataset = load_from_disk(gcs_uri)

    print(f"dataset at {gcs_uri} has {len(dataset[split])} records")
    dataset = dataset[split].shuffle()

    # Convert dataset to OAI messages
    # dataset = dataset[split].map(create_conversation, batched=False)

    # split dataset into 10,000 training samples and 2,500 test samples
    #dataset = dataset.train_test_split(test_size=2500/12500)

    return dataset

def load_dataset(dataset_id, split='test'):
    # Load dataset from the hub
    #dataset_id = "philschmid/gretel-synthetic-text-to-sql" 
    dataset = load_from_disk(f"dataset/{dataset_id}/{split}")
    dataset = dataset.shuffle()

    # Convert dataset to OAI messages
    #dataset = dataset.map(create_conversation, remove_columns=dataset.features,batched=False)

    # split dataset into 10,000 training samples and 2,500 test samples
    dataset = dataset.train_test_split(test_size=2500/12500)

    # Print formatted user prompt -- test
    # print(dataset["train"][345]["messages"][0]["content"])
    # print(dataset["train"][345]["messages"][1]["content"])

    return dataset

