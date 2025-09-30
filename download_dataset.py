from datasets import load_dataset, load_from_disk
from google.cloud import storage
from google.api_core.exceptions import NotFound

import os
import sys

GCS_BUCKET_NAME = "jkwng-hf-datasets"  
GCS_DESTINATION_PATH = "datasets" # The folder path inside your GCS bucket

dataset_id = "philschmid/gretel-synthetic-text-to-sql"

if not os.path.exists(f"dataset/{dataset_id}"):
  # Load dataset from the hub
  dataset = load_dataset(dataset_id)

  # save to disk
  dataset.save_to_disk(f"dataset/{dataset_id}")

dataset = load_from_disk(f"dataset/{dataset_id}")
#dataset = dataset.shuffle().select(range(12500))

# Convert dataset to OAI messages
#dataset = dataset.map(create_conversation, remove_columns=dataset.features,batched=False)
# split dataset into 10,000 training samples and 2,500 test samples
#dataset = dataset.train_test_split(test_size=2500/12500)

# Print formatted user prompt
#print(dataset["train"][345]["messages"][0]["content"])
#print(dataset["train"][345]["messages"][1]["content"])

# write the dataset to gcs
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

# Construct the full GCS URI
gcs_uri = f"gs://{GCS_BUCKET_NAME}/{GCS_DESTINATION_PATH}/{dataset_id}"

print(f"\nSaving dataset to {gcs_uri}...")
try:
  dataset.save_to_disk(gcs_uri)
  print("Dataset successfully saved to GCS!")
except Exception as e:
  print(f"Failed to save dataset to GCS: {e}")
  sys.exit(1)
