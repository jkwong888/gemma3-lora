from datasets import load_dataset, load_from_disk
from google.cloud import storage
from google.api_core.exceptions import NotFound

import os
import sys
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Download and upload dataset to GCS")
parser.add_argument("--gcs-bucket-name", type=str, default="jkwng-hf-datasets",
                    help="GCS bucket name (default: jkwng-hf-datasets)")
parser.add_argument("--gcs-destination-path", type=str, default="datasets",
                    help="GCS destination path inside bucket (default: datasets)")
parser.add_argument("--dataset-id", type=str, default="philschmid/gretel-synthetic-text-to-sql",
                    help="Hugging Face dataset ID (default: philschmid/gretel-synthetic-text-to-sql)")
args = parser.parse_args()

GCS_BUCKET_NAME = args.gcs_bucket_name
GCS_DESTINATION_PATH = args.gcs_destination_path
dataset_id = args.dataset_id

if not os.path.exists(f"dataset/{dataset_id}"):
  # Load dataset from the hub
  dataset = load_dataset(dataset_id)

  # save to disk
  dataset.save_to_disk(f"dataset/{dataset_id}")
else:
  dataset = load_from_disk(f"dataset/{dataset_id}")

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
