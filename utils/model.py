import asyncio
from huggingface_hub import snapshot_download
from google.cloud import storage
from google.api_core.exceptions import NotFound
import os
import sys


def download_from_hf(model_id, local_dir):
    """
    Downloads a model from Hugging Face Hub to a local directory.

    Args:
        model_id (str): The ID of the model on Hugging Face Hub.
        local_dir (str): The local directory to download the model to.
    """
    print(f"downloading model {model_id} to folder {local_dir} ...")
    snapshot_download(repo_id=model_id, local_dir=local_dir)


async def upload_to_gcs(gcs_bucket_name, gcs_destination_path, model_id, local_dir):
    """
    Uploads a model from a local directory to a GCS bucket.

    Args:
        gcs_bucket_name (str): The name of the GCS bucket.
        gcs_destination_path (str): The destination path in the GCS bucket.
        model_id (str): The ID of the model.
        local_dir (str): The local directory where the model is saved.
    """
    try:
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(gcs_bucket_name)
        print(f"Successfully accessed bucket: '{gcs_bucket_name}'")
    except NotFound:
        print(f"Error: Bucket '{gcs_bucket_name}' not found.")
        print("Please ensure the bucket exists and you have entered the correct name.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while accessing the bucket: {e}")
        print("Please ensure you have authenticated correctly ('gcloud auth application-default login')")
        print("and have the 'Storage Object Admin' or 'Storage Admin' role on the bucket.")
        sys.exit(1)

    gcs_uri = f"gs://{gcs_bucket_name}/{gcs_destination_path}/{model_id}"
    print(f"Target GCS URI: {gcs_uri}")

    print(f"Uploading files from {local_dir} to {gcs_bucket_name}...")
    semaphore = asyncio.Semaphore(8)
    tasks = []
    for root, _, files in os.walk(local_dir):
        for filename in files:
            local_path = os.path.join(root, filename)
            relative_path = os.path.relpath(local_path, local_dir)
            gcs_path = os.path.join(gcs_destination_path, relative_path).replace(os.path.sep, '/')
            task = _upload_blob_async(bucket, semaphore, local_path, gcs_path)
            tasks.append(task)

    if not tasks:
        return

    await asyncio.gather(*tasks)


async def load_model_from_gcs(gcs_uri, local_dir):
    """
    Downloads a model from a GCS URI to a local directory.

    Args:
        gcs_uri (str): The GCS URI of the model.
        local_dir (str): The local directory to download the model to.
    """
    try:
        storage_client = storage.Client()
        bucket_name, prefix = gcs_uri.replace("gs://", "").split("/", 1)
        bucket = storage_client.get_bucket(bucket_name)
        print(f"Successfully accessed bucket: '{bucket_name}'")
    except NotFound:
        print(f"Error: Bucket '{bucket_name}' not found.")
        print("Please ensure the bucket exists and you have entered the correct name.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while accessing the bucket: {e}")
        print("Please ensure you have authenticated correctly ('gcloud auth application-default login')")
        print("and have the 'Storage Object Admin' or 'Storage Admin' role on the bucket.")
        sys.exit(1)

    print(f"Downloading files from {gcs_uri} to {local_dir}...")
    semaphore = asyncio.Semaphore(8)
    tasks = []
    for blob in bucket.list_blobs(prefix=prefix):
        local_path = os.path.join(local_dir, os.path.relpath(blob.name, prefix))
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        task = _download_blob_async(semaphore, blob, local_path)
        tasks.append(task)

    if not tasks:
        return

    await asyncio.gather(*tasks)


async def _upload_blob_async(bucket, semaphore, local_path, gcs_path):
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


async def _download_blob_async(semaphore, blob, local_path):
    """
    Asynchronously downloads a single file from GCS, managed by a semaphore.
    
    Args:
        semaphore (asyncio.Semaphore): The semaphore to control concurrency.
        blob (storage.Blob): The GCS blob object to download.
        local_path (str): The local path to download the file to.
    """
    async with semaphore:
        try:
            await asyncio.to_thread(
                blob.download_to_filename, local_path
            )
            print(f"  Downloaded '{local_path}'")
        except Exception as e:
            print(f"  FAILED to download '{local_path}'. Reason: {e}")
