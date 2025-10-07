# gemma3-lora
playing with gemma3 and lora


# pre-requisites

1. install GPU drivers into base Debian 12 machine
   [install gpu drivers](https://cloud.google.com/compute/docs/gpus/install-drivers-gpu#install-script)

   - `sudo systemctl stop google-cloud-ops-agent`
   - `curl -L https://storage.googleapis.com/compute-gpu-installation-us/installer/latest/cuda_installer.pyz --output cuda_installer.pyz`
   - `sudo python3 cuda_installer.pyz install_driver --installation-mode=binary --installation-branch=prod`
     - (reboot)
   - `sudo python3 cuda_installer.pyz install_driver --installation-mode=binary --installation-branch=prod`
   - `sudo python3 cuda_installer.pyz install_cuda --installation-mode=binary --installation-branch=prod`
     - (reboot)

2. install python virtualenv and create one

   ```
   sudo apt install python3.11-venv
   python3 -mvenv venv
   source venv/bin/activate
   ```

3. install requirements

   ```
   pip install -r requirements.txt
   ```


# script

Note - most of these scripts assume the `unsloth` dynamic quant of Gemma3 12B, and will pull the dataset from my GCS bucket in `jkwng-hf-datasets` and the model in `jkwng-model-data`.  You can update these at the top of each file.

- `download_dataset.py` - download the dataset to (local dir and) GCS
- `download_model.py` - download the model from huggingface to (local dir and) GCS
- `load_model_and_test_vllm.py` - load model using VLLM and batch request the dataset to it
- `sft_trainer.py` - attempt to fine tune the model using huggingface `trl` library