# 11-667 Homework 3: Tool Use and Retrieval (ver 2024.1.2)

## Setting up

### AWS
If you do not already have access to GPUs, you may need an AWS virtual
  machine for model training.
[Here are the instructions for setting that up.](https://docs.google.com/presentation/d/1Tw_klO84R9G7CZ3cINAKgy4BfdNm-8dlnRXSBIVD_3A/edit?usp=sharing) 

### Python environment
1. Install conda: `bash setup-conda.sh && source ~/.bashrc`
2. Create conda environment:
   If you run into error like `UnavailableInvalidChannel: HTTP 403 FORBIDDEN for channel <some channel>` on your EC2 instance, you can solve it by running `conda config --remove channels <some channel>`, and make sure you have the default channel by running `conda config --add channels defaults`.
<!-- ```bash
conda create -n cmu-llms-hw3 python=3.11
conda activate cmu-llms-hw3
pip install -r requirements.txt
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
pip install ninja
pip install flash-attn --no-build-isolation
pip install wandb
pip install -e . -->
```
