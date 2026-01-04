# Exploring LLMs

- Exploring models like gemma-3, gpt-oss, LLaMA-3.1-8b,  GLM-4.5-Air, DeepSeek-3.1 etc for research purpose
- Frameworks like llama.cpp, vLLM, sglang, ollama for running models optimizing for performance (power usage, speed, stability, resoure utilization) on a small scale
- Synthetic data generation using meta-llama's synthetic data kit for model finetuning
- https://github.com/meta-llama/synthetic-data-kit/pull/68


## Finetuning models

- Used synthetic-data-kit to generate QA pairs from AWS documentation, HTB writeups, other cybersecurity focused content
- Finetuned gemma-3-12b, llama-3.2-3b models using QLoRA on a custom dataset built from the most relevant data from the above 

## Usage by projects

- Pulse Insight 


## Hardware setup

- LLM VM set to boot automatically on proxmox boot (Always On)
- `~32 GB RAM` without ballooning 
- plenty of storage space on High Speed SSD (Proxmox zpool)
- `8TB Intel p4510` SSD mounted at `/home/ubuntu/.cache/` for huggingface models storage
- 1x `RTX 3090` PCIe passthrough (`gen 4 x16 Lanes`)
- 32 "cores" passed through with processor type set to `host`
- machine type `q35` 
- `AMD_SEV` disabled 