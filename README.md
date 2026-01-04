
# Why

- The homelab allows me to experiment on various things like LLMs, malwares, tinkering with stuff etc. 
- Works as a "proxy" of a small company for me to play around with. _Cosplaying as a SysAdmin_
- Helps me learn a LOT of things (from various services like hashicorp vault, terraform to k3s or CUDA)

![](assets/Pasted%20image%2020251229225254.png)

# Setup
- [Hardware Specifications](Hardware-Specifications.md) for the various nodes in the home (and cloud) lab
- [Storage Layout](Storage%20Layout.md) details
- [k3s cluster](k3s-cluster.md) documentation
- [Network Infrastructure](Network%20Infrastructure.md) - Details about the global network setup
- [Exploring LLMs](Exploring-LLMs.md) - Diving into self hosting & using Large Language Models

# Services

Here are some services that I run in my homelab:

- [Forgejo](https://forgejo.org/) as self-hosted github alternative
- [Jenkins](https://www.jenkins.io/) for CI / CD of various services
- [vLLM](https://github.com/vllm-project/vllm) for running large language models on the GPU
- [llama.cpp](https://github.com/ggml-org/llama.cpp) for running LLMs on CPU + GPU
- [immich](https://immich.app/) as google photos alternative
- [SeaweedFS](https://github.com/seaweedfs/seaweedfs) for exposing storage as an S3 API
- [PostgreSQL](https://www.postgresql.org/) for storing relational data
- [Grafana](https://grafana.com/) for observability / metrics
- [Pyroscope](https://pyroscope.io/) for runtime profiling of custom services
- [Metabase](https://www.metabase.com/) for dashboarding
- [registry](https://hub.docker.com/_/registry) as a private docker registry using S3 backend (with seaweedfs S3 API)
- [Temporal](https://temporal.io/) for some workflows


![](assets/Pasted%20image%2020251229224941.png)