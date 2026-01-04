
- 2 Nodes are run & managed completely by me. Remaining 5 nodes are VMs running across the globe.
- 1 new node to be added to the lab (still under process)

| Node Name |     Type      | vCPU | Memory in GB | Internet Traffic Limit | Storage |
| --------- | :-----------: | ---- | ------------ | ---------------------- | ------- |
| Kraken    |    HomeLab    | 32   | 128          | 3.3 TB / month         | 30 TB   |
| Remote-1  | Dell Optiplex | 4    | 8            | 3.3 TB / month         | 1 TB    |
| ARM-1     | Free tier VM  | 4    | 24           | 10 TB upstream         | 200 GB  |
| ARM-2     | Free tier VM  | 4    | 24           | 10 TB upstream         | 200 GB  |
| Blackbird |   Paid VPS    | 2    | 10           | Unlimited              | 80 GB   |
| Sherlock  |   Paid VPS    | 4    | 16           | 40 TB / month          | 160 GB  |
| Watson    |   Paid VPS    | 4    | 16           | 40 TB / month          | 160 GB  |
| - TBD -   |    Server     | 48   | 512          | 4 TB / month           | - TBD - |

### Kraken

- Ryzen 7950 x3D Build with 128GB DDR5 RAM and lots of storage
- Runs Proxmox with multiple VMs (full k3s-stage cluster, k3s-prod-control-node)
- Also has 1x RTX 3090 GPUs for accelerating LLM performance (1 more to be added soon)
- Details : https://medium.com/@harsh8v/homelab-v2-migrating-to-onprem-91020b163117
- Functions as a swiss-army-knife for all PoCs and running most workloads that do not need lots of internet traffic

### Remote-1
- Refurbished Dell Optiplex with an old HDD mirrored with an old SSD 
- Runs Proxmox Host, TrueNAS VM
- TrueNAS VM runs S3 API and immich
- Functions mostly as a backup server for images & as remote backup server for kraken.
