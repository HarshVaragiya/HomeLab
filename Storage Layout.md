

- Kraken Node provides centralized storage including high speed SSDs and high capacity HDDs exposed via various services (S3 API, Postgres etc)

![](assets/Pasted%20image%2020251229222112.png)

- TrueNAS VM running virtualized inside Proxmox gives access to 22TB of usable space after the above ZFS layout.
- It tries to balance performance (read/write speeds & latency), space and potential upgrade path for storage expansion without compromising redundancy. 


### Storage Services

- SeaweedFS running on TrueNAS as an application with S3 API Filer
- https://github.com/seaweedfs/seaweedfs

![](assets/Pasted%20image%2020251229223211.png)

- Postgres database running as a container on a dedicated Proxmox VM with 1.5 TB disk 

![](assets/Pasted%20image%2020251229223704.png)

