

- Running k3s instead of full k8s due to the limited availability of resources on various nodes.
- K3S Setup with 1 control plane node, 5 worker nodes, flannel backend and modified max pod limit per node of 200.
- Due to the various different types of nodes (x86, ARM, CPU/Memory Ratio, internet traffic) the workloads running on these nodes needs to be adjusted to be scheduled on the correct node.
- Each node has labels that describe the kinds of workloads that can be run on the instance (like allowing compute workloads but not internet-heavy workloads on HomeLab instances due to limited internet traffic).

- Example node labels for workload scheduling (for ARM-2 node):

```
workload=compute
disk-heavy=yes
internet-heavy=yes
memory-burst=yes
```

- Then, workload that needs access to internet heavy instances can be run on this node like:

```yaml
--- 
apiVersion: apps/v1
kind: Deployment 
metadata: 
	name: "sample-workload"
spec:
	template:
		spec:
			nodeSelector:
			internet-heavy: "yes"
# rest of the manifest		
---
```

