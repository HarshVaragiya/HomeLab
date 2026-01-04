# Network Infrastructure

## Overview

The homelab network consist of nodes spread across the globe, interconnected using Tailscale VPN (based on WireGuard protocol) to create a single mesh VPN. This allows any device on the network to communicate with any other device seamlessly (as long as the Tailscale network ACL allows it to).

## Network Topology

- **Nodes**: Various physical and virtual machines located in different geographic locations
- **VPN**: Tailscale VPN provides secure, mesh-style connectivity between all nodes
- **Firewall Configuration**: All nodes have firewalls configured to allow traffic on a limited set of ports to minimize the attack surface
- **DNS**: While i am not managing my own DNS server for now, that might change in the future.