# Playbooks

# Installation

## First Install
- The article at https://www.digitalocean.com/community/tutorials/how-to-set-up-an-openvpn-server-on-ubuntu-18-04 shows how to setup the CA and certificates

- Post that, update the vars.yml file with the files and variables you want...

## Ansible Installation
- Install OpenVPN Server and configure it on a remote machine
```
ansible-playbook -i inventory.ini server-playbook.yml
```
- Generate a client config and configure it to use static ip
```
./gen_client_conf.sh <client_name> <static_ip>
```
- Just configure a static ip for a client
```
ansible-playbook -i inventory.ini static-ip.yml -e 'client_name=<client_name>' -e 'static_ip=<static_ip>'
```
- Configure remote client to use the given openvpn configuration
```
ansible-playbook -i <client>, update-client.yml -e 'client_name=<client_name>' 
```

## IpTables Editing for Ubuntu
- ORACLE UBUNTU VMs DO NOT FOLLOW UFW RULES CORRECTLY. [Stack Overflow Question](https://askubuntu.com/questions/1299752/enabling-port-80-with-iptables-works-but-does-not-work-with-ufw)
- Allow all traffic from a network interface 
```
sudo iptables -I INPUT -i tun0 -j ACCEPT
```
- Save existing rules into rules.v4 , v6
```
sudo netfilter-persistent save
```
