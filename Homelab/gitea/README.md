# Gitea on Raspberry Pi 
- Based on work by **PiMyLifeUp**

# Installation
## Installation via Ansible
- Find the version URL you want to install from https://dl.gitea.io/gitea/
- Check the ARM CPU Version for your Raspberry Pi (ex. Pi Zero has arm-5) [how to find arm cpu version](https://raspberrypi.stackexchange.com/questions/9912/how-do-i-see-which-arm-cpu-version-i-have)
- Update the variable "package_url" in the "playbook.yml" with the link to the binary (raw binary) for the CPU package.
- Update the example-inventory.ini file Run the following command:
```bash
ansible-playbook -i example-inventory.ini playbook.yml
```

## Manual Installation Guide 
https://pimylifeup.com/raspberry-pi-gitea/

# Setup
- navigate to http://<raspberrypi-ip>:3000/ on a web browser.
- set up the gitea instance how you want (sqlite is lightweight, URL configuration, admin user creation etc).
- if you do change the server port from 3000 - not that you will also have to update the UFW configuration to reflect the changes.
- save the config and give it a minute to restart the service with the updated config.


