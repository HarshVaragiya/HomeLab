#!/bin/bash

# First argument: Client identifier

# Variables

CLIENT_DEFAULT_CONF="clients/${1}/default.conf"
KEY_DIR="clients/${1}"
OUTPUT_DIR="clients/${1}"
BASE_CONFIG="clients/base.conf"
OUTPUT_CONF="${OUTPUT_DIR}/${1}.ovpn"
CERTS_DIR="certs"

REMOTE_SERVER="my-vpn-domain.ddns.net"

# Checks

if [ -z "${1}" ] || [ -z "${REMOTE_SERVER}" ]
then
    echo "Client Name (first argument) or 'REMOTE_SERVER' (exported variable) is not defined."
    exit -1
fi

if [ -f "${OUTPUT_CONF}" ]
then
    echo "Client Configuration with name ${1} already exists. cannot continue."
    exit -1
fi

# Execution

mkdir ${KEY_DIR}

echo "Generating Client Configuration for OpenVPN. Client : ${1} on OpenVPN Server: ${REMOTE_SERVER}"

# Copy boilerplate config
cp "${CERTS_DIR}/ca.crt" "${OUTPUT_DIR}/"
cp "${CERTS_DIR}/ta.key" "${OUTPUT_DIR}/"

# Generate Key and Certificate for Client
cd "${CERTS_DIR}/EasyRSA-3.0.4/"

echo "Press ENTER to continue"
./easyrsa gen-req ${1} nopass
# Press enter

mv "pki/private/${1}.key" "../../${OUTPUT_DIR}/"
mv "pki/reqs/${1}.req" "../../${OUTPUT_DIR}/"

./easyrsa import-req ../../${OUTPUT_DIR}/${1}.req ${1}
echo "type 'YES' to continue."
./easyrsa sign-req client ${1}
# Type "yes"
mv "pki/issued/${1}.crt" "../../${OUTPUT_DIR}/"

cd "../../"

# One File Configuration

cp ${BASE_CONFIG} ${CLIENT_DEFAULT_CONF}
sed -i "s/REMOTE_SERVER/${REMOTE_SERVER}/g" ${CLIENT_DEFAULT_CONF}

cat ${CLIENT_DEFAULT_CONF} <(echo -e '<ca>') ${KEY_DIR}/ca.crt <(echo -e '</ca>\n<cert>') \
    ${KEY_DIR}/${1}.crt <(echo -e '</cert>\n<key>') ${KEY_DIR}/${1}.key \
    <(echo -e '</key>\n<tls-auth>') ${KEY_DIR}/ta.key <(echo -e '</tls-auth>') \
    > ${OUTPUT_CONF}

echo "Output file saved at : ${OUTPUT_CONF}"

# Static IP Configuration
if [ -z "${2}" ]
then
    exit 0
fi

echo "Setting Static IP for ${1} to be ${2}"
ansible-playbook -i inventory.ini -e "client_name=${1}" -e "static_ip=${2}" static-ip.yml