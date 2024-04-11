#!/usr/bin/env bash

# Unminimizing the container
unminimize

# Adding package repository for OneAPI
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null || exit 1
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list || exit 1

# Updating package index, upgrading, and installing required packages
sudo apt update || exit 1
sudo apt upgrade -y || exit 1
sudo apt install -y intel-basekit-2023.2.0 libboost-dev cmake || exit 1
source /opt/intel/oneapi/setvars.sh
echo "source /opt/intel/oneapi/setvars.sh" >> ~/.bashrc

# Installing Julia & instantiating the project
curl -fsSL https://install.julialang.org > /tmp/juliaup.sh
chmod +x /tmp/juliaup.sh
/tmp/juliaup.sh -y
source ~/.bashrc
julia --project -e "using Pkg; Pkg.instantiate()"
