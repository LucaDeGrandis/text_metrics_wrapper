#!/bin/bash

# Set the environment variable
export TEXT_METRICS_WRAPPER_DIR=$(dirname $(readlink -f $0))

# Add the environment variable to /etc/environment
echo "export TEXT_METRICS_WRAPPER_DIR='$TEXT_METRICS_WRAPPER_DIR'" | sudo tee -a /etc/environment

# Reload the environment variables
source /etc/environment

# Go to the correct folder
cd $TEXT_METRICS_WRAPPER_DIR

# Install Google language
git clone https://github.com/google-research/bleurt.git
cd bleurt
pip install .
wget https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip .
unzip BLEURT-20.zip