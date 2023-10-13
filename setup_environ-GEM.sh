#!/bin/bash

# Set the environment variable
export TEXT_METRICS_WRAPPER_DIR=$(dirname $(readlink -f $0))

# Add the environment variable to /etc/environment
echo "export TEXT_METRICS_WRAPPER_DIR='$TEXT_METRICS_WRAPPER_DIR'" | sudo tee -a /etc/environment

# Reload the environment variables
source /etc/environment

# Go to the correct folder
cd $TEXT_METRICS_WRAPPER_DIR

# Create a txt file
touch example.txt

# Install GEM
git clone https://github.com/GEM-benchmark/GEM-metrics.git
cd GEM-metrics
git checkout 8162210
pip install -r requirements.txt
cd ..

# Install heavy requirements for GEM
pip install bert_score==0.3.13
pip install pyemd==1.0.0

# Get the MoverScore requirements
git clone https://github.com/LucaDeGrandis/moverscore_modified.git
cd moverscore_modified
pip install .
cd ..
