#!/bin/bash

# Set the environment variable
export TEXT_METRICS_WRAPPER_DIR=$(dirname $(readlink -f $0))

# Add the environment variable to /etc/environment
echo "export TEXT_METRICS_WRAPPER_DIR='$TEXT_METRICS_WRAPPER_DIR'" | sudo tee -a /etc/environment

# Reload the environment variables
source /etc/environment

# Go to the correct folder
cd $TEXT_METRICS_WRAPPER_DIR

# Install nltk package from github
git clone https://github.com/nltk/nltk.git
cd nltk
git checkout e2d368e
pip install .
cd ..
