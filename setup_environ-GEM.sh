#!/bin/bash

# Set the environment variable
export TEXT_METRICS_WRAPPER_DIR=$(dirname $(readlink -f $0))

# Add the environment variable to /etc/environment
echo "export TEXT_METRICS_WRAPPER_DIR='$TEXT_METRICS_WRAPPER_DIR'" | sudo tee -a /etc/environment

# Reload the environment variables
source /etc/environment

# Go to the correct folder
cd $TEXT_METRICS_WRAPPER_DIR

# Install GEM
git clone https://github.com/GEM-benchmark/GEM-metrics
cd GEM-metrics
pip install -r requirements.txt
cd ..

# Get the MoverScore requirements
cd "/content/drive/Shareddrives/LLM_test/Luca_De_Grandis_test/Table_summarization/emnlp19-moverscore"
pip install .
cd ..

%%shell
cd "/content/drive/Shareddrives/LLM_test/Luca_De_Grandis_test/Table_summarization/GEM-metrics"
pip install -r requirements-heavy.txt
cd ..

