#!/bin/bash

# Set the environment variable
export TEXT_METRICS_WRAPPER_DIR=$(dirname $(readlink -f $0))

# Add the environment variable to /etc/environment
echo "export TEXT_METRICS_WRAPPER_DIR='$TEXT_METRICS_WRAPPER_DIR'" | sudo tee -a /etc/environment

# Reload the environment variables
source /etc/environment

# Go to the correct folder
cd $TEXT_METRICS_WRAPPER_DIR

##############
### BLEURT ###
##############

# Install Google language
git clone https://github.com/google-research/bleurt.git
cd bleurt
git checkout cebe7e6
pip install .
wget https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip .
unzip BLEURT-20.zip
cd ..


###########
### GEM ###
###########

# Install heavy requirements for GEM
pip install bert_score==0.3.13
pip install pyemd==1.0.0

# Get the MoverScore requirements
git clone https://github.com/LucaDeGrandis/moverscore_modified.git
cd moverscore_modified
pip install .
cd ..


############
### NLTK ###
############

# Install nltk package from github
git clone https://github.com/nltk/nltk.git
cd nltk
git checkout e2d368e
pip install .
cd ..
