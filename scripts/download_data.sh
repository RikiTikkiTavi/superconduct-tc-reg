#!/bin/bash

# Check if the argument (data dir) is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <data_dir>"
  exit 1
fi

# Assign the first argument to the 'data_dir' variable
data_dir="$1"

# Download the data.zip file
curl -L -o data.zip https://archive.ics.uci.edu/static/public/464/superconductivty+data.zip

# Create the necessary directory structure
mkdir -p "${data_dir}/raw/superconductivity_data"

# Unzip the data.zip file into the specified directory
unzip data.zip -d "${data_dir}/raw/superconductivity_data"

# Remove the zip file after extraction
rm data.zip

echo "Data downloaded and extracted to ${data_dir}/raw/superconductivity_data"