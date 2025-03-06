#!/bin/bash

# Check if the argument (data dir) is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <data_dir>"
  exit 1
fi

# Assign the first argument to the 'data_dir' variable
data_dir="$1"

python -m superconduct_tc_reg.data.process preprocessing=ho_hd_scale_std_cfs dir.data=${data_dir}