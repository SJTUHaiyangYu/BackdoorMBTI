#!/bin/bash

input_dir="../data/hmdb51/rars/"
output_dir="../data/hmdb51/data/"


mkdir -p "$output_dir"

for rar_file in "$input_dir"*.rar; do
    # 检查文件是否存在
    if [ -f "$rar_file" ]; then
        unrar x "$rar_file" "$output_dir"
    fi
done

# mkdir -p ../data/hmdb51/data
# mkdir -p ../data/hmdb51/test_train_splits


# wget -O ../data/hmdb51/hmdb51_org.rar http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar
# unrar x ../data/hmdb51/hmdb51_org.rar ../data/hmdb51/rars/


# wget -O ../data/hmdb51/test_train_splits.rar http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/test_train_splits.rar
# unrar x ../data/hmdb51/test_train_splits.rar ../data/hmdb51/test_train_splits/