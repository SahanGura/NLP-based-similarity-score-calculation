#!/bin/sh

#INSTRUCTIONS
#!bash -i /content/sample_data/run.sh /content/sample_data/data/data.json /content/sample_data/data/top.json 6

input_file=$1
output_file=$2
n_skill=$3

echo "Input dataset path: $input_file";
echo "Output file path: $output_file";
echo "Number of Skills: $n_skill";

python ./scripts/embeddings.py $input_file $output_file $n_skills
#!python /content/sample_data/scripts/embeddings.py /content/sample_data/data/data.json /content/sample_data/data/top.json 6