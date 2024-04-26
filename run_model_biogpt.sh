#!/usr/bin/bash
str1="train"
str2="test"
if [ $1 == $str1 ]; then
python3 biogpt_train.py $2 $3
else
python3 biogpt_infer.py $2 $3 $4
python3 clean.py $4
fi