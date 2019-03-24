#!/bin/bash

python3 ./learning/model/keras_model_resave.py

model_folders="./learning/model/saved_models/*/";
for model_folder in $model_folders; do
    mmconvert -sf keras -iw "$model_folder/model.h5" -in "$model_folder/model.json" -df pytorch -om "$model_folder/model.pth";
    echo "$model_folder was convert to PyTorch"; done

