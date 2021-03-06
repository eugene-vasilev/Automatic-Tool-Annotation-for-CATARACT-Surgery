#!/bin/bash

python3 ./model/keras_model_resave.py

model_folders="./model/saved_models/*/";
for model_folder in $model_folders; do
    mmconvert -sf keras -iw "$model_folder/model.h5" -in "$model_folder/model.json" -df onnx -om "$model_folder/model.onnx";
    echo "$model_folder was convert to ONNX"; done
