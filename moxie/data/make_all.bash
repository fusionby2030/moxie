#!/usr/bin/env bash

mkdir -p ../../data/raw && mkdir -p ../../data/processed && python Gather_From_JET.py && python Create_Processed_Dataset.py && python Create_Train_Val_Test_set.py
