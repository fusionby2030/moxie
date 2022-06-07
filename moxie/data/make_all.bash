#!/usr/bin/env bash
python Gather_From_JET.py && python Create_Processed_Dataset.py && python Create_Train_Val_Test_set.py