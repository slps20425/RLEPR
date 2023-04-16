#!/bin/bash

echo "Running step A -- stock price update and preprocess and AE Feature exporting"
python data/stockPriceUpdate.py
python AE/AE_preprocess.py
python AE/exportFeature.py --if_AEtrain "${ARGUMENT1}"

