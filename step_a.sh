#!/bin/bash

echo "Running step A -- stock price update and preprocess and AE Feature exporting"
python data/stockPriceUpdate.py 2>&1 | ts "[%Y-%m-%d %H:%M:%S]" >> data/stockPriceUpdate.log
python AE/AE_preprocess.py 2>&1 | ts "[%Y-%m-%d %H:%M:%S]" >> AE/AE_preprocess.log
python AE/exportFeature.py --if_AEtrain "${ARGUMENT1}" 2>&1 | ts "[%Y-%m-%d %H:%M:%S]" >> AE/exportFeature.log

