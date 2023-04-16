#!/bin/bash

echo "Running step B -- RL evaluation"
python RL/elegantrl_test.py --if_RLtrain "${ARGUMENT2}" --encoder_dataPath ./data/latest_composite_lstm_45tic_30d_25F@5F_eleganRl.pkl --data_trend_path ./data/latest_composite_lstm_45tic_30d_25F@5F_trend_eleganRl.pkl --latest_priceBook_path ./data/latest_45tic_priceBook.pkl 2>&1 | ts "[%Y-%m-%d %H:%M:%S]" >> RL/output.log


