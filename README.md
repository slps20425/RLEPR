# RLEPR project

# Entrypoint Script
The entrypoint script, entrypoint.sh, performs the following actions:
Sets default values for if_AETrain{ARGUMENT1} and if_RLTrain{ARGUMENT2}.

# Runs step_a.sh every Sunday at 8:00 am.
-- run /data/stockPriceUpdate.py to retrieve latest stock price
-- run /AE/AE_preprocess.py to generate technical indicators
-- run /AE/exportFeature.py by using {best_model.h5} to generate below 

# Encoding history stock price
1. /data/latest_composite_lstm_45tic_30d_25F@5F_eleganRl.pkl 
# Trend by AE LSTM AutoEncoder predictor
2. /data/latest_composite_lstm_45tic_30d_25F@5F_trend_eleganRl.pkl


# Runs step_b.sh in the background.
this is the main entry for RL
-- run elegantrl_test.py 

# Runs step_c.sh in the background.
-- simply for web displaying
Monitors data/latest_test_live_update.pkl for changes and re-runs step_c.sh whenever the file is updated or modified.

Usage
To run the project, you need to build the Docker image and run a Docker container using the image. The ARGUMENT1 and ARGUMENT2 values can be passed as arguments to the docker run command.

For example, to run the project with ARGUMENT1=True and ARGUMENT2=False, you would run the following command:
docker build -t <image name> .
docker run --rm <image name> True False
or pull down latest from docker hub
pull down the latest container image
docker pull slps20425/rlepr:latest

Logs
The output of each script is logged to a file in the data directory. The logs can be used to monitor the progress of the scripts and troubleshoot any issues.

