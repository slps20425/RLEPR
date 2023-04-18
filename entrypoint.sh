#!/bin/bash

# Set default values for ARGUMENT1 and ARGUMENT2
ARGUMENT1="$1"
ARGUMENT2="$2"

# Run step B
./step_b.sh &

# Run step_c.sh the first time
./step_c.sh &

# Monitor data/latest_test_live_update.pkl for changes
while inotifywait -e modify,create,delete data/latest_test_live_update.pkl; do
  # Re-run step_c.sh whenever data/latest_test_live_update.pkl is updated or modified
  ./step_c.sh &
done

# Check if it is Sunday at 8:00 am, and if so, run step A
while true; do
  current_day=$(date +%u)
  current_hour=$(date +%H)
  current_minute=$(date +%M)

  if [[ $current_day == 7 ]] && [[ $current_hour == 08 ]] && [[ $current_minute == 00 ]]; then
    ./step_a.sh
  fi

  # Sleep for a minute before checking again
  sleep 60
done