#!/bin/bash

# Set default values for ARGUMENT1 and ARGUMENT2
ARGUMENT1="False"
ARGUMENT2="False"

# Read the arguments
while [ $# -gt 0 ]; do
  case "$1" in
    --if_RLtrain)
      ARGUMENT2="$2"
      shift 2
      ;;
    --if_AEtrain)
      ARGUMENT1="$2"
      shift 2
      ;;
    *)
      printf "Unknown argument: %s\n" "$1" >&2
      exit 1
      ;;
  esac
done


# Run step B
./step_b.sh &

# Run step C in the background
./step_c.sh &

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

