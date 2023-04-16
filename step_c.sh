#!/bin/bash

echo "Running step C --dash demo"
python dash_demo.py 2>&1 | ts "[%Y-%m-%d %H:%M:%S]" >>web_demo.log 

