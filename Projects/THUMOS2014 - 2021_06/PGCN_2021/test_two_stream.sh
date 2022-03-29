#!/bin/bash

python eval_detection_results.py sleep results/flow_sleep_result results/rgb_sleep_result --score_weights 1.2 1 --nms_threshold 0.32
