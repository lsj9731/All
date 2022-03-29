#! /bin/bash/

python pgcn_test.py sleep $1 sleep_result -j7 | tee -a test_results.txt
python eval_detection_results.py sleep sleep_result --nms_threshold 0.35 | tee -a test_results.txt
