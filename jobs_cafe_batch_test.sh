#!/bin/bash

echo "Launching --> CAFE RNN jobs: "


bsub -J debugtest_0 ~/NPS_s/jobs_cafe/cafe_rnn_test.sh
bsub -J debugtest_1 -w "ended(debugtest_0)" ~/NPS_s/jobs_cafe/cafe_rnn_test.sh
bsub -J debugtest_2 -w "ended(debugtest_1)" ~/NPS_s/jobs_cafe/cafe_rnn_test.sh

