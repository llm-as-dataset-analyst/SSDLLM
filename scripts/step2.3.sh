#!/bin/bash
dataset_config=${1}
summary_config=${2}
mllm_name=${3}
llm_name=${4}
output_root=${5}

python3 step2_criteria_initialization/Step2.3_InitializingCriterias.py \
--dataset-config ${dataset_config} \
--summary-config ${summary_config} \
--llm ${llm_name} \
--mllm ${mllm_name} \
--output-root ${output_root} \