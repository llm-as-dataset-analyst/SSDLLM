dataset_config=${1}
mllm_name=${2}

python step2_criteria_initialization/Step2.1_PreprocessData.py --dataset-config ${dataset_config} --mllm-name ${mllm_name}