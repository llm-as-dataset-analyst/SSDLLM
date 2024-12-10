mllm_name=llava1.5-7b
llm_name=gpt-3.5-turbo  # gpt-4, gpt-3.5-turbo

output_dir=output
summary_config=config/0_summary.yaml

for class_name in mood
do
    dataset_config=config/${class_name}.yaml

    echo -e "\n #### Step2.1 Preprocess [${class_name}] Data #####"
    bash scripts/step2.1.sh ${dataset_config} ${mllm_name}

    echo -e "\n #### Step2.2 Suggesting Dimension #####"
    bash scripts/step2.2.sh ${dataset_config} ${summary_config} ${mllm_name} ${llm_name} ${output_dir}

    echo -e "\n #### Step2.3 Initializing Criterias #####"
    bash scripts/step2.3.sh ${dataset_config} ${summary_config} ${mllm_name} ${llm_name} ${output_dir}

    echo -e "\n #### Step3 Refining Criterias #####"
    bash scripts/step3.sh ${dataset_config} ${summary_config} ${mllm_name} ${llm_name} ${output_dir}

    echo -e "\n #### Step4 Assigning #####"
    bash scripts/step4.sh ${dataset_config} ${summary_config} ${mllm_name} ${llm_name} ${output_dir}
done