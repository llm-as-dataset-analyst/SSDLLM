import random
import openai
import json
import numpy as np
import time
import utils
from utils import *
import os
import argparse
from tqdm.contrib.concurrent import process_map
import csv
from flag_embedding import find_most_similar_indices

def default_argument_parser():
    parser = argparse.ArgumentParser(description="suggest-dimensions")
    parser.add_argument(
        "--dataset-config", default="", help="dataset-config")
    parser.add_argument(
        "--summary-config", default="", help="summary-config")
    parser.add_argument(
        "--llm", default="gpt-3.5-turbo", type=str, help="llm")
    parser.add_argument(
        "--mllm", default="llava1.5-7b", type=str, help="mllm")
    parser.add_argument(
        "--output-root", default="", help="output root")
    return parser

def get_features(dimension, captions, main_subject, llm, num_samples_each_round, i, num_sample_rounds):
    # [num_samples_each_round] Samples of captions
    Caption_Samples = random.sample(captions, num_samples_each_round)
    caption_samples = ""
    for sample in Caption_Samples:
        caption_samples = caption_samples + '[' + sample + ']'
    
    # Load the prompt from the file
    get_features_prompt = load_prompt_from_file("step2_criteria_initialization/prompt/2.3a_get_features.txt")
    
    # Format the prompt with actual values
    Initialization_Prompt = get_features_prompt.format(main_subject=main_subject, dimension=dimension, caption_samples=caption_samples)
    
    # Sending prompt to GPT model
    Initialization_Response = get_completion(Initialization_Prompt, llm)
    if Initialization_Response is None:
        print("Error in openai response... Continue")
        return None
    Suggestions = extract_set(Initialization_Response)
    Suggestions = Suggestions.lower()

    # Summarize the attributes
    summarize_attributes_prompt = load_prompt_from_file("step2_criteria_initialization/prompt/2.3b_summarize_attributes.txt")
    Summarize_Attributes_Prompt = summarize_attributes_prompt.format(dimension=dimension, Suggestions=Suggestions)
    
    Summarize_Attributes = get_completion(Summarize_Attributes_Prompt, llm)
    if Summarize_Attributes is None:
        print("Error in openai response... Continue")
        return None
    Summarize_Attributes = extract_dimensions(Summarize_Attributes)
    Summarize_Attributes = Summarize_Attributes.lower()

    print(f"Finding the Attributes for Dimension [{dimension}]. Round: {i+1} / {num_sample_rounds}. Getting Responses from GPT...")
    print("Unsummarized: ", Suggestions)
    print("Summarized: ", Summarize_Attributes)
    print("\n")
    
    return extract_feature(Summarize_Attributes)

def process_dimensions(args):
    dimension, captions, main_subject, num_class, llm, num_sample_rounds, num_samples_each_round = args
    features = []
    for i in range(num_sample_rounds):
        feature_tmp = get_features(dimension, captions, main_subject, llm, num_samples_each_round, i, num_sample_rounds)
        if feature_tmp is not None:
            features += feature_tmp

    features = ', '.join(features)  # list -> str
    features = features.strip()
    print(f"The Unsummarized Criteria Initialized for {dimension}:", features, '\nLength:', len(features.split(",")), '\n')

    summarize_attributes_prompt = load_prompt_from_file("step2_criteria_initialization/prompt/2.3b_summarize_attributes.txt")
    
    Summarize_Attributes_Prompt = summarize_attributes_prompt.format(dimension=dimension, Suggestions=features)

    if num_class != 0:  # 这边还有一个prompt, 之后去处理一下
        Summarize_Attributes_Prompt += f" I require that the number of final summarized attributes should be equal to {num_class} strictly!!"
    
    Summarize_Attributes = get_completion(Summarize_Attributes_Prompt, llm)
    Summarize_Attributes = extract_dimensions(Summarize_Attributes)
    features = Summarize_Attributes.lower().strip()

    print(main_subject)
    print(f"\nThe Summarized Criteria Initialized for {dimension}: ", features, '\nLength:', len(features.split(",")), '\n')

    features = features.split(', ')

    return {"dimension": dimension, "features": features}

def main():
    # random.seed(42)
    args = default_argument_parser().parse_args()

    ############# Config ###############
    dataset_cfg = load_yaml(args.dataset_config)
    summary_cfg = load_yaml(args.summary_config)

    ############# Task ############
    task_name = dataset_cfg["task_name"]

    ############# Dataset ############
    if task_name == "ictc":
        main_subject_list = [dataset_cfg["dataset_name"]]
    elif task_name in ["slice_discovery", "subpopulation_shift"]:
        main_subject_list = dataset_cfg["class_list"]
    for main_subject in main_subject_list:
        captions_root = os.path.dirname(dataset_cfg["caption_csv"].format(mllm_name=args.mllm, class_name=main_subject))
        output_root = args.output_root
        if task_name in ["slice_discovery", "subpopulation_shift"]:
            output_root = os.path.join(output_root, dataset_cfg["dataset_name"])
        captions, attributes_list = get_captions(main_subject, captions_root, True)
        print("Dataset is loaded. # of samples = %d"%len(captions))

        num_sample_rounds_1 = summary_cfg["num_sample_rounds_1"]
        num_samples_each_round_1 = summary_cfg["num_samples_each_round_1"]
        num_sample_rounds_2 = summary_cfg["num_sample_rounds_2"]
        num_samples_each_round_2 = summary_cfg["num_samples_each_round_2"]
        thresh = summary_cfg["ci_thresh_2"]

        num_refining_rounds = summary_cfg["num_refining_rounds"]
        num_testing = summary_cfg["num_testing"]
        num_suggestions = summary_cfg["num_suggestions"]
        num_class = dataset_cfg["num_class"]

        ######## Create Output Dir  ####################
        # 构建 output_dir 路径
        output_dir = os.path.join(
                        output_root, 
                        main_subject, 
                        args.llm, 
                        args.mllm,
                        f"dim-{num_sample_rounds_1}-{num_samples_each_round_1}_"
                        f"attribute-{num_sample_rounds_2}-{num_samples_each_round_2}_"
                        f"refine-{num_refining_rounds}-{num_testing}-{num_suggestions}"
                    )

        with open(os.path.join(output_dir, '%s_selected_summarized_final_dimensions.csv'%main_subject), 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                Dimensions = ','.join(row)
                print(Dimensions)
        Dimensions = Dimensions.replace(" ", "").split(',')
        print("Initial Criteria:", Dimensions)
        n_dim = len(Dimensions)

        ############# Initialization of the Criteria ############
        records = process_map(process_dimensions, list(zip(Dimensions, n_dim*[captions], n_dim*[main_subject], 
                            n_dim*[num_class], n_dim*[args.llm], 
                            n_dim*[num_sample_rounds_2], n_dim*[num_samples_each_round_2])), max_workers=n_dim)
        result = {}
        for record in records:
            result[record['dimension']] = record['features']
        
        merged_results = merge_dimensions(result, thresh)
        print("Merged Results:", merged_results)
        
        json.dump(merged_results, open(os.path.join(output_dir, '%s_criteria_unrefined.json'%main_subject),"w"), indent=4)
        print("\nAll Results:", result)

        if task_name == "ictc":
            for dim, attr_list in merged_results.items():
                Expected_list = attributes_list
                Expected_list = [expected.replace("_", " ") for expected in Expected_list]
                Summarized_list = attr_list
                most_idx_list, second_idx_list = find_most_similar_indices(Summarized_list, Expected_list)
            
            print(Expected_list)
            for ours_topic, most_idx in zip(Summarized_list, most_idx_list):
                gt_topic = Expected_list[most_idx]
                print(f"{ours_topic} - {gt_topic}")


if __name__ == "__main__":
    main()