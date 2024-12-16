import openai
import random
import json
import csv
import time
import numpy as np
import os
import argparse
from utils import *
import tqdm
from tqdm.contrib.concurrent import process_map
from multiprocessing import Pool
import random

num_testing = 1
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

def identity_leaf(args):
    caption, features, dimension, llm = args
    
    # Load the prompt from the file
    identity_leaf_prompt = load_prompt_from_file("step4_image_assignment/prompt/4_identity_leaf.txt")
    
    # Format the prompt with actual values
    Testing_Prompt = identity_leaf_prompt.format(caption=caption, features=",".join(features), dimension=dimension)
    
    classification_results = []
    
    for j in range(num_testing):
        while True:
            Result = get_completion(Testing_Prompt, llm)
            if Result is not None:
                break
        classification_results.append(extraction(Result))
    
    # Vote out the best one
    freqs = {x: classification_results.count(x) for x in set(classification_results)}
    assignment = max(freqs, key=freqs.get)
    
    return assignment

def process_caption(args):
    caption_obj, Dimensions, Features, out_path, main_subject, llm = args
    n_path = os.path.join(out_path, caption_obj['n_label'])
    cap_file = os.path.join(n_path, caption_obj['img_name']+'.txt')

    if os.path.exists(cap_file):
        return

    caption = caption_obj['caption']
    # print(sample_1)
    n_dim = len(Dimensions)
    with Pool(n_dim) as p:
        assignments = p.map(identity_leaf, list(zip([caption]*n_dim, Features, Dimensions, n_dim*[llm])))
    for i in range(n_dim):
        caption_obj['attr'][Dimensions[i]] = assignments[i]
    
    if not os.path.exists(n_path):
        os.makedirs(n_path)

    with open(cap_file,"w") as f:
        json.dump(caption_obj, f)
    print(caption_obj)
    return caption_obj


def main():
    # random.seed(42)
    # print("WARNING: RUNNING UNREFINED VERSION")
    args = default_argument_parser().parse_args()
    
    ############# Config ###############
    dataset_cfg = load_yaml(args.dataset_config)
    summary_cfg = load_yaml(args.summary_config)

    ############# Task ############
    task_name = dataset_cfg["task_name"]

    ############# Dataset ############
    dataset_name = dataset_cfg["dataset_name"]
    if task_name == "ictc":
        main_subject_list = [dataset_name]
    elif task_name in ["slice_discovery", "subpopulation_shift"]:
        main_subject_list = dataset_cfg["class_list"]
    for main_subject in main_subject_list:
        captions_root = os.path.dirname(dataset_cfg["caption_csv"].format(mllm_name=args.mllm, class_name=main_subject))
        if task_name in ["slice_discovery", "subpopulation_shift"]:
            output_root = os.path.join(args.output_root, dataset_name)
        else:
            output_root = args.output_root
        captions = get_captions(main_subject, captions_root)
        print("Dataset is loaded. # of samples = %d" % len(captions))

        num_sample_rounds_1 = summary_cfg["num_sample_rounds_1"]
        num_samples_each_round_1 = summary_cfg["num_samples_each_round_1"]
        num_sample_rounds_2 = summary_cfg["num_sample_rounds_2"]
        num_samples_each_round_2 = summary_cfg["num_samples_each_round_2"]

        num_refining_rounds = summary_cfg["num_refining_rounds"]
        num_testing = summary_cfg["num_testing"]
        num_suggestions = summary_cfg["num_suggestions"]

        ######## Create Output Dir  ####################
        # Construct output_dir path
        output_dir = os.path.join(
                        output_root, 
                        main_subject, 
                        args.llm, 
                        args.mllm,
                        f"dim-{num_sample_rounds_1}-{num_samples_each_round_1}_"
                        f"attribute-{num_sample_rounds_2}-{num_samples_each_round_2}_"
                        f"refine-{num_refining_rounds}-{num_testing}-{num_suggestions}"
                    )


        with open(os.path.join(captions_root, main_subject+".json"), "r") as read_file:
            captions_obj = json.load(read_file)
            captions_obj = captions_obj[main_subject]
        captions = []
        for n_label, captions_obj in captions_obj.items():
            for img_name, caption in captions_obj.items():
                captions.append({"n_label": n_label, "img_name": img_name, "caption": caption,"attr": {}})

        ############# Testing and Refining the Criteria ############
        # Load the Criteria
        Dimensions = []
        Features = []
        criteria_path = os.path.join(output_dir, '%s_criteria.json'%main_subject)
        json_obj = json.load(open(criteria_path))
        for dim,feats in json_obj.items():
            Dimensions.append(dim)
            Features.append(feats)

        # Start to Assign
        n = len(captions)
        # n_selected = round(n*0.1)
        n_selected = n
        n_dim = len(Dimensions)

        print("Found %d captions, selected %d captions, found %d dimension"%(n, n_selected, n_dim))

        random.shuffle(captions)
        
        captions = captions[:n_selected]

        results = []
        for i, caption in enumerate(captions):
            print(f"Assigning [{main_subject}] | [{dataset_name}]... Round: {i+1} / {n}. Getting Responses from GPT...")
            result = process_caption((caption, Dimensions, Features, os.path.join(os.path.join(output_dir, "assign")), main_subject, args.llm))
            results.append(result)
            print("\n")

        print(results)


if __name__ == "__main__":
    main()