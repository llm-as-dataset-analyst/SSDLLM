import random
from re import I
import numpy as np
import csv
from utils import *
import argparse
import os
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

def main():
    # random.seed(42)
    args = default_argument_parser().parse_args()
    
    # Load prompts from separate text files
    get_dimension_prompt_file = "step2_criteria_initialization/prompt/2.2a_get_dimension.txt"
    summarize_dimension_prompt_file = "step2_criteria_initialization/prompt/2.2b_summarize_dimension.txt"
    
    get_dimension_prompt = load_prompt_from_file(get_dimension_prompt_file)
    summarize_dimension_prompt = load_prompt_from_file(summarize_dimension_prompt_file)

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
        captions = get_captions(main_subject, captions_root)
        print("Dataset is loaded. # of samples = %d" % len(captions))

        num_sample_rounds_1 = summary_cfg["num_sample_rounds_1"]
        num_samples_each_round_1 = summary_cfg["num_samples_each_round_1"]
        thresh = summary_cfg["ci_thresh_1"]
        num_sample_rounds_2 = summary_cfg["num_sample_rounds_2"]
        num_samples_each_round_2 = summary_cfg["num_samples_each_round_2"]
        num_refining_rounds = summary_cfg["num_refining_rounds"]
        num_testing = summary_cfg["num_testing"]
        num_suggestions = summary_cfg["num_suggestions"]

        num_of_selection_threshold = int(num_sample_rounds_1 * thresh)  # May need to adjust the threshold later, but not in most cases

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
        # Create directory, equivalent to `mkdir -p`, will create parent directories if they do not exist
        os.makedirs(output_dir, exist_ok=True)
        cfg_dir = os.path.join(output_dir, "cfg")
        os.makedirs(cfg_dir, exist_ok=True)

        ######### Save Dataset Config and Summary Config ####################
        # save_dict_to_yaml(dataset_cfg, output_dir, args.dataset_config.split("/")[-1])
        # save_dict_to_yaml(summary_cfg, output_dir, args.summary_config.split("/")[-1])
        copy_file(args.dataset_config, cfg_dir)
        copy_file(args.summary_config, cfg_dir)

        ############# Initialization of the Criteria ############
        Dimensions = ''
        for i in range(num_sample_rounds_1):
            caption_samples_all = random.sample(captions, num_samples_each_round_1)
            caption_samples = ""
            for sample in caption_samples_all:
                caption_samples += f'[{sample}]'
            
            # Build the prompt for dimensions
            if task_name == "ictc":
                Get_Dimension_Prompt = get_dimension_prompt.format(batch_size=num_samples_each_round_1, main_subject="dimensions", caption_samples=caption_samples)
            else:
                Get_Dimension_Prompt = get_dimension_prompt.format(batch_size=num_samples_each_round_1, main_subject=main_subject, caption_samples=caption_samples)
            
            print(f"Finding the Dimensions. Round: {i + 1} / {num_sample_rounds_1}. Getting Responses from GPT...")
            Dimension_Response = get_completion(Get_Dimension_Prompt, args.llm)
            
            if Dimension_Response is None:
                print("Error in openai response... Continue")
                continue
            
            # Extract out the dimensions suggested
            Suggestions = extract_dimensions(Dimension_Response)
            Suggestions = Suggestions.lower()
            print("Dimensions Suggested:", Suggestions, '\n')
            Dimensions += Suggestions + ','
        
        print("The Dimensions Suggested:", Dimensions, 'Length:', len(Dimensions.split(",")))

        np.savetxt(os.path.join(output_dir, main_subject + '_suggested_dimensions.csv'), [Dimensions], fmt='%s')

        with open(os.path.join(output_dir, '%s_suggested_dimensions.csv' % main_subject), 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                Dimensions = ','.join(row)
        
        Dimensions = Dimensions.split(',')
        Dimensions = get_elements_over_count(Dimensions, num_of_selection_threshold)  
        Dimensions = ','.join(Dimensions)
        print("Here are the dimensions suggested after majority counting:", Dimensions)

        # Summarize dimensions using GPT
        Summarize_Dimension_Prompt = summarize_dimension_prompt.format(dimensions=Dimensions)
        Summarized_Dimensions = get_completion(Summarize_Dimension_Prompt, args.llm)
        Summarized_Dimensions = extract_dimensions(Summarized_Dimensions)
        Summarized_Dimensions = Summarized_Dimensions.lower()

        print("Unsummarized:", Dimensions)
        print("Summarized:", Summarized_Dimensions)

        np.savetxt(os.path.join(output_dir, main_subject + '_selected_dimensions.csv'), [Dimensions], fmt='%s')
        np.savetxt(os.path.join(output_dir, main_subject + '_selected_summarized_dimensions.csv'), [Summarized_Dimensions], fmt='%s')

        if task_name == "ictc":
            Summarized_Dimensions_list = Summarized_Dimensions.split(", ")
            if main_subject in ["cifar10", "stl10", "cifar100"]:
                Expected_list = ["object"]
            else:
                Expected_list = [main_subject]

            most_idx_list, second_idx_list = find_most_similar_indices(Expected_list, Summarized_Dimensions_list)
            for topic, most_idx in zip(Expected_list, most_idx_list):
                print(f"{topic}-{Summarized_Dimensions_list[most_idx]}")
                final_topic = Summarized_Dimensions_list[most_idx]
                np.savetxt(os.path.join(output_dir, main_subject + '_selected_summarized_final_dimensions.csv'), [final_topic], fmt='%s')
        elif task_name in ["slice_discovery", "subpopulation_shift"]:
            np.savetxt(os.path.join(output_dir, main_subject + '_selected_summarized_final_dimensions.csv'), [Summarized_Dimensions], fmt='%s')  # Unified format


if __name__ == "__main__":
    main()