import openai
import random
import json
import csv
import time
import numpy as np
import os
import argparse
from utils import *
from tqdm.contrib.concurrent import process_map
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

def validate_new_keywords(criterion, add_keywords, llm):
    validation_prompt = load_prompt_from_file("step3_criteria_refinement/prompt/3c_validation.txt")
            
    # Format the validation prompt with actual values
    Validation_Prompt = validation_prompt.format(criteria=criterion, new_attribute=add_keywords)
    
    # Check if the new attribute fits into the criteria
    INDI = get_completion(Validation_Prompt, llm)
    INDI = extraction(INDI)

    return INDI

def refine(dimension, features, test_results, sample, llm, num_suggestions):
    # Create the criteria string
    criteria = dimension + ": " + ", ".join(features)
    test_results = ", ".join(test_results)
    
    # Load the refinement prompt from the file
    refinement_prompt = load_prompt_from_file("step3_criteria_refinement/prompt/3b_refinement.txt")
    
    # Format the prompt with actual values
    Refinement_Prompt = refinement_prompt.format(sample=sample, criteria=criteria, test_results=test_results)

    # Generate the refined criteria
    Refinement_Result = get_completion(Refinement_Prompt, llm).replace("'", '"')
    print(f"Refinement_Result: {Refinement_Result}")
    Refinement_Result = json.loads(Refinement_Result)

    for failure_pattern, improved_content in Refinement_Result.items():
        if failure_pattern.lower() == "hallucination" and improved_content == []:
            print(f"failure_pattern: {failure_pattern}")
        
        elif failure_pattern.lower() == "hard_case" and improved_content == []:
            print(f"failure_pattern: {failure_pattern}")
            
        elif failure_pattern.lower() == "redundant":
            print(f"failure_pattern: {failure_pattern}, test_results: {test_results}, redundant_results: {test_results}")

            add_keywords = improved_content[0].replace("_", " ")
            criteria = dimension + ": " + ", ".join(features)
            INDI = validate_new_keywords(criteria, add_keywords, llm)  # only suggest one new keyword
            
            if INDI == "Yes":
                features.append(add_keywords)
                print(add_keywords + " is added to the criteria")
            else:
                print(f"After evaluation, new keyword {add_keywords} is not reasonable enough, so is not added to the criteria")
            
        elif failure_pattern.lower() == "missing":
            add_keywords = improved_content[0].replace("_", " ")
            print(f"failure_pattern: {failure_pattern}, test_results: {test_results}, missing_content: {add_keywords}")
                # Load the validation prompt from the file
            
            criteria = dimension + ": " + ", ".join(features)
            INDI = validate_new_keywords(criteria, add_keywords, llm)
            
            if INDI == "Yes":
                features.append(add_keywords)
                print(add_keywords + " is added to the criteria")
            else:
                print(f"After evaluation, new keyword {add_keywords} is not reasonable enough, so is not added to the criteria")

        else:
            print(f"failure_pattern: {failure_pattern} that cannot be handled")
    
    return features

def process_dimensions(args):
    sample_1, dimension, features, llm, num_suggestions, num_testing = args
    
    # Load the testing prompt from the file
    testing_prompt = load_prompt_from_file("step3_criteria_refinement/prompt/3a_testing.txt")
    
    # Format the prompt with actual values
    Testing_Prompt = testing_prompt.format(dimension=dimension, sample=sample_1, features=",".join(features))
    
    test_results = []
    for j in range(num_testing):
        # Get classification from the model
        Result = get_completion(Testing_Prompt, llm)
        test_results.append(extraction(Result))
    
    print(f"dimension: {dimension}, test_result: {test_results}")
    # Check if all classification results are consistent
    if test_results.count(test_results[0]) == len(test_results):
        print("All classification results are consistent")
        # If consistent, check if the result is in the features
        if test_results[0] in features:
            pass
        else:
            features.append(test_results[0])
    else:
        print("Classification results are inconsistent")
        # If inconsistent, refine the features
        features = refine(dimension=dimension, features=features, test_results=test_results, sample=sample_1, llm=llm, num_suggestions=num_suggestions)
    
    return features

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
        if task_name in ["slice_discovery", "subpopulation_shift"]:
            output_root = os.path.join(args.output_root, dataset_cfg["dataset_name"])
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
        num_class = dataset_cfg["num_class"]

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

        ############# Testing and Refining the Criteria ############
        # Load the Criteria
        Dimensions = []
        Features = []
        unrefined_path = os.path.join(output_dir, '%s_criteria_unrefined.json' % main_subject)
        json_obj = json.load(open(unrefined_path, 'rb'))
        for dim, feats in json_obj.items():
            Dimensions.append(dim)
            Features.append(feats)
        
        # Start to Refine
        for iteration in range(num_refining_rounds):
            print("\nTesting and Refining Stage. Total Iteration: %d." % iteration)
            
            # Sampling
            sample_1 = random.sample(captions, 1)[0]
            print(sample_1)
            
            n_dim = len(Dimensions)
            Features = process_map(process_dimensions, list(zip(n_dim * [sample_1], Dimensions, Features, 
                                                                n_dim * [args.llm], n_dim * [num_suggestions], n_dim * [num_testing])), 
                                                                max_workers=n_dim)
            
            # Saving the results of this round
            result_obj = {}
            for idx, dimension in enumerate(Dimensions):
                result_obj[Dimensions[idx]] = Features[idx]
        
        # final post processing
        for dimension, features in result_obj.items():
            features = ', '.join(features)  # list -> str
            features = features.strip()
            summarize_attributes_prompt = load_prompt_from_file("step2_criteria_initialization/prompt/2.3b_summarize_attributes.txt")
            Summarize_Attributes_Prompt = summarize_attributes_prompt.format(dimension=dimension, Suggestions=features)
            if num_class != 0:  # There is another prompt here, handle it later
                Summarize_Attributes_Prompt += f" I require that the number of final summarized attributes should be equal to {num_class} strictly!!"
            
            Summarize_Attributes = get_completion(Summarize_Attributes_Prompt, args.llm)
            Summarize_Attributes = extract_dimensions(Summarize_Attributes)
            features = Summarize_Attributes.lower().strip().replace(".", "")
            result_obj[dimension] = features.split(", ")

        json.dump(result_obj, open(os.path.join(output_dir, '%s_criteria.json' % main_subject), "w"), indent=4)

if __name__ == "__main__":
    main()