import csv
import json
from tqdm import tqdm
import argparse
import yaml
from utils import load_yaml
        
def default_argument_parser():
    parser = argparse.ArgumentParser(description="suggest-dimensions")
    parser.add_argument(
        "--dataset-config", default="", help="dataset config")
    parser.add_argument(
        "--mllm-name", default="", help="main subject")
    return parser

def main():
    args = default_argument_parser().parse_args()
    dataset_cfg_file_path = args.dataset_config
    dataset_cfg = load_yaml(dataset_cfg_file_path)
    
    ############# Task ############
    task_name = dataset_cfg["task_name"]
    if task_name == "ictc":
        main_subject_list = [dataset_cfg["dataset_name"]]
    elif task_name in ["slice_discovery", "subpopulation_shift"]:
        main_subject_list = dataset_cfg["class_list"]
    for main_subject in main_subject_list:
        csv_file_path = dataset_cfg["caption_csv"].format(mllm_name=args.mllm_name, class_name=main_subject)

        class_name = main_subject  # "dimension"
        csv_name = csv_file_path.split("/")[-1]
        json_file_path = csv_file_path.replace(csv_name, f"{class_name}.json")

        def convert_csv_to_json(class_name, csv_file_path, json_file_path):
            data = {class_name: {}}
            
            with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
                csv_reader = csv.DictReader(csv_file)
                for row in tqdm(csv_reader):
                    second_id = row[class_name]
                    img_name = row['img_name'].replace("/", "-")
                    caption = row['caption']
                    
                    if second_id not in data[class_name]:
                        data[class_name][second_id] = {}
                    data[class_name][second_id][img_name] = caption
            
            with open(json_file_path, 'w', encoding='utf-8') as json_file:
                json.dump(data, json_file, ensure_ascii=False, indent=2)

        convert_csv_to_json(class_name, csv_file_path, json_file_path)

if __name__ == "__main__":
    main()