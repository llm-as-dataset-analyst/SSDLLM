# pip install accelerate
import imp
import requests
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
import os
import json
import argparse
import math
from tqdm import tqdm
import pandas as pd
import torch

def get_df_chunk(df, n, k):
    # Assume df is your DataFrame, and you've already read the CSV file using pd.read_csv
    # Get the total number of rows
    total_rows = len(df)
    # Calculate the size of each chunk, ensuring that each chunk has as equal number of rows as possible
    chunk_size = math.ceil(total_rows / n)
    # Calculate the start and end row numbers for the current chunk
    start_index = k * chunk_size
    end_index = min((k + 1) * chunk_size, total_rows)

    # Use Pandas' iloc function to get the rows within the specified range of row numbers
    chunk_df = df.iloc[start_index:end_index].reset_index(drop=True)

    return chunk_df, start_index


def eval_model(args):
    model_path = args.model_path  # "llava-hf/llava-1.5-7b-hf"
    model_name = args.model_name
    model = LlavaForConditionalGeneration.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    dataset_dir = os.path.join(args.dataset_root, args.dataset_name)
    ds_name = args.dataset_name
    label_index_mapping_json_path = os.path.join(dataset_dir, "dg_label_id_mapping.json")  # for nicopp, waterbirds, metashift
    with open(label_index_mapping_json_path, "r") as f:
        label_index_mapping = json.load(f)
    label_index = {v : k for k, v in label_index_mapping.items()}  # class_name: class_index

    if ds_name == "nicopp":
        csv_path = os.path.join(dataset_dir, f"metadata_0.1.csv")
        csv_save_path = os.path.join(dataset_dir, f"split/{model_name}/metadata_caption_{model_name}_0.1_{args.chunk_idx}.csv")
    else:
        csv_path = os.path.join(dataset_dir, f"metadata_{ds_name}.csv")
        csv_save_path = os.path.join(dataset_dir, f"split/{model_name}/metadata_{ds_name}_caption_{model_name}_{args.chunk_idx}.csv")
    df_origin = pd.read_csv(csv_path)
    print(args.num_chunks, args.chunk_idx)
    df, start_index = get_df_chunk(df_origin, args.num_chunks, args.chunk_idx)
    new_col_list = [0] * len(df["filename"])

    for idx, filename in enumerate(tqdm(df["filename"])):
        class_name = label_index[int(df['y'][idx])]
        question = args.query.format(class_name=class_name)  # Please describe the image in detail with the keyword {class_name}
        
        if ds_name == "waterbirds":
            img_path = os.path.join(dataset_dir, "waterbird_complete95_forest2water2", filename)
        else:
            img_path = filename

        raw_image = Image.open(img_path).convert('RGB')

        inputs = processor(images=raw_image, text=question, return_tensors="pt").to(device)

        # autoregressively complete prompt
        output = model.generate(**inputs, max_new_tokens=512)
        generated_text = processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        generated_text = generated_text.replace("\n", "")
        print(generated_text)

        new_col_list[idx] = generated_text

        df["caption"] = new_col_list
            
        df.to_csv(csv_save_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--num-chunks", type=int, default=1)
    
    args = parser.parse_args()

    eval_model(args)