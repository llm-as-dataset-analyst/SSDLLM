import random
import openai
import json
import numpy as np
import os
import time
from openai import OpenAI
import httpx
import yaml
import shutil

def get_client(model="gpt-3.5-turbo"):
    if model == "gpt-3.5-turbo":
        client = OpenAI(
            base_url="https://api.xty.app/v1", 
            api_key="",
            http_client=httpx.Client(
                base_url="https://api.xty.app/v1",
                follow_redirects=True,
            ),
        )
    elif model == "gpt-4":
        client = OpenAI(
            base_url="https://api.xty.app/v1", 
            api_key="",
            http_client=httpx.Client(
                base_url="https://api.xty.app/v1",
                follow_redirects=True,
            ),
        )

    return client

def get_captions(main_subject, captions_root, return_attribute=False):
    json_path = os.path.join(captions_root, main_subject+".json")
    print("\nLoaded from %s"%json_path)
    with open(json_path, "r") as read_file:
        raw = json.load(read_file)
        raw = raw[main_subject]
    captions = []
    for _, captions_obj in raw.items():
        for caption in captions_obj.values():
            captions.append(caption)
    attributes = raw.keys()
    if return_attribute:
        return captions, attributes
    else:
        return captions

# def get_completion(prompt, model="gpt-4"):
def get_completion(prompt, model="gpt-3.5-turbo"):
    client = get_client(model) 
    try:
        messages = [{"role": "user", "content": prompt}]
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0
        )
        result = response.choices[0].message.content
        return result
    except Exception as e: 
        print(e)
        return None


def extraction(completion):
    completion = completion.split('{')[-1]
    completion = completion.split('}')[0]
    return completion


def extract_dimensions(completion):
    completion = completion.split('{')[-1]
    completion = completion.split('}')[0]
    completion = completion.split(':')[-1]
    completion = completion.split('[')[-1]
    completion = completion.split(']')[0]
    completion = completion.replace("'",'')
    return completion



def extract_set(criteria):
    set = criteria.split('[')[-1]
    set = set.split(']')[0]
    return set


def extract_criteria(set):
    criteria = ''
    set = set.split(',')
    while len(set) > 0:
        point = set[0]
        if set.count(point) >= 12:
            criteria += point+','
        set = [i for i in set if i!=point]
    return criteria

def get_elements_over_count(lst, n):
    result = []
    counts = {}

    for elem in lst:
        if elem not in counts:
            counts[elem] = 0
        counts[elem] += 1

    for elem, count in counts.items():
        if count > n:
            result.append(elem)

    return result


def extract_feature(set):
    features = []
    set = set.split(':')[-1]
    set = set.split(',')
    for f in set:
        f = f.replace("'",'')
        features.append(f.strip())
    return features


def merge_dimensions(data, thresh):
    # New dictionary to store the results
    new_dict = {}
    
    # Find all keys to avoid modifying the dictionary during iteration
    keys = list(data.keys())
    n = len(keys)
    
    # Keep track of which dimensions have been merged
    merged = set()
    
    for i in range(n):
        key1 = keys[i]
        if key1 in merged:
            continue  # Skip if this has been merged into another dimension
        
        # Start with the original attributes, using set for easy operations
        current_attrs = set(data[key1])
        
        for j in range(i + 1, n):
            key2 = keys[j]
            if key2 in merged:
                continue  # Skip if already merged
            
            # Calculate intersection
            attrs1 = set(data[key1])
            attrs2 = set(data[key2])
            intersection = attrs1.intersection(attrs2)
            
            # Calculate if intersection is more than 30% of the smaller set
            if len(intersection) / min(len(attrs1), len(attrs2)) > thresh:  # 先不处理合并的事情
                # Merge attributes from key2 to key1
                current_attrs.update(attrs2)
                # Mark key2 as merged
                merged.add(key2)
        
        # Save the updated attributes for key1
        new_dict[key1] = list(current_attrs)
    
    # Add the keys that were not merged at all
    for key in keys:
        if key not in merged and key not in new_dict:
            new_dict[key] = data[key]
    
    return new_dict

def load_yaml(file_path):
    """Load a YAML file and return the content as a Python dictionary."""
    with open(file_path, 'r') as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as exc:
            print(f"Error reading YAML file: {exc}")
            return None

def load_prompt_from_file(filepath):
    """Read a single-line prompt from a text file."""
    with open(filepath, 'r') as file:
        return file.read().strip()

def save_dict_to_yaml(data, output_dir, filename="output.yaml"):
    """
    将一个字典保存到指定路径的 YAML 文件中。
    
    :param data: 要保存的字典
    :param output_dir: 保存文件的目录
    :param filename: 保存文件的名称，默认是 output.yaml
    """
    # 确保目录存在，如果不存在则创建
    os.makedirs(output_dir, exist_ok=True)
    
    # 构建完整的文件路径
    file_path = os.path.join(output_dir, filename)
    
    # 将字典写入 YAML 文件
    with open(file_path, 'w') as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False, allow_unicode=True)
    
    print(f"Config file have been saved in {file_path}")

def copy_file(source_file, destination_dir):
    """
    复制文件到指定目录。如果目录不存在，则创建该目录。

    :param source_file: 源文件的路径
    :param destination_dir: 目标目录
    """
    # 确保目标目录存在，如果不存在则创建
    os.makedirs(destination_dir, exist_ok=True)
    
    # 获取目标文件路径
    destination_file = os.path.join(destination_dir, os.path.basename(source_file))
    
    # 复制文件
    try:
        shutil.copy2(source_file, destination_file)
        print(f"Config file have been saved in : {destination_file}")
    except FileNotFoundError:
        print(f"File Not Found Error: {source_file}")
    except Exception as e:
        print(f"Error when copying file: {e}")