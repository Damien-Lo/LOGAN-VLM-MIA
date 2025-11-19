import numpy as np 
import pandas as pd
import json
import random
from pathlib import Path
from datasets import load_dataset, Dataset
from datasets import Features, Value 
import os

'''
get_random_subset()
Given a JSON file of form [{'image': <img_path>, 'label': <label>},...], get a random subset of num_samples
    @param json_path: JSON file with format: [{'image': <img_path>, 'label': <label>},...]
    @param out_path: output directory for JSON subset of same format
    @param num_samples: number of random samples
'''
def get_random_subset(data_path, num_samples, out_path=None):
    with open(data_path, "r") as f:
        data = json.load(f)
        

    def random_int_list(m: int, n: int):
        if m > n + 1:
            raise ValueError("m cannot exceed the size of the range (n+1).")
        return random.sample(range(n + 1), m)

    desired_idxs = random_int_list(num_samples, len(data))
    result = list()
    for idx, sample in enumerate(data):
        if idx in desired_idxs:
            result.append(sample)
    if out_path != None:
        with Path(out_path).open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, sort_keys=True)
    return result
        
        
'''
json_to_dataset
Given a json file of [{'image': <img_path>, 'label': <label>},...], convert into a dataset object (from huggyface) with image being "image path" objects
    @param json_path: JSON file with format: [{'image': <img_path>, 'label': <label>},...]
    @param out_path: output directory for JSON subset of same format if none not saved locally
Return: dataset object
'''
def json_to_dataset(json_path, out_path=None):

    ds = load_dataset("json", data_files=json_path)["train"]

    features = Features({
        "image": Value("string"),
        "label": Value("int64"),
    })
    ds = ds.cast(features)
    
    if out_path != None:
        ds.save_to_disk(out_path)
        
        data_info_path = os.path.join(out_path, "dataset_info.json")
        state_path = os.path.join(out_path, "state.json")
        
        os.remove(data_info_path)
        os.remove(state_path)
        
    return ds
        

'''
get_class_subset
Given a json file with {idxs: [], data: []} and corresponding class labels, return the json file in the same form with the desired class labels
'''
def get_class_subset(json_path, label_path, desired_label, out_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    with open(label_path, "r") as f:
        labels = json.load(f)
    if (len(data['idxs']) != len(data['sentences'])):
        print("Label and data lengths do not match")
        return

    df = pd.DataFrame(data)
    df['label'] = labels
    desired = df[df['label'] == desired_label].drop('label', axis=1)
    
    def dump_cols_as_lists(df: pd.DataFrame, path: str):
        df = df.copy()
        for c in df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns:
            df[c] = df[c].astype("datetime64[ns]").dt.strftime("%Y-%m-%dT%H:%M:%S")
        df = df.replace({np.nan: None})
        data = df.to_dict(orient="list")
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
            
    dump_cols_as_lists(desired, out_path)
    
    
def build_mixed_target_set(size_of_set, member_data_path, nonmember_data_path, ratio, out_path):
    
    with open(member_data_path, "r") as f:
        member_data = json.load(f)
    with open(nonmember_data_path, "r") as f:
        nonmember_data = json.load(f)
    
    def random_int_list(m: int, n: int):
        if m > n + 1:
            raise ValueError("m cannot exceed the size of the range (n+1).")
        return random.sample(range(n), m)

    # for ratio in ratios:
    num_of_mem = int(300 * ratio)
    num_of_nonmem = 300 - num_of_mem
    
    mixed_target_list = list()
    true_val_labels = list()
    
    mem_idxs = random_int_list(num_of_mem, len(member_data))
    nonmem_idxs = random_int_list(num_of_nonmem, len(nonmember_data))
    
    for idx in mem_idxs:
        mixed_target_list.append(member_data[idx])
        true_val_labels.append(member_data[idx])
        
    for idx in nonmem_idxs:
        mixed_target_list.append({'image': nonmember_data[idx]['image'], 'label': 1})
        true_val_labels.append({'image': nonmember_data[idx]['image'], 'label': 0})
        
    with open(os.path.join(out_path,f"mixed_subset_memrat_{ratio}_test.json"), "w") as f:
        json.dump(mixed_target_list, f, indent=4)
    with open(os.path.join(out_path,f"mixed_subset_memrat_{ratio}_val.json"), "w") as f:
        json.dump(true_val_labels, f, indent=4)


def main():
    print("Running")
    


if __name__ == "__main__":
    main()
