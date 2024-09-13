import datasets
from glob import glob
import json 
from tqdm import tqdm 

all_data = glob("/home/maverick/openui-translation/data/raw/*/*.json")
dataset = []
for data_path in tqdm(all_data):
    with open(data_path, 'r') as f:
        data = json.load(f)['corpus']
        data = {
            "Korean": data['ko_info'][0]['ko_txt'],
            "Vietnamese": data['vi_info'][0]['vi_txt'],
            "Indonesian": data['in_info'][0]['in_txt'],
            "Thai": data['th_info'][0]['th_txt'],
            "Cambodian": data['ca_info'][0]['ca_txt'],
        }
    dataset.append(data)

ds = datasets.Dataset.from_list(dataset)
ds_split = ds.train_test_split(test_size=0.05, seed=42)

ds_split.save_to_disk(
    "/home/maverick/openui-translation/data/raw/raw_dataset"
)