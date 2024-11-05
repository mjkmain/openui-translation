import datasets
import json 
import glob
from tqdm import tqdm 

all_data = glob.glob("/raid/MLP/maverick/openui-translation/translation/dataset/*/*.json")
dataset = []
for data_path in tqdm(all_data):
    with open(data_path, 'r') as f:
        data = json.load(f)
        data = {
            "Korean": data['corpus']['ko_info'][0]['ko_txt'],
            "Vietnamese": data['corpus']['vi_info'][0]['vi_txt'],
            "Indonesian": data['corpus']['in_info'][0]['in_txt'],
            "Thai": data['corpus']['th_info'][0]['th_txt'],
            "Cambodian": data['corpus']['ca_info'][0]['ca_txt'],
            "meta_data": data['metadata'],
        }
    dataset.append(data)

ds = datasets.Dataset.from_list(dataset)
ds_split = ds.train_test_split(test_size=0.05, seed=42)

ds_split.save_to_disk(
    "/raid/MLP/maverick/openui-translation/translation/dataset/raw_dataset"
)