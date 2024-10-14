import jsonlines 
import os
import re
import argparse
from evaluate import load
from transformers import AutoTokenizer
import numpy as np
import warnings
import math 

warnings.filterwarnings('ignore')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str)
    parser.add_argument("--lang", type=str)
    parser.add_argument("--tokenizer_name_or_path", type=str, default="google-bert/bert-base-multilingual-cased")
    args = parser.parse_args()

    return args

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
    # Load Evaluator
    bleu = load("bleu")
    rouge = load("rouge")
    # bertscore = load("bertscore")

    lang = args.lang
    result_path = args.result_path

    pred_dir = f"{result_path}/{lang}/"
    pred_files = os.listdir(pred_dir)
    
    data = []
    for pred_file in pred_files:
        with jsonlines.open(os.path.join(pred_dir, pred_file)) as f:
            for x in f.iter():
                data.append(x)

    predictions = []
    references = []

    for d in data:
        predictions.append(d['response'])
        references.append(d['gt'])
    
    bleu_results = bleu.compute(predictions=predictions, references=references, tokenizer=lambda x: tokenizer.tokenize(x))
    # bleu_results = bleu.compute(predictions=predictions, references=references, tokenizer=lambda x: x.split(" "))
    # rouge_results = rouge.compute(predictions=predictions, references=references, tokenizer=lambda x: tokenizer.tokenize(x))
    # bertscore_results = bertscore.compute(predictions=predictions, references=references, model_type="bert-base-multilingual-cased")

    print(f"#### BLEU SCORE #####")
    print(f"BLEU :: {math.floor(bleu_results['bleu']*100)}\n")
    # print("\n#### ROUGE SCORE ####")
    # for k, v in rouge_results.items():
        # print(f"{k} :: {v}")
    # print("\n#### BERT SCORE ####")
    # print(f"F1 :: {np.mean(bertscore_results['f1'])}")
    # print(f"PRECISION :: {np.mean(bertscore_results['precision'])}")
    # print(f"RECALL :: {np.mean(bertscore_results['recall'])}")

if __name__=="__main__":
    args = get_args()
    
    main(args)
    