from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import os
import json
import math 
import torch
from translation.data_utils import build_dataset, PAD_TOKEN_ID
from tqdm import tqdm
import datasets

cwd = os.getcwd()

class ResponseGenerator:
    def __init__(
        self,
        model_name_or_path,
        tokenizer_name_or_path,
        tgt_language,
    ):  
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation="flash_attention_2",
            token=hf_read_token,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            token=hf_read_token,
        )
        self.tokenizer.pad_token_id = PAD_TOKEN_ID
        
        self.generation_config = {
            "bos_token_id" : self.tokenizer.bos_token_id,
            "eos_token_id" : self.tokenizer.eos_token_id,
            "pad_token_id" : self.tokenizer.pad_token_id,
            "use_cache" : True,
            "do_sample" : False,
            "max_new_tokens" : 1024,
            "temperature" : None,
            "top_p" : None,
            "top_k" : None,
        }
        
        self.tgt_language = tgt_language
        self.save_dir = os.path.join(os.getcwd(), "../results", tgt_language)
        os.makedirs(self.save_dir, exist_ok=True)
        
        set_seed(42)
        
    def write_to_file(self, data, file_path):
        with open(file_path, 'a') as file:
            json.dump(data, file, ensure_ascii=False)
            file.write('\n')
            
    def load_and_split_dataset(
        self, 
        language, 
        gpu_id=0,
        num_gpus=1,
    ):
        raw_ds = datasets.load_from_disk(
            "/home/maverick/openui-translation/data/raw/raw_dataset"
        )
        
        ds = build_dataset(
            raw_ds,
            tokenizer=self.tokenizer,
            tgt_language=language,
            split="test",
        )
        ### Divide dataset for each GPUs
        ds = list(ds)
        chunk_size = math.ceil(len(ds) / num_gpus)
        start_index = chunk_size * gpu_id
        end_index = chunk_size * (gpu_id + 1)
        ds = ds[start_index:end_index]

        return ds


    @torch.inference_mode
    def response_generation(
        self, 
        gpu_id=0,
        num_gpus=1,
    ):
        print(f"Start inference {self.tgt_language}.")
        from timeit import default_timer as timer
        import time
        start = timer()

        def print_time(current_time):
            return f"Elapsed time: {current_time - start:.2f} s"
            
        eval_dataset = self.load_and_split_dataset(
            language=self.tgt_language, 
            gpu_id=gpu_id,
            num_gpus=num_gpus,
        )

        for i, eval_data in enumerate(eval_dataset): #tqdm(eval_dataset, total=len(eval_dataset)):

            input_ids = eval_data['input_ids'].unsqueeze(0)

            if input_ids.size(-1) > self.model.config.max_position_embeddings:
                continue

            attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
            
            output_ids = self.model.generate(
                input_ids=input_ids.to(self.model.device),
                attention_mask=attention_mask.to(self.model.device),
                **self.generation_config
            )[0]
            
            response = self.tokenizer.decode(
                output_ids[len(input_ids[0]):], skip_special_tokens=True
            ).strip()
        
            result = {
                "korean": eval_data['text_src'],
                "gt": eval_data['text_tgt'],
                "response": response,
                "input_prompt": self.tokenizer.decode(eval_data['input_ids'])
            }
            current_time = timer()
            
            print(f"""\
                
============================================================= Sample number {i+1} ============================================================================
## Korean 
{eval_data['text_src']}

## Ground Truth 
{eval_data['text_tgt']}

## Predict 
{response}
=============================================================== {print_time(current_time)} ===================================================================

""")

            # save_path = os.path.join(
            #     self.save_dir,
            #     f"response_{gpu_id}.jsonl"
            # )
            # self.write_to_file(
            #     data=result,
            #     file_path=save_path,
            # )
            
def main(args):
    generator = ResponseGenerator(
        model_name_or_path=args.model_name_or_path,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        tgt_language=args.target_language,        
    )
    generator.response_generation(
        gpu_id=args.gpu_id,
        num_gpus=args.num_gpus,
    )

if __name__=="__main__":
    import argparse
    import warnings
    warnings.filterwarnings("error")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--tokenizer_name_or_path", type=str)
    parser.add_argument("--target_language", type=str)
    parser.add_argument("--gpu_id", type=int)
    parser.add_argument("--num_gpus", type=int)
    args = parser.parse_args()
    
    main(args)