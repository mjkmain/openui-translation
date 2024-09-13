import datasets
from dataclasses import dataclass
import transformers
import torch

def build_dataset(
    low_dataset,
    tokenizer,
    tgt_language,
    split="train",
):
    assert tgt_language in ['Vietnamese', 'Indonesian', 'Thai', 'Cambodian']

    def tokenize_dataset(examples):
        tokenized_src = []
        tokenized_tgt = []
        
        for korean, target_lang in zip(examples['Korean'], examples[tgt_language]):
            prompt = tokenizer.apply_chat_template(
                conversation=[
                    {"role": "system", "content": f"You are a useful translation AI. Please translate the sentence given in Korean into {tgt_language}."},
                    {"role": "user", "content": korean},
                ],
                tokenize=True,
                add_generation_prompt=True,
            )

            label = tokenizer(target_lang, add_special_tokens=False)['input_ids'] + [tokenizer.eos_token_id]
            
            tokenized_src.append(prompt)
            tokenized_tgt.append(label)
        
        all_input_ids = []
        all_labels = []
        for src, tgt in zip(tokenized_src, tokenized_tgt):
            input_ids = torch.LongTensor(src + tgt)
            labels = torch.LongTensor([-100] * len(src) + tgt)
            assert len(input_ids) == len(labels)
            
            all_input_ids.append(input_ids)
            all_labels.append(labels)

        return{
            "input_ids": all_input_ids,
            "labels": all_labels,
        }
    
    try:
        ds = datasets.load_from_disk(
            f"/home/maverick/openui-translation/data/tokenized_v1/{tgt_language.lower()}/{split}",
        )
    

    except:
        ds = low_dataset[split].map(
            tokenize_dataset,
            batched=True,
            num_proc=48,
            remove_columns=low_dataset[split].column_names,
        )

        ds.save_to_disk(
            f"/home/maverick/openui-translation/data/tokenized_v1/{tgt_language.lower()}/{split}",
        )

    ds.set_format('torch')
    return ds
    
@dataclass
class DataCollatorForSFT:
    tokenizer: transformers.PreTrainedTokenizer
    
    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        
        return dict(
            input_ids = input_ids,
            attention_mask = attention_mask,
            labels = labels,
        )