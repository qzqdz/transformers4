from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, DataCollatorWithPadding,BertForSequenceClassification
from datasets import load_dataset
from torch.utils.data import DataLoader
import tqdm
import numpy as np
import torch
import pandas as pd
import os

DATA_PATH = r'E:\data\nlpcct5\nlpcct5.py'
# MODEL_PATH = r'E:\model\transformers4\bertrdrop1_lr20_bs8_256_lv1_1'
MODEL_PATH = r'E:\model\transformers4\bertflbce_lr20_bs8_256_lv1'
# MODEL_PATH = r'E:\model\transformers4\bertfl_lr20_bs8_256_lv1'
# MODEL_PATH = r'E:\model\transformers4\xlnetbase_lr5_bs2_1024_lv1_1'

if not os.path.exists(os.path.join(MODEL_PATH, 'eval_res')):
    os.mkdir(os.path.join(MODEL_PATH, 'eval_res'))
EVAL_RES_PATH = os.path.join(MODEL_PATH, 'eval_res')
max_length = 256

# 默认不padding补到最长
pad_to_max_length = False
per_device_eval_batch_size = 8


def main():

    sentence1_key, sentence2_key = 'title', 'abstract'
    data = load_dataset(DATA_PATH)
    eval_data = data['validation']

    label_list = eval_data.column_names
    label_list.sort()
    label_list.remove('title')
    label_list.remove('abstract')
    num_labels = len(label_list)

    label_to_id = {v: i for i, v in enumerate(label_list)}
    config = AutoConfig.from_pretrained(MODEL_PATH, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH, config=config)


    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=False, max_length=max_length, truncation=True, add_special_tokens=True)

        if "label" in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]
        elif 'title' in examples and 'abstract' in examples:
            labels_lst = []
            batch_length = len(examples['title'])
            for lb in examples:
                assert batch_length == len(examples[lb]), 'error!'
            for i in range(batch_length):
                row_label = []
                for lab in label_list:
                    # print(labels_lst)
                    row_label.append(examples[lab][i])
                labels_lst.append(row_label)
            result['labels'] = labels_lst
            result['title'] = examples['title']
        return result


    processed_datasets = eval_data.map(
        preprocess_function,
        batched=True,
        remove_columns=eval_data.column_names,
        desc="Running tokenizer on dataset"
    )


    df = pd.DataFrame(columns=['title','predictions','references'])

    pred_lst = []
    label_lst = []
    title_lst = []

    for step, data in enumerate(processed_datasets):
        if step%500==0:
            print(f'iterated to {step} steps now.totally {len(processed_datasets)} steps.')
        txt_len = len(data['input_ids'])
        input = {'input_ids': torch.tensor(data['input_ids']).reshape(1,txt_len), 'token_type_ids': torch.tensor(data['token_type_ids']).reshape(1,txt_len),
                 'attention_mask': torch.tensor(data['attention_mask']).reshape(1,txt_len), 'labels': torch.tensor(data['labels']).reshape(1,num_labels)}

        outputs = model(**input).logits.tolist()
        pred_lst.append(outputs[0])
        label_lst.append(input['labels'].type(torch.int).tolist()[0])
        title_lst.append(data['title'])
        df.loc[step,'title']=title_lst[-1]
        df.loc[step,'predictions']=pred_lst[-1]
        df.loc[step,'references']=label_lst[-1]

    df.to_excel(os.path.join(EVAL_RES_PATH,'eval_res.xlsx'))
    with open(os.path.join(EVAL_RES_PATH,'labels.txt'),'w',encoding='utf-8') as f:
        f.write(str(label_list))

if __name__ == '__main__':
    main()