from torch.utils.data import DataLoader
import tqdm
import numpy as np
import torch
import pandas as pd
import os

from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.schedulers import ASHAScheduler
from datasets import load_dataset, load_metric
from transformers import (BertForSequenceClassification, AutoTokenizer,AutoConfig,
                          Trainer, TrainingArguments)
from sklearn.metrics import accuracy_score,recall_score,f1_score,precision_score,precision_recall_fscore_support

# from .run_glue_no_trainer import muticheck


def accuracy_cal(y_true, y_pred):
    count = 0
    for i in range(y_true.shape[0]):
        p = sum(np.logical_and(y_true[i], y_pred[i]))
        q = sum(np.logical_or(y_true[i], y_pred[i]))
        count += p / q
    return count / y_true.shape[0]
# metric 对象
class muticheck:
    def __init__(self, check_method=['accuracy']):
        self.check_method = check_method
        self.predictions = []
        self.references = []
        self.eval_metric = {}

    def check(self,predictions_=None):
        if predictions_ is not None:
            predictions = np.array(predictions_.cpu().numpy())
        else:
            predictions = np.array([p.cpu().numpy() for p in self.predictions])

        label = np.array([r.cpu().numpy() for r in self.references])
        self.eval_metric['suset_accuracy'] = accuracy_score(label,predictions)
        self.eval_metric['accuracy'] = accuracy_cal(label,predictions)
        self.eval_metric['precision'],self.eval_metric['recall'],self.eval_metric['f1'],_ = precision_recall_fscore_support(label,predictions, average='samples')
        self.eval_metric['micro-precision'],self.eval_metric['micro-recall'],self.eval_metric['micro-f1'],_ = precision_recall_fscore_support(label,predictions, average='micro',zero_division=0)


        # for metric_way in self.check_method:
        #     metric = evaluate.load(metric_way)
        #     metric.add_batch(
        #         predictions=self.predictions,
        #         references=self.references
        #     )
        #     self.eval_metric.update(metric.compute())
        # print(self.eval_metric)
        # self.predictions = predictions
        # self.references = label
        # print(f"r-macro:{recall_score(self.predictions,self.references,average='macro')}")
        # print(f"r-micro:{recall_score(self.predictions,self.references,average='micro')}")
        # # print(f"r-binary:{recall_score(self.predictions, self.references)}")
        # print(f"r-samples:{recall_score(label, predictions, average='samples')}")
        # print(f"f1-macro:{f1_score(self.predictions, self.references, average='macro',zero_division=0)}")
        # print(f"f1-micro:{f1_score(self.predictions, self.references, average='micro',zero_division=0)}")
        # # print(f"f1-binary:{f1_score(self.predictions, self.references)}")
        # print(f"f-samples:{f1_score(label, predictions, average='samples')}")
        # print(f"p-macro:{precision_score(self.predictions, self.references, average='macro')}")
        # print(f"p-micro:{precision_score(self.predictions, self.references, average='micro')}")
        # # print(f"p-binary:{precision_score(self.predictions, self.references)}")
        # print(f"p-samples:{precision_score(label,predictions, average='samples')}")
        # print(f"acc:{accuracy_score(self.predictions, self.references)}")
        # print(f"p,r,f:{precision_recall_fscore_support(self.predictions, self.references,average='binary')}")


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# DEVICE = 'cpu'

# c1
DATA_PATH = r'E:\data\nlpcct5\nlpcct5.py'
# MODEL_PATH = r'E:\model\transformers4\bertbase_lr20_bs8_256_lv1_1'
# MODEL_PATH = r'E:\model\transformers4\bertflbce_lr20_bs8_256_lv1'
# MODEL_PATH = r'E:\model\transformers4\bertfl_lr20_bs8_256_lv1'
# MODEL_PATH = r'E:\model\transformers4\xlnetbase_lr5_bs2_1024_lv1_1'
# MODEL_PATH = r'E:\model\transformers4\bertcdloss_md_lr20_bs8_256_lv1'
# MODEL_PATH=r'E:\model\transformers4\bert_for_test\output_dir'
MODEL_PATH=r'E:\model\transformers4\bertsimcse_lr20_bs8_256_lv1_1'


# c2

if not os.path.exists(os.path.join(MODEL_PATH, 'eval_res')):
    os.mkdir(os.path.join(MODEL_PATH, 'eval_res'))
EVAL_RES_PATH = os.path.join(MODEL_PATH, 'eval_res')
max_length = 256

# 默认不padding补到最长
pad_to_max_length = False
per_device_eval_batch_size = 8

task = 'log'

def log_into_file(processed_datasets,label_list,num_labels,model):
    df = pd.DataFrame(columns=['title','predictions','references'])

    pred_lst = []
    label_lst = []
    title_lst = []
    for step, data in enumerate(processed_datasets):
        if step % 500 == 0:
            print(f'iterated to {step} steps now.totally {len(processed_datasets)} steps.')
        txt_len = len(data['input_ids'])
        input = {'input_ids': torch.tensor(data['input_ids']).reshape(1, txt_len),
                 'token_type_ids': torch.tensor(data['token_type_ids']).reshape(1, txt_len),
                 'attention_mask': torch.tensor(data['attention_mask']).reshape(1, txt_len),
                 'labels': torch.tensor(data['labels']).reshape(1, num_labels)}

        outputs = model(**input).logits.tolist()
        pred_lst.append(outputs[0])
        label_lst.append(input['labels'].type(torch.int).tolist()[0])
        title_lst.append(data['title'])
        df.loc[step, 'title'] = title_lst[-1]
        df.loc[step, 'predictions'] = pred_lst[-1]
        df.loc[step, 'references'] = label_lst[-1]

    df.to_excel(os.path.join(EVAL_RES_PATH, 'eval_res.xlsx'))
    with open(os.path.join(EVAL_RES_PATH, 'labels.txt'), 'w', encoding='utf-8') as f:
        f.write(str(label_list))




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
    # 获取隐藏层
    config.output_hidden_states=True
    config.loss_mess = {
        'loss1': {},
        'loss2': {}
    }
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    def model_init():
        return BertForSequenceClassification.from_pretrained(MODEL_PATH, config=config)

    if task=='tune':
        def preprocess_function(examples):
            # Tokenize the texts
            texts = (
                (examples[sentence1_key],) if sentence2_key is None else (
                examples[sentence1_key], examples[sentence2_key])
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
            return result


        processed_datasets = data.map(
            preprocess_function,
            batched=True,
            remove_columns=data["train"].column_names,
            desc="Running tokenizer on dataset",
        )
    else:
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

    if task=='log':
        model = model_init()
        log_into_file(processed_datasets, label_list, num_labels, model)
        return

    if task=='test':
        for step, data in enumerate(processed_datasets):
            txt_len = len(data['input_ids'])
            input = {'input_ids': torch.tensor(data['input_ids']).reshape(1, txt_len),
                     'token_type_ids': torch.tensor(data['token_type_ids']).reshape(1, txt_len),
                     'attention_mask': torch.tensor(data['attention_mask']).reshape(1, txt_len),
                     'labels': torch.tensor(data['labels']).reshape(1, num_labels)}
            model = model_init()
            out = model(**input)
            print(out.hidden_states[-1].size())
            last = out.hidden_states[-1].transpose(1, 2)
            print(last.size())
            print(torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1).size())
            break
        return

    metric = muticheck(['accuracy', 'precision', 'f1'])

    def compute_metrics(eval_pred):
        metric.predictions, metric.references = eval_pred
        best_th = 0.5
        default_th = 0.4
        best_dir = {}
        thresholds = (np.array(range(-11, 10)) / 100) + default_th
        best_f1 = 0
        metric.predictions = torch.tensor(metric.predictions,
                                          device='cuda' if torch.cuda.is_available() else 'cpu')
        for threshold in thresholds:
            metric.eval_metric = {}
            predictions = torch.ge(metric.predictions, threshold).type(torch.int)
            # print(predictions)
            metric.check(predictions)
            if metric.eval_metric['f1'] > best_f1:
                best_f1 = metric.eval_metric['f1']
                best_dir = metric.eval_metric
                best_th = threshold
        if best_dir:
            metric.eval_metric = best_dir
        metric.eval_metric['threshold'] = best_th
        return metric.eval_metric['mirco-f1']

    training_args = TrainingArguments(
        MODEL_PATH,
        evaluation_strategy="steps",
        eval_steps=500,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        disable_tqdm=True,
        label_names=label_list,

    )

    train_dataset = processed_datasets["train"].shuffle(seed=666).select(range(3000))
    eval_dataset = processed_datasets["validation"].shuffle(seed=666).select(range(500))

    trainer = Trainer(
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        model_init=model_init,
        compute_metrics=compute_metrics,
    )



    best_trial = trainer.hyperparameter_search(
        direction="maximize",

        # Choose among many libraries:
        # https://docs.ray.io/en/latest/tune/api_docs/suggestion.html
        search_alg=HyperOptSearch(metric="objective", mode="max"),
        # Choose among schedulers:
        # https://docs.ray.io/en/latest/tune/api_docs/schedulers.html
        scheduler=ASHAScheduler(metric="objective", mode="max"),
        n_trials=100,
    )
    print(best_trial)






if __name__ == '__main__':
    main()