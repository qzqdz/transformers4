# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning a 🤗 Transformers model for sequence classification on GLUE."""
import argparse
import json
import logging
import math
import os
import random
from pathlib import Path

import datasets
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import evaluate
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from huggingface_hub import Repository
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import check_min_version, get_full_repo_name, send_example_telemetry
from transformers.utils.versions import require_version
import numpy as np
from sklearn.metrics import accuracy_score,recall_score,f1_score,precision_score,precision_recall_fscore_support


from ray import tune, air
from ray.air import session
from ray.tune.search.optuna import OptunaSearch



# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.23.0.dev0")

logger = get_logger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="The name of the glue task to train on.",
        choices=list(task_to_keys.keys()),
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )

    parser.add_argument(
        "--do_train",
        type=bool,
        default=False,
        help="do train or not",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"` and `"comet_ml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )

    parser.add_argument(
        "--multicheck",
        type=bool,
        default=True,
        help="metric model by calucating acc,p,r,f1",
    )

    parser.add_argument(
        "--child_tune",
        action="store_true",
        help="to child tune the model.",
    )
    parser.add_argument(
        "--train_mode",
        type=str,
        default='default',
        help="the train mode.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="set the fixed threshold.",
    )


    args = parser.parse_args()

    # Sanity checks
    if args.task_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json", 'py'], "`train_file` should be a py, csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json", 'py'], "`validation_file` should be a py, csv or a json file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


def accuracy_cal(y_true, y_pred):
    count = 0
    for i in range(y_true.shape[0]):
        p = sum(np.logical_and(y_true[i], y_pred[i]))
        q = sum(np.logical_or(y_true[i], y_pred[i]))
        count += p / q
    return count / y_true.shape[0]





# metric 对象
class multicheck:
    def __init__(self, check_method=['accuracy']):
        self.check_method = check_method
        self.predictions = []
        self.references = []
        self.eval_metric = {}

    def check(self,predictions_=None):

        label = np.array([r.cpu().numpy() for r in self.references])
        multilabel = type(label.tolist()[0]) is list

        if predictions_ is not None:
            if type(predictions_) is not list:
                predictions = np.array(predictions_.cpu().numpy())
            else:
                predictions = predictions_
        else:
            predictions = np.array([p.cpu().numpy() for p in self.predictions])

        if multilabel:

            # print(label[0],predictions[0])
            self.eval_metric['suset_accuracy'] = accuracy_score(label,predictions)
            self.eval_metric['accuracy'] = accuracy_cal(label,predictions)
            # self.eval_metric['precision'],self.eval_metric['recall'],self.eval_metric['f1'],_ = precision_recall_fscore_support(label,predictions, average='samples')
            self.eval_metric['precision'],self.eval_metric['recall'],self.eval_metric['f1'],_ = precision_recall_fscore_support(label,predictions, average='macro')
            self.eval_metric['micro-precision'],self.eval_metric['micro-recall'],self.eval_metric['micro-f1'],_ = precision_recall_fscore_support(label,predictions, average='micro',zero_division=0)
            # self.eval_metric['micro-precision'], self.eval_metric['micro-recall'], self.eval_metric['micro-f1'], _ = precision_recall_fscore_support(label, predictions, average='macro', zero_division=0)
        else:
            # print(label)
            # print(predictions)
            self.eval_metric['accuracy'] = accuracy_score(label,predictions)
            self.eval_metric['precision'], self.eval_metric['recall'], self.eval_metric[
                'f1'], _ = precision_recall_fscore_support(label, predictions,average='binary')

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



def main():
    args = parse_args()
    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_glue_no_trainer", args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator = (
        Accelerator(log_with=args.report_to, logging_dir=args.output_dir) if args.with_tracking else Accelerator()
    )
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).

    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.

    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset("glue", args.task_name)
    elif args.train_file.split('.')[-1]=='py':
        if 'reuters21578' in args.train_file:
            raw_datasets = load_dataset(args.train_file,'ModHayes')

        elif 'web_of_science' in args.train_file:
            raw_datasets = load_dataset(args.train_file, 'WOS46985')
        else:
            raw_datasets = load_dataset(args.train_file)
    else:
        # Loading the dataset from local csv or json file.
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = (args.train_file if args.train_file is not None else args.validation_file).split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    # 多标签读入数据的时候需要处理一下这里


    if args.task_name is not None:
        is_regression = args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names

            num_labels = len(label_list)
        else:
            num_labels = 1

    # 数据读取规则,注册处
    elif 'title' in raw_datasets['train'].column_names and 'abstract' in raw_datasets['train'].column_names:
        label_list = raw_datasets['train'].column_names
        label_list.sort()
        label_list.remove('title')
        label_list.remove('abstract')
        num_labels = len(label_list)
        is_regression=False
    elif ('title' not in raw_datasets['train'].column_names) and ('abstract' in raw_datasets['train'].column_names):
        label_list = raw_datasets['train'].column_names
        label_list.sort()
        label_list.remove('abstract')
        num_labels = len(label_list)
        is_regression=False
    elif 'title1' in raw_datasets['train'].column_names and 'abstract1' in raw_datasets['train'].column_names:
        label_list = raw_datasets['train'].column_names
        label_list.sort()
        label_list.remove('title1')
        label_list.remove('abstract1')
        label_list.remove('title2')
        label_list.remove('abstract2')
        num_labels = len(label_list)
        is_regression=False
    elif ('title1' not in raw_datasets['train'].column_names) and ('abstract1' in raw_datasets['train'].column_names):
        label_list = raw_datasets['train'].column_names
        label_list.sort()
        label_list.remove('abstract1')
        label_list.remove('abstract2')
        num_labels = len(label_list)
        is_regression=False

    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        # print(raw_datasets["train"])
        print('---------------------')
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # print(num_labels)
    # print('check here!!')
    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    try:
        print('the label list is here!')
        print(label_list)
    except:
        pass

    # 定义模型处
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels,
                                        finetuning_task=args.task_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        ignore_mismatched_sizes=args.ignore_mismatched_sizes,
    )

    #数据注册处
    # Preprocessing the datasets
    if args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        elif 'title' in non_label_column_names and 'abstract' in non_label_column_names:
            sentence1_key,sentence2_key = 'title', 'abstract'
        elif ('title' not in non_label_column_names) and ('abstract' in non_label_column_names):
            sentence1_key,sentence2_key = 'abstract',None

        elif 'title1' in non_label_column_names and 'abstract1' in non_label_column_names:
            sentence1_key, sentence2_key, sentence3_key, sentence4_key = 'title1', 'abstract1', 'title2', 'abstract2'
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            logger.info(
                f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
                "Using it!"
            )
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}


    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    padding = "max_length" if args.pad_to_max_length else False
    # 此处为添加标签处
    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )

# 静默处1
        '''
        if args.train_mode=='simcse':
            if sentence2_key is None:
                mid_lst = [*zip(examples[sentence1_key],examples[sentence1_key])]
                mid_lst_ = []
                for ml in mid_lst:
                    for i in range(len(ml)):
                        mid_lst_.append(ml[i])
                mid_lst = mid_lst_
                del mid_lst_
                texts = (mid_lst,)
            else:
                mid_lst1 = [*zip(examples[sentence1_key],examples[sentence1_key])]
                mid_lst1_ = []
                for ml in mid_lst1:
                    for i in range(len(ml)):
                        mid_lst1_.append(ml[i])
                mid_lst1 = mid_lst1_
                del mid_lst1_

                mid_lst2 = [*zip(examples[sentence2_key],examples[sentence2_key])]
                mid_lst2_ = []
                for ml in mid_lst2:
                    for i in range(len(ml)):
                        mid_lst2_.append(ml[i])
                mid_lst2 = mid_lst2_
                del mid_lst2_

                # texts = ([row[i] for i in range(len(mid_lst1[0])) for row in mid_lst1],[row[i] for i in range(len(mid_lst2[0])) for row in mid_lst2])
                texts = (mid_lst1,mid_lst2)
        '''

        if args.train_mode == 'simcse_sup' and ('zhihu' in args.train_file or 'new' in args.train_file or 'waimai' in args.train_file):
            mid_lst = [*zip(examples[sentence1_key], examples[sentence2_key])]
            mid_lst_ = []
            for ml in mid_lst:
                for i in range(len(ml)):
                    mid_lst_.append(ml[i])
            mid_lst = mid_lst_
            del mid_lst_
            texts = (mid_lst,)

        elif args.train_mode=='simcse_sup':
            # [(1,1),(2,2)]->[1,1,2,2]

            mid_lst1 = [*zip(examples[sentence1_key], examples[sentence3_key])]
            mid_lst1_ = []
            for ml in mid_lst1:
                for i in range(len(ml)):
                    mid_lst1_.append(ml[i])
            mid_lst1 = mid_lst1_
            del mid_lst1_

            mid_lst2 = [*zip(examples[sentence2_key], examples[sentence4_key])]
            mid_lst2_ = []
            for ml in mid_lst2:
                for i in range(len(ml)):
                    mid_lst2_.append(ml[i])
            mid_lst2 = mid_lst2_
            del mid_lst2_

            texts = (mid_lst1, mid_lst2)




        result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True,add_special_tokens=True)

        if "label" in examples:
            if ('zhihu' in args.train_file or 'waimai' in args.train_file) and 'simcse' in args.train_mode:
                labels_lst = []
                for i in range(len(examples['label'])):
                    labels_lst.append(examples['label'][i])
                    labels_lst.append(examples['label'][i])
                result["labels"] = labels_lst
            elif label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples["label"]

            # 静默处2
            # if args.train_mode=='simcse':
            #     labels_lst = result["labels"]
            #     labels_lst = [*zip(labels_lst, labels_lst)]
            #     labels_lst = [row[i] for i in range(len(labels_lst[0])) for row in labels_lst]
            #     result['labels'] = labels_lst


        elif ('abstract' in examples) or ('abstract1' in examples):
            labels_lst = []
            try:
                batch_length = len(examples['abstract'])
            except:
                batch_length = len(examples['abstract1'])
            for lb in examples:
                assert batch_length==len(examples[lb]),'error!'


            if args.train_mode=='simcse' or args.train_mode=='simcse_sup':

                # if 'zhihu' in args.train_file:
                #     for i in range(batch_length):
                #         labels_lst.append(examples['label'][i])
                #         labels_lst.append(examples['label'][i])
                # else:
                for i in range(batch_length):
                    row_label = []
                    for lab in label_list:
                        # print(labels_lst)
                        row_label.append(examples[lab][i])

                    labels_lst.append(row_label)
                    labels_lst.append(row_label)
                # print(len(labels_lst))

            else:
                for i in range(batch_length):
                    row_label = []
                    for lab in label_list:
                        # print(labels_lst)

                        row_label.append(examples[lab][i])
                    labels_lst.append(row_label)

            result['labels']=labels_lst

        return result

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
        )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation_matched" if args.task_name == "mnli" else "validation"]


    if args.child_tune:
        train_dataset = train_dataset.select(range(3000))
        eval_dataset = eval_dataset.select(range(500))
    # eval_dataset = eval_dataset.select(range(2))
    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # 加collector
    class unsup_CSECollator(object):
        def __init__(self,
                     tokenizer,
                     features=("input_ids", "attention_mask", "token_type_ids","labels"),
                     max_len=100):
            self.tokenizer = tokenizer
            self.features = features
            self.max_len = max_len

        def collate(self, batch):
            new_batch = []
            for example in batch:
                for i in range(2):
                    # 每个句子重复两次
                    new_batch.append({fea: example[fea] for fea in self.features})
            new_batch = self.tokenizer.pad(
                new_batch,
                padding=True,
                max_length=self.max_len,
                return_tensors="pt"
            )

            return new_batch

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
        if args.train_mode=='simcse':
            collator = unsup_CSECollator(tokenizer, max_len=args.max_length)
            data_collator = collator.collate
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))

        if args.train_mode=='simcse':
            collator = unsup_CSECollator(tokenizer, max_len=args.max_length)
            data_collator = collator.collate


    # 开始定义训练相关对象
    if args.train_mode=='simcse' or args.train_mode=='simcse_sup':
        shuffle=False
    else:
        shuffle=True
    train_dataloader = DataLoader(
        train_dataset, shuffle=shuffle, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # 优化器所在地
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("glue_no_trainer", experiment_config)


    # Get the metric function
    # 读入测试函数

    if args.task_name is not None:
        metric = evaluate.load("glue", args.task_name)
    elif args.multicheck:
        metric = multicheck(['accuracy', 'precision', 'f1'])
    else:
        metric = evaluate.load("accuracy")


    if args.do_train:
        # Train!
        total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")


        def train_cycle(model_config=None):
            if not model_config:
                model.config.loss_mess = {
                    'loss1':{},
                    'loss2':{}
                }
            else:
                model.config.loss_mess = model_config
            if args.with_tracking:
                accelerator.log(
                    {
                        "eval_res": {'suset_accuracy': 0, 'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0},
                        "train_loss": 0,
                        "epoch": 0,
                        "step": 0,
                    },
                    step=0,
                )

            # Only show the progress bar once on each machine.
            progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
            completed_steps = 0
            starting_epoch = 0
            # Potentially load in the weights and states from a previous save
            if args.resume_from_checkpoint:
                if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
                    accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
                    accelerator.load_state(args.resume_from_checkpoint)
                    path = os.path.basename(args.resume_from_checkpoint)
                else:
                    # Get the most recent checkpoint
                    dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
                    dirs.sort(key=os.path.getctime)
                    path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
                # Extract `epoch_{i}` or `step_{i}`
                training_difference = os.path.splitext(path)[0]

                if "epoch" in training_difference:
                    starting_epoch = int(training_difference.replace("epoch_", "")) + 1
                    resume_step = None
                else:
                    resume_step = int(training_difference.replace("step_", ""))
                    starting_epoch = resume_step // len(train_dataloader)
                    resume_step -= starting_epoch * len(train_dataloader)

            # 训练循环
            for epoch in range(starting_epoch, args.num_train_epochs):
                if args.with_tracking:
                    total_loss = 0
                for step, batch in enumerate(train_dataloader):
                    # print(batch)
                    model.train()
                    # We need to skip steps until we reach the resumed step
                    if args.resume_from_checkpoint and epoch == starting_epoch:
                        if resume_step is not None and step < resume_step:
                            completed_steps += 1
                            continue
                    outputs = model(**batch)

                    loss = outputs.loss
                    # We keep track of the loss at each epoch
                    if args.with_tracking:
                        total_loss += loss.detach().float()

                    loss = loss / args.gradient_accumulation_steps
                    if isinstance(checkpointing_steps, int):
                        if completed_steps % checkpointing_steps == 0:
                            logger.info(f"loss: {loss}")

                    accelerator.backward(loss)
                    if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()
                        progress_bar.update(1)
                        completed_steps += 1

                    if isinstance(checkpointing_steps, int):
                        if completed_steps % checkpointing_steps == 0:
                            output_dir = f"step_{completed_steps }"
                            if args.output_dir is not None:
                                output_dir = os.path.join(args.output_dir, output_dir)
                            accelerator.save_state(output_dir)

                            model.eval()

                            if args.multicheck:
                                # not supprot accelerator
                                check_batch_mun = 100
                                metric.predictions=[]
                                metric.references=[]
                                if len(label_list) > 1:
                                    for step, batch in enumerate(tqdm(eval_dataloader)):
                                        if check_batch_mun<=0:
                                            break
                                        with torch.no_grad():
                                            outputs = model(**batch)

                                        if num_labels<=2:
                                            predictions =outputs.logits.argmax(dim=-1)

                                            metric.predictions.extend(predictions)
                                        else:
                                            predictions = torch.sigmoid(outputs.logits.squeeze())
                                             # predictions = torch.ge(predictions, 0.5).type(torch.int)
                                            metric.predictions.extend(predictions.cpu().detach().tolist())
                                        metric.references.extend(batch["labels"].type(torch.int))
                                        # check_batch_mun-=1

                                else:
                                    for step, batch in enumerate(tqdm(eval_dataloader)):
                                        if check_batch_mun<=0:
                                            break
                                        with torch.no_grad():
                                            outputs = model(**batch)
                                        predictions = outputs.logits.argmax(
                                            dim=-1) if not is_regression else outputs.logits.squeeze()
                                        metric.predictions.extend(predictions)
                                        metric.references.extend(batch["labels"])
                                        # check_batch_mun -= 1

                                if len(label_list) > 2:
                                    best_th = 0.5
                                    default_th = 0.4
                                    best_dir = {}
                                    thresholds = (np.array(range(-20, 20)) / 100) + default_th
                                    best_f1 = 0
                                    metric.predictions = torch.tensor(metric.predictions,
                                                                      device='cuda' if torch.cuda.is_available() else 'cpu')


                                    for threshold in thresholds:
                                        metric.eval_metric = {}
                                        predictions = torch.ge(metric.predictions,threshold).type(torch.int)
                                        # print(predictions)
                                        metric.check(predictions)
                                        if metric.eval_metric['f1']>best_f1:
                                            best_f1 = metric.eval_metric['f1']
                                            best_dir = metric.eval_metric
                                            best_th = threshold
                                    if best_dir:
                                        metric.eval_metric = best_dir
                                    logger.info(f"best checkpoint:{metric.eval_metric};threshold:{best_th}")
                                    metric.eval_metric['threshold'] = best_th

                                else:
                                    metric.eval_metric = {}
                                    metric.check()
                                    logger.info(f"checkpoint:{metric.eval_metric}")
                                if args.with_tracking:
                                    accelerator.log(
                                        {
                                            "eval_res": metric.eval_metric,

                                            "train_loss": total_loss.item() / len(train_dataloader),
                                            "epoch": epoch,
                                            "step": completed_steps,
                                        },
                                        step=completed_steps,
                                    )
                            else:
                                samples_seen = 0
                                for step, batch in enumerate(tqdm(eval_dataloader)):
                                    with torch.no_grad():
                                        outputs = model(**batch)
                                    predictions = outputs.logits.argmax(
                                        dim=-1) if not is_regression else outputs.logits.squeeze()
                                    predictions, references = accelerator.gather((predictions, batch["labels"]))
                                    # If we are in a multiprocess environment, the last batch has duplicates
                                    if accelerator.num_processes > 1:
                                        if step == len(eval_dataloader) - 1:
                                            predictions = predictions[: len(eval_dataloader.dataset) - samples_seen]
                                            references = references[: len(eval_dataloader.dataset) - samples_seen]
                                        else:
                                            samples_seen += references.shape[0]

                                    metric.add_batch(
                                        predictions=predictions,
                                        references=references,
                                    )

                                eval_metric = metric.compute()

                                logger.info(f"epoch {epoch}: {eval_metric}")

                                if args.with_tracking:
                                    accelerator.log(
                                        {
                                            "accuracy" if args.task_name is not None else "glue": eval_metric,
                                            "train_loss": total_loss.item() / len(train_dataloader),
                                            "epoch": epoch,
                                            "step": completed_steps,
                                        },
                                        step=completed_steps,
                                    )

                        if args.child_tune:
                            tune.report(mmicro_f1=metric.eval_metric['f1'])

                    if completed_steps >= args.max_train_steps:
                        break

                model.eval()
                if args.multicheck:
                    metric.predictions = []
                    metric.references = []
                    # not supprot accelerator
                    if len(label_list)>1:
                        for step, batch in enumerate(tqdm(eval_dataloader)):
                            with torch.no_grad():
                                outputs = model(**batch)
                            if num_labels <= 2:
                                predictions = outputs.logits.argmax(dim=-1)
                                # print(predictions)
                                metric.predictions.extend(predictions)
                            else:
                                predictions = torch.sigmoid(outputs.logits.squeeze())
                                # predictions = torch.ge(predictions, 0.5).type(torch.int)
                                metric.predictions.extend(predictions.cpu().detach().tolist())
                            metric.references.extend(batch["labels"].type(torch.int))
                    else:
                        for step, batch in enumerate(tqdm(eval_dataloader)):
                            with torch.no_grad():
                                outputs = model(**batch)
                            predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
                            metric.predictions.extend(predictions)
                            metric.references.extend(batch["labels"])


                    if len(label_list) > 2:
                        default_th = 0.5
                        best_th = 0.5
                        best_dir = {}
                        thresholds = (np.array(range(-20, 20)) / 100) + default_th
                        best_f1 = 0
                        metric.predictions = torch.tensor(metric.predictions,
                                                          device='cuda' if torch.cuda.is_available() else 'cpu')
                        for threshold in thresholds:
                            metric.eval_metric = {}
                            predictions = torch.ge(metric.predictions, threshold).type(torch.int)
                            metric.check(predictions)
                            if metric.eval_metric['f1'] > best_f1:
                                best_f1 = metric.eval_metric['f1']
                                best_dir = metric.eval_metric
                                best_th = threshold
                        if best_dir:
                            metric.eval_metric = best_dir
                        logger.info(f"epoch {epoch}: {metric.eval_metric};best_th:{best_th}")
                        metric.eval_metric['threshold'] = best_th
                    else:
                        metric.eval_metric = {}
                        metric.check()
                        logger.info(f"epoch {epoch}: {metric.eval_metric}")

                    if args.with_tracking:
                        accelerator.log(
                            {
                                "eval_res": metric.eval_metric,
                                "train_loss": total_loss.item() / len(train_dataloader),
                                "epoch": epoch,
                                "step": completed_steps,
                            },
                            step=completed_steps,
                        )
                else:
                    samples_seen = 0
                    for step, batch in enumerate(tqdm(eval_dataloader)):
                        with torch.no_grad():
                            outputs = model(**batch)
                        predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
                        predictions, references = accelerator.gather((predictions, batch["labels"]))
                        # If we are in a multiprocess environment, the last batch has duplicates
                        if accelerator.num_processes > 1:
                            if step == len(eval_dataloader) - 1:
                                predictions = predictions[: len(eval_dataloader.dataset) - samples_seen]
                                references = references[: len(eval_dataloader.dataset) - samples_seen]
                            else:
                                samples_seen += references.shape[0]

                        metric.add_batch(
                            predictions=predictions,
                            references=references,
                        )

                    eval_metric = metric.compute()

                    logger.info(f"epoch {epoch}: {eval_metric}")

                    if args.with_tracking:
                        accelerator.log(
                            {
                                "accuracy" if args.task_name is not None else "glue": eval_metric,
                                "train_loss": total_loss.item() / len(train_dataloader),
                                "epoch": epoch,
                                "step": completed_steps,
                            },
                            step=completed_steps,
                        )

                if args.push_to_hub and epoch < args.num_train_epochs - 1:
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(
                        args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
                    )
                    if accelerator.is_main_process:
                        tokenizer.save_pretrained(args.output_dir)
                        repo.push_to_hub(
                            commit_message=f"Training in progress epoch {epoch}", blocking=False, auto_lfs_prune=True
                        )

                if args.checkpointing_steps == "epoch":
                    output_dir = f"epoch_{epoch}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)



            if args.with_tracking:
                accelerator.end_training()

            if args.output_dir is not None:
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(
                    args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
                )
                if accelerator.is_main_process:
                    tokenizer.save_pretrained(args.output_dir)
                    if args.push_to_hub:
                        repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)

        def hm_train_cycle(model_config=None):
            if not model_config:
                model.config.loss_mess = {
                    'loss1':{},
                    'loss2':{}
                }
            else:
                model.config.loss_mess = model_config
            if args.with_tracking:
                accelerator.log(
                    {
                        "eval_res": {'suset_accuracy': 0, 'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0},
                        "train_loss": 0,
                        "epoch": 0,
                        "step": 0,
                    },
                    step=0,
                )

            # Only show the progress bar once on each machine.
            progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
            completed_steps = 0
            starting_epoch = 0
            # Potentially load in the weights and states from a previous save
            if args.resume_from_checkpoint:
                if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
                    accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
                    accelerator.load_state(args.resume_from_checkpoint)
                    path = os.path.basename(args.resume_from_checkpoint)
                else:
                    # Get the most recent checkpoint
                    dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
                    dirs.sort(key=os.path.getctime)
                    path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
                # Extract `epoch_{i}` or `step_{i}`
                training_difference = os.path.splitext(path)[0]

                if "epoch" in training_difference:
                    starting_epoch = int(training_difference.replace("epoch_", "")) + 1
                    resume_step = None
                else:
                    resume_step = int(training_difference.replace("step_", ""))
                    starting_epoch = resume_step // len(train_dataloader)
                    resume_step -= starting_epoch * len(train_dataloader)

            # 训练循环
            for epoch in range(starting_epoch, args.num_train_epochs):
                if args.with_tracking:
                    total_loss = 0
                for step, batch in enumerate(train_dataloader):
                    # print(batch['labels'])
                    # print('debug labels!')
                    model.train()
                    # We need to skip steps until we reach the resumed step
                    if args.resume_from_checkpoint and epoch == starting_epoch:
                        if resume_step is not None and step < resume_step:
                            completed_steps += 1
                            continue
                    outputs = model(**batch)

                    loss = outputs['loss']
                    # We keep track of the loss at each epoch
                    if args.with_tracking:
                        total_loss += loss.detach().float()

                    loss = loss / args.gradient_accumulation_steps
                    if isinstance(checkpointing_steps, int):
                        if completed_steps % checkpointing_steps == 0:
                            logger.info(f"loss: {loss}")

                    accelerator.backward(loss)
                    if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()
                        progress_bar.update(1)
                        completed_steps += 1

                    if isinstance(checkpointing_steps, int):
                        if completed_steps % checkpointing_steps == 0:
                            output_dir = f"step_{completed_steps }"
                            if args.output_dir is not None:
                                output_dir = os.path.join(args.output_dir, output_dir)
                            accelerator.save_state(output_dir)

                            model.eval()

                            if args.multicheck:
                                # not supprot accelerator
                                check_batch_mun = 100
                                metric.predictions=[]
                                metric.references=[]

                                for step, batch in enumerate(tqdm(eval_dataloader)):
                                    if check_batch_mun<=0:
                                        break
                                    with torch.no_grad():
                                        outputs = model(**batch)

                                    predictions = outputs['outputs']
                                     # predictions = torch.ge(predictions, 0.5).type(torch.int)
                                    metric.predictions.extend(predictions.cpu().detach().tolist())
                                    metric.references.extend(batch["labels"].type(torch.int))
                                    # check_batch_mun-=1
                                    # print(metric.references[-1])
                                    # print('debug references!')

                                if 'wos' in args.train_file:
                                    predictions = []
                                    lv1_mask = np.array([1] * 7 + [0] * (num_labels - 7))
                                    lv2_mask = (lv1_mask + 1) % 2
                                    for pred in metric.predictions:
                                        pred_ = np.array(pred)
                                        lv1_pred = np.argmax(lv1_mask * pred_)
                                        lv2_pred = np.argmax(lv2_mask * pred_)
                                        res = num_labels * [0]
                                        res[lv1_pred] = 1
                                        res[lv2_pred] = 1
                                        predictions.append(res)
                                    # metric.predictions = predictions
                                    metric.check(predictions)
                                    logger.info(f"best checkpoint:{metric.eval_metric}")
                                else:

                                    if 'wos' in args.train_file:
                                        predictions = []
                                        lv1_mask = np.array([1] * 7 + [0] * (num_labels - 7))
                                        lv2_mask = (lv1_mask + 1) % 2
                                        for pred in metric.predictions:
                                            pred_ = np.array(pred)
                                            lv1_pred = np.argmax(lv1_mask * pred_)
                                            lv2_pred = np.argmax(lv2_mask * pred_)
                                            res = num_labels * [0]
                                            res[lv1_pred] = 1
                                            res[lv2_pred] = 1
                                            predictions.append(res)
                                        # metric.predictions = predictions
                                        metric.check(predictions)
                                        logger.info(f"best checkpoint:{metric.eval_metric}")
                                    else:
                                        best_th = 0.5
                                        default_th = 0.4
                                        best_dir = {}
                                        thresholds = (np.array(range(-20, 20)) / 100) + default_th
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
                                        logger.info(f"best checkpoint:{metric.eval_metric};threshold:{best_th}")
                                        metric.eval_metric['threshold'] = best_th

                                if args.with_tracking:
                                    accelerator.log(
                                        {
                                            "eval_res": metric.eval_metric,

                                            "train_loss": total_loss.item() / len(train_dataloader),
                                            "epoch": epoch,
                                            "step": completed_steps,
                                        },
                                        step=completed_steps,
                                    )

                        if args.child_tune:
                            tune.report(mmicro_f1=metric.eval_metric['f1'])

                    if completed_steps >= args.max_train_steps:
                        break

                model.eval()
                if args.multicheck:
                    metric.predictions = []
                    metric.references = []
                    # not supprot accelerator
                    if len(label_list)>1:
                        for step, batch in enumerate(tqdm(eval_dataloader)):
                            with torch.no_grad():
                                outputs = model(**batch)

                            predictions = outputs['outputs']
                            # predictions = torch.ge(predictions, 0.5).type(torch.int)
                            metric.predictions.extend(predictions.cpu().detach().tolist())
                            metric.references.extend(batch["labels"].type(torch.int))


                    if len(label_list) > 2:

                        if 'wos' in args.train_file:
                            predictions = []
                            lv1_mask = np.array([1] * 7 + [0] * (num_labels - 7))
                            lv2_mask = (lv1_mask + 1) % 2
                            for pred in metric.predictions:
                                pred_ = np.array(pred)
                                lv1_pred = np.argmax(lv1_mask * pred_)
                                lv2_pred = np.argmax(lv2_mask * pred_)
                                res = num_labels * [0]
                                res[lv1_pred] = 1
                                res[lv2_pred] = 1
                                predictions.append(res)
                            # metric.predictions = predictions
                            metric.check(predictions)
                            logger.info(f"best checkpoint:{metric.eval_metric}")
                        else:
                            best_th = 0.5
                            default_th = 0.4
                            best_dir = {}
                            thresholds = (np.array(range(-20, 20)) / 100) + default_th
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
                            logger.info(f"best checkpoint:{metric.eval_metric};threshold:{best_th}")
                            metric.eval_metric['threshold'] = best_th

                    else:
                        metric.eval_metric = {}
                        metric.check()
                        logger.info(f"epoch {epoch}: {metric.eval_metric}")

                    if args.with_tracking:
                        accelerator.log(
                            {
                                "eval_res": metric.eval_metric,
                                "train_loss": total_loss.item() / len(train_dataloader),
                                "epoch": epoch,
                                "step": completed_steps,
                            },
                            step=completed_steps,
                        )

                if args.push_to_hub and epoch < args.num_train_epochs - 1:
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(
                        args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
                    )
                    if accelerator.is_main_process:
                        tokenizer.save_pretrained(args.output_dir)
                        repo.push_to_hub(
                            commit_message=f"Training in progress epoch {epoch}", blocking=False, auto_lfs_prune=True
                        )

                if args.checkpointing_steps == "epoch":
                    output_dir = f"epoch_{epoch}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)



            if args.with_tracking:
                accelerator.end_training()

            if args.output_dir is not None:
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(
                    args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
                )
                if accelerator.is_main_process:
                    tokenizer.save_pretrained(args.output_dir)
                    if args.push_to_hub:
                        repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)


        def simcse_train_cycle():
            model.config.pooling='cls'
            model.config.loss_mess = {
                'loss1': {},
                'loss2': {}
            }

            model.config.train_mode = args.train_mode

            if args.with_tracking:
                accelerator.log(
                    {
                        "eval_res": {'suset_accuracy': 0, 'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0},
                        "train_loss": 0,
                        "epoch": 0,
                        "step": 0,
                    },
                    step=0,
                )

            # Only show the progress bar once on each machine.
            progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
            completed_steps = 0
            starting_epoch = 0
            # Potentially load in the weights and states from a previous save
            if args.resume_from_checkpoint:
                if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
                    accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
                    accelerator.load_state(args.resume_from_checkpoint)
                    path = os.path.basename(args.resume_from_checkpoint)
                else:
                    # Get the most recent checkpoint
                    dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
                    dirs.sort(key=os.path.getctime)
                    path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
                # Extract `epoch_{i}` or `step_{i}`
                training_difference = os.path.splitext(path)[0]

                if "epoch" in training_difference:
                    starting_epoch = int(training_difference.replace("epoch_", "")) + 1
                    resume_step = None
                else:
                    resume_step = int(training_difference.replace("step_", ""))
                    starting_epoch = resume_step // len(train_dataloader)
                    resume_step -= starting_epoch * len(train_dataloader)

            # 训练循环
            for epoch in range(starting_epoch, args.num_train_epochs):
                if args.with_tracking:
                    total_loss = 0
                for step, batch in enumerate(train_dataloader):
                    # print(batch)
                    model.train()
                    # We need to skip steps until we reach the resumed step
                    if args.resume_from_checkpoint and epoch == starting_epoch:
                        if resume_step is not None and step < resume_step:
                            completed_steps += 1
                            continue

                    outputs = model(**batch)

                    loss = outputs.loss
                    # We keep track of the loss at each epoch
                    if args.with_tracking:
                        total_loss += loss.detach().float()

                    loss = loss / args.gradient_accumulation_steps
                    if isinstance(checkpointing_steps, int):
                        if completed_steps % checkpointing_steps == 0:
                            logger.info(f"loss: {loss}")
                            if args.with_tracking:
                                accelerator.log(
                                    {
                                        "loss": loss,
                                        "epoch": epoch,
                                        "step": step,
                                    }
                                )

                    accelerator.backward(loss)

                    if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()
                        progress_bar.update(1)
                        completed_steps += 1

                    if isinstance(checkpointing_steps, int):
                        if completed_steps % checkpointing_steps == 0:
                            output_dir = f"step_{completed_steps}"
                            if args.output_dir is not None:
                                output_dir = os.path.join(args.output_dir, output_dir)
                            accelerator.save_state(output_dir)
                            if args.with_tracking:
                                accelerator.log(
                                    {
                                        "loss": loss,
                                        "epoch": epoch,
                                        "step": step,
                                    }
                                )
                    with open(os.path.join(args.output_dir,'loss_log.txt'),'a',encoding='utf=8') as f:
                        f.write(f'<{step}:{loss}>\n')

                    if completed_steps >= args.max_train_steps:
                        break

        def easy_train_cycle(model_config=None):
            # metric = multicheck(['accuracy', 'precision', 'f1'])
            metric = multicheck(['f1'])

            model.config.loss_mess=model_config
            # Only show the progress bar once on each machine.
            progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
            completed_steps = 0
            starting_epoch = 0
            # Potentially load in the weights and states from a previous save

            # 训练循环
            for epoch in range(starting_epoch, args.num_train_epochs):
                if args.with_tracking:
                    total_loss = 0
                for step, batch in enumerate(train_dataloader):
                    model.train()

                    outputs = model(**batch)

                    loss = outputs.loss
                    # We keep track of the loss at each epoch
                    loss = loss / args.gradient_accumulation_steps

                    accelerator.backward(loss)
                    if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()
                        progress_bar.update(1)
                        completed_steps += 1

                model.eval()
                if args.multicheck:
                    metric.predictions = []
                    metric.references = []
                    # not supprot accelerator
                    if len(label_list)>1:
                        for step, batch in enumerate(tqdm(eval_dataloader)):
                            with torch.no_grad():
                                outputs = model(**batch)
                            if num_labels <= 2:
                                predictions = outputs.logits.argmax(dim=-1)

                                metric.predictions.extend(predictions)
                            else:
                                predictions = torch.sigmoid(outputs.logits.squeeze())
                                # predictions = torch.ge(predictions, 0.5).type(torch.int)
                                metric.predictions.extend(predictions.cpu().detach().tolist())
                            metric.references.extend(batch["labels"].type(torch.int))
                    else:
                        for step, batch in enumerate(tqdm(eval_dataloader)):
                            with torch.no_grad():
                                outputs = model(**batch)
                            predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
                            metric.predictions.extend(predictions)
                            metric.references.extend(batch["labels"])

                    if len(label_list) > 2:

                        if 'wos' in args.train_file:
                            predictions = []
                            lv1_mask = np.array([1] * 7 + [0] * (num_labels - 7))
                            lv2_mask = (lv1_mask + 1) % 2
                            for pred in metric.predictions:
                                pred_ = np.array(pred)
                                lv1_pred = np.argmax(lv1_mask * pred_)
                                lv2_pred = np.argmax(lv2_mask * pred_)
                                res = num_labels * [0]
                                res[lv1_pred] = 1
                                res[lv2_pred] = 1
                                predictions.append(res)
                            # metric.predictions = predictions
                            metric.check(predictions)
                            logger.info(f"best checkpoint:{metric.eval_metric}")
                        else:
                            best_th = 0.5
                            default_th = 0.4
                            best_dir = {}
                            thresholds = (np.array(range(-20, 20)) / 100) + default_th
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
                            logger.info(f"best checkpoint:{metric.eval_metric};threshold:{best_th}")
                            metric.eval_metric['threshold'] = best_th



                    else:
                        metric.eval_metric = {}
                        metric.check()
                        logger.info(f"epoch {epoch}: {metric.eval_metric}")

                else:
                    return ValueError('we need multicheck')


        if args.child_tune:
            x = {
                    'reweight_func':tune.choice(['CB','rebalance',None]),
                    'focal':True,
                    'alpha':tune.grid_search([i/10 for i in range(3,16,3)]),
                    'CB_loss':True,
                    'CB_loss_alpha':tune.grid_search([i/10 for i in range(6,16,3)]),
                    'map_param':True,
                    'map_alpha':tune.grid_search([i/100 for i in range(6,16,3)]),
                    'map_beta':tune.grid_search([i*10. for i in range(3,13,3)]),
                    'map_gamma':0.05
                }
            model_config={
                'loss1':x,
                'loss2':x
            }

            analysis = tune.run(
                easy_train_cycle,
                mode='max',
                config=model_config, local_dir='./')
            res_para = analysis.get_best_config(metric="f1", mode='max')
            print("Best config: ", res_para)
            with open(os.path.join(args.output_dir,'best_para.txt'),'w',encoding='utf-8') as f:
                f.write(res_para)

        elif args.train_mode=='simcse' or args.train_mode=='simcse_sup':
            simcse_train_cycle()

            if args.output_dir is not None:
                accelerator.save_state(args.output_dir)
            return
        elif args.train_mode == 'hm12':
            model.config.train_mode = args.train_mode
            hm_train_cycle()
            if args.output_dir is not None:
                accelerator.save_state(args.output_dir)
            return

        else:
            train_cycle()

    # 开始评估
    print('-----------------------------------------------------------------------------')
    print('start eval')
    if args.task_name == "mnli":
        # Final evaluation on mismatched validation set
        eval_dataset = processed_datasets["validation_mismatched"]
        eval_dataloader = DataLoader(
            eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
        )
        eval_dataloader = accelerator.prepare(eval_dataloader)

    model.eval()
    if args.multicheck:
        metric.predictions = []
        metric.references = []
        # not supprot accelerator
        if num_labels > 1:
            for step, batch in enumerate(tqdm(eval_dataloader)):
                with torch.no_grad():
                    outputs = model(**batch)
                if num_labels <= 2:
                    predictions = outputs.logits.argmax(dim=-1)
                    metric.predictions.extend(predictions)

                else:
                    if args.train_mode=='hm12':
                        predictions = outputs['outputs']
                    else:
                        predictions = torch.sigmoid(outputs.logits.squeeze())
                    # predictions = torch.ge(predictions, 0.5).type(torch.int)
                    metric.predictions.extend(predictions.cpu().detach().tolist())
                metric.references.extend(batch["labels"].type(torch.int))

        else:
            for step, batch in enumerate(tqdm(eval_dataloader)):
                with torch.no_grad():
                    outputs = model(**batch)
                predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
                metric.predictions.extend(predictions)
                metric.references.extend(batch["labels"])

        if len(label_list) > 2:

            if 'wos' in args.train_file:
                # print('debug here!!!!')
                predictions = []
                lv1_mask = np.array([1] * 7 + [0] * (num_labels - 7))
                lv2_mask = (lv1_mask + 1) % 2
                for pred in metric.predictions:
                    pred_ = np.array(pred)
                    lv1_pred = np.argmax(lv1_mask * pred_)
                    lv2_pred = np.argmax(lv2_mask * pred_)
                    res = num_labels * [0]
                    res[lv1_pred] = 1
                    res[lv2_pred] = 1
                    predictions.append(res)
                    # print(res)
                # metric.predictions = predictions
                metric.check(predictions)

                logger.info(f"best checkpoint:{metric.eval_metric}")
            else:
                metric.predictions = torch.tensor(metric.predictions,
                                                  device='cuda' if torch.cuda.is_available() else 'cpu')
                if int(args.threshold)==1:
                    best_th = 0.5
                    default_th = 0.4
                    best_dir = {}
                    thresholds = (np.array(range(-20, 20)) / 100) + default_th
                    best_f1 = 0

                    for threshold in thresholds:
                        metric.eval_metric = {}
                        predictions = torch.ge(metric.predictions, threshold).type(torch.int)
                        # print(predictions)
                        metric.check(predictions)
                        f1 = 'micro-f1'
                        # f1 = 'macro-f1'
                        if metric.eval_metric[f1] > best_f1:
                            best_f1 = metric.eval_metric[f1]
                            best_dir = metric.eval_metric
                            best_th = threshold
                    if best_dir:
                        metric.eval_metric = best_dir

                    logger.info(f"best checkpoint:{metric.eval_metric};threshold:{best_th}")
                    metric.eval_metric['threshold'] = best_th
                else:
                    metric.eval_metric = {}
                    predictions = torch.ge(metric.predictions, args.threshold).type(torch.int)
                    # print(predictions)
                    metric.check(predictions)

                    logger.info(f"best checkpoint:{metric.eval_metric};threshold:{args.threshold}")
                    metric.eval_metric['threshold'] = args.threshold

        else:
            metric.eval_metric = {}
            metric.check()
            logger.info(f"eval_res:{metric.eval_metric}")

        if args.output_dir is not None and int(args.threshold) != 1:
            with open(os.path.join(args.output_dir, "test_results.json"), "w") as f:
                print(metric.eval_metric)
                json.dump(metric.eval_metric, f)
        elif args.output_dir is not None:
            with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                print(metric.eval_metric)
                json.dump(metric.eval_metric, f)
    else:
        for step, batch in enumerate(tqdm(eval_dataloader)):
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            metric.add_batch(
                predictions=accelerator.gather(predictions),
                references=accelerator.gather(batch["labels"]),
            )
        eval_metric = metric.compute()
        logger.info(f"eval_res: {eval_metric}")

        if args.output_dir is not None:
            with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                json.dump({"eval_accuracy": eval_metric["accuracy"]}, f)


if __name__ == "__main__":
    main()
