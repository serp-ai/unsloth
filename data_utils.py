#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import io
import json
import os
import gc
import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Callable, List, Tuple, Union, Any
import transformers

import torch
from torch.utils.data import Dataset

import warnings

warnings.filterwarnings("ignore")


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"


class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        # print('Saving PEFT checkpoint...')
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(
                state.best_model_checkpoint, "adapter_model"
            )
        else:
            checkpoint_folder = os.path.join(
                args.output_dir, f"checkpoint-{state.global_step}"
            )

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        model = kwargs["model"]
        model.save_pretrained(peft_model_path)

        moe_state = {}
        for param_tensor in model.state_dict():
            if "adapter" in param_tensor:
                moe_state.update({param_tensor: model.state_dict()[param_tensor]})
            # if "adapter" in param_tensor or "norm" in param_tensor:
            #     moe_state.update({param_tensor: model.state_dict()[param_tensor]})
        moe_model_path = os.path.join(checkpoint_folder, "moe_model.bin")
        # print(moe_state.keys())
        torch.save(moe_state, moe_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        def touch(fname, times=None):
            with open(fname, "a"):
                os.utime(fname, times)

        touch(os.path.join(args.output_dir, "completed"))
        self.save_model(args, state, kwargs)


def _tokenize_fn(
    strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [
        _tokenize_fn(strings, tokenizer) for strings in (examples, sources)
    ]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()

        logging.warning("Loading data: {}".format(data_path))
        data_list = jload(data_path)

        # Preprocess Data
        logging.warning("Processing data")
        self.tokenizer = tokenizer
        self.sources = []
        self.targets = []

        system_str = "<|im_start|>system\n{message}<|im_end|>\n"
        user_str = "<|im_start|>user\n{message}<|im_end|>\n"
        assistant_str = "<|im_start|>assistant\n{message}<|im_end|>\n"

        # pylint: disable-next=consider-using-enumerate
        for idx in range(len(data_list)):
            data = data_list[idx]
            corpus = data.get("corpus", "")
            if corpus != "":
                # pretrain mode
                source = f"{tokenizer.bos_token}"
                self.sources.append(source)

                target = f"{corpus}{tokenizer.eos_token}"
                self.targets.append(target)
            else:
                conversations = data["conversations"]
                conversation_str = f"{tokenizer.bos_token}"
                for conversation in conversations[:-1]:
                    if conversation["from"] in ["human", "user"]:
                        conversation_str += user_str.format(
                            message=conversation["value"]
                        )
                    elif conversation["from"] in ["gpt", "assistant"]:
                        conversation_str += assistant_str.format(
                            message=conversation["value"]
                        )
                    elif conversation["from"] in ["system", "instruction"]:
                        conversation_str += system_str.format(
                            message=conversation["value"]
                        )
                    else:
                        raise ValueError(
                            f"Unknown conversation type: {conversation['from']}"
                        )
                conversation_str += "<|im_start|>assistant\n"
                self.sources.append(conversation_str)

                last_message = conversations[-1]
                self.targets.append(f"{last_message['value']}{tokenizer.eos_token}")

        del data_list
        gc.collect()
        logging.warning("there are {} data in dataset".format(len(self.sources)))

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, i):
        source = [self.sources[i]]
        target = [self.targets[i]]
        data_dict = preprocess(source, target, self.tokenizer)

        input_ids = data_dict["input_ids"][0]
        labels = data_dict["labels"][0]

        return dict(input_ids=input_ids, labels=labels)


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_path: str
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jdump(obj, f, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict
