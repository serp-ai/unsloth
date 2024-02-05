from bitsandbytes.nn import Linear4bit as Bnb_Linear4bit
from peft.tuners.lora import Linear4bit as Peft_Linear4bit

from unsloth.models.sparsetral import FastSparsetralModel
import torch
from torch import nn
import math

from data_utils import make_supervised_data_module, SavePeftModelCallback

max_seq_length = 4096
dtype = torch.bfloat16
load_in_4bit = True

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.1-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
    "unsloth/llama-2-7b-bnb-4bit",
    "unsloth/llama-2-13b-bnb-4bit",
    "unsloth/codellama-34b-bnb-4bit",
    "unsloth/tinyllama-bnb-4bit",
]

model, tokenizer = FastSparsetralModel.from_pretrained(
    model_name="unsloth/mistral-7b-instruct-v0.2-bnb-4bit",  # Choose ANY! eg teknium/OpenHermes-2.5-Mistral-7B
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

model = FastSparsetralModel.get_peft_model(
    model,
    r=64,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",  # Supports any, but = "none" is optimized
    use_gradient_checkpointing=True,
    random_state=741,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
)

for name, module in model.named_modules():
    if "adapter" in name or "router" in name:
        if isinstance(module, (Bnb_Linear4bit, Peft_Linear4bit)):
            # Create a new Linear module
            new_module = (
                torch.nn.Linear(module.in_features, module.out_features, bias=False)
                .to(model.device)
                .to(torch.bfloat16)
            )
        else:
            new_module = module.to(torch.bfloat16)
        # Get the attribute name to set the new module
        parent_name, child_name = name.rsplit(".", 1)
        parent_module = dict(model.named_modules())[parent_name]

        # Replace the old module with the new one
        setattr(parent_module, child_name, new_module)

# Zero Init
for n, p in model.named_parameters():
    if "adapter_up" in n:
        nn.init.zeros_(p)
    if "adapter_down" in n:
        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
    if "router" in n:
        nn.init.kaiming_uniform_(p, a=math.sqrt(5))

for n, p in model.named_parameters():
    if "adapter" in n or "router" in n:
        p.requires_grad = True


data_module = make_supervised_data_module(tokenizer, "data/openhermes2_5.json")

# from trl import SFTTrainer
from transformers import TrainingArguments, Trainer

tokenizer.model_max_length = max_seq_length
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=128,
        warmup_steps=200,
        num_train_epochs=1,
        # ddp_find_unused_parameters=False, If using DDP and you get an error, uncomment this line
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=741,
        output_dir="outputs",
        save_strategy="steps",
        save_steps=1000,
    ),
    **data_module,
)
trainer.add_callback(SavePeftModelCallback)

# alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriat>

# ### Instruction:
# {}

# ### Input:
# {}

# ### Response:
# {}"""

# EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN


# def formatting_prompts_func(examples):
#     instructions = examples["instruction"]
#     inputs = examples["input"]
#     outputs = examples["output"]
#     texts = []
#     for instruction, input, output in zip(instructions, inputs, outputs):
#         # Must add EOS_TOKEN, otherwise your generation will go on forever!
#         text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
#         texts.append(text)
#     return {
#         "text": texts,
#     }


# pass

# from datasets import load_dataset

# dataset = load_dataset("yahma/alpaca-cleaned", split="train")
# dataset = dataset.map(
#     formatting_prompts_func,
#     batched=True,
# )


# from trl import SFTTrainer
# from transformers import TrainingArguments

# trainer = SFTTrainer(
#     model=model,
#     tokenizer=tokenizer,
#     train_dataset=dataset,
#     dataset_text_field="text",
#     max_seq_length=max_seq_length,
#     dataset_num_proc=2,
#     packing=False,  # Can make training 5x faster for short sequences.
#     args=TrainingArguments(
#         per_device_train_batch_size=8,
#         gradient_accumulation_steps=4,
#         warmup_steps=5,
#         max_steps=50,
#         learning_rate=2e-4,
#         fp16=not torch.cuda.is_bf16_supported(),
#         bf16=torch.cuda.is_bf16_supported(),
#         logging_steps=1,
#         optim="adamw_8bit",
#         weight_decay=0.01,
#         lr_scheduler_type="linear",
#         seed=741,
#         output_dir="outputs",
#     ),
# )


# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")


trainer_stats = trainer.train()

model.save_pretrained("outputs")


# @title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")
