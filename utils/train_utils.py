from bitsandbytes.optim.ademamix import AdEMAMix
from tqdm.auto import tqdm

import torch
from torch import nn
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
import os

import gc

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader

def memory_cleanup():
    """Perform thorough memory cleanup"""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def destruct_module_optimized(module: torch.nn.Module) -> torch.nn.Module:
    """Efficiently destroy module and clear memory."""
    module.to_empty(device="meta")
    memory_cleanup()

def create_model_tokenizer(
    model_id: str,
    load_model: bool = True,
    hidden_size: int = 512
):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    
    embed_tokens, lm_head, norm, hidden_size = None, None, None, hidden_size
    if load_model:
        if hidden_size is not None:
            print("hidden_size will be ignored as load_model is True")
        model = AutoModelForCausalLM.from_pretrained(model_id)

        embed_tokens = deepcopy(model.model.embed_tokens)
        lm_head = deepcopy(model.lm_head)
        norm = deepcopy(model.model.norm)
        hidden_size = embed_tokens.weight.shape[-1]

        destruct_module_optimized(model)
        memory_cleanup()
    else:
        assert hidden_size is not None, "hidden_size must be provided if load_model is False"

    # IMPORTANT for batched generation with this architecture
    tokenizer.padding_side = 'right'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    vocab_size = tokenizer.vocab_size

    return tokenizer, embed_tokens, lm_head, norm, vocab_size, hidden_size

def batch_tokenize(tokenizer, texts, padding="max_length", batch_size=256, max_length=512, device='cuda'):
    tokenized_batch = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        
        if padding and max_length:
            return_pt=True
            tokenized = tokenizer(batch, padding=padding, truncation=True, max_length=max_length, return_tensors='pt')['input_ids']
            tokenized_batch.append(tokenized)
        else:
            return_pt=False
            tokenized = tokenizer(batch)['input_ids']
            tokenized_batch.extend(tokenized)
    
    if return_pt:
        return torch.cat(tokenized_batch, dim=0)
    return tokenized_batch


def create_dataset(
    dataset_id: str,
    split: str = "train",
    field: str = "text",
    num_train_samples: int = 100000,
    num_test_samples: int = 10000,
):
    dataset = load_dataset(dataset_id)
    raw_train_set = list(dataset[split].select(range(num_train_samples))[field])
    raw_test_set = list(dataset[split].select(range(num_train_samples, num_train_samples + num_test_samples))[field])
    return raw_train_set, raw_test_set