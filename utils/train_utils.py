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

def destruct_module_optimized(module: torch.nn.Module) -> torch.nn.Module:
    """Efficiently destroy module and clear memory."""
    module.to_empty(device="meta")
    gc.collect()
    torch.cuda.empty_cache()

def memory_cleanup():
    """Perform thorough memory cleanup"""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def create_model(
    model_id: str,
):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_id)

    embed_tokens = deepcopy(model.model.embed_tokens)
    lm_head = deepcopy(model.lm_head)
    norm = deepcopy(model.model.norm)

    destruct_module_optimized(model)
    memory_cleanup()

    # IMPORTANT for batched generation with this architecture
    tokenizer.padding_side = 'right'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    vocab_size = tokenizer.vocab_size
    hidden_size = embed_tokens.weight.shape[-1]
    return tokenizer, embed_tokens, lm_head, norm, vocab_size, hidden_size

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

def train_loop(
    module,
    run_name,
    do_compile,
    tokenizer,
    device,
    dtype,
    train_set,
    test_set,
    epochs,
    batch_size,
    max_length,
    embed_tokens,
    lm_head,
    norm,
):
    # module is trainable, embed_tokens, lm_head, norm are frozen
    module.train()
    for parameters in embed_tokens.parameters():
        parameters.requires_grad = False

    for parameters in lm_head.parameters():
        parameters.requires_grad = False

    for parameters in norm.parameters():
        parameters.requires_grad = False

    ## Move to device and ensure consistent dtype
    module = module.to(device=device, dtype=dtype)

    embed_tokens = embed_tokens.to(device=device, dtype=dtype)
    lm_head = lm_head.to(device=device, dtype=dtype)
    norm = norm.to(device=device, dtype=dtype)

    ## Compile module
    if do_compile:
        module = torch.compile(module)

    ## Create optimizer and loss
    optimizer = AdEMAMix(module.parameters(), lr=3e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Determine a reasonable number of workers for parallel data loading
    num_workers = os.cpu_count() // 2 if os.cpu_count() else 0
    
    if num_workers > 0:
        print(f"Using {num_workers} workers for DataLoader.")
    else:
        print("Using 0 workers for DataLoader (main process).")

    # Calculate total number of steps based on DataLoader length
    n_steps = len(train_set) // batch_size * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=1e-6)

    # TensorBoard writer
    writer = SummaryWriter(log_dir=f"runs/{run_name}")

    ## Train for multiple epochs with batching
    num_batches = len(train_set) // batch_size

    for epoch in tqdm(range(epochs), desc="Epochs"):
        epoch_loss = 0.0
        # Iterate directly over the train_dataloader
        for i in tqdm(range(num_batches), desc="Batches", leave=False):
            input_ids = train_set[i * batch_size : (i + 1) * batch_size].to(device)

            # embed
            x = embed_tokens(input_ids)

            # forward pass
            y = module(x)

            # norm
            y = norm(y)
            logits = lm_head(y)

            # loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()

            shift_logits = shift_logits.view(-1, tokenizer.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fn(shift_logits, shift_labels)

            # backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            epoch_loss += loss.item()

            # Log batch loss to TensorBoard
            global_step = epoch * num_batches + i
            writer.add_scalar('Training/Batch Loss', loss.item(), global_step)

            # Update progress bar with current loss
            tqdm.write(f"Batch {i+1}/{num_batches} - Loss: {loss.item():.4f}", end='\r')

        # Calculate average loss for the epoch
        avg_epoch_loss = epoch_loss / num_batches
        print(f"\nEpoch {epoch+1}/{epochs} - Average Loss: {avg_epoch_loss:.4f}")

        # Log epoch loss to TensorBoard
        writer.add_scalar('Training/Epoch Loss', avg_epoch_loss, epoch)

        # Evaluation after each epoch
        module.eval()
        eval_loss = 0.0
        
        num_batches = len(test_set) // batch_size
        with torch.no_grad():
            # Iterate directly over the test_dataloader
            for i in range(num_batches):
                input_ids = test_set[batch_size * i : batch_size * (i + 1)].to(device)

                x = embed_tokens(input_ids)
                y = module(x)
                y = norm(y)
                logits = lm_head(y)

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                shift_logits = shift_logits.view(-1, tokenizer.vocab_size)
                shift_labels = shift_labels.view(-1)
                shift_labels = shift_labels.to(shift_logits.device)
                eval_loss += loss_fn(shift_logits, shift_labels).item()

        avg_eval_loss = eval_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs} - Evaluation Loss: {avg_eval_loss:.4f}")

        # Log evaluation loss to TensorBoard
        writer.add_scalar('Evaluation/Loss', avg_eval_loss, epoch)

        module.train()

    # Close TensorBoard writer
    writer.close()

    # Save model state
    os.makedirs(f"model_states/", exist_ok=True)
    torch.save(module.state_dict(), f"model_states/{run_name}.pth")

    destruct_module_optimized(module)
    memory_cleanup()