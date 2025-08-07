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
    accumulation_steps=4,  # Add accumulation_steps parameter
):
    # module is trainable, embed_tokens, lm_head, norm are frozen

    ## Move to device and ensure consistent dtype
    module = module.to(device=device, dtype=dtype)

    ## Compile module
    if do_compile:
        module = torch.compile(module)

    ## Create optimizer and loss
    optimizer = AdEMAMix(module.parameters(), lr=2e-4)
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

    # Checkpointing setup
    checkpoint_dir = f"checkpoints/{run_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_interval = 200
    
    last_checkpoint = None
    best_loss = float('inf')
    spike_threshold =1.5  # If loss increases by 3x, consider it a spike

    # Moving average setup
    moving_avg_window = 64  # Changed from 50 to 8
    recent_moving_avg_window = 8
    
    loss_history = []
    
    moving_avg_loss = None
    moving_avg_short_loss = None

    ## Train for multiple epochs with batching
    num_batches = len(train_set) // batch_size
    global_step = 0

    for epoch in tqdm(range(epochs), desc="Epochs"):
        epoch_loss = 0.0
        # Iterate directly over the train_dataloader
        for i in tqdm(range(num_batches), desc="Batches", leave=False):
            input_ids = train_set[i * batch_size : (i + 1) * batch_size].to(device)

            # forward pass
            logits = module(input_ids)

            # loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()

            shift_logits = shift_logits.view(-1, tokenizer.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fn(shift_logits, shift_labels) / accumulation_steps  # Scale loss by accumulation_steps

            # backward pass
            loss.backward()

            # Update loss history and calculate moving average
            loss_history.append(loss.item())  # Scale back for logging
            
            if len(loss_history) > max(moving_avg_window, recent_moving_avg_window):
                loss_history.pop(0)

            moving_avg_loss = sum(loss_history) / len(loss_history)
            moving_avg_short_loss = sum(loss_history[-8:]) / 8

            # Check for loss spike using moving average
            if len(loss_history) >= moving_avg_window and moving_avg_short_loss > spike_threshold * moving_avg_loss:
                print(f"Loss spike detected at step {global_step}. Restoring last checkpoint.")
                if last_checkpoint:
                    module.load_state_dict(torch.load(last_checkpoint))
                    optimizer.load_state_dict(torch.load(f"{last_checkpoint}.optim"))
                    # Revert step counter and continue from last checkpoint
                    global_step = int(last_checkpoint.split("_")[-1].split(".")[0])
                    # Reset loss history after restoring checkpoint
                    loss_history = []
                    moving_avg_loss = None
                    continue

            # Perform optimizer step and zero grad every accumulation_steps
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                
                global_step += 1

                # Save checkpoint every 200 steps
            if (i+1) % checkpoint_interval == 0:
                checkpoint_path = f"{checkpoint_dir}/checkpoint_{global_step}.pth"
                torch.save(module.state_dict(), checkpoint_path)
                torch.save(optimizer.state_dict(), f"{checkpoint_path}.optim")
                # Keep track of the last checkpoint
                last_checkpoint = checkpoint_path

            epoch_loss += loss.item() * accumulation_steps  # Scale back for logging

            # Log batch loss to TensorBoard
            writer.add_scalar('Training/Batch Loss', loss.item() * accumulation_steps, global_step)

            # Update progress bar with current loss
            tqdm.write(f"Batch {i+1}/{num_batches} - Loss: {loss.item() * accumulation_steps:.4f}", end='\r')

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

                logits = module(input_ids)

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

    destruct_module_optimized(module.stacked_mixin_block)
    memory_cleanup()