# Recurrent Hyper Network

## Definitions

Principle:
- let T1, T2, ..., TK be the tokens of a sequence
- let 1 < n < depth be the index of depth in a network
- let 1 < k < seq_len be the index of a sequence

We can define X_k:n the latent representation of the token k after layer n.
We can also define the FFN_n the feed forward network at layer n.

## Other definitions
- GQA_n : Grouped Query Attention at layer n
- RMSNorm : Root Mean Square Layer Normalization
- Embedding : Embedding layer
- Unembedding : Unembedding layer
- Dora : Dynamic Rank Adaptation - Dora is a method to adapt the rank of a matrix through lora combined with a magnitude vector
- Transformer Block (TB) : standard transformer block
- Transformer : standard transformer

## Transformer pseudo code

TB pseudo code with x of shape (batch_size, seq_len, hidden_size):

```python
residual = x
x = RMSNorm(x)
x = GQA(x)
x = residual + x
residual = x
x = RMSNorm(x)
x = FFN(x)
x = residual + x
```

Or in sequential form:

```python
residual = x
x = RMSNorm(x)
x = GQA(x) # mixin block
x = residual + x
residual = x
x = RMSNorm(x)
for k in seq_len:
  x[:, k, :] = FFN(x[:, k, :]) # mixin block
x = residual + x
```

Transformer pseudo code:
```python
x = Embedding(x)
for 1 < n < depth:
  x = TB_n(x)
x = RMSNorm(x)
x = Unembedding(x)
```

Transformer pseudo code with sequential form:

```python
x = Embedding(x)
for 1 < n < depth:
  residual = x
  x = RMSNorm(x)
  x = GQA(x) # mixin block
  x = residual + x
  residual = x
  x = RMSNorm(x)
  for k in seq_len:
    x[:, k, :] = FFN(x[:, k, :]) # mixin block
  x = residual + x
x = RMSNorm(x)
x = Unembedding(x)
```

## Implementation
### Naive Implementation
#### RHNBlock : Recurent Hyper Network Block pseudo code
```python
residual = x
x = RMSNorm(x)
for k in seq_len:
  if k = 0:
    x[:, k, :] = FFN(x[:, k, :]) # mixin block
  else:
    dora_a, dora_b, magnitude, beta = BHN(x[:, k-1, :]) # Backward Hyper Network
    x[:, k, :] = DoraFFN(x[:, k, :], dora_a, dora_b, magnitude) # mixin block
x = residual + x
```

FFN contains the following linear layers:
- gate_proj : (hidden_size, intermediate_size)
- up_proj : (hidden_size, intermediate_size)
- down_proj : (intermediate_size, hidden_size)

with:
- dora_a of shape [
    gate_proj_A, # (batch_size, rank, intermediate_size)
    up_proj_A, # (batch_size, rank, intermediate_size)
    down_proj_A # (batch_size, rank, hidden_size)
],
- dora_b of shape [
    gate_proj_B, # (batch_size, intermediate_size, rank)
    up_proj_B, # (batch_size, intermediate_size, rank)
    down_proj_B # (batch_size, hidden_size, rank)
],
- magnitude of shape [
    gate_proj_magnitude, # (batch_size, intermediate_size)
    up_proj_magnitude, # (batch_size, intermediate_size)
    down_proj_magnitude # (batch_size, hidden_size)
]
- beta # (batch_size, 1)

FFN pseudo code (x of shape (batch_size, hidden_size)):
- gate = F.silu(gate_proj(x))
- up = up_proj(x)
- down = down_proj(gate * up)
- return down

DoraFFN pseudo code (x of shape (batch_size, hidden_size)):
- gate = SILU(Dora(x, gate_proj, dora_a[0], dora_b[0], magnitude[0]) + beta)
- up = Dora(x, up_proj, dora_a[1], dora_b[1], magnitude[1])
- down = Dora(gate * up, down_proj, dora_a[2], dora_b[2], magnitude[2])

#### Naive RHN pseudo code
```python
x = Embedding(x)
for 1 < n < depth:
  x = RHNBlock(x)
x = RMSNorm(x)
x = Unembedding(x)
```

#### Naive RHN pseudo code With detail
```python
x = Embedding(x)
for 1 < n < depth:
    x = RMSNorm(x)
    for k in seq_len:
    if k = 0:
        x[:, k, :] = FFN(x[:, k, :]) # mixin block
    else:
        dora_a, dora_b, magnitude, beta = BHN(x[:, k-1, :]) # Backward Hyper Network
        x[:, k, :] = DoraFFN(x[:, k, :], dora_a, dora_b, magnitude) # mixin block
    x = residual + x
x = RMSNorm(x)
x = Unembedding(x)
```

### Optimized Implementation
#### Training RHN pseudo code
During training, the block decomposition of RHN is far from optimal, as we need to wait for each every intermediate state at depth n to be able to start computing depth n+1.

Instead, to minimize memory footprint and maximize parallelism, we should process states with the same n + k index in parallel, additionnaly, dora adapters should be calculated dynamically 

Adapted pseudo code for training is:

```python
states = {}

FFNs = [FFN_1, FFN_2, ..., FFN_depth]
BHNs = [BHN_1, BHN_2, ..., BHN_depth]

x_embedded = Embedding(x)

for k in 0..seq_len-1:
    states[(k, 0)] = x_embedded[:, k, :]

for s in 1..(depth + seq_len - 1):
    for n in 1..depth:
        k = s - n
        
        if 0 <= k < seq_len:
        
            x_input = states[(k, n - 1)]
            residual = x_input
            x_norm = RMSNorm(x_input)
            if k == 0:
                x_processed = FFNs[n-1](x_norm)
            
            else:
                recurrent_state = states[(k - 1, n)]
                
                dora_a, dora_b, magnitude, beta = BHNs[n-1](recurrent_state)
            
                x_processed = DoraFFN(x_norm, FFNs[n-1], dora_a, dora_b, magnitude, beta)

            states[(k, n)] = residual + x_processed
            
    keys_to_prune = [key for key in states.keys() if key[0] + key[1] < s - 1]
    for key in keys_to_prune:
        del states[key]

final_hidden_states = []
for k in 0..seq_len-1:
    final_hidden_states.append(states[(k, depth)])

output_tensor = stack(final_hidden_states, dim=1)

output_tensor = RMSNorm(output_tensor)
logits = Unembedding(output_tensor)

return logits
```

### Inference: Batch 1 Autoregressive Generation with Caching

For autoregressive generation (e.g., in a chatbot or single-user scenario), we generate one token at a time. The key to efficient generation is to cache the intermediate states of the *previous token* so they don't need to be recomputed at every step.

In a standard Transformer, the cache holds the Key and Value matrices for each layer. For RHN, the cache needs to hold the hidden state of the previous token at each layer, `X_{k-1}:n`, which is the input to the `BHN`.

#### Principle

1.  **Prompt Processing**: The initial prompt is processed in a single forward pass. The hidden states of the *last token* of the prompt, `X_{prompt_len-1}:1, ..., X_{prompt_len-1}:depth`, are saved into a cache.
2.  **Iterative Generation**: For each new token `k`, we perform a single forward pass through the `depth` of the network.
3.  **Cache Usage**: At each layer `n`, the `BHN` uses the cached state `X_{k-1}:n` to generate the DoRA parameters for the current token's FFN.
4.  **Cache Update**: After computing the new token's state `X_k:n`, it replaces the old `X_{k-1}:n` in the cache, making it ready for generating token `k+1`.
5.  **Sampling**: After computing the final state `X_k:depth`, we apply the final normalization and unembedding layers to get logits. Standard sampling techniques (temperature, top-p, top-k, etc.) are then used to select the next token.

#### Recurrent Cache Definition
-   `recurrent_cache`: A list or array of tensors, of size `depth`.
-   `recurrent_cache[n]` stores the hidden state `X_{k-1}:n+1` of the previously processed token `k-1` at layer `n+1`.
-   The shape of each tensor in the cache is `(hidden_size)`.

#### Autoregressive Generation Pseudo Code

```python
function generate(prompt_tokens, max_new_tokens, sampling_params):
    # 1. Prompt Processing
    # Process the initial prompt using the optimized training-style forward pass
    # to populate the initial hidden states.
    prompt_states = RHN_forward_pass(prompt_tokens)
    
    # Initialize the cache with the states of the LAST prompt token
    recurrent_cache = []
    last_prompt_token_idx = len(prompt_tokens) - 1
    for n in 1..depth:
        # Get the state of the last token at layer n
        cache_state = prompt_states.get_state(last_prompt_token_idx, n)
        recurrent_cache.append(cache_state)
    
    # The first token for the loop is the last token of the prompt
    next_token = prompt_tokens[-1]
    generated_sequence = prompt_tokens
    
    # 2. Iterative Generation Loop
    for _ in 1..max_new_tokens:
        # Input is a single token of shape (1, 1)
        x_k_0 = Embedding(next_token) # Shape: (1, hidden_size)
        
        # This will hold the state of the current token as it passes through layers
        current_token_state = x_k_0
        new_recurrent_cache = []

        # Pass the single token through all layers
        for n in 1..depth:
            residual = current_token_state
            x_norm = RMSNorm(residual) # Layer-specific RMSNorm
            
            # Get the previous token's state for this layer from the cache
            recurrent_input = recurrent_cache[n-1]
            
            # Generate DoRA parameters from the previous token's state
            dora_a, dora_b, magnitude, beta = BHNs[n-1](recurrent_input)
            
            # Apply the DoraFFN
            x_processed = DoraFFN(x_norm, FFNs[n-1], dora_a, dora_b, magnitude, beta)
            
            # Compute the current token's state at this layer
            current_token_state = residual + x_processed
            
            # Store this new state to update the cache after the loop
            new_recurrent_cache.append(current_token_state)

        # Update the cache for the next generation step
        recurrent_cache = new_recurrent_cache
        
        # 3. Finalization and Sampling
        final_state = RMSNorm(current_token_state)
        logits = Unembedding(final_state) # Shape: (1, vocab_size)
        
        # Apply sampling strategies (temperature, top_p, top_k, etc.)
        next_token = sample(logits, sampling_params)
        
        # Check for stop condition (e.g., EOS token)
        if next_token == EOS_TOKEN:
            break
            
        generated_sequence.append(next_token)

    return generated_sequence
```

### Inference: Batch N Continuous Batching Generation

To serve multiple generation requests concurrently and maximize GPU utilization, we use continuous batching. This technique processes a dynamic batch of tokens, where each token belongs to a different, independent generation request.

#### Principle

The core logic is an extension of the single-batch generation. Instead of a single `recurrent_cache`, a `CacheManager` tracks a separate cache for each ongoing request.

1.  **Request Management**: A scheduler groups active requests into a micro-batch for processing in the current step.
2.  **Batched Gathering**: For the micro-batch, we gather the last token of each request and their corresponding `recurrent_cache` states from the `CacheManager`. All inputs are now batched.
3.  **Batched Forward Pass**: The model performs a single-token forward pass on the entire batch. All operations (RMSNorm, BHN, DoraFFN) are now batched over the number of active requests.
4.  **Batched Cache Update**: The newly computed hidden states for each request in the batch are collected.
5.  **Batched Sampling**: Logits are produced for the entire batch. Sampling is applied individually to each request's logit vector, respecting its specific sampling parameters.
6.  **State Update**: The `CacheManager` is updated with the new `recurrent_cache` for each request, and the newly generated tokens are appended to their respective sequences.

#### Cache Manager Definition
-   `CacheManager`: A key-value store (e.g., a dictionary) that maps a unique `request_id` to its `recurrent_cache`.
-   `CacheManager.get_caches(request_ids)`: Gathers the caches for a batch of requests and stacks them into batched tensors.
-   `CacheManager.update_caches(request_ids, new_caches)`: Updates the caches for a batch of requests with their new states.

#### Continuous Batching Generation Step Pseudo Code

```python
# This function represents a single iteration of the server's generation loop.

function process_batch_step(active_requests, cache_manager):
    # active_requests: a list of request_ids to process in this batch.
    
    # 1. Gather Inputs
    # Gather the last token from each active request
    token_batch = gather_last_tokens([req.sequence for req in active_requests])
    # Shape: (num_requests)
    
    # Gather the recurrent caches for each request and stack them
    # recurrent_cache_batch is a list of `depth` tensors.
    # Each tensor has shape (num_requests, hidden_size).
    recurrent_cache_batch = cache_manager.get_caches([req.id for req in active_requests])

    # 2. Batched Forward Pass
    x_k_0_batch = Embedding(token_batch) # Shape: (num_requests, hidden_size)
    
    current_token_states_batch = x_k_0_batch
    new_recurrent_cache_batch = [None] * depth

    for n in 1..depth:
        residual_batch = current_token_states_batch
        x_norm_batch = RMSNorm(residual_batch)
        
        # Get the batched recurrent input for this layer
        recurrent_input_batch = recurrent_cache_batch[n-1]
        
        # BHN and DoraFFN are now batched operations
        dora_a, dora_b, magnitude, beta = BHNs[n-1](recurrent_input_batch)
        x_processed_batch = DoraFFN(x_norm_batch, FFNs[n-1], dora_a, dora_b, magnitude, beta)
        
        current_token_states_batch = residual_batch + x_processed_batch
        
        # Store the new states for batched cache update
        new_recurrent_cache_batch[n-1] = current_token_states_batch

    # 3. Batched Finalization and Sampling
    final_states_batch = RMSNorm(current_token_states_batch)
    logits_batch = Unembedding(final_states_batch) # Shape: (num_requests, vocab_size)
    
    # Apply sampling per request, as each may have different parameters
    next_tokens = []
    for i, req in enumerate(active_requests):
        logits_i = logits_batch[i, :]
        token_i = sample(logits_i, req.sampling_params)
        next_tokens.append(token_i)

    # 4. Update State and Caches
    # Update each request with its new token
    update_request_sequences(active_requests, next_tokens)

    # Update the cache manager with the newly computed states
    cache_manager.update_caches([req.id for req in active_requests], new_recurrent_cache_batch)

    # Return finished/ongoing requests for the scheduler
    return active_requests
```