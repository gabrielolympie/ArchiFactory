# ArchiFactory
Repo aiming at benchmarking several variant of transformer architectures for pre training, on datasets TinyStories and Pints 1.5

## How it works
Most modern transformer architecture follow a simple pattern:
- An embedding layer : its role is to convert int tokens into n-th dimensionnal vectors
- A series of layers, each layer being composed of two residual blocks with a layer normalization on top of each
  - A mixin block, in the initial transformer, it was a multi head attention block, in modern architecture it can be grouped query attention (mistral, smollm), multy head latent attention (deepseek) or even a recurrent block (mamba, rwkv, retnet, linear attention, ffn attention etc.)
  - A feed forward network block, in the initial transformer, it was a simple linear layer with a gelu activation, in modern architecture it can be a gated MLP with Silu activation (mistral, smollm) or a sparse mixture of experts (mixtral, deepseek, qwen3-moe)
- A final layer normalization, often a simple RMS NORM
- A final linear layer, to convert the n-th dimensionnal vectors into logits

## Purpose of this repo:
LLM pretraining incur a prohibitive cost, and the most impactfull elements in pretraining are the architecture, the scale of that architecture and the dataset.
After pretraining, the model must be post trained to acquire instruction following, ethical grounding and reasoning capabilities.

Due to that, today LLM benchmarks compare apples and tomatoes given that the pretraining dataset and postraining methods are differents.

As a result, the purpose of this repo is to enable testing several architecture with ablation study made on a single module at a time, with always the same pretraining dataset and post training methods.

## Structure of the repo
-root
--modules : key modules to test
---archi_modules.py : modules to define the architecture as a whole, ie how many layers, what mixin, what ffn etc.
---ffn_modules.py : modules to define the feed forward network architecture, as of now FFN and MOE are implemented (and DoraMOE + HyperMOE are in the works)
---hyper_modules.py : modules to define the hypernetwork architecture, work in progress
---mixin_modules.py : modules to define the mixin architecture, as of now GQA, LSTM, RNN, Mamba2, RWKV6 and retentive network are implemented, mostly based on the FLA library
---positionnal_modules.py : modules to define the positionnal embedding architecture, as of now only naive positionnal embedding is implemented
--utils : training loop used to pretrain the architecture

## How to use:
You can check the notebook Benchmark.py, it is a simple example of how to use the modules to define an architecture and train it.

To define a model, you simply need to use the StackedMixinForCausalLM class, which takes as input the following arguments:

StackedMixinForCausalLM(
    num_layers | number of layers in the model
    hidden_size | hidden size of the model (if a pretrained embedding module or lm head is provided the size should match)
    initializer_range | initializer range for the weights
    embedding_module | embedding module to use
    lm_head_module=lm_head | lm head module to use
    final_norm_module | final norm module to use
    freeze_lm_modules | whether to freeze the lm modules or not, usefull if you want to leverage some parts of an already pretrained model
    vocab_size | vocab size of the model
    mixin_module | mixin module to use
    ffn_module | ffn module to use
)

The Benchmark notebook will then guide you to run the training loop.

## Standard Datasets.
As our compute budget is very limited, we will use simple datasets to benchmark the models.

### Pretraining
- architecture initial test : 1M samples from the tiny stories dataset, this runs in roughly 20 minutes in a single 4090 for a 30M model
- large testt : full Pints 1.5 dataset, that can be run in a few days for a 1B model

### Instruct tuning
- dolphin r1 : provides both thinking and non thinkgin high quality samples
- limo : 1k sample extreme quality dataset

### Thinking / GRPO
TBD

## Standard benchmarks
- MMLU
- Humaneval

## First observations:
- So far, RWKKV6 have slight advantage over other, then come Mamba2, GQA and RetNet roughtly bringing the same results. and finally legacy LSTM and RNN that are far behind

# TODO
- [ ] Add more positionnal embedding modules
- [ ] Create a simple benchmarking script
- [ ] Add GPU parallelism and a lightning script to improve training speed for upcoming larger runs
- [ ] Add Recurent Hyper Transformer Modules
- [ ] Add a generate function on top of the model
- [ ] Add caching to optimize inference on compatibles architectures
- [ ] Add early benchmarks to readme