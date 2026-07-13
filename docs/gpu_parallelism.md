# Running Ollama Efficiently Across 4 GPUs

This note summarizes how to get good utilization out of a 4-GPU machine
running Ollama, which is relevant both to the team's shared GPU machine and
to any local multi-GPU setup used for development.

## The key fact

Ollama does not implement tensor parallelism: a single request is not split
across GPUs to make that one request faster. What multi-GPU does provide is
extra VRAM (to fit larger models) and the ability to serve multiple
concurrent requests without queuing behind a single inference stream. With
4 GPUs, the realistic expectation is "many independent requests handled at
once," not "one request answered four times faster."

## Default behavior

- If a model fits entirely on one GPU, Ollama loads it on a single GPU
  (this avoids unnecessary cross-GPU data transfer and is usually faster).
- If a model does not fit on one GPU, Ollama splits it across the visible
  GPUs.
- If it still does not fit, part of the model spills to system RAM and
  speed drops noticeably (`ollama ps` shows a `CPU/GPU` split in that case).

## Two ways to use 4 GPUs

### Option A: one Ollama instance, all 4 GPUs visible

Useful for large models (for example a 70B-class model) that do not fit on
a single GPU, and/or for handling many concurrent requests against the same
model.

```
CUDA_VISIBLE_DEVICES=0,1,2,3 ollama serve
OLLAMA_NUM_PARALLEL=4
OLLAMA_SCHED_SPREAD=1
```

`OLLAMA_NUM_PARALLEL` controls how many requests a model processes at once
(default auto-selects 1 or 4 depending on available memory).
`OLLAMA_SCHED_SPREAD` asks the scheduler to spread models/requests more
aggressively across GPUs; it is an experimental knob, not a guarantee of
linear speedup.

### Option B: one Ollama instance per GPU

Useful for genuine task specialization: run a different model on each GPU
and route requests to the right one.

```
CUDA_VISIBLE_DEVICES=0 OLLAMA_HOST=127.0.0.1:11434 ollama serve
CUDA_VISIBLE_DEVICES=1 OLLAMA_HOST=127.0.0.1:11435 ollama serve
CUDA_VISIBLE_DEVICES=2 OLLAMA_HOST=127.0.0.1:11436 ollama serve
CUDA_VISIBLE_DEVICES=3 OLLAMA_HOST=127.0.0.1:11437 ollama serve
```

For example: GPU 0 for the coder model (`qwen2.5-coder:14b`), GPU 1 for the
large semantic model (`qwen3:32b`), GPU 2 for embeddings, GPU 3 kept free
for whichever model has the most concurrent demand at a given time.

## Relevant environment variables

| Variable | Purpose |
|---|---|
| `CUDA_VISIBLE_DEVICES` | Selects which GPUs Ollama can see (GPU UUIDs from `nvidia-smi -L` are more stable than numeric indices) |
| `OLLAMA_NUM_PARALLEL` | Number of concurrent requests a loaded model will process |
| `OLLAMA_MAX_LOADED_MODELS` | Number of different models kept resident in memory at once |
| `OLLAMA_SCHED_SPREAD` | Encourages spreading models/requests across more GPUs (experimental) |

Useful commands to check what is actually happening: `ollama ps` (shows
where a model is loaded and the CPU/GPU split), `nvidia-smi` (VRAM usage per
GPU), and `docker logs -f <container>` if Ollama runs inside a container.

## Common misconceptions

- **"Two 12GB GPUs behave like one 24GB GPU."** Not quite — cross-GPU
  access has overhead, so a large model split across two GPUs will usually
  be slower than the same model fitting entirely on one large-VRAM GPU.
- **"More GPUs always means faster."** Not if the model already fits on
  one GPU; forcing a split adds PCIe transfer overhead for no benefit.
- **"NVLink is required."** No, ordinary PCIe multi-GPU setups work; PCIe
  bandwidth mainly matters when a large model is being loaded, less so
  once it is resident in VRAM.

## Relevance to this project

The shared GPU machine already serves multiple models
(`qwen2.5-coder:14b`, `qwen3:32b`) to multiple users concurrently, which is
consistent with Option A above (or a variant of it). From the client side
(this project's code), the practical levers are the ones already reflected
in `config.py`: use the lighter model for development/testing
(`OLLAMA_LIGHT_MODEL`) and the larger model only where the task actually
needs it (`OLLAMA_DEFAULT_MODEL`), which keeps load on the shared GPUs
predictable regardless of how many GPUs back the service.
