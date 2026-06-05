# Qwen3-8B-AWQ — vLLM Inference Server (NVIDIA Jetson AGX Thor)

A high-throughput, OpenAI-compatible LLM server running **`Qwen/Qwen3-8B-AWQ`** under
[vLLM](https://docs.vllm.ai) on an NVIDIA Jetson AGX Thor device, deployed with Docker Compose.

This directory is self-contained:

| File | Purpose |
|------|---------|
| [`docker-compose.yml`](docker-compose.yml) | The full server definition — image, command, ports, volumes, healthcheck. |
| [`api_doc.md`](api_doc.md) | Endpoint reference + how to call the API from another machine over an SSH tunnel. |
| `README.md` | This file — research notes, setup, tuning, and troubleshooting. |

---

## 1. Overview

| | |
|---|---|
| **Model** | `Qwen/Qwen3-8B-AWQ` (4-bit AWQ quantized) |
| **Serving engine** | vLLM (NGC image `nvcr.io/nvidia/vllm:26.02-py3`) |
| **Quantization kernel** | `awq_marlin` (fused dequant + matmul) |
| **API** | OpenAI-compatible (`/v1/chat/completions`, `/v1/completions`, …) |
| **Listen address** | `127.0.0.1:9999` (loopback only — reached via SSH tunnel) |
| **Max context** | 8192 tokens |
| **GPU memory budget** | `--gpu-memory-utilization 0.12` (≈ 20 GiB total: ~6 GiB weights + ~12 GiB KV cache) |
| **Concurrency** | ~11 simultaneous full-context (8K) requests |

---

## 2. Hardware target & why the image matters

This deployment targets the **Jetson AGX Thor** specifically:

- GPU compute capability **`sm_110`** (Blackwell-class)
- **CUDA 13.0**, driver 580, L4T **R38 / JetPack 7**, `aarch64`

This matters because the container's CUDA runtime must support `sm_110`. The
[`docker-compose.yml`](docker-compose.yml) header comment records the key lesson:

> NVIDIA's official vLLM build for CUDA 13 / Thor (sm_110). The dustynv JetPack-6
> (CUDA 12.9) image fails on Thor with CUDA error 801.

**Takeaway:** use a **CUDA-13 / JetPack-7** image. A JetPack-6 (CUDA 12.9) image has no
kernels for `sm_110` and dies at the first CUDA call with `cudaErrorNotSupported (801)`.

---

## 3. The `docker-compose.yml`, explained

Every non-obvious line in [`docker-compose.yml`](docker-compose.yml) encodes a decision:

```yaml
image: nvcr.io/nvidia/vllm:26.02-py3   # CUDA-13/Thor build (see §2)
runtime: nvidia                        # expose the GPU to the container
ipc: host                              # shared memory for the engine workers
restart: unless-stopped                # auto-recover after a crash/reboot

ports:
  - "127.0.0.1:9999:9999"              # loopback-only; access is via SSH tunnel only

environment:
  - NVIDIA_VISIBLE_DEVICES=all
  - HUGGING_FACE_HUB_TOKEN=${HF_TOKEN:-}   # unused for this public model; here for gated models later

volumes:
  - ~/.cache/huggingface:/root/.cache/huggingface   # persist downloaded weights across restarts

command: >
  vllm serve "Qwen/Qwen3-8B-AWQ"       # NGC entrypoint is `exec "$@"`, so we pass the full command
  --quantization awq_marlin            # fast fused kernel (NOT plain `awq`, see §6)
  --host 0.0.0.0
  --port 9999
  --gpu-memory-utilization 0.12        # caps total footprint at ~20 GiB (see §7)
  --max-model-len 8192
  --trust-remote-code

healthcheck:                           # `/health` probe so Compose reports (healthy)
  test: ["CMD", "bash", "-lc", "curl -fsS http://localhost:9999/health || exit 1"]
  interval: 30s
  timeout: 5s
  retries: 5
  start_period: 180s                   # generous: first boot compiles CUDA graphs
```

### Notable points (each one cost a debugging cycle)
1. **The NGC entrypoint runs `exec "$@"`** — it does *not* auto-prepend `vllm serve`. The
   command must start with `vllm serve`, otherwise the container tries to exec `--model` /
   `--quantization` as a program and exits immediately.
2. **Model id is `Qwen/Qwen3-8B-AWQ`** (no `-Instruct`). A wrong id returns HTTP 401 from
   Hugging Face (it masks "not found" as "unauthorized"), which looks like a token problem
   but isn't — this is a **public** model needing **no token**.
3. **Loopback bind (`127.0.0.1:9999`)** is intentional: the server is only meant to be
   reached through an SSH tunnel, never exposed on the network.

---

## 4. Setup & run (on the Thor device)

```bash
cd ~/vllm_benchmark/Qwen_serve

# Pull + start (first run downloads ~6 GiB of weights into ~/.cache/huggingface)
docker compose up -d

# Watch startup — first boot compiles CUDA graphs (can take several minutes)
docker compose logs -f
# Ready when you see:  INFO ... Application startup complete.  /  Uvicorn running on http://0.0.0.0:9999

# Confirm health
curl http://localhost:9999/health        # -> 200 OK
curl http://localhost:9999/v1/models      # -> lists Qwen/Qwen3-8B-AWQ
```

> If the NGC image pull returns `unauthorized`, run `docker login nvcr.io`
> (username `$oauthtoken`, password = your NGC API key), then retry.

### Day-to-day operations
```bash
docker compose ps           # status (look for "healthy")
docker compose logs -f      # live logs
docker compose restart      # restart
docker compose down         # stop & remove the container
docker compose up -d        # (re)start detached — also applies any compose edits
```

The model cache lives in `~/.cache/huggingface` on the host (mounted into the container),
so restarts do **not** re-download weights.

---

## 5. Accessing the API from another machine (SSH tunnel)

The server is loopback-only on Thor, so reach it with SSH local port forwarding:

```bash
# Forward YOUR localhost:9999  ->  Thor's localhost:9999
ssh -N -L 9999:localhost:9999 -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes thor-box-seqato
```

Then, on your machine, the base URL is `http://localhost:9999`.

> ⚠️ **Match the ports.** `-L <local>:localhost:<remote>` — the **left** number is the port you
> call locally. `-L 8080:localhost:9999` means you must `curl http://localhost:8080`, not 9999.
> `ExitOnForwardFailure=yes` makes ssh fail loudly if the local port is already taken instead of
> connecting without the forward.

A minimal request (see [`api_doc.md`](api_doc.md) for the full endpoint list and examples):
```bash
curl http://localhost:9999/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "Qwen/Qwen3-8B-AWQ",
  "messages": [{"role": "user", "content": "Explain SSH port forwarding in two sentences."}],
  "max_tokens": 256
}'
```

---

## 6. Performance: the `awq_marlin` decision

Single-stream generation speed hinges on the quantization kernel:

| Kernel | Config flag | Single-stream speed |
|--------|-------------|---------------------|
| Reference AWQ | `--quantization awq` | ~4 tok/s (compute-bound, slow dequant) |
| **Marlin AWQ** | `--quantization awq_marlin` | **~38–40 tok/s** |

Forcing plain `awq` makes the GPU 98%-busy doing inefficient dequantization. `awq_marlin`
fuses dequant into the matmul and is ~**9× faster** here. vLLM will even log a hint at startup
that the model *can* run with `awq_marlin`.

### Clocks (host-side, run on Thor)
The GPU can be throttled by the power mode. For max performance, uncap the power budget and
lock clocks (requires sudo, persists until reboot):
```bash
sudo nvpmodel -m 0      # MAXN power mode
sudo jetson_clocks      # lock GPU/CPU/memory clocks to max
```

---

## 7. Memory & the `--gpu-memory-utilization` knob

vLLM pre-allocates a fixed pool at startup: **model weights + KV cache**. The knob scales it:

| `--gpu-memory-utilization` | Total VRAM | KV cache | Max concurrency (8K ctx) | Single-req speed |
|---|---|---|---|---|
| 0.30 | ~42 GiB | ~35 GiB | ~31× | ~38 tok/s |
| 0.15 | ~24 GiB | ~16 GiB | ~14× | ~40 tok/s |
| **0.12** *(current)* | **~20 GiB** | **~12 GiB** | **~11×** | **~40 tok/s** |

Key facts:
- **KV-cache size does NOT affect single-request speed** — only how many requests run in
  parallel and how long a context fits. Shrinking it is "free" for latency.
- The reservation is **fixed at startup and does not grow** per request — it is not a leak.
- To resize, edit `--gpu-memory-utilization` in [`docker-compose.yml`](docker-compose.yml)
  and `docker compose up -d`. Raise it for more concurrency; lower it to free RAM.

For ~5 concurrent users, `0.12` (≈11× concurrency) is comfortable with headroom.

> **Do not** run `echo 3 > /proc/sys/vm/drop_caches` to "free" RAM. Linux page cache
> (`buff/cache`) is reclaimable and evicted automatically under pressure; dropping it only
> forces slow re-reads. The number that matters is `available` in `free -h`, not `used`.

---

## 8. Troubleshooting

| Symptom | Cause / Fix |
|---|---|
| `CUDA error 801` / `cudaErrorNotSupported` at startup | Wrong image for the GPU. Use a CUDA-13 / Thor (`sm_110`) image (§2). |
| `exec: "--model": ... not found in $PATH` or `exec: --: invalid option` | Command missing the launcher. Must start with `vllm serve` (§3, note 1). |
| `401 Unauthorized` / `RepositoryNotFound` for the model | Wrong repo id. Use `Qwen/Qwen3-8B-AWQ` exactly (public, no token needed). |
| Startup "stalls" with no logs for minutes | First-boot CUDA-graph compilation. Normal; wait. `docker stats` shows CPU pegged = progressing. |
| Slow generation (~4 tok/s) | Plain `awq` kernel and/or throttled clocks. Use `awq_marlin` + `jetson_clocks` (§6). |
| Container crashes during graph capture | Memory pressure at the peak-allocation phase. Lower `--gpu-memory-utilization` (§7). |
| `curl` from laptop refused, but works on Thor | SSH tunnel port mismatch or tunnel down. Match ports (§5); verify `ssh -N -L ...` is running. |
| Response truncated mid-reasoning (`finish_reason: length`) | Qwen3 "thinking" mode ate the token budget. See §9. |

Inspect what's wrong:
```bash
docker compose ps
docker logs --tail 50 vllm-qwen-agent
docker inspect -f '{{.State.Health.Status}} {{.State.ExitCode}}' vllm-qwen-agent
```

---

## 9. Qwen3 "thinking" mode

Qwen3 emits internal reasoning in `<think>...</think>` blocks, which can consume your entire
`max_tokens` before the real answer appears (`finish_reason: "length"`). To get a direct answer,
disable thinking per request:

```bash
curl http://localhost:9999/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "Qwen/Qwen3-8B-AWQ",
  "messages": [{"role": "user", "content": "write a 50 word poem."}],
  "max_tokens": 256,
  "chat_template_kwargs": {"enable_thinking": false}
}'
```

Alternatively, raise `max_tokens` generously so the model has room to both think and answer.

---

## 10. Quick reference

```bash
# Start / stop (on Thor)
cd ~/vllm_benchmark/Qwen_serve && docker compose up -d
docker compose down

# Tunnel (from your machine)
ssh -N -L 9999:localhost:9999 -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes thor-box-seqato

# Smoke test (through the tunnel)
curl http://localhost:9999/v1/models

# Per-container GPU memory (on Thor)
nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv
```

See [`api_doc.md`](api_doc.md) for the complete endpoint catalog and client code samples.
