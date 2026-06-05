# Qwen3-8B-AWQ Inference API

A vLLM OpenAI-compatible server running on the **thor-box-seqato** device (NVIDIA Jetson AGX Thor).
This document tells another engineer how to reach the server from their own machine and how to call it.

| | |
|---|---|
| **Model served** | `Qwen/Qwen3-8B-AWQ` |
| **Engine** | vLLM `0.15.1` (NGC image `nvcr.io/nvidia/vllm:26.02-py3`) |
| **Host device** | `thor-box-seqato` |
| **Listen address** | `127.0.0.1:9999` (loopback only — **not** exposed on the network) |
| **Max context length** | 8192 tokens |
| **Quantization** | AWQ |
| **Auth** | None (no API key required) |
| **API style** | OpenAI-compatible (`/v1/...`) |

> ⚠️ The server is deliberately bound to `127.0.0.1` on Thor. It is **only** reachable
> through an SSH tunnel — there is no open network port. This is the intended security model.

---

## 1. Connect from your machine (SSH port forwarding)

The server listens on Thor's `localhost:9999`. Forward a local port on your machine to it:

```bash
# Forward YOUR localhost:9999  ->  Thor's localhost:9999
ssh -N -L 9999:localhost:9999 thor-box-seqato
```

- `-N` = don't open a shell, just hold the tunnel open.
- `-L 9999:localhost:9999` = local port `9999` → (over SSH) → Thor's `localhost:9999`.
- Keep this command running in a terminal for as long as you need access.

Want a different local port (e.g. 9999 is taken on your side)? Change only the left number:

```bash
ssh -N -L 8080:localhost:9999 thor-box-seqato   # then use http://localhost:8080
```

> `thor-box-seqato` must be defined in your `~/.ssh/config` (Host, HostName, User, IdentityFile).
> If it isn't, use `ssh -N -L 9999:localhost:9999 <user>@<thor-ip>` instead.

**Tip — keep the tunnel alive across drops:**
```bash
ssh -N -L 9999:localhost:9999 -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes thor-box-seqato
```

Once the tunnel is up, the base URL on your machine is:

```
http://localhost:9999
```

### Quick sanity check
```bash
curl http://localhost:9999/health        # -> 200 OK (empty body)
curl http://localhost:9999/v1/models      # -> JSON listing Qwen/Qwen3-8B-AWQ
```

---

## 2. Endpoints

Base URL (through the tunnel): `http://localhost:9999`

### Primary (you'll use these)
| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/chat/completions` | **Main endpoint.** Chat-style generation (system/user/assistant messages). |
| `POST` | `/v1/completions` | Raw text completion (single prompt string, no chat template). |
| `GET`  | `/v1/models` | List served models + their IDs and `max_model_len`. |
| `POST` | `/v1/messages` | Anthropic-Messages-compatible endpoint. |
| `POST` | `/v1/responses` | OpenAI "Responses" API style. |

### Utility
| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/health` | Liveness probe (200 when ready). |
| `GET`  | `/ping` / `POST /ping` | Same as health. |
| `GET`  | `/version` | vLLM version. |
| `GET`  | `/load` | Current server load stats. |
| `GET`  | `/metrics` | Prometheus metrics (throughput, latency, queue, KV-cache usage). |
| `POST` | `/tokenize` | Convert text → token IDs. |
| `POST` | `/detokenize` | Convert token IDs → text. |

### Other available routes (less common for this model)
`/v1/embeddings`, `/pooling`, `/classify`, `/score` & `/v1/score`,
`/rerank` & `/v1/rerank` & `/v2/rerank`, `/v1/audio/transcriptions`,
`/v1/audio/translations`, `/v1/responses/{id}` (GET) & `/v1/responses/{id}/cancel`,
`/invocations` (SageMaker), `/pause` · `/resume` · `/is_paused`,
`/scale_elastic_ep` · `/is_scaling_elastic_ep`.

> These exist because vLLM registers a generic route table. For a causal-LM like
> Qwen3-8B-AWQ, the chat/completions endpoints are what you want; embeddings/rerank/audio
> require a model that supports those tasks and will error otherwise.

### Interactive docs
With the tunnel running, open in a browser:
- Swagger UI: `http://localhost:9999/docs`
- ReDoc: `http://localhost:9999/redoc`
- Raw spec: `http://localhost:9999/openapi.json`

---

## 3. Usage examples

The model id for every request is **`Qwen/Qwen3-8B-AWQ`**.

### 3a. Chat completion — curl
```bash
curl http://localhost:9999/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-8B-AWQ",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain SSH port forwarding in two sentences."}
    ],
    "max_tokens": 256,
    "temperature": 0.7
  }'
```

### 3b. Streaming (token-by-token, SSE)
```bash
curl http://localhost:9999/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-8B-AWQ",
    "messages": [{"role": "user", "content": "Write a haiku about GPUs."}],
    "stream": true
  }'
```

### 3c. Python — OpenAI SDK
```python
from openai import OpenAI

# Point the OpenAI client at the tunnel. api_key is required by the SDK but unused.
client = OpenAI(base_url="http://localhost:9999/v1", api_key="not-needed")

resp = client.chat.completions.create(
    model="Qwen/Qwen3-8B-AWQ",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Give me 3 uses for an edge LLM."},
    ],
    max_tokens=300,
    temperature=0.7,
)
print(resp.choices[0].message.content)
```

### 3d. Python — streaming
```python
stream = client.chat.completions.create(
    model="Qwen/Qwen3-8B-AWQ",
    messages=[{"role": "user", "content": "Count to five slowly."}],
    stream=True,
)
for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
```

### 3e. Raw completion (no chat template)
```bash
curl http://localhost:9999/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-8B-AWQ",
    "prompt": "The capital of France is",
    "max_tokens": 16
  }'
```

### Common sampling parameters
| Field | Meaning | Typical |
|---|---|---|
| `max_tokens` | Max tokens to generate | 128–1024 |
| `temperature` | Randomness (0 = deterministic) | 0.0–1.0 |
| `top_p` | Nucleus sampling | 0.8–1.0 |
| `stream` | Stream tokens via SSE | `true`/`false` |
| `stop` | Stop string(s) | e.g. `["\n\n"]` |

> **Context budget:** prompt + `max_tokens` must stay ≤ **8192**. Larger requests are rejected.

---

## 4. Reasoning / thinking note

Qwen3 supports an internal "thinking" mode. If you see `<think>...</think>` blocks in
the output and don't want them, instruct the model in the system prompt (e.g.
"Answer directly without showing reasoning.") or strip the `<think>` block client-side.

---

## 5. Troubleshooting

| Symptom | Cause / Fix |
|---|---|
| `curl: (7) Failed to connect to localhost port 9999` | SSH tunnel isn't running. Start the `ssh -N -L ...` command. |
| Tunnel command hangs with no prompt | That's normal — `-N` holds it open. Leave it; open another terminal to make requests. |
| `bind: Address already in use` on tunnel start | Local port 9999 is taken. Use a different left-hand port, e.g. `-L 8080:localhost:9999`. |
| `404` `model not found` | Set `"model": "Qwen/Qwen3-8B-AWQ"` exactly. Check with `GET /v1/models`. |
| Request hangs then errors on length | Prompt + `max_tokens` exceeded 8192. Shorten input or `max_tokens`. |
| `Connection refused` even with tunnel up | Server may be restarting/compiling on Thor. On Thor: `docker compose logs -f` and wait for `Application startup complete`. |

### Server-side ops (run ON thor-box-seqato)
```bash
cd ~/vllm_benchmark/Qwen_serve
docker compose ps           # status (look for "healthy")
docker compose logs -f      # live logs
docker compose restart      # restart
docker compose down         # stop
docker compose up -d        # start (detached)
```

---

*Server config lives in `~/vllm_benchmark/Qwen_serve/docker-compose.yml` on thor-box-seqato.*
