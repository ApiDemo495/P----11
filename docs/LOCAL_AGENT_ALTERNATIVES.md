# Qwen won't load or download — every working alternative

_Short answer: you almost certainly do not need to download anything. Any
local model server you already have works with this app over a standard API,
and the app now finds it by itself._

---

## Why the Qwen download fails

The GGUF route needs **three** things at once:

1. `huggingface.co` reachable from this host
2. a 2 GB transfer that survives to completion
3. a `llama-cpp-python` runtime built for this exact Python/platform

Either one failing kills the local agent — and the way it failed was the
problem. On a network that blocks the hub, TLS is closed at the CONNECT
stage, so every attempt looked like an endless **"downloading…"** and the
only advice the panel gave was *press download again*. Verified here against
the real endpoint:

```
GET https://huggingface.co/api/models/Qwen/Qwen2.5-Coder-3B-Instruct-GGUF/tree/main
  -> TLS/SSL connection has been closed (EOF)   (retried 5x, then gave up)
```

Nothing was wrong with the install, the RAM tier, or the model name. The
network simply cannot reach the host the file lives on, and **no amount of
retrying changes that**.

---

## Option 1 — a local server you already have (no download)

One wire format covers nearly every runner people install: `POST
/v1/chat/completions` + `GET /v1/models`. The app now probes for it **before**
trying anything that downloads or compiles:

| server | default port | start it | load a model with |
|---|---|---|---|
| **Ollama** | 11434 | `ollama serve` | `ollama pull qwen2.5-coder:1.5b` |
| **LM Studio** | 1234 | GUI → *Start Server* | any model, from its search |
| **llama.cpp** | 8080 | `llama-server -m model.gguf --port 8080` | the `.gguf` you have |
| **Jan** | 1337 | GUI | any model |
| **GPT4All** | 4891 | GUI → *Enable API server* | any model |
| **LocalAI** | 8081 | `local-ai run` | any model |
| **vLLM** | 8000 | `vllm serve <hf-model>` | any HF model |
| **text-generation-webui** | 5000 | `--api` | any model |
| **KoboldCpp** | 5001 | `--api` | any `.gguf` |

Start one, then **Agents tab → "use this server"**, or just press *re-probe*
— autodetection checks all nine ports in well under a second (a refused
localhost port answers instantly; measured 0.00 s for all nine).

A server on another machine:

```bash
PREDICT_LOCAL_ENDPOINT=http://192.168.1.9:8080
```

or POST `{"endpoint": "http://192.168.1.9:8080"}` to
`/api/agents/local/endpoints`. Comma-separated lists are accepted.

### Why this is the best answer

* **no 2 GB transfer** — the model is already on your disk, managed by
  whatever GUI you prefer
* **no C++ toolchain** — no 15-minute compile that can time out
* **works behind the blocked network** — it is a localhost socket
* **Qwen is still available** — `ollama pull qwen2.5-coder:1.5b` or load the
  same GGUF in LM Studio. You get the model you wanted; you just skip the
  download step that was failing.

---

## Option 2 — a free cloud key (nothing to install)

| provider | free tier | setting | example model |
|---|---|---|---|
| **Groq** | generous | `openai_base=groq` | `llama-3.3-70b-versatile` |
| **Google Gemini** | generous | `gemini_key` | `gemini-2.5-flash` |
| **OpenRouter** | some `:free` models | `openai_base=openrouter` | any `:free` |
| **DeepSeek** | cheap | `openai_base=deepseek` | `deepseek-chat` |
| **Together / Mistral / Cerebras / Fireworks** | varies | `openai_base=<name>` | see their docs |

The `openai` slot is now a **generic** OpenAI-compatible slot — pointing it
at a nickname is enough, and a full URL works too:

```
openai_base = groq
openai_base = https://api.groq.com/openai/v1/chat/completions
openai_base = http://127.0.0.1:1234        # your own LM Studio
```

---

## Option 3 — drop in a `.gguf` you downloaded elsewhere

A browser on another machine, a friend's USB stick, whatever:

```
models/qwen2-0_5b-instruct-q4_k_m.gguf          # the exact expected path
```

The app validates the `GGUF` magic bytes and hashes it once. The path is
shown on the Agents panel, so there is no guessing which filename it wants.

---

## What changed in the code

* **`app/agents.py`**
  * `discover_local_endpoints()` / `probe_openai_endpoint()` — one prober for
    all nine servers, memoised, with a 2.5 s sweep budget
  * `LocalModel` gained an `openai` backend, tried **first** in
    `try_load()` — before the GGUF load and before Ollama
  * the installer's runtime ladder gained **rung −1**: a running server needs
    no pip install, so it is checked before the wheel/build rungs
  * `download_gguf()` now reports `blocked`, unwrapping `URLError` to see the
    TLS close, and returns a `hint` naming the routes that actually work
  * failure text no longer says "press download again" — it says the hub is
    unreachable and lists the alternatives
  * `_alternatives()` — the no-download routes as data, rendered on the panel
  * `openai_base_url()` — the `openai` slot can be Groq, DeepSeek,
    OpenRouter, Together, Mistral, Cerebras, Fireworks, a local server, or
    plain OpenAI
* **`app/main.py`** — `GET/POST /api/agents/local/endpoints` to inspect and
  set the endpoint live, without a restart
* **`static/app.js`** — an "Works without any download" block on the Agents
  panel listing what is running, how to start each server, the free-cloud
  keys, the manual path, and an input to point at a server by URL
* **`tests/conftest.py`** — autodetect is disabled for the whole suite; a
  developer's running LM Studio must not change a test's result
  (it did: failures read `'openai' != 'llama_cpp'`)
* **`tests/test_agent_alternatives.py`** — 18 tests for discovery, the
  blocked-vs-flaky distinction, the sweep budget, and the URL mapping

Suite: **1156 passed, 2 skipped.**
