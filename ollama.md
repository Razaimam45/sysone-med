# Cheatsheet

## 🖥️ Server (the daemon)

```bash
# ── START, single GPU, FOREGROUND (logs in terminal, Ctrl-C stops) ──
CUDA_VISIBLE_DEVICES=1 OLLAMA_HOST=127.0.0.1:11435 \
  ~/.local/share/ollama-install/bin/ollama serve

# ── START, single GPU, DETACHED (survives terminal close) ──
CUDA_VISIBLE_DEVICES=1 OLLAMA_HOST=127.0.0.1:11435 \
  nohup ~/.local/share/ollama-install/bin/ollama serve >> ~/.ollama-personal.log 2>&1 &

# ── START, ALL GPUs (default, no restriction) ──
OLLAMA_HOST=127.0.0.1:11435 \
  nohup ~/.local/share/ollama-install/bin/ollama serve >> ~/.ollama-personal.log 2>&1 &

# ── STOP ──
pkill -f "ollama-install/bin/ollama serve"

# ── STATUS: is it up? which GPU? ──
curl -s http://127.0.0.1:11435/api/version
pgrep -af "ollama-install/bin/ollama serve"
tr '\0' '\n' < /proc/$(pgrep -f "ollama-install/bin/ollama serve" | head -1)/environ | grep CUDA_VISIBLE
```

> To change GPU later: **stop → start again** with the new `CUDA_VISIBLE_DEVICES`. You can't change it on a running server.

## 📦 Models

```bash
export OLLAMA_HOST=http://127.0.0.1:11435
export OLLA=~/.local/share/ollama-install/bin/ollama   # shortcut

$OLLA pull <model>          # download
$OLLA list                  # list
$OLLA rm <model>            # delete
```

## 💬 Inference

```bash
# Interactive chat (NO --num-ctx flag exists)
$OLLA run <model>

# One-shot via API — this is where you set num_ctx
curl http://127.0.0.1:11435/api/chat -d '{
  "model": "<model>",
  "messages": [{"role":"user","content":"hi"}],
  "options": {"num_ctx": 32768}
}'
```

**To make 32k context the default**, bake it into a model variant:
```bash
cat > /tmp/Modelfile <<'EOF'
FROM hf.co/unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL
PARAMETER num_ctx 32768
EOF
$OLLA create qwen32k -f /tmp/Modelfile
$OLLA run qwen32k        # now always 32k ctx
```

**Python (OpenAI-compatible):**
```python
from openai import OpenAI
c = OpenAI(base_url="http://127.0.0.1:11435/v1", api_key="ollama")
r = c.chat.completions.create(
    model="<model>",
    messages=[{"role":"user","content":"hi"}],
    extra_body={"options": {"num_ctx": 32768}},
)
print(r.choices[0].message.content)
```

## ⚡ Quick reference

| Task | Command |
|---|---|
| Start (GPU 1, detached) | `CUDA_VISIBLE_DEVICES=1 OLLAMA_HOST=127.0.0.1:11435 nohup ~/.local/share/ollama-install/bin/ollama serve >> ~/.ollama-personal.log 2>&1 &` |
| Stop | `pkill -f "ollama-install/bin/ollama serve"` |
| Is it up? | `curl -s http://127.0.0.1:11435/api/version` |
| Which GPU? | `tr '\0' '\n' < /proc/$(pgrep -f "ollama-install/bin/ollama serve"\|head -1)/environ \| grep CUDA` |
| Set context | API `options.num_ctx` or Modelfile `PARAMETER num_ctx` |

Want me to save this as a `~/.ollama-cheatsheet.md` file, or wrap the start/stop into a tiny `ollama-local` script so it's one command?

## Check which user is using which GPU
```bash
ps -f -p $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr '\n' ',' | sed 's/,$//')
```