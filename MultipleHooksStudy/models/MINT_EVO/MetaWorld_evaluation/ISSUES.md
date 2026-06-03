# Known Issues: `launch_eval.py` + `Mod_server.py`

Files under review:
- `models/MINT_EVO/MetaWorld_evaluation/launch_eval.py`
- `models/MINT_EVO/Evo_1/scripts/Mod_server.py`

---

## ISSUE 1 — `wait_for_server` uses raw TCP instead of a WebSocket probe, causing a spurious server-side error and masking real startup failures

**Severity**: High  
**File**: `launch_eval.py`, lines 53–63 and 157–160

**Root cause**:
```python
def wait_for_server(host="127.0.0.1", port=9000, timeout=300):
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2):
                print("  Server is ready.")
                return True
        except OSError:
            time.sleep(5)
    return False
```
`socket.create_connection` opens a raw TCP connection and closes it immediately (zero bytes sent). The WebSocket server on the other side expects an HTTP upgrade request; instead it gets EOF, which causes:
```
EOFError: stream ends after 0 bytes, before end of line
websockets.exceptions.InvalidMessage: did not receive a valid HTTP request
```
This is printed to the server log every time the probe fires.

**Second problem in the same function**: there is no `server_proc.poll()` check. If `Mod_server.py` crashes at startup (e.g., wrong checkpoint format, CUDA unavailable, missing files), the function does not detect the dead process and continues polling for the full 300-second timeout before giving up. The caller at lines 157–160:
```python
if not wait_for_server(port=args.port, timeout=300):
    print(f"[ERROR] Server did not start in time. Check log: {server_log}")
    server_proc.terminate()
    sys.exit(1)
```
...will not surface the real error for 5 minutes.

**Scenario A — Confirmed on Colab**:
`Mod_server.py` loads the model, starts the WebSocket server, prints "EVO_1 server running at ws://0.0.0.0:9001". `wait_for_server` connects via raw TCP, immediately closes. Server logs the `InvalidMessage` traceback. User sees the traceback and thinks the server crashed. It has not — the eval continues normally, but user confidence is destroyed and time is wasted investigating a non-error.

**Scenario B — Confirmed on any runtime where model load fails**:
`Mod_server.py` exits in 2 seconds due to a `RuntimeError` (e.g., key mismatch in `load_state_dict`, CUDA not available). `wait_for_server` does not detect the dead process. Polls every 5 seconds for 300 seconds. User waits 5 minutes to see an error that was available immediately in the server log.

**Fix required**:
1. Replace the raw TCP probe with a proper WebSocket connection attempt using `websockets.connect` with a short timeout; close it immediately on success.
2. Inside the polling loop, add `if server_proc.poll() is not None: return False` so a dead server is detected immediately without waiting out the full timeout.

---

## ISSUE 2 — `--timesteps` argument is accepted by `launch_eval.py` but never passed to `Mod_server.py`; server always runs at 32 timesteps

**Severity**: Medium (silent correctness issue)  
**File**: `launch_eval.py`, lines 82 and 134–138

**Root cause**:
```python
p.add_argument("--timesteps", type=int, default=32, ...)   # line 82 — parsed but unused
```
```python
server_cmd = [
    sys.executable, _SERVER_SCRIPT,
    "--ckpt_dir", ckpt_dir,
    "--port",     str(args.port),
    # args.timesteps is NEVER added here
]
```
`Mod_server.py` also has no `--timesteps` argument in its own `argparse` and hardcodes `config["num_inference_timesteps"] = 32` at line 72. Even if you add it to `server_cmd`, `Mod_server.py` will not accept or use it.

**Scenario**:
User runs `python launch_eval.py --step 80000 --timesteps 16` to benchmark faster inference or reduce latency per step. The server starts with 32 timesteps. Results are collected for 32 timesteps. All timing measurements and any comparisons across timestep counts are silently wrong with no error or warning. The `--timesteps` flag appears to work (no error) but does nothing.

**Fix required**:
1. Add `p.add_argument("--timesteps", type=int, default=32)` to `Mod_server.py`'s `argparse` (in the `if __name__ == "__main__"` block).
2. Pass `args.timesteps` into `load_model_and_normalizer` and use it to set `config["num_inference_timesteps"]` instead of the hardcoded `32` on line 72.
3. Add `"--timesteps", str(args.timesteps)` to `server_cmd` in `launch_eval.py`.

---

## ISSUE 3 — `_CLIENT_SCRIPT` hard-codes a cross-repo path into `Evo1StateExperiments`; does not exist on a standalone clone of `MINT_EVO`

**Severity**: High  
**File**: `launch_eval.py`, lines 30–33 and 172

**Root cause**:
```python
_CLIENT_SCRIPT = os.path.join(_REPO_ROOT, "models", "Evo1StateExperiments",
                               "MetaWorld_evaluation", "mt50_eval_client.py")
_CLIENT_CWD    = os.path.join(_REPO_ROOT, "models", "Evo1StateExperiments",
                               "MetaWorld_evaluation")
```
`_REPO_ROOT` resolves two levels above `MINT_EVO` to the monorepo root. The client script lives inside the `Evo1StateExperiments` subproject. If the user runs this on Colab with only the `MINT_EVO` subtree checked out, or if the repo is restructured, this path does not exist.

The client subprocess is launched at line 172:
```python
subprocess.run(client_cmd, cwd=_CLIENT_CWD, env={**os.environ})
```
There is no `check=True`.

**Scenario A — Missing path, completely silent failure**:
`_CLIENT_SCRIPT` does not exist on the filesystem. `subprocess.run([sys.executable, _CLIENT_SCRIPT, ...])` invokes Python with a nonexistent script path. Python prints to stderr:
```
can't open file '/path/Evo1StateExperiments/.../mt50_eval_client.py': [Errno 2] No such file or directory
```
`subprocess.run` returns with exit code 2. No exception is raised because there is no `check=True`. The `finally` block runs, the server is terminated, `[server] Done.` is printed. The eval produced zero results. Nothing in `launch_eval.py`'s stdout indicates failure.

**Scenario B — Wrong working directory for data files**:
Even when the file exists, `_CLIENT_CWD` is set to `Evo1StateExperiments/MetaWorld_evaluation/`. Any relative file path inside `mt50_eval_client.py` — specifically `mt50_order.json` and `tasks.jsonl` — resolves from that directory, not from `MINT_EVO/MetaWorld_evaluation/`. The MINT_EVO-specific versions of those files (which may have different task ordering, task prompts, or difficulty groupings) are silently ignored. Eval results are computed with the wrong task ordering.

**Fix required**:
1. Copy or symlink `mt50_eval_client.py` into `models/MINT_EVO/MetaWorld_evaluation/` so the client lives alongside `launch_eval.py`.
2. Update `_CLIENT_SCRIPT` to `os.path.join(_HERE, "mt50_eval_client.py")` and `_CLIENT_CWD` to `_HERE`.
3. Add `check=True` to the `subprocess.run(client_cmd, ...)` call at line 172 so client failures raise a `CalledProcessError` instead of silently returning.

---

## ISSUE 4 — `Mod_server.py` `strict=True` weight load crashes on multi-GPU training checkpoints due to `module.` key prefix mismatch

**Severity**: High  
**File**: `Mod_server.py`, line 75

**Root cause**:
```python
model.load_state_dict(load_file(os.path.join(ckpt_dir, "model.safetensors")), strict=True)
```
`train.py` saves checkpoints via `accelerator.save_state()`. When training was launched with `accelerate launch --num_processes > 1` (multi-GPU DDP), the underlying DDP wrapper may cause state dict keys to be prefixed with `module.`. Loading these into a bare `EVO1()` instance with `strict=True` raises immediately:
```
RuntimeError: Unexpected key(s) in state_dict: 'module.embedder.model.vision_model.encoder...'
Missing key(s) in state_dict: 'embedder.model.vision_model.encoder...'
```
The server process exits with code 1. Because of Issue 1, `launch_eval.py` will not detect this for 5 minutes.

**Scenario**:
Training ran on 4 GPUs (`accelerate launch --num_processes 4 train.py ...`). Checkpoint saved normally. User runs `python launch_eval.py --step 80000`. Server starts, attempts `load_state_dict`, throws `RuntimeError` on the first key. Server exits in under 5 seconds. `launch_eval.py` waits 300 seconds (Issue 1), then prints `[ERROR] Server did not start in time`. User checks the server log, finds the key mismatch error, realises the checkpoint format is incompatible.

**Fix required**:
After `load_file(...)` and before `load_state_dict`, strip any `module.` prefix from the loaded state dict:
```python
state_dict = load_file(os.path.join(ckpt_dir, "model.safetensors"))
state_dict = {
    (k[len("module."):] if k.startswith("module.") else k): v
    for k, v in state_dict.items()
}
model.load_state_dict(state_dict, strict=True)
```

---

## ISSUE 5 — Any unhandled exception during inference silently drops the WebSocket connection, crashing the entire eval run with no aggregate results written

**Severity**: High  
**File**: `Mod_server.py`, lines 122–132

**Root cause**:
```python
async def handle_request(websocket, model, normalizer):
    try:
        async for message in websocket:
            json_data = json.loads(message)
            actions = infer_from_json_dict(json_data, model, normalizer)
            await websocket.send(json.dumps(actions))
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected.")
```
Only `websockets.exceptions.ConnectionClosed` is caught. Any other exception originating from `infer_from_json_dict` propagates uncaught out of the `async for` loop. This includes:

- `AssertionError` from `assert len(images) == 3` at line 94 (if client sends wrong number of images)
- `AssertionError` from `assert img.shape == (3, 448, 448)` at line 96
- `RuntimeError` from `action.reshape(1, -1, 24)` at line 117 if `model.run_inference` returns unexpected shape
- `torch.cuda.OutOfMemoryError` if GPU memory is exhausted partway through a long eval run
- `json.JSONDecodeError` at line 128 if a malformed WebSocket payload arrives

When this exception propagates, the websockets library terminates the connection with code 1011. On the client side (`mt50_eval_client.py`), `await ws.recv()` raises `websockets.exceptions.ConnectionClosedError`. This is not caught anywhere in `eval_mt50`. It propagates through `while steps < args.horizon`, through `for ep in range(args.episodes)`, through `for idx in ordered`, and exits the `async with websockets.connect(...) as ws` block. The per-task summary and aggregate success-rate block at lines 322–340 of `mt50_eval_client.py` **never executes**. The log file has partial per-task lines but no totals.

**Scenario — GPU OOM partway through MT50**:
Eval runs tasks 0–29 without issue. On task 30, after hours of running, GPU memory is sufficiently fragmented that inference throws `torch.cuda.OutOfMemoryError`. Server logs the traceback to `server_log`, closes the connection. Client crashes. 30 tasks of data are in the log but there is no overall success rate. The full MT50 run must be restarted from scratch.

**Fix required**:
Catch all exceptions during inference in `handle_request`, log them server-side, and send an error marker to the client so the client can skip the step and continue:
```python
async def handle_request(websocket, model, normalizer):
    try:
        async for message in websocket:
            try:
                json_data = json.loads(message)
                actions = infer_from_json_dict(json_data, model, normalizer)
                await websocket.send(json.dumps({"actions": actions}))
            except Exception as e:
                import traceback
                traceback.print_exc()
                await websocket.send(json.dumps({"error": str(e)}))
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected.")
```
The client must also be updated to check for `{"error": ...}` in the response and handle it by skipping the current action step rather than crashing.

---

## ISSUE 6 — `load_model_and_normalizer` opens config files without context managers, leaking file descriptors on repeated server restarts

**Severity**: Low  
**File**: `Mod_server.py`, lines 67–68

**Root cause**:
```python
config = json.load(open(os.path.join(ckpt_dir, "config.json")))
stats  = json.load(open(os.path.join(ckpt_dir, "norm_stats.json")))
```
No `with` statement. File handles are not explicitly closed. CPython GC will eventually close them, but timing is not guaranteed in a long-running async process.

**Scenario**:
On a Colab session, the user iterates over multiple checkpoints by restarting `launch_eval.py` many times without restarting the Python kernel or Colab runtime. Over 15–20 restarts, leaked handles accumulate in the server process. When the fd limit is reached, the next `open()` call anywhere in the process throws:
```
OSError: [Errno 24] Too many open files
```
This can happen during model loading (crashing the server at startup) or during a subsequent inference request (crashing mid-eval and triggering Issue 5).

**Fix required**:
```python
with open(os.path.join(ckpt_dir, "config.json")) as f:
    config = json.load(f)
with open(os.path.join(ckpt_dir, "norm_stats.json")) as f:
    stats = json.load(f)
```

---

## ISSUE 7 — `Mod_server.py` hardcodes `"cuda"` throughout with no `--device` flag; crashes immediately on CPU-only runtimes

**Severity**: Medium  
**File**: `Mod_server.py`, lines 72 (implicit via `model.to("cuda")`), 76, 87, 91

**Root cause**:
```python
model = model.to("cuda")                              # line 76
return transforms.ToTensor()(pil).to("cuda")          # line 87
device = "cuda"                                       # line 91
```
`Mod_server.py`'s `argparse` only exposes `--ckpt_dir` and `--port`. There is no `--device` flag. `launch_eval.py` does parse `--device` (default `"cuda"`) but never passes it to `server_cmd` (related to Issue 2).

**Scenario**:
User is on a Colab CPU-only runtime (free tier with no GPU allocated, or GPU runtime not selected). `Mod_server.py` reaches `model.to("cuda")` and throws:
```
RuntimeError: No CUDA GPUs are available
```
or
```
AssertionError: Torch not compiled with CUDA enabled
```
Server exits at startup. Because of Issue 1, `launch_eval.py` waits 300 seconds before surfacing this.

**Fix required**:
1. Add `p.add_argument("--device", default="cuda")` to `Mod_server.py`'s `argparse`.
2. Pass `args.device` into `load_model_and_normalizer(ckpt_dir, device=args.device)` and replace all hardcoded `"cuda"` strings throughout `Mod_server.py` with the `device` variable.
3. Add `"--device", args.device` to `server_cmd` in `launch_eval.py` (alongside the fix for Issue 2).

---

## Summary Table

| # | File | Lines | Severity | Crashes? | Silent? | Trigger |
|---|------|--------|----------|----------|---------|---------|
| 1 | `launch_eval.py` | 53–63, 157–160 | High | Hangs 5 min on dead server | Partially | Any server startup failure |
| 2 | `launch_eval.py` | 82, 134–138 | Medium | No | Yes | `--timesteps` flag passed by user |
| 3 | `launch_eval.py` | 30–33, 172 | High | Yes — 0 eval results | Yes | Standalone MINT_EVO clone on Colab |
| 4 | `Mod_server.py` | 75 | High | Yes — server startup crash | No (error in log) | Multi-GPU training checkpoint |
| 5 | `Mod_server.py` | 122–132 | High | Yes — mid-eval, no summary written | Partial | OOM, shape error, malformed payload |
| 6 | `Mod_server.py` | 67–68 | Low | Eventually (fd exhaustion) | Yes | Many server restarts in one session |
| 7 | `Mod_server.py` | 76, 87, 91 | Medium | Yes — server startup crash | No (error in log) | CPU-only Colab runtime |
