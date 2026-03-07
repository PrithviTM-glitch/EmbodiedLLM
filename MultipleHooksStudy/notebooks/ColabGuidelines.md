# Colab Notebook Design Guidelines

## File System

- **Runtime is ephemeral** — all files in `/content/` are lost on disconnect/restart. Never assume persistence.
- Working directory is `/content/`. Write all temp files here.
- `/content/sample_data/` exists by default; ignore or delete as needed.
- Do **not** rely on `/tmp/` across cells — same session only.

---

## Getting Code & Files Into the Runtime

**Clone a repo** (preferred for local scripts/modules):
```python
!git clone https://github.com/user/repo.git
import sys; sys.path.insert(0, '/content/repo')
```

**Install from a private repo** (use token):
```python
!git clone https://<TOKEN>@github.com/user/private-repo.git
```

**Copy a single file in:**
```python
!wget -q https://example.com/file.py -O /content/file.py
```

---

## Ports & Networking

- **Outbound HTTP/HTTPS works fine.** Restricted protocols (raw sockets, etc.) may be blocked.
- **Port 9000 is restricted** — do not use for socket communication.
- For internal server/client implementations, `localhost` works normally within the runtime — no tunneling needed.

---

## Persistent Storage Options

### Google Drive (simple, interactive)
```python
from google.colab import drive
drive.mount('/content/drive')
# Access files at /content/drive/MyDrive/
```
- Requires manual auth click. Not suitable for headless/automated runs.

### GCS via `gsutil`
```python
!gsutil cp gs://my-bucket/file.txt /content/file.txt
!gsutil cp /content/output.csv gs://my-bucket/output.csv
!gsutil -m cp -r gs://my-bucket/folder/ /content/folder/  # recursive
```
- Auth is handled automatically if running under a GCP service account or after `gcloud auth`.

### GCS via Python (`google-cloud-storage`)
```python
from google.cloud import storage
client = storage.Client(project='my-project')
bucket = client.bucket('my-bucket')
bucket.blob('output.csv').upload_from_filename('/content/output.csv')
```

### Authenticate GCP (if needed)
```python
from google.colab import auth
auth.authenticate_user()  # triggers OAuth flow
```
Or use a service account key:
```python
import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/content/key.json'
```

---

## GCFS (GCS FUSE) — Mount a Bucket as a Filesystem
```python
!echo "deb http://packages.cloud.google.com/apt gcsfuse-focal main" > /etc/apt/sources.list.d/gcsfuse.list
!apt-get install -qq gcsfuse
!mkdir -p /content/gcs_mount
!gcsfuse my-bucket /content/gcs_mount
```
- Useful for treating a bucket like a local directory.
- Unmount with `!fusermount -u /content/gcs_mount`.

---

## Secrets & Credentials

- Use **Colab Secrets** (left sidebar 🔑) — access via:
```python
from google.colab import userdata
api_key = userdata.get('MY_API_KEY')
```
- Never hardcode credentials in cells.

---

## Key Restrictions Summary

| Constraint | Detail |
|---|---|
| No persistent local disk | Save outputs to Drive or GCS immediately |
| No inbound ports | Use ngrok or Colab port forwarding |
| Session timeout | ~90 min idle, ~12 hr max (free tier) |
| GPU/TPU not guaranteed | Request via Runtime > Change runtime type |
| No systemd / init system | Use `!command &` for background processes |
| Limited RAM | ~12 GB free tier; monitor with `!free -h` |

---

## Best Practices

- **Pin library versions** — Colab updates packages; use `pip install lib==x.y.z`.
- **Cell order matters** — design notebooks to run top-to-bottom without skipping.
- **Save checkpoints** to GCS/Drive after expensive steps.
- Use `%%capture` to suppress noisy install output.
- Add a **"Setup" cell** at the top that handles all installs, clones, and mounts.