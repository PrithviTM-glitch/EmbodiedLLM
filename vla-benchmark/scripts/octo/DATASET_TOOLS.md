# Dataset Tools - Quick Reference

This guide covers the new dataset inspection and download tools.

## 🔍 Inspector Script (`inspect_tfrecord.py`)

Inspect TFRecord files, validate dataset structure, and monitor downloads in real-time.

### Quick Start

```bash
cd /Users/tmprithvi/Code/EmbodiedLLM/vla-benchmark/scripts/octo

# Inspect the debug dataset (no args needed)
python inspect_tfrecord.py

# Inspect a specific TFRecord file
python inspect_tfrecord.py --tfrecord /path/to/file.tfrecord-00000-of-00001

# Inspect a TFDS dataset
python inspect_tfrecord.py --dataset bridge_dataset --data-dir ~/tensorflow_datasets

# Monitor a directory during download (run in separate terminal)
python inspect_tfrecord.py --monitor --data-dir ../../data/open-x/bridge_dataset --interval 5
```

### Use Cases

**1. Debug local TFRecord files**
```bash
# Inspect the OCTO debug dataset
python inspect_tfrecord.py

# Output shows:
# - Dataset metadata (splits, sizes)
# - Feature structure (keys, shapes, dtypes)
# - Sample values
# - Number of records
```

**2. Monitor downloads in real-time**
```bash
# Terminal 1: Start download
python download_oxe_dataset.py --dataset fractal20220817_data

# Terminal 2: Monitor progress
python inspect_tfrecord.py --monitor --data-dir ../../data/open-x/fractal20220817_data --interval 5
```

Shows:
- New files as they appear
- File sizes growing
- Total dataset size

**3. Validate dataset after download**
```bash
python inspect_tfrecord.py --dataset fractal20220817_data \
    --data-dir ../../data/open-x
```

Confirms:
- Dataset is readable
- Feature structure is correct
- Can load samples

---

## 📥 Download Script (`download_oxe_dataset.py`)

Download Open X-Embodiment datasets from Google Cloud to local storage.

### Quick Start

```bash
cd /Users/tmprithvi/Code/EmbodiedLLM/vla-benchmark/scripts/octo

# List available datasets
python download_oxe_dataset.py --list

# Download a small dataset (recommended for testing)
python download_oxe_dataset.py --dataset fractal20220817_data

# Download to custom location
python download_oxe_dataset.py --dataset kuka --output-dir /my/custom/path

# Download partial dataset (faster)
python download_oxe_dataset.py --dataset bridge_dataset --split "train[:5%]"
```

### Recommended Datasets for Testing

| Dataset | Size | Description | Command |
|---------|------|-------------|---------|
| `fractal20220817_data` | ~2GB | Small, quick download | `python download_oxe_dataset.py --dataset fractal20220817_data` |
| `kuka` | ~5GB | Standard manipulation | `python download_oxe_dataset.py --dataset kuka` |
| `bridge_dataset` | ~20GB | Diverse tasks (large) | `python download_oxe_dataset.py --dataset bridge_dataset --split "train[:10%]"` |

### Authentication Setup

OXE datasets are on Google Cloud Storage. You may need authentication:

```bash
# Show auth instructions
python download_oxe_dataset.py --setup-auth

# Or manually:
# 1. Install gcloud
brew install --cask google-cloud-sdk

# 2. Authenticate
gcloud auth login
gcloud auth application-default login

# 3. Test access
gsutil ls gs://gresearch/robotics/
```

### Download Workflow

**Option A: Simple download (foreground)**
```bash
python download_oxe_dataset.py --dataset fractal20220817_data
# Wait for completion...
```

**Option B: Background download with monitoring**
```bash
# Terminal 1: Start download in background
python download_oxe_dataset.py --dataset fractal20220817_data &

# Terminal 2: Monitor progress
python inspect_tfrecord.py --monitor \
    --data-dir ../../data/open-x/fractal20220817_data \
    --interval 5
```

**Option C: Partial download for testing**
```bash
# Download only 5% of training data
python download_oxe_dataset.py --dataset kuka --split "train[:5%]"
```

### After Download

Once downloaded, use the dataset with benchmarks:

```bash
# Inspect to verify
python inspect_tfrecord.py --dataset fractal20220817_data \
    --data-dir ../../data/open-x

# Run benchmark
python run_openx_benchmark.py \
    --dataset fractal20220817_data \
    --data-path ../../data/open-x \
    --max-episodes 10
```

---

## 🚀 Complete Workflow Example

Here's a complete workflow from download to benchmark:

```bash
cd /Users/tmprithvi/Code/EmbodiedLLM/vla-benchmark/scripts/octo

# 1. List available datasets
python download_oxe_dataset.py --list

# 2. Download a small dataset
python download_oxe_dataset.py --dataset fractal20220817_data

# 3. Verify download
python inspect_tfrecord.py --dataset fractal20220817_data \
    --data-dir ../../data/open-x

# 4. Run synthetic debug test first (no dataset needed)
python run_openx_benchmark.py --max-episodes 5 --use-debug

# 5. Run real benchmark on downloaded data
python run_openx_benchmark.py \
    --dataset fractal20220817_data \
    --max-episodes 10

# 6. Check results
ls -lh ../../results/openx/
cat ../../results/openx/OpenX_*.json | python -m json.tool | head -50
```

---

## 📊 Expected Output

### Inspector Output (debug dataset)

```
================================================================================
Debug Dataset Metadata
================================================================================
{
  "name": "bridge_dataset",
  "version": "1.0.0",
  "splits": [
    {
      "name": "train",
      "numBytes": "3474742",
      "shardLengths": ["25"]
    }
  ]
}

================================================================================
Inspecting TFRecord File: bridge_dataset-train.tfrecord-00000-of-00001
================================================================================
📁 Path: .../bridge_dataset/1.0.0/bridge_dataset-train.tfrecord-00000-of-00001
📊 Size: 3393.30 KB

🔍 Reading first 2 records...

📦 Record 1:
   Raw size: 1234567 bytes
   Features (15):
     - observation/image: bytes_list (length: 1)
     - observation/state: float_list (length: 7)
     - action: float_list (length: 7)
     - language_instruction: bytes_list (length: 1)
     ...
```

### Download Output

```
================================================================================
Downloading Dataset: fractal20220817_data
================================================================================

📁 Output directory: /Users/tmprithvi/Code/EmbodiedLLM/vla-benchmark/data/open-x/fractal20220817_data
📊 Split: full dataset
⏰ Started at: 2025-11-17 14:30:00

[1/3] Creating dataset builder...
✅ Builder created
   Dataset: fractal20220817_data
   Version: 0.1.0

[2/3] Downloading and preparing dataset...
Downloading: 100%|██████████| 2.1G/2.1G [05:23<00:00, 6.5MB/s]
Extracting: 100%|██████████| 150/150 [01:15<00:00, 2.0files/s]

✅ Download completed in 384.5s (6.4 minutes)

[3/3] Verifying download...
✅ Verification successful
   Sample keys: ['observation', 'action', 'task']

================================================================================
Download Summary
================================================================================
Dataset: fractal20220817_data
Location: .../data/open-x/fractal20220817_data
Download time: 6.4 minutes
TFRecord files: 150
Total size: 2.05 GB

✅ Dataset ready for use!
```

### Monitor Output

```
================================================================================
📡 Monitoring Directory: ../../data/open-x/bridge_dataset
   Check interval: 5s
   Press Ctrl+C to stop
================================================================================

⚡ Change detected at 14:32:15
   📥 New files (12):
      + dataset_info.json (2.34 KB)
      + bridge_dataset-train.tfrecord-00000-of-00100 (25.6 MB)
      + bridge_dataset-train.tfrecord-00001-of-00100 (25.6 MB)
      ...
   💾 Total size: 156.34 MB

⚡ Change detected at 14:32:20
   📥 New files (8):
      + bridge_dataset-train.tfrecord-00012-of-00100 (25.6 MB)
      ...
   💾 Total size: 362.78 MB
```

---

## 🛠️ Troubleshooting

### Problem: "Dataset not found"

```bash
# Check if dataset name is correct
python download_oxe_dataset.py --list

# Verify OCTO is installed
python list_oxe_datasets.py
```

### Problem: "Permission denied" or "Access denied"

```bash
# Setup GCloud authentication
python download_oxe_dataset.py --setup-auth

# Or manually
gcloud auth login
gcloud auth application-default login
```

### Problem: Download is very slow

```bash
# Download only a subset
python download_oxe_dataset.py --dataset kuka --split "train[:5%]"

# Or use synthetic debug mode instead
python run_openx_benchmark.py --use-debug
```

### Problem: Disk space issues

```bash
# Check available space
df -h

# Download smaller dataset
python download_oxe_dataset.py --dataset fractal20220817_data  # ~2GB

# Or download partial
python download_oxe_dataset.py --dataset bridge_dataset --split "train[:1%]"
```

---

## 📝 Summary

**For quick testing (no download):**
```bash
python run_openx_benchmark.py --use-debug --max-episodes 5
```

**For real evaluation:**
```bash
# 1. Download dataset
python download_oxe_dataset.py --dataset fractal20220817_data

# 2. Run benchmark
python run_openx_benchmark.py --dataset fractal20220817_data --max-episodes 10
```

**For debugging:**
```bash
# Inspect local files
python inspect_tfrecord.py

# Monitor downloads
python inspect_tfrecord.py --monitor --data-dir ../../data/open-x/{dataset}
```
