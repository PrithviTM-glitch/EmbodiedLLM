# Downloading Open X-Embodiment Datasets

The Open X-Embodiment (OXE) datasets are large robotic manipulation datasets hosted on Google Cloud Storage. They use the RLDS (Robotic Learning Dataset Standard) format and require special handling to download.

## ⚠️ Important Notes

1. **Large Datasets**: OXE datasets are very large (2GB - 100GB+). Make sure you have sufficient disk space and network bandwidth.
2. **Authentication Required**: You need Google Cloud authentication to download from `gs://gresearch/robotics`.
3. **Time-Consuming**: Downloads can take hours depending on dataset size and network speed.

## 🚀 Recommended Approach: Use Synthetic Debug Mode

For testing the benchmark pipeline, **use the synthetic debug mode instead of downloading real data**:

```bash
cd scripts/octo
conda run -n octo python run_openx_benchmark.py --use-debug --max-episodes 5
```

This creates small synthetic episodes that exercise the full pipeline without requiring downloads.

## 📥 Option 1: Download via RLDS (Recommended for Real Data)

The OXE datasets are available through the RLDS format. To download:

### Setup
```bash
# Install RLDS
pip install rlds

# Authenticate with Google Cloud
gcloud auth login
gcloud auth application-default login
```

### Download Using Python
```python
import tensorflow_datasets as tfds

# Available datasets: fractal20220817_data, bridge_dataset, kuka, etc.
dataset_name = 'fractal20220817_data'

# This will download to vla-benchmark/data/open-x/ by default
# Or specify custom path with data_dir parameter
builder = tfds.builder(dataset_name, data_dir='../../data/open-x')
builder.download_and_prepare()

# Load dataset
ds = builder.as_dataset(split='train[:10%]')
```

### Specify Custom Download Location
```python
import tensorflow_datasets as tfds

data_dir = 'vla-benchmark/data/open-x'  # Project data directory
builder = tfds.builder('fractal20220817_data', data_dir=data_dir)
builder.download_and_prepare()
```

## 📥 Option 2: Use Pre-Downloaded Debug Dataset

OCTO includes a small debug dataset for testing. This is already in your repo:

```bash
cd vla-benchmark
ls models/octo/tests/debug_dataset/bridge_dataset/
```

To use it with the inspector:
```bash
cd scripts/octo
conda run -n octo python inspect_tfrecord.py
```

## 📥 Option 3: Manual Download from GCS

If you have `gsutil` installed and GCS access:

```bash
# Install Google Cloud SDK
# https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth login

# Download specific dataset
gsutil -m cp -r gs://gresearch/robotics/fractal20220817_data /path/to/local/data/

# This can take hours for large datasets!
```

## 🎯 Using Downloaded Data with Benchmark

Once you have data downloaded to a local directory:

```bash
cd scripts/octo

# Option 1: Use default TFDS data directory
conda run -n octo python run_openx_benchmark.py --dataset fractal20220817_data --max-episodes 10

# Option 2: Specify custom data directory (update benchmark_config.yaml)
# Edit: config/benchmark_config.yaml
# Add: data_path: /path/to/your/tfds/data

conda run -n octo python run_openx_benchmark.py --dataset fractal20220817_data --max-episodes 10
```

## 📊 Available Datasets

### Small Datasets (Good for Testing)
- **fractal20220817_data** (~2GB) - Manipulation with fractal objects
- **kuka** (~5GB) - Standard manipulation tasks

### Medium Datasets
- **bridge_dataset** (~20GB) - Diverse manipulation tasks
- **taco_play** (~10GB) - Tabletop manipulation

### Large Datasets
- **austin_buds_dataset** (~50GB+)
- **bc_z** (~100GB+)

## 🔍 Checking Dataset Availability

To see which datasets are configured in OCTO:

```bash
cd scripts/octo
conda run -n octo python -c "from octo.data.oxe import OXE_DATASET_CONFIGS; print(list(OXE_DATASET_CONFIGS.keys())[:20])"
```

## 🐛 Troubleshooting

### "Dataset not found" Error
- The dataset hasn't been downloaded yet
- Use `--use-debug` mode for testing
- Download using one of the methods above

### "No registered data_dirs" Error
- TFDS can't find the downloaded data
- Check that data is in TFDS format (has dataset_info.json, etc.)
- Verify the data_dir path in your config

### Authentication Errors
```bash
# Re-authenticate
gcloud auth login
gcloud auth application-default login

# Verify credentials
gcloud auth list
```

### Slow Downloads
- Normal for large datasets (GB to 100GB+)
- Use smaller datasets for testing
- Consider using synthetic debug mode
- Monitor progress with inspector script

## 💡 Best Practice Workflow

1. **Development/Testing**: Use `--use-debug` mode (no downloads needed)
   ```bash
   python run_openx_benchmark.py --use-debug --max-episodes 5
   ```

2. **Quick Validation**: Download small dataset (fractal, 2GB)
   ```python
   import tensorflow_datasets as tfds
   builder = tfds.builder('fractal20220817_data')
   builder.download_and_prepare()
   ```

3. **Full Evaluation**: Download full dataset (hours/days)
   - Use background process
   - Monitor disk space
   - Plan for long download times

## 📚 Additional Resources

- [Open X-Embodiment Website](https://robotics-transformer-x.github.io/)
- [RLDS Documentation](https://github.com/google-research/rlds)
- [TensorFlow Datasets](https://www.tensorflow.org/datasets)
- [OCTO Documentation](https://github.com/octo-models/octo)
