# CCDS GPU Cluster (TC1) — Quick Reference Guide
> Last Updated: 12 February 2026

---

## Table of Contents
1. [Introduction & Architecture](#introduction--architecture)
2. [Logging Into the Cluster](#logging-into-the-cluster)
3. [File Transfer (SFTP)](#file-transfer-sftp)
4. [Managing Your Home Directory](#managing-your-home-directory)
5. [Conda Package & Environment](#conda-package--environment)
6. [Using CUDA in TC1](#using-cuda-in-tc1)
7. [SLURM User Guide](#slurm-user-guide)
8. [Job Submission Guidelines](#job-submission-guidelines)
9. [Shared Resources](#shared-resources)
10. [SSH Tunneling & Jupyter](#ssh-tunneling--jupyter)
11. [Other Resources (Applying for More)](#other-resources-applying-for-more)
12. [Important Notices](#important-notices)

---

## Introduction & Architecture

The TC1 GPU Cluster has two node types:

| Node Type | Purpose |
|-----------|---------|
| **Head Node** (`CCDS-TC1`) | Main access point via SSH. No GPU. No code execution allowed here. |
| **Compute Nodes** (`TC1N01`–`TC1N07`) | Where GPU computation happens. Jobs are dispatched here by SLURM. |

**Key rule:** You must **never** run code or GPU commands on the Head Node. Always submit jobs via SLURM.

### User Workflow
1. SSH into the Head Node (home directory created on first login)
2. Upload job scripts, code, and datasets via SFTP
3. Set up your Conda environment(s)
4. Verify your Conda environment via Jupyter on a Compute Node
5. Submit non-interactive jobs via SLURM (`sbatch`)
6. Download output files via SFTP

### Client Tools

| Tool | Platform | Purpose |
|------|----------|---------|
| **PuTTY** | Windows | SSH client |
| **WinSCP** | Windows | SFTP file transfer |
| **FileZilla** | macOS/Linux | SFTP file transfer |
| Terminal (`ssh`) | macOS/Linux | SSH built-in |

---

## Logging Into the Cluster

> The cluster is only accessible **within NTU network**. Off-campus users must connect via [NTU VPN](https://vpngate.ntu.edu.sg).

| Field | Value |
|-------|-------|
| **Hostname** | `CCDS-TC1` |
| **IP Address** | `10.96.189.11` |
| **Port** | `22` |
| **Credentials** | NTU Network Account ID (lowercase) + Password |

### SSH Commands
```bash
ssh -l <username_lowercase> 10.96.189.11
# or
ssh -p 22 <username_lowercase>@10.96.189.11
```

- On first login, accept the host key prompt.
- To exit: type `exit` or press `Ctrl + D`

---

## File Transfer (SFTP)

### WinSCP (Windows)
1. Launch WinSCP → set **File Protocol** to `SFTP`
2. Host name: `10.96.189.11`, Port: `22`
3. Enter NTU credentials
4. Verify your home directory path with:
   ```bash
   pwd
   ```
5. Drag-and-drop files between local and remote windows

### FileZilla (macOS/Linux)
- Download from https://filezilla-project.org

---

## Managing Your Home Directory

- Default disk quota: **at least 100 GB** (varies by account)
- If quota usage exceeds **98%**: unable to login, compute, or save data

### Check Disk Usage
```bash
ncdu        # Interactive disk usage viewer
# Press 'q' to exit
```

### Best Practices
- Regularly transfer and backup data to your local machine
- Remove unwanted data and unused Conda environments
- **Do NOT delete** system files prefixed with `.` (e.g. `.bashrc`, `.bash_profile`, `.config`)
- The `.conda` folder stores your environments — remove unused ones to free space

---

## Conda Package & Environment

### Module Commands
```bash
module avail                    # List available shared applications
module show <module_name>       # View description of a module
module load <module_name>       # Load a module
module list                     # List currently loaded modules
module unload <module_name>     # Unload a module
module purge                    # Unload all modules
module --help                   # Help
```

> **Note:** Load the `slurm` module on every login. Required for SLURM job commands.
> ```bash
> module load slurm
> ```

### Loading Anaconda / Miniconda
```bash
module load anaconda            # Load Anaconda
# or
module purge
module load miniconda/py39      # Load Miniconda with Python 3.9

whereis conda                   # Find conda path
which conda

conda init bash                 # First-time setup: adds conda to ~/.bashrc
source .bashrc                  # Apply changes immediately
conda info                      # View loaded Anaconda info
conda list                      # List packages in current env
conda deactivate                # Exit conda module/environment
```

> You are **not allowed** to install packages in the base environment.

### Managing Conda Environments

```bash
# Create and activate
conda create -n TestEnv
conda activate TestEnv

# Create at custom path
conda create --prefix /scratch-shared/myteam/envs/TestEnv
conda activate /scratch-shared/myteam/envs/TestEnv

# Exit environment
conda deactivate

# List environments
conda env list

# Rename (clone + remove old)
conda create -n NewEnv --clone OldEnv
conda env remove -n OldEnv

# Export / backup
conda env export > TestEnv.yml

# Create from yml
conda env create -n Env002 -f TestEnv.yml

# Update from yml
conda env update --prefix ./TestEnv --file TestEnv.yml --prune

# Remove environment
conda env remove -n TestEnv
# or directly remove folder:
rm -Rf ~/.conda/envs/<folder>
```

### Managing Environment Variables
```bash
conda env config vars list                      # List variables
conda env config vars set my_var1=value         # Set variable
conda env config vars unset my_var1             # Remove variable
conda activate TestEnv                          # Re-activate after changes
```

### Managing Packages
```bash
conda create -n TestEnv
conda activate TestEnv

conda search python                             # Search available packages
conda install python=3.11                       # Install specific version

# Third-party channel (e.g. conda-forge)
conda search -c conda-forge <package>
conda install -c conda-forge <package>
conda install conda-forge::<package>

conda list                                      # List installed packages
conda install python=3.10                       # Install/replace version
conda uninstall python=3.9                      # Remove package
```

> **Important:** Prefer `conda install` over `pip install` to avoid incompatibility. Use `pip` only if the package is unavailable via conda.

---

## Using CUDA in TC1

> GPU cards compatible with CUDA are in **Compute Nodes only** — never in the Head Node.  
> Do **not** run `nvidia-smi` or `nvcc --version` on the Head Node.

### System-Installed CUDA (via Modules)
```bash
module avail | grep cuda        # List available CUDA modules
module load cuda/12.9           # Example: load CUDA 12.9
```

### Install CUDA in Your Own Conda Environment
```bash
# Search for CUDA toolkit from NVIDIA
conda search -c nvidia cuda-toolkit

# Search from Anaconda
conda search cudatoolkit

# Install specific version
conda install -c nvidia cuda-toolkit=<version>

# Install NVIDIA CUDA Compiler (nvcc)
conda install -c nvidia cuda-nvcc=<version>
```

### Verify CUDA on Head Node
```bash
conda activate <env_name>
conda list | grep cuda
```

### Verify CUDA on Compute Node (via Jupyter Terminal)
```bash
conda env list                  # List environments
conda activate <env_name>
conda list | grep cuda          # Check CUDA packages
nvcc -V                         # Check CUDA compiler version
nvidia-smi                      # Check GPU info (assigned GPU only)
```

---

## SLURM User Guide

**SLURM** (Simple Linux Utility for Resource Management) schedules and manages jobs on the cluster.

### Check Your QoS Assignment
```bash
sacctmgr show user <username> withassoc format=user,qos
```

### Check QoS Resource Limits
```bash
sacctmgr -P show qos <QoS_name> withassoc format=name,MaxTRESPU,MaxJobsPU,MaxWall
```

**Example QoS "normal" limits:**
| Resource | Limit |
|----------|-------|
| CPUs | 20 |
| GPU cards | 1 |
| Memory | 64 GB |
| Max concurrent jobs | 2 |
| Max wall time | 6 hours |

### Common SLURM Commands

| Command | Usage | Description |
|---------|-------|-------------|
| `sbatch` | `sbatch job.sh` | Submit job script (background execution) |
| `squeue` | `squeue` | Show jobs in queue |
| `squeue -la` | | More detailed queue view |
| `squeue -u <user>` | | Jobs for a specific user |
| `scancel` | `scancel <jobid>` | Cancel a job |
| `sinfo` | `sinfo -N -l` | Show compute node states |
| `sacct` | `sacct -u <user>` | View job history |
| `scontrol` | `scontrol show -d jobid <id>` | View running job details |
| `seff` | `seff <jobid>` | Resource efficiency report for completed job |
| `MyTCinfo` | `MyTCinfo` | Custom script: view your account + QoS info |
| `MyJobHistory` | `MyJobHistory` | Custom script: view today's job history |
| `TC1RunningJob` | `TC1RunningJob` | Custom script: view current running jobs |

### Node States

| State | Meaning |
|-------|---------|
| `idle` | Free, ready to accept jobs |
| `mixed` | Jobs running, but capacity available |
| `down` | SLURM service is down |
| `completing` | Finishing last job |
| `mixed+drain` / `draining` | Jobs running but node has failing tasks |
| `drained` | Failed tasks, cannot accept jobs — needs admin reset |

---

## Job Submission Guidelines

### Rules
1. **Never** run code on the Head Node — offenders may be banned
2. Only use your **assigned QoS**
3. Keep resource requests **within your QoS limits**

### Job Script Flags

| Resource | Flag | Notes |
|----------|------|-------|
| Partition* | `--partition=UGGPU-TC1` | Required |
| QoS* | `--qos=normal` | Must match your assigned QoS |
| GPU* | `--gres=gpu:1` | Omit if not using GPU |
| Memory* | `--mem=8G` or `--mem=8000M` | Required |
| Nodes* | `--nodes=1` | Max 1 per job |
| Time* | `--time=60` | Minutes, or `HH:MM:SS`, max 6hr |
| Job name* | `--job-name=MyJob` | |
| Output file* | `--output=output_%x_%j.out` | `%x`=job name, `%j`=job ID |
| Error file* | `--error=error_%x_%j.err` | |
| CPUs (optional) | `--ntasks-per-node=2` | Default = 1 |
| CPUs (optional) | `--cpus-per-tasks=2` | Default = 1 |
| Node (optional) | `--nodelist=TC1N04` | Pin to specific node |

### Example Job Script: `job.sh`
```bash
#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=60
#SBATCH --job-name=MyJob
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

module load cuda/12.2
module load anaconda
source activate TestEnv
python test.py
```

### Submit & Monitor
```bash
sbatch job.sh                               # Submit
squeue                                      # Check queue
squeue -u <username>                        # Your jobs only
sacct -u <username>                         # Job history
seff <jobid>                                # Resource utilization
scancel <jobid>                             # Cancel job
```

### Common Pending (PD) Reasons

| Reason Code | Fix |
|-------------|-----|
| `QOSMaxWallDurationPerJobLimit` | Reduce `--time` value |
| `QOSMaxJobsPerUserLimit` | Cancel jobs to stay under MaxJobsPU |
| `QOSMaxMemoryPerUser` | Reduce `--mem` across all your jobs |
| `QOSMaxCpuPerUserLimit` | Reduce `--ntasks-per-node` |
| `QOSMaxGRESPerUser` | Already using 1 GPU — omit `--gres` from second job |
| `Priority` | Waiting; will run when resources free up |
| `Resources` | Node resources unavailable; will run eventually |

> **Note:** Pending jobs are auto-removed from queue when wait time exceeds MaxWall.

### GPU Memory Note
GPU cards are **NVIDIA Tesla V100 32GB** — memory is fixed at 32GB and cannot be adjusted in the job script. Control GPU memory allocation within your code.

---

## Shared Resources

Shared Conda environments are available under `/tc1apps/2_conda_env/`:

| Environment | Python | Key Packages | Notes |
|-------------|--------|--------------|-------|
| `CZ4042_v4` | 3.10 | PyTorch 2.4, CUDA 12.4, TensorFlow 2.9, cudnn 8.9 | Set `TF_ENABLE_ONEDNN_OPTS=0` for TF |
| `CZ4042_v5` | 3.11.4 | PyTorch 2.0.1, CUDA 11.7, Jupyter | |
| `CZ4042_v6` | 3.12.3 | PyTorch 2.5.1, CUDA 11.8, Jupyter | **Latest** — for SC4001/CZ4042 |
| `Jupyter` | 3.12 | JupyterLab + Notebook | For interactive sessions |

### Clone the CZ4042_v6 Environment Locally
```bash
cd
cp /tc1apps/3_conda_yml/SC4001.yml .
conda env create -f SC4001.yml -n mySC4001
```

### Launch Jupyter from Shared Environment
Add to your job script:
```bash
module load anaconda
source activate /tc1apps/2_conda_env/Jupyter
jupyter-lab --ip=$(hostname -i) --port=8881
```

> Check if a port is free: `netstat -tp | grep <port_number>`

---

## SSH Tunneling & Jupyter

> Jupyter requires SSH tunneling because the cluster is only accessible via SSH.

### Full Setup Steps

#### 1. Create a Conda Environment for Jupyter
```bash
module load anaconda
conda create -n RunJupyter
conda activate RunJupyter
conda install -c conda-forge jupyterlab notebook
conda install -c conda-forge ipykernel
conda deactivate
```

#### 2. Create and Submit the Job Script (`run1.sh`)
```bash
#!/bin/sh
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --mem=10G
#SBATCH --gres=gpu:1
#SBATCH --time=360
#SBATCH --nodes=1
#SBATCH --job-name=run1
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

module load anaconda
source activate RunJupyter
jupyter-lab --ip=$(hostname -i) --port=8882
# or for Notebook:
# jupyter-notebook --ip=$(hostname -i) --port=8882
```
```bash
sbatch run1.sh
```

#### 3. Get Access Info from Error Log
```bash
tail error_run1_<jobid>.err
# Note the IP address, port, and URL with token
```

#### 4. Set Up SSH Tunnel in PuTTY
- Right-click session title bar → **Change Settings**
- Navigate to **Connection > SSH > Tunnels**
- Source port: `8882`
- Destination: `10.128.10.11:8882` (use IP from error log)
- Click **Add**, then **Apply**

#### 5. Access Jupyter in Browser
```
http://127.0.0.1:8882/lab?token=<your_token>
```

#### 6. Shut Down Jupyter
- Via browser: **File → Shut Down**
- Via terminal: `scancel <jobid>`

> **Best practice:** Cancel your Jupyter job when done to free resources for others.

### Useful Commands Inside Jupyter Terminal
```bash
conda env list                  # List your environments
conda activate <env_name>       # Activate an environment
nvidia-smi                      # GPU info (assigned GPU only)
nvtop                           # Real-time GPU monitoring (press Q to exit)
free -h                         # Memory info
top -i                          # Real-time processes
```

---

## Log Files for Monitoring

| Log Directory | File Format | Content |
|---------------|-------------|---------|
| `/tc1share2/Log-RunHistory` | `Running_YYYY-MM-DD-HH-mm` | Running job GPU usage, 5-min intervals, last 24H |
| `/tc1share2/Log-JobHistory` | `JobHistory_YYYY-MM-DD-HH-mm` | Jobs computed in last 48H |
| `/tc1share2/Log-TC1Status` | `TC1_ClusterInfo`, `ActiveNode_YYYY-MM-DD-HH-mm` | Overall resource utilization, last 28H |

```bash
more /tc1share2/Log-JobHistory/JobHistory_2025-01-09-*
more /tc1share2/Log-TC1Status/TC1_ClusterInfo
```

---

## Other Resources (Applying for More)

### Additional QoS
- Options: 8hr, 12hr, 24hr, up to 48hr; more CPU/Memory/GPU
- Max usage period: 1 month
- Email: **CCDSgpu-tc@ntu.edu.sg** with reason + proof of current usage

### Additional Storage Space
- Minimum request: >100 GB
- Max usage period: 4 months
- Email: **CCDSgpu-tc@ntu.edu.sg** — a form will be sent for completion

---

## Important Notices

1. Resources are for **academic use only** — misuse results in account termination
2. All activities in TC1 may be **monitored and logged** by authorized personnel
3. **Do NOT access** directories outside your Home Directory and assigned Shared folders
4. Jobs with unauthorized QoS will be **terminated without notice**; repeat offenders are banned
5. **No data backup service** — you are responsible for your own data
6. For coursework/project issues: consult your Supervisor or TA
7. For technical issues: **CCDSgpu-tc@ntu.edu.sg**

---

*Reference compiled from CCDS GPU Cluster (TC1) User Guide — Last Updated 12 February 2026*

