
### 1. What you need before SSH works

1. Your UCSD B200 account must be created and your **public key** must be registered by the admin (Junda / Lanxiang).
   - If they don’t have your key yet, send them the output of:
     ```bash
     cat ~/.ssh/lightning_rsa.pub
     ```

2. You must be on the **UCSD VPN** (Cisco AnyConnect → `vpn.ucsd.edu`, login with Duo).

Once those two are true, set up SSH on your Mac.

---

### 2. Configure `~/.ssh/config`

Your `.ssh` directory is `/Users/aaronfeng/.ssh`. Do this in Terminal:

```bash
cd ~
ls .ssh
```

Fix permissions (good practice):

```bash
chmod 700 ~/.ssh
chmod 600 ~/.ssh/lightning_rsa
chmod 600 ~/.ssh/config 2>/dev/null || true
```

Now edit the config:

```bash
nano ~/.ssh/config
```

Add this block (or edit an existing `Host b200` block to match):

```
Host b200
    User zhf004
    HostName mlsys-b200.ucsd.edu
    IdentityFile /Users/aaronfeng/.ssh/lightning_rsa
    IdentitiesOnly yes
```

Save in nano: `Ctrl+O`, Enter, then `Ctrl+X`.

---

### 3. Commands to connect

When you are on UCSD VPN, simply run:

```bash
ssh b200
```

First time, you’ll be asked to accept the host key (`yes`), then enter your passphrase if you set one on `lightning_rsa`.

If you ever want to bypass the config and connect directly:

```bash
ssh -i ~/.ssh/lightning_rsa zhf004@mlsys-b200.ucsd.edu
```

---

### 4. Report on the rest of the commands in the guide

Below is what each major command/concept in the doc is for, grouped by topic.

#### 4.1 Hugging Face models and storage layout

- **`/raid/huggingface`**  
  Shared directory where Hugging Face models should live so everyone can reuse downloads instead of redownloading into `$HOME`.  
- **Soft link from your home to `/raid/huggingface`**  
  Typically something like:
  ```bash
  ln -s /raid/huggingface ~/.cache/huggingface
  ```
  so your tools think they’re writing to your home but actually use the shared RAID space.

- **Storage paths**:
  - `/` (root + home) – 2TB, scarce; avoid filling this.  
  - `/raid` – main large local storage; use e.g. `/raid/users/zhf004`.  
  - `/mnt/mlsys` – NFS archive storage (slower, for long-term stuff).

- To **check root space**:
  ```bash
df -h | grep /dev/md0   # focuses on the root md0 device
df -h                   # full disk usage for all mounts
  ```

If your home is filling up, move heavy stuff to `/raid`:
```bash
mkdir -p /raid/users/zhf004
mv ~/big_folder /raid/users/zhf004/
```

#### 4.2 GPU contention / monitoring

To see who is using GPUs:

- `nvitop`, `nvtop` – interactive TUI GPU monitors (like `htop` for GPUs).  
- `nvidia-smi` – standard NVIDIA status tool; shows GPU utilization, memory, running processes.

Use these before starting huge jobs to avoid stepping on others.

#### 4.3 Docker usage

All sample commands are `docker run` invocations to start containers with GPUs attached.

General structure:

```bash
docker run -it --gpus <which> --name <container_name> [ -v host:container ] <image> bash
```

Key options:

- `-it` – interactive terminal.
- `--gpus all` – expose all GPUs.
- `--gpus 0` – expose only GPU 0.
- `--name $(whoami)-sglang-blackwell` – names the container (here using your username).
- `-v $HOME/:/workspace` – mounts your home directory into `/workspace` inside the container.
- `<image>` – Docker image to use:
  - `lmsysorg/sglang:blackwell` – image for SGLang on Blackwell GPUs.
  - `nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04` – base CUDA 12.8.1 + cuDNN dev image.

Example commands from the doc:

1. **All GPUs, SGLang:**
   ```bash
   docker run -it --gpus all --name $(whoami)-sglang-blackwell lmsysorg/sglang:blackwell bash
   ```

2. **Only GPU 0, SGLang, with home mounted:**
   ```bash
   docker run -it --gpus 0 --name $(whoami)-sglang-blackwell -v $HOME/:/workspace lmsysorg/sglang:blackwell bash
   ```

3. **All GPUs, raw CUDA dev container:**
   ```bash
   docker run -it --gpus all --name $(whoami)-cuda nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 bash
   ```

To **detach** from a running container without stopping it:

- Press `Ctrl+P` then `Ctrl+Q`.

You can later reattach using:

```bash
docker attach <container_name>
```

#### 4.4 vLLM / PyTorch installation

Working approach (inside some Python env in the container):

1. Install **PyTorch 2.7.0 nightly** with CUDA 12.8:
   ```bash
   uv pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/test/cu128
   ```
   - `uv pip` is the `uv` package manager’s pip front-end (faster resolver).
   - `--index-url` points to the PyTorch test wheel index for that CUDA version.

2. Install **vLLM from source**:
   ```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -e .
   ```
   - `git clone vllm` – download the repo from GitHub.
   - `pip install -e .` – editable install; imports vLLM from the local source tree.

Alternative (admin shortcut): copy a prebuilt vLLM wheel / site-packages directory from Junda instead of building.

Non-working commands they explicitly warn about:

```bash
uv pip install -U vllm --extra-index-url https://wheels.vllm.ai/nightly
```

This pulls a vLLM version that downgrades Torch back to 2.6.0, which conflicts with the desired 2.7.0 setup.

---

If you want, next step after SSH works can be setting up a default workspace on `/raid/users/zhf004` and a standard Docker+conda or uv workflow so you’re not improvising each time you log in.
