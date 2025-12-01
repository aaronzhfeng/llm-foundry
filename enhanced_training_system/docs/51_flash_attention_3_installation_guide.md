
# Guide to Installing FlashAttention 3

## Overview

FlashAttention 3 is the third iteration of the FlashAttention algorithm and is engineered specifically for NVIDIA Hopper GPUs (such as the **H100** and **H800**).  It takes advantage of new Hopper features (Warpgroup Matrix Multiply‑Accumulate, Tensor Memory Accelerator and low‑precision FP8) to deliver up to **1.5–2×** faster attention than FlashAttention 2.  According to the maintainers, FlashAttention 3 is currently released as a **beta** library and is not yet integrated into popular frameworks such as Hugging Face Transformers https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/174457385/Installing%20flash-attention%203%20for%20hopper#:~:text=FlashAttention3%20has%20not%20yet%20been,attention%20functions%20from%20%60flash_attn_interface.  You must call its functions directly via `flash_attn_interface`.

### Hardware and software requirements

- **GPU** – FlashAttention 3 only runs on Hopper-class GPUs (H100/H800).  It does not work on earlier GPUs like the A100 or RTX 3090.  The official README notes that the beta release is “optimized for Hopper GPUs” and lists the requirement as **H100/H800**https://github.com/Dao-AILab/flash-attention#:~:text=FlashAttention.
- **CUDA Toolkit** – Requires **CUDA 12.3** or newer (the maintainers recommend CUDA 12.8 for best performance)https://github.com/Dao-AILab/flash-attention#:~:text=Requirements%3A%20H100%20%2F%20H800%20GPU%2C,12.3.
- **PyTorch** – Works with **PyTorch 2.2 or later**https://github.com/Dao-AILab/flash-attention#:~:text=Installation%20and%20features.  The SURF knowledge‑base example uses PyTorch 2.7 with CUDA 12.6/12.4 moduleshttps://servicedesk.surf.nl/wiki/spaces/WIKI/pages/174457385/Installing%20flash-attention%203%20for%20hopper#:~:text=module%20load%202024%20Python%2F3.12.3,2.7.
- **Operating system** – Linux is required.  Windows support is experimental and requires manual compilation; the README notes that Windows compilation still needs more testing https://github.com/Dao-AILab/flash-attention#:~:text=,reach%20out%20via%20Github%20issue.
- **Compiler tools** – An NVCC compiler from the CUDA toolkit and a C++ compiler.  The installation uses the build tool **ninja**; ensure `ninja` is installed and functioning (run `ninja --version`).  The maintainers warn that without ninja the compile may take hours, whereas with ninja it completes in minutes https://github.com/Dao-AILab/flash-attention#:~:text=,core%20machine%20using%20CUDA.
- **Python packages** – You need `packaging` and `ninja`https://github.com/Dao-AILab/flash-attention#:~:text=Installation%20and%20features along with standard packages such as `torch`, `numpy`, `pytest`, `einops` and `setuptools` (these are installed in the SURF example) https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/174457385/Installing%20flash-attention%203%20for%20hopper#:~:text=python%20,pytest%20packaging%20setuptools%20einops%20ninja.

## Step‑by‑step installation from source

### 1. Prepare the environment

1. **Install CUDA and NVCC.**  Make sure the CUDA version (>=12.3) matches your intended PyTorch build.  In HPC environments, modules such as `CUDA/12.6.0` or `CUDA/12.4.0` are loaded before building FlashAttention 3 https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/174457385/Installing%20flash-attention%203%20for%20hopper#:~:text=module%20load%202024%20Python%2F3.12.3,2.7.
2. **Create a Python virtual environment.**  Use `python -m venv venv` or `conda create`.  Activate it with `source venv/bin/activate`https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/174457385/Installing%20flash-attention%203%20for%20hopper#:~:text=python%20,venv%2Fbin%2Factivate.
3. **Install dependencies.**  Inside the virtual environment install PyTorch (with the appropriate CUDA version) along with other packages:

   ```bash
   python -m pip install torch numpy pytest packaging setuptools einops ninja
   ```

   The `packaging` and `ninja` packages are explicitly required by the FlashAttention build system https://github.com/Dao-AILab/flash-attention#:~:text=Installation%20and%20features.  Verify that `ninja` works by running `ninja --version`; reinstall it if needed https://github.com/Dao-AILab/flash-attention#:~:text=,core%20machine%20using%20CUDA.

### 2. Fetch the source code

Clone the official repository and move into the **hopper** directory, which contains FlashAttention 3:

```bash
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention/hopper
```

The maintainers emphasise that the FA3 code lives under `hopper` and that you should run your commands from that directory.  In a support issue they clarified that the correct sequence is `cd flash-attention/hopper` before running `python setup.py install` https://github.com/Dao-AILab/flash-attention/issues/1050#:~:text=Assuming%20you%27re%20referring%20to%20FlashAttention,release.

### 3. Compile and install

Run the installation script using Python:

```bash
python setup.py install
```

This command invokes a C++/CUDA build that compiles the kernels for your GPU.  The README lists this as the installation procedure for FlashAttention 3 https://github.com/Dao-AILab/flash-attention#:~:text=FlashAttention.  Important tips:

- **Parallel jobs.**  The build uses all CPU cores by default and can consume significant memory.  If your machine has limited RAM, set `MAX_JOBS` to a smaller number:

  ```bash
  MAX_JOBS=4 python setup.py install
  ```

  to limit parallel compile jobs, as recommended in the README https://github.com/Dao-AILab/flash-attention#:~:text=If%20your%20machine%20has%20less,MAX_JOBS.  Conversely, on large machines you can speed up compilation by increasing `MAX_JOBS`.  In a February 2025 issue Tri Dao suggested using `MAX_JOBS=32`, `64` or even `128` (depending on RAM).  Setting `MAX_JOBS=128` reduced the build time to 10‑15 minutes https://github.com/Dao-AILab/flash-attention/issues/1486#:~:text=Member.
- **Disable unused features.**  If you only need certain head dimensions and want a faster build, you can disable features using environment variables.  The maintainer shared a list that disables most features except head dimension 128 https://github.com/Dao-AILab/flash-attention/issues/1486#:~:text=Member.  For example:

  ```bash
  export FLASH_ATTENTION_DISABLE_BACKWARD=FALSE
  export FLASH_ATTENTION_DISABLE_SPLIT=TRUE
  export FLASH_ATTENTION_DISABLE_LOCAL=TRUE
  export FLASH_ATTENTION_DISABLE_PAGEDKV=TRUE
  export FLASH_ATTENTION_DISABLE_FP16=TRUE
  export FLASH_ATTENTION_DISABLE_FP8=TRUE
  export FLASH_ATTENTION_DISABLE_APPENDKV=TRUE
  export FLASH_ATTENTION_DISABLE_VARLEN=TRUE
  export FLASH_ATTENTION_DISABLE_CLUSTER=FALSE
  export FLASH_ATTENTION_DISABLE_PACKGQA=TRUE
  export FLASH_ATTENTION_DISABLE_SOFTCAP=TRUE
  export FLASH_ATTENTION_DISABLE_HDIM64=TRUE
  export FLASH_ATTENTION_DISABLE_HDIM96=TRUE
  export FLASH_ATTENTION_DISABLE_HDIM128=FALSE
  export FLASH_ATTENTION_DISABLE_HDIM192=TRUE
  export FLASH_ATTENTION_DISABLE_HDIM256=TRUE
  MAX_JOBS=128 python setup.py install
  ```

  This reduces compile time to about a minute on systems with many cores https://github.com/Dao-AILab/flash-attention/issues/1486#:~:text=Member.  Omit environment variables for features you require.

- **Build anywhere** – you do **not** need Hopper GPUs to compile the library.  Tri Dao confirmed that you can build on any machine as long as it has the CUDA compiler `nvcc` https://github.com/Dao-AILab/flash-attention/issues/1486#:~:text=I%20have%20one%20more%20question%3A,use%20later%20on%20H100%20cluster.  However, you must run the compiled library on a Hopper GPU.

### 4. Run tests

After installation, verify that the kernels work.  The README suggests adding the project root to your `PYTHONPATH` and running the provided tests:

```bash
export PYTHONPATH=$PWD  # inside the hopper directory
pytest -q -s test_flash_attn.py
```

The test command is documented in the README as the way to check the build https://github.com/Dao-AILab/flash-attention#:~:text=To%20run%20the%20test%3A.  In HPC environments, the SURF knowledge‑base example executes the same test after installation https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/174457385/Installing%20flash-attention%203%20for%20hopper#:~:text=git%20clone%20https%3A%2F%2Fgithub.com%2FDao,py%20install.

### 5. Using FlashAttention 3

Once installed, import the interface and call its functions directly:

```python
import flash_attn_interface
# call the FP16/BF16 attention function
output = flash_attn_interface.flash_attn_func(q, k, v, dropout_p=0.0)
```

The README shows that after installation the module can be imported with `import flash_attn_interface` and functions like `flash_attn_func()` can be called https://github.com/Dao-AILab/flash-attention#:~:text=Once%20the%20package%20is%20installed%2C,can%20import%20it%20as%20follows.  The SURF knowledge‑base notes that FA3 is not yet integrated into frameworks like Hugging Face and must be used via these interface functions https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/174457385/Installing%20flash-attention%203%20for%20hopper#:~:text=FlashAttention3%20has%20not%20yet%20been,attention%20functions%20from%20%60flash_attn_interface.

## Alternative installation: prebuilt wheels (community builds)

As of November 2025 there is **no official pip wheel** for FlashAttention 3.  A GitHub issue raised in September 2025 requested prebuilt wheels https://github.com/Dao-AILab/flash-attention/issues/1896#:~:text=Issue%20body%20actions.  In response, community members published unofficial wheels:

- **Hugging Face dataset** `malaysia-ai/Flash-Attention3-wheel`.  It provides wheels built on H100 GPUs for specific combinations of PyTorch and CUDA versions.  Installation requires downloading the wheel matching your environment and renaming it to remove the version suffix before installing.  The dataset page provides an example:

  ```bash
  wget https://huggingface.co/datasets/mesolitica/Flash-Attention3-whl/resolve/main/flash_attn_3-3.0.0b1-cp39-abi3-linux_x86_64-2.7.1-12.8.whl -O \
        flash_attn_3-3.0.0b1-cp39-abi3-linux_x86_64.whl
  pip3 install flash_attn_3-3.0.0b1-cp39-abi3-linux_x86_64.whl
  ```

  This method is documented on the dataset page along with compatibility notes (e.g., PyTorch 2.6.0/2.7.0/2.7.1 with CUDA 12.6 or 12.8 and Python 3.9+) https://huggingface.co/datasets/malaysia-ai/Flash-Attention3-wheel#:~:text=Build%20using%20H100.

- **GitHub Pages repository** `windreamer/flash-attention3-wheels`.  According to a comment on the pip-wheel issue, you can install with:

  ```bash
  pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch280 \
      --extra-index-url https://download.pytorch.org/whl/cu128
  ```

  (This repository also hosts weekly-built wheels and provides installation instructions.)

**Caution:** these wheels are community built and not officially maintained by the FlashAttention authors.  They may not include the latest optimizations or may be incompatible with future versions.  If reliability is critical, build FlashAttention 3 from source using the steps above.

## Troubleshooting and tips

- **Ensure dependencies match** – mismatched CUDA/PyTorch versions cause compilation failures.  Align your PyTorch binary with the CUDA toolkit version you plan to use.
- **Memory consumption** – the build process is memory intensive; on machines with less than 96 GB of RAM, restrict `MAX_JOBS` to avoid out‑of‑memory errors https://github.com/Dao-AILab/flash-attention#:~:text=If%20your%20machine%20has%20less,MAX_JOBS.
- **ninja issues** – if `ninja --version` returns a non-zero exit code, reinstall it (`pip uninstall -y ninja && pip install ninja`) https://github.com/Dao-AILab/flash-attention#:~:text=,core%20machine%20using%20CUDA.
- **Testing** – always run the provided tests after installation to verify correctness https://github.com/Dao-AILab/flash-attention#:~:text=To%20run%20the%20test%3Ahttps://servicedesk.surf.nl/wiki/spaces/WIKI/pages/174457385/Installing%20flash-attention%203%20for%20hopper#:~:text=git%20clone%20https%3A%2F%2Fgithub.com%2FDao,py%20install.
- **Feature flags** – disable unused features to shorten build time and reduce binary size using the environment variables described above https://github.com/Dao-AILab/flash-attention/issues/1486#:~:text=Member.
- **Compilation on non-Hopper GPUs** – you can compile on any machine that has the CUDA toolkit installed, but the resulting library will only run on Hopper-class GPUs https://github.com/Dao-AILab/flash-attention/issues/1486#:~:text=I%20have%20one%20more%20question%3A,use%20later%20on%20H100%20cluster.

## Summary

FlashAttention 3 brings significant speed improvements for attention kernels on NVIDIA Hopper GPUs.  Because it is still a beta and not yet integrated into major AI frameworks, installation requires compiling from source.  After ensuring you have a Hopper GPU, CUDA 12.3+, and PyTorch 2.2+, clone the repository, enter the `hopper` directory, and run `python setup.py install`.  Use the `MAX_JOBS` and feature-disabling environment variables to tailor build time and resource usage.  Once installed, test the kernels and call them via the `flash_attn_interface` module.  Community-built wheels exist for convenience, but building from source provides the most control and ensures compatibility.
