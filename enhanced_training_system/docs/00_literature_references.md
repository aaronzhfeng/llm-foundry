MFU (Model FLOPs Utilization)

nanoGPT (MFU implementation) – the GPT.estimate_mfu function lives in model.py, and it’s used in bench.py:

nanoGPT/model.py (estimate_mfu): https://github.com/karpathy/nanoGPT/blob/master/model.py

nanoGPT/bench.py (uses estimate_mfu): https://github.com/karpathy/nanoGPT/blob/master/bench.py

scaling_laws (uses MFU from nanoGPT) – the repository replicates Kaplan et al.’s scaling-law experiments; MFU is estimated via NanoGPT:

Repository homepage: https://github.com/shehper/scaling_laws

Scaling laws and compute‑optimal training

Chinchilla toolkit – provides functions to fit and use the Chinchilla scaling law:

Repository homepage: https://github.com/kyo-takano/chinchilla

README.md (formulation overview): https://github.com/kyo-takano/chinchilla/blob/master/README.md

core.py (adjust_D_to_N, allocate_compute, predict_loss): https://github.com/kyo-takano/chinchilla/blob/master/chinchilla/core.py

scaling_laws (Kaplan law replication):

Repository homepage: https://github.com/shehper/scaling_laws

schedules-and-scaling (NeurIPS 2024, scaling with learning‑rate schedules):

Repository homepage: https://github.com/epfml/schedules-and-scaling

scaling-with-vocab (NeurIPS 2024, scaling vocabulary size):

README.md (describes the three approaches): https://github.com/sail-sg/scaling-with-vocab/blob/main/README.md

resolving-scaling-law-discrepancies (2024 analysis):

README.md: https://github.com/formll/resolving-scaling-law-discrepancies/blob/main/README.md

steplaw (Predictable Scale: Part I – hyperparameter scaling law, 2025):

README.md: https://github.com/step-law/steplaw/blob/main/README.md

These links point directly to the relevant files or repository overviews discussed previously.