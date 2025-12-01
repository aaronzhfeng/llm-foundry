# Scaling Laws and Compute Constant **C** for Large Language Models

## Introduction

Scaling laws describe how the performance of large language models (LLMs) varies as the **model size (N)**, **dataset size (D)** and **compute budget (C)** change.  Originally observed in OpenAI’s 2020 paper *“Scaling Laws for Neural Language Models”*, these relationships help researchers plan training budgets and make informed trade‑offs between model parameters and data.  More recently, DeepMind’s *“Training Compute‑Optimal Large Language Models”* (also known as **Chinchilla**) refined these insights with a parametric loss model and explicit formulas for the optimum allocation of compute between N and D.

## Compute **C** and the **6ND** law

Both Kaplan et al. and Hoffmann et al. base their scaling analyses on a simple estimate of training compute.  For a decoder‑only Transformer trained with back‑propagation, the total floating‑point operations per **token** can be approximated as \(6N\).  The constant 6 arises because the forward pass accounts for roughly \(2N\) FLOPs and the backward pass costs about \(4N\) FLOPs:contentReference[oaicite:0]{index=0}.  Multiplying by the number of tokens processed (D) gives the **compute law**:

\[
C \approx 6\,N\,D,
\]

where **C** is measured in floating‑point operations (FLOPs).  This relation underpins many scaling laws: fixing compute implies a trade‑off between model size and data size, while measuring compute utilisation (MFU) requires comparing actual FLOPs/s against the peak values:contentReference[oaicite:1]{index=1}.

## Kaplan (OpenAI) scaling law

OpenAI’s 2020 study fitted a power‑law relationship between test loss \(L\) and model size \(N\) or dataset size \(D\).  Empirically, they found that loss decayed as a power of either variable and proposed a **combined scaling law**:

\[
L(N, D) = \left[\left(\frac{N_c}{N}\right)^{\alpha_N/\alpha_D} + \frac{D_c}{D}\right]^{\alpha_D},
\]

where \(N_c, D_c\) and exponents \(\alpha_N, \alpha_D\) are constants fitted to data.  Using this model they showed that, for a fixed compute budget \(C\), the **optimal model size** scales as \(N_{\mathrm{opt}} \propto C^{0.73}\):contentReference[oaicite:2]{index=2}.  This means larger compute budgets should be allocated more to increasing the model parameters than to training on more data.

### Key observations

- **Power‑law decay:** Test loss decreases roughly as \(N^{-\alpha_N}\) or \(D^{-\alpha_D}\) until saturation.  For GPT‑style models, \(\alpha_N \approx 0.076\) and \(\alpha_D \approx 0.095\) (values vary by dataset).
- **Combined law:** Combining \(N\) and \(D\) yields smoother fits across regimes; the exponents capture the relative importance of model size versus data:contentReference[oaicite:3]{index=3}.
- **Optimal scaling:** The best way to spend compute is to scale \(N\) slightly faster than \(D\) (exponent ~0.72).

## Chinchilla (DeepMind) scaling law

DeepMind’s 2022 paper argued that the 2020 scaling law under‑utilised data; they proposed a **parametric loss model**:

\[
\hat{L}(N, D) = E + A \cdot N^{-\alpha} + B \cdot D^{-\beta},
\]

where \(E\) is the irreducible loss, and \(A,B,\alpha,\beta\) are fitted coefficients:contentReference[oaicite:4]{index=4}.  Subject to the compute constraint \(C \approx 6ND\), the optimal allocations of compute between N and D can be solved analytically:

\[
N_{\mathrm{opt}} = G \left(\frac{C}{6}\right)^{\frac{\beta}{\alpha+\beta}}, \quad D_{\mathrm{opt}} = \frac{1}{G} \left(\frac{C}{6}\right)^{\frac{\alpha}{\alpha+\beta}},
\]

where \(G = \left( \frac{\alpha A}{\beta B} \right)^{\frac{1}{\alpha+\beta}}\):contentReference[oaicite:5]{index=5}.  The exponents \(\frac{\beta}{\alpha+\beta}\) and \(\frac{\alpha}{\alpha+\beta}\) often turn out to be ~0.5, meaning Chinchilla recommends increasing model and data sizes **in tandem**.  The paper demonstrated that GPT‑3‑sized models were dramatically under‑trained on data and that using the Chinchilla allocation could achieve the same performance with far fewer parameters.

### Key observations

- **Parametric loss:** Loss decomposes into contributions from finite model size (\(A N^{-\alpha}\)) and finite dataset size (\(B D^{-\beta}\)).  The irreducible loss \(E\) sets a lower bound:contentReference[oaicite:6]{index=6}.
- **Compute‑optimal frontier:** By enforcing \(C = 6ND\), one can derive explicit formulas for the optimal N and D given any compute budget:contentReference[oaicite:7]{index=7}.
- **Implications:** For typical language modelling experiments, \(\alpha\) and \(\beta\) are roughly equal, leading to nearly symmetrical scaling.  As a result, many existing large models (e.g., GPT‑3) are **under‑trained** relative to their size.

## Extensions and recent developments

Research since 2022 has explored new axes of scaling:

- **Learning‑rate schedules:** The NeurIPS 2024 work “Scaling Laws and Compute‑Optimal Training Beyond Fixed Training Durations” investigated constant versus cosine schedules and found that optimal compute allocation depends on the schedule.  Their codebase includes FLOP calculators and training scripts.
- **Vocabulary size:** The 2024 paper “Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies” argued that vocabulary parameters should scale more slowly than non‑vocabulary parameters.  Their repository provides three methods to choose the optimal vocabulary size as a function of compute budget:contentReference[oaicite:8]{index=8}.
- **Hyper‑parameter scaling law (Step Law):** The 2025 “Predictable Scale: Part I” paper proposes power‑law relationships for optimal learning rate and batch size.  Their repository offers a tool to predict these hyper‑parameters given model size and data:contentReference[oaicite:9]{index=9}.

## Role of **C** in training efficiency and MFU

The constant **C** encapsulates the *total* floating‑point budget for training.  In MFU (Model FLOPs Utilization) calculations, one measures the **actual** FLOPs per second achieved (throughput × FLOPs per token) and divides it by the *theoretical* peak FLOPs of the hardware.  The theoretical FLOP requirement comes from the same compute law \(C \approx 6ND\): if a training loop processes \(D\) tokens on a model of size \(N\), then the expected FLOP consumption is \(6ND\).  Comparing this with the GPU’s rated capability yields a utilization score.  To improve MFU, engineers can reduce overheads, ensure high batch sizes, or use efficient kernels.

## Conclusion

The concept of **scaling laws** has evolved from simple power‑law fits (OpenAI’s 2020 work) to analytic, compute‑constrained allocations (DeepMind’s Chinchilla).  Central to all these formulations is the **compute constant** \(C\), linking model parameters, data size and FLOP budgets via \(C \approx 6ND\):contentReference[oaicite:10]{index=10}.  Understanding how **C** governs training efficiency enables practitioners to design models that make optimal use of available compute, whether measured by loss minimization or hardware utilization.

