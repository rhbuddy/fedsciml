# Complete Study Notes — Federated Scientific Machine Learning (FedSciML)

**Paper:** *Federated scientific machine learning for approximating functions and solving differential equations with data heterogeneity*
**Authors:** Handi Zhang, Langchen Liu, Lu Lu (University of Pennsylvania & Yale University)
**Source:** arXiv:2410.13141v1, 17 Oct 2024

> These notes are written to be fully self-contained — you should be able to understand the whole paper
> from these notes alone, without opening the PDF. Each section is one page of the paper.
> Concepts are explained in plain language first, then formulas and details.

---

## Page 1 — Title & Abstract (What this paper is about)

### The Big Idea (read this first)
- **Scientific machine learning (SciML)** uses neural networks to solve problems governed by **partial differential equations (PDEs)** — e.g., heat flow, fluid motion, wave propagation.
- In real life, the data needed for such problems is often **spread across different organizations** (hospitals, companies, countries) and is **private/confidential** — it cannot be gathered into one central database.
- **Federated learning (FL)** is a way to train one shared model across many parties **without anyone sharing their raw data**. Only model updates (weights) are exchanged, so privacy is preserved.
- This paper **combines FL with SciML** — a new field the authors call **FedSciML** — and studies how well it works when the data on different parties is different (non-iid).

### What the paper contributes (2 new models)
1. **FedPINN** — Federated **Physics-Informed Neural Networks**: neural networks that solve PDEs (like PINNs) but trained collaboratively across clients.
2. **FedDeepONet** — Federated **Deep Operator Networks**: networks that learn mappings between functions (operators, e.g., "given initial condition → solution at all times") trained collaboratively.

### What the paper contributes (tools & theory)
3. **Data generation methods** — ways to split data among clients to simulate controlled amounts of "non-iid-ness" (data that differs between clients).
4. **1-Wasserstein distance (W1)** — a number that measures *how different* two clients' data distributions are. Smaller W1 = more similar = more iid.
5. **Weight divergence** — a measure of how far the federated model's weights drift from a centrally-trained model, plus a theoretical **upper bound** on this drift.

### Experiments & headline result
- **10 experiments total:** 2 function-approximation + 5 PDE problems (FedPINN) + 3 operator-learning problems (FedDeepONet).
- **Main result:** federated models **beat models trained on only one client's data**, and get close to the accuracy of a centralized model that trains on *all* the data.

**Keywords:** federated learning; scientific machine learning; function approximation; physics-informed neural networks; operator learning; data heterogeneity

---

## Page 2 — Background: PINNs and DeepONets

### Physics-Informed Neural Networks (PINNs)
- Traditional PDE solvers (finite difference, finite element) are accurate but **computationally expensive**.
- **PINNs** replace them with a neural network that predicts the solution u(x).
- Trick: the **PDE equation itself is added into the network's loss (error) function**, so the network learns to satisfy the physics, not just match data.
- Works for both **forward problems** (given equation → find solution) and **inverse problems** (given partial data → find hidden equation/parameters).
- Known improvements to vanilla PINN (already in literature):
  - meta-learning to find better loss functions
  - gradient-enhanced PINNs (add derivative info of the residual to the loss)
  - self-adaptive loss weights (automatically balance PDE term vs boundary term)
  - residual-based adaptive sampling, multiscale Fourier features
  - libraries/toolboxes (e.g., **DeepXDE**)

### DeepONet / Operator Learning
- Instead of learning a *function*, learn a **mapping between function spaces** (an *operator*). Example: given the source term v(x) of an equation, output the solution u(x,t).
- **Key advantage:** after training, a **single forward pass** gives the solution for a *new* input function — even extrapolation works.
- Applications in the literature: bubble dynamics, brittle fracture, solar-thermal forecasting, electroconvection, fast multiscale modeling.
- Variants: POD-DeepONet, physics-informed DeepONet, multifidelity DeepONet, MIONet, Fourier-DeepONet, UQ-DeepONet, and D2NO (distributed/hybrid-input DeepONet).

### Why federated learning is needed
- Example: **atmospheric science** — data comes from satellites, radars, weather stations, aircraft, ships, buoys — all owned by different agencies, stored in different forms.
- Example: **subsurface energy** — oil companies' data is proprietary, protected by confidentiality clauses and regulation.
- Existing SciML models cannot handle decentralized data → need **federated learning**, introduced by **McMahan et al. 2017**.

---

## Page 3 — Background: Federated Learning + This Paper's Contributions

### What federated learning is
- Many clients (devices/companies) train one **global model** under a central **server**, while **raw data never leaves each client**.
- Each round: server sends current model → clients train locally on their private data → clients send back only model updates → server aggregates them into a new global model.
- Benefits: privacy preserved, low communication cost, robustness.
- Real uses: healthcare, finance, IoT.
- Privacy mechanisms: differential privacy, random sub-sampling, noise injection.
- Communication-efficient algorithms: **FedAvg**, FedAvg+momentum, adaptive optimizers (**FedAdagrad, FedAdam, FedYogi**).

### What THIS paper does (full contribution list)
1. Proposes **FedAvg-Adam** — federated averaging where each client uses **Adam** optimizer locally.
2. Builds **FedPINN** and **FedDeepONet**.
3. Uses **W1 distance** to measure data heterogeneity in function-approximation & PINN tasks.
4. Designs **data generation methods**: 1D & 2D domain partitioning for function approximation/PINNs; **different functional spaces** (Chebyshev polynomials) for operator learning — all to control the non-iid level.
5. Proves a **theoretical upper bound** on weight divergence.
6. Experiments show: federated beats extrapolation (local-only) models; approaches centralized model as W1 → small; weight divergence correlates with non-iid; **FedDeepONet is insensitive to communication frequency**.

### Paper organization (roadmap)
- **Sec. 2 (Methods):** FedSciML, FedAvg-Adam algorithm, data generation, W1 distance.
- **Sec. 3 (Theory):** weight divergence definition + growth-bound theorem.
- **Sec. 4 (Experiments):** function approximation — 1D & 2D, two and multi clients.
- **Sec. 5 (Experiments):** FedPINN — Poisson, Helmholtz, Allen-Cahn, inverse Navier-Stokes, inverse diffusion-reaction.
- **Sec. 6 (Experiments):** FedDeepONet — antiderivative, diffusion-reaction, Burgers + communication-frequency test.
- **Sec. 7:** conclusions & future work.

---

## Page 4 — Methods Overview (Figure 1: FedSciML workflow)

- **Section 2 plan:** Sec. 2.1 = FedSciML + non-iid setting + federated averaging; Sec. 2.2 = W1 distance + data partitioning methods.
- **Figure 1 — the FedSciML workflow (described fully):**
  - **Top:** the full training dataset is split and distributed across **K clients**. Each client holds only its own private slice.
  - **Bottom:** each client owns a model + its dataset. The models are trained **collaboratively** via two steps repeated many times:
    1. **Aggregation (local → server):** each client sends its updated model weights to the server; the server averages them.
    2. **Broadcast (server → local):** the server sends the new global model back to all clients.
  - No raw data is ever exchanged — only model weights.

---

## Page 5 — FedSciML Setup, Problem Formulation & Types of Non-IID

### FedSciML definition
- Applying FL to SciML: learn a **function, a PDE solution, or an operator** (possibly with data heterogeneity) from multiple local datasets, **without direct communication between clients**.
- Clients only interact indirectly via aggregation + broadcast; the final server model generalizes well while preserving local privacy.

### Optimization problem (formal)
- Goal: minimize the **federated server loss**:
  - `L(θ) = Σ_{k=1..K} (N_k / N) · L_k(θ)`
- where the **k-th client's loss** is:
  - `L_k(θ) = (1 / N_k) Σ_{dᵢ ∈ D_k} ℓ(θ; dᵢ)`
- Notation:
  - K = number of clients
  - D_k = dataset of client k; N_k = |D_k| (its size)
  - N = total data across all clients = Σ_k N_k
  - ℓ(θ; dᵢ) = pointwise loss of the model (weights θ) on data point dᵢ
  - In FedSciML, ℓ is usually the **squared error (MSE)** — so each client's loss is just its mean squared error.
- Intuition: each client's contribution is weighted by how much data it has.

### Types of non-iid data (Table 1)
"Non-iid" = non-identical distributions across clients. Pᵢ(x) = feature distribution of client i, Pᵢ(y) = label distribution.

| Type | Feature dist. | Conditional | Meaning |
|---|---|---|---|
| **Covariance shift** | Pᵢ(x) ≠ Pⱼ(x) | Pᵢ(y\|x) = Pⱼ(y\|x) | Same label-given-input rule, but inputs differ between clients |
| **Prior probability shift** | Pᵢ(y) ≠ Pⱼ(y) | Pᵢ(x\|y) = Pⱼ(x\|y) | Different label frequencies, same input-given-label rule |
| **Concept drift** | Pᵢ(y) = Pⱼ(y) | Pᵢ(x\|y) ≠ Pⱼ(x\|y) | Same labels, but the data generating each label differs |
| **Concept shift** | Pᵢ(x) = Pⱼ(x) | Pᵢ(y\|x) ≠ Pⱼ(y\|x) | Same inputs, but different label rules |

- Besides distribution shifts, **quantity skew / dataset imbalance** also causes non-iid.
- **This paper focuses on covariance shift** (the first row) — clients have data in different regions of the input space.

---

## Page 6 — FedAvg Algorithm & the FedOPT Framework

### FedAvg setup
- C = fraction of clients selected each round (set to **100%** here — all clients participate → exactly one broadcast + one aggregation per round).
- η = learning rate; E = number of local epochs (gradient steps per client per round).

### FedAvg — the 3 steps at global epoch l
1. **(a) Broadcast:** each client's model is initialized to the current server model:
   - `θ_k^{l,0} = θ^l` for k = 1, …, K
2. **(b) Local training (E steps):** each client computes local gradients of its own loss:
   - `g_k^i = ∇L_k(θ_k^{l,i−1})` for i = 1, …, E
3. **(c) Global aggregation:** the server combines all local updates, weighted by data size:
   - `θ^{l+1} ← θ^l − η Σ_{k=1..K} (N_k/N) Σ_{i=1..E} g_k^i` …**(Eq. 1)**
- Notation: θ_k^{l,i} = client k's model at global epoch l, local step i; θ^l = global model at epoch l.

### FedOPT — the generalized framework
- The general method replaces "gradient descent" with arbitrary **Client-OPT** and **Server-OPT** routines:
  - Local update: `θ_k^{l,i+1} = Client-OPT(θ_k^{l,i}, g_k^{i+1}, η, l)`
  - Total local change: `Δ_k^l = θ_k^{l,E} − θ_k^{l,0}`
  - Aggregation of changes: `Δ^{l+1} = Σ_k (N_k/N) Δ_k^l`, then `θ^{l+1} = Server-OPT(θ^l, −Δ^{l+1}, η, l)` …**(Eq. 2)**
- **Vanilla FedAvg (Eq. 1) is a special case of FedOPT** where both Client-OPT and Server-OPT are plain gradient descent.

---

## Page 7 — FedAvg-Adam (Algorithm 1)

### Why Adam
- Empirical observation in this paper: **Adam optimizer performs better than SGD** for these problems.

### FedAvg-Adam definition
- Take the FedOPT framework and set:
  - **Client-OPT = Adam** (each client updates locally with Adam)
  - **Server-OPT = gradient descent**
- In practice the authors **average the local models directly** (not the "changes" Δ) to update the server.

### Algorithm 1 (FedAvg-Adam) — step by step
```
Input: initial model θ0, K clients, local epochs E, learning rate η
For each global round l = 0, 1, 2, … :
    For each client k = 1..K in parallel:
        θ_k^{l,0} ← θ^l                     # copy server model
        For i = 1..E:
            compute gradient g_k(θ_k^{l,i−1})
            θ_k^{l,i} ← Adam(g_k^i, η)      # Adam local update
        Δ_k^l ← θ_k^{l,E} − θ_k^{l,0}       # local change
    # Global aggregation
    θ^{l+1} ← θ^l + Σ_{k=1..K} (N_k/N) Δ_k^l
    #  (equivalently: θ^{l+1} ← Σ_{k=1..K} (N_k/N) θ_k^{l,E})
```

### Data generation overview (Sec. 2.2)
- Before experiments, define how to measure & create non-iid:
  - **Measure** heterogeneity with **Wasserstein distance** (regression tasks).
  - **Create** it by: partitioning the domain into subdomains (function approx & PDE, Sec. 2.2.2–2.2.3), or sampling functions from different functional spaces (operator learning, Sec. 2.2.4).

---

## Page 8 — 1-Wasserstein Distance & 1D Data Generation

### 1-Wasserstein distance (the heterogeneity metric)
- Given two probability distributions µ and ν on a metric space (M, d), the **p-th Wasserstein distance** is:
  - `W_p(µ, ν) = [ inf_{γ ∈ Γ(µ,ν)} ∫ d(x,y)^p dγ(x,y) ]^{1/p}`
  - Γ(µ, ν) = the set of all **couplings** of µ and ν — joint distributions whose marginals are µ and ν. Intuition: all the ways to "move mass" from µ to ν.
- When **p = 1**, W1 is the **Earth Mover's Distance (EMD)**: the minimum total "work" to transform one distribution into the other.
  - `W1(µ, ν) = inf_{γ} ∫ |x − y| dγ(x, y)` …**(Eq. 3)**
- **How it's computed in this paper:** with the **Python Optimal Transport (POT)** package, on discrete samples:
  - `W1(µ_{n1}, ν_{n2}) = min_γ ⟨γ, M⟩_F`  subject to  `γ1 = a`, `γ^T 1 = b`, `γ ≥ 0`
  - M = n1×n2 Euclidean distance matrix between sample points (m_ij = distance between point i of µ and point j of ν)
  - a, b = sample weights (mass) of each distribution; γ = transport plan
  - The optimization finds the cheapest way to ship mass from µ to ν.

### Mean pairwise W1 for many clients (K ≥ 3)
- Direct W1 is between two distributions. For K clients, average over all pairs:
  - `W1 = (1 / ((K−1)(K−2))) Σ_{1 ≤ i < j ≤ K} W1(µ_i, µ_j)` …**(Eq. 4)**

### 1D data generation (function approximation & PDEs)
- Have N data points {x₁,…,x_N} on an interval I ⊂ ℝ.
- Divide the interval into **n subdomains per client**; clients **alternately** receive subdomains (Fig. 2A). So each client gets data spread over several disjoint chunks.
- Remaining points (if N not divisible by n·K) are assigned sequentially to clients.
- Local datasets: {Ω_k}, k=1..K, sizes N_k.
- **Effect:** more subdomains n ⇒ the two clients' data overlaps more ⇒ **W1 decreases ⇒ data becomes more iid**.

---

## Page 9 — Figure 2 (Data Generation Visualized)

**Figure 2 shows (described):**
- **(A)** 1D generation, two clients. Left → right: the iid level increases (more, smaller subdomains interleaved between clients). Last column: plot of **W1 vs number of subdomains** — W1 drops as n increases.
- **(B)** 2D **x-partition**, two clients — vertical strips alternate between clients.
- **(C)** 2D **xy-partition**, two clients — checkerboard of square blocks alternate between clients.
- **(D)** 2D **xy-partition**, three clients. Last column: **mean pairwise W1** for 3, 4, and 5 clients vs number of subdomains.
- **(E)** Operator-learning generation: functions sampled from **different functional spaces** (first → third column = different Chebyshev subsets).

---

## Page 10 — 2D Data Generation + Operator-Learning Data Generation

### 2D generation methods (two options)
- **x-partition:** split only the **x** coordinate into n subintervals, y unchanged → vertical strips (like 1D, just on the first axis). Preferred for **time-dependent problems that are 1D in space**, so each client can be identified purely by spatial domain.
- **xy-partition:** split **both** x and y into n subintervals → n² square subdomains, assigned alternately to clients. Used for **2D time-independent problems**.
  - Data is sampled over the whole domain using **Hammersley sampling** (a low-discrepancy / evenly-spread sampling sequence).
- **Finding:** the W1 difference between xy- and x-partition becomes negligible as n grows → the method choice just follows the problem type.

### Multi-client extension
- For 3, 4, 5 clients use the **mean pairwise W1 (Eq. 4)**. Same trend as 2-client case, but **non-iid-ness becomes more pronounced with more clients**.

### Data generation for operator learning (Sec. 2.2.4)
- Goal: control non-iid for learning operators, where training data = **functions**, not points.
- Setup: N total training functions, K clients → each client gets N_k = N/K functions, generated from **different functional spaces**.
- Generating space = **Chebyshev polynomials**:
  - `p(x) = Σ_{i=0..M} a_i T_i(x)`,  a_i ∈ [−1,1],  T_i = Chebyshev polynomial of first kind, **M = 10** (keeps computational cost low).
- **Non-iid control:** vary the number of **nonzero basis functions** n each client uses:
  - **Two clients:** Client 1 = **forward** (first n terms): `Σ_{i=0..n} a_i T_i(x)`; Client 2 = **inverse** (last n terms): `Σ_{i=M−n−1..M−1} a_i T_i(x)`.
  - n ∈ [1,10].
- **Rationale:** larger n ⇒ each client's functions use more of the full Chebyshev basis ⇒ closer to the full space ⇒ **more iid**.

---

## Page 11 — Three-Client Chebyshev + Weight Divergence Definition

### Three clients (Chebyshev)
- Client 1 (forward): `Σ_{i=0..n} a_i T_i(x)` — first n terms
- Client 2 (middle): `Σ_{i=⌊(M−n)/2⌋..⌊(M+n)/2⌋} a_i T_i(x)` — middle n terms
- Client 3 (inverse): `Σ_{i=M−n−1..M−1} a_i T_i(x)` — last n terms

### Section 3 (Theory) overview
- Goal: quantify the gap between **federated** and **centralized** learning.
- Sec. 3.1: define **weight divergence**. Sec. 3.2: prove its **growth bound**.

### Weight divergence — the definition
- First introduced in Ref. [44] (Zhao et al.):
  - `E_WD := (θ_NN1 − θ_NN0) / ||θ_NN0||`  → this is the **relative weight divergence** (normalized).
- This paper uses the **absolute** version (they still call it "weight divergence"):
  - `E_WD := θ_NN1 − θ_NN0`
- Here NN1 = the **FedAvg** model, NN0 = the **centralized SGD** model:
  - `E_WD := θ_FedAvg − θ_SGD`
- Notation **E_WD^{l,E}**: weight divergence between the federated model at (l-th global epoch, E-th local epoch) and the centralized model at the (l × E)-th epoch — i.e., both models have seen the same total number of gradient steps → **fair comparison**.

---

## Page 12 — Weight Divergence Visualized & Why It Matters

### Figure 3 (described)
- Two clients, two communication rounds, gradient descent.
- Black dashed line = **centralized model** trajectory.
- Blue & orange = the **two clients' local** trajectories.
- Green = the **FedAvg global model** trajectory.
- **Key observation:** the distance between the centralized model and FedAvg **grows as more communication rounds pass**.

### Context from prior work
- FL convergence studied in [44, 45, 46]; PINN convergence in [47]; DeepONet convergence in [48].
- **Weight divergence [44]** is a key factor that *degrades* federated model accuracy. It remains **bounded** after finitely many synchronization rounds.
- **FedProx [45]** adds a proximal/regularization term that penalizes large weight divergence — shown effective across tasks.

### This paper's contribution (preview of Theorem)
- Under certain assumptions, after E rounds of local updates the weight divergence is **linearly bounded by the local epoch count E**.
- **Notably, the bound is independent of the data heterogeneity** across clients (surprising/useful result).

---

## Page 13 — Growth Bound of Weight Divergence (Theorem 3.1)

### Setup for the theorem
- Problem: minimize L(x) with dataset D, using **centralized SGD for E epochs** vs **FedAvg with E local epochs per global sync**, same learning rate η, same initialization, full participation (K clients), D = ∪_k D_k.

### Assumption 1 (bounded gradients)
- There exists M > 0 such that for all weights θ and all data points d ∈ D:
  - `||∇ℓ(θ, d)|| ≤ M`
- (A standard, reasonable assumption — gradients don't blow up during training.)

### Theorem 3.1
- After **one** global aggregation with E local epochs:
  - `E_WD^{1,E} ≤ 2ηME`
- Generalizing to the **l-th global epoch** (centralized model trained l×E epochs):
  - `E_WD^{l,E} ≤ 2ηMEl`
- **Meaning:** the drift between federated and centralized training grows **at most linearly** in the number of local steps, and does **not** depend on how heterogeneous the clients' data is.

### Section 4 begins: Function Approximation setup
- All experiments implemented with **DeepXDE** (Lu et al.); code at github.com/lu-group/federated-sciml.
- **Two baselines (to bracket the federated error):**
  - **Centralized NN:** trained on the *whole* dataset → the **lower error bound** reference (best possible).
  - **Extrapolation models:** each client trained *only* on its local data, **no communication** → the **upper error bound** reference (worst case).
- **Expectation:** federated error lies between the two, and all three converge as data becomes more iid.

### Example 1: Gramacy & Lee (2012) function (1D)
- Original: `f(x) = (x−1)⁴ + sin(10πx)/(2x)`, on x ∈ [0.5, 2.5]
- Normalized to [−1, 1]: `f(x) = (x+0.5)⁴ − sin(10πx)/(2x+3)`

---

## Page 14 — 1D Function Approximation Results

### Setup
- 200 uniformly-spaced sample points on [−1,1]. Number of partitions per client varies **1 → 50**. Uses the 1D two-client generation (Fig. 2A).

### Results
- **W1 ↓ ⇒ L2 relative error ↓** (Fig. 4A). So the heterogeneity metric directly predicts accuracy.
- **Heterogeneity ↔ weight divergence:** data heterogeneity is **positively correlated** with the weight divergence between the two clients' networks (Fig. 4B).

### Figure 4 (described)
- (A) L2 relative error vs number of subdomains and vs W1.
- (B) weight divergence of hidden layers vs subdomains/W1.
- (C) Example approximations: **left = 2 subdomains** (high heterogeneity), **right = 100 subdomains** (low heterogeneity). Shaded areas = which data belongs to which client.

---

## Page 15 — 2D Function Approximation (Schaffer) + Multi-Client

### Why extrapolation baselines fail
- A client trained only on its own half of the domain approximates its half well but **fails on the other half** (no data there).
- The **federated model wins because aggregation + broadcast lets every client "see" the whole domain** through the shared model.
- When client distributions are nearly identical (iid), federated ≈ extrapolation ≈ centralized.

### Example 2: Schaffer function (2D)
- `f(x,y) = 0.5 + (sin²(x²−y²) − 0.5) / [1 + 0.001(x²+y²)]²`,  x,y ∈ [−1,1]
- The function is symmetric, so to create heterogeneity they sample only the **asymmetric subdomain [0,1]×[0,1]** (upper-right corner) (Fig. 5C).
- x-partition method, number of partitions **1 → 25**.

### Results
- More partitions ⇒ W1 ↓ ⇒ L2 error drops from ~10⁻¹ to below 10⁻² (Fig. 5A). Same overall trend as the 1D case.
- Positive correlation between heterogeneity and weight divergence (Fig. 5B).
- Federated beats both extrapolation baselines under high AND low heterogeneity (Figs. 5D/E).

### Multi-client experiments (3, 4, 5 clients)
- Use **mean pairwise W1 (Eq. 4)**.
- Same trend holds (larger W1 ⇒ larger L2 error); the gap between extrapolation and federated **widens with more clients** (Figs. 6A–C).
- All models' errors grow as clients increase; **2-client federated model has the smallest error vs the 5-client model** (Fig. 6D).
- Note: W1 (2-client) and mean pairwise W1 (multi-client) have different ranges, so the four error lines are not aligned in Fig. 6D.

---

## Page 16 — Figure 5 (Schaffer, two clients)

**Figure 5 (described):**
- (A) L2 relative error vs # subdomains & W1.
- (B) weight divergence of hidden layers vs # subdomains & W1.
- (C) ground truth Schaffer function.
- (D) most non-iid scenario: federated model vs two extrapolation baselines.
- (E) most iid scenario: same comparison. Black dashed line = x-partition boundaries for the two clients.

---

## Page 17 — Figure 6 (Schaffer, multiple clients)

**Figure 6 (described):**
- (A) three, (B) four, (C) five clients: L2 relative error vs # subdomains and mean pairwise W1.
- (D) comparison of the centralized baseline vs federated models with 2, 3, 4, 5 clients.

---

## Page 18 — PINNs: Introduction & FedPINN Setup + 1D Poisson

### What PINNs are (review)
- A class of deep learning that **integrates data + PDE knowledge**. The PDE is baked into the loss, which gives accuracy, interpretability, and robustness even for extrapolation/generalization.
- For a well-posed PDE on a bounded domain: `F[u](x) = f(x)` on Ω, with BC/IC. We use a neural network û(θ) for the solution.
- **Residual (PDE) loss:** `L_r(θ; x) = ||F[û(θ)](x) − f(x)||²₂` — forces the network to satisfy the equation.
- **Training details:** **hard boundary constraints** enforced for ALL examples in this paper (Ref. [51] method). Default: **tanh** activation, **Adam**, learning rate 0.001. Full hyperparameters in Table 4.

### FedPINN vs classic domain decomposition (important conceptual point)
- Domain decomposition (e.g., cPINN, XPINN) **requires** knowing the partition — each subdomain knows its neighbors, or a central server coordinates them.
- **FedPINN needs NO knowledge of other clients' domains** — domain locations stay **hidden**. Data partition happens implicitly by who owns which data.

### Example: 1D Poisson problem
- PDE: `−Δu = Σ_{i=1..4} i·sin(ix) + 8 sin(8x)`,  x ∈ [0, π]
- Dirichlet BC: `u(0) = 0`, `u(π) = π`
- Exact solution: `u(x) = x + Σ_{i=1..4} sin(ix)/i + sin(8x)/8`

---

## Page 19 — 1D Poisson Results (Figure 7)

### Setup
- 32 uniformly-distributed collocation points in [0, π]; 1D data generation with subdomain counts n ∈ [6, 8, 10, 16, 32].

### Results
- Accuracy of **both** the extrapolation clients and the federated model **degrades as W1 increases** (Fig. 7A).
- Federated **outperforms extrapolation models**; when W1 is small, extrapolation models do about as well as the centralized baseline.
- Weight divergence of **all four hidden layers grows with heterogeneity** (Fig. 7B).
- Federating the client models **significantly improves solving the 1D Poisson system** (Figs. 7C/D).

### Figure 7 (described)
- (A) L2 relative error vs # subdomains & W1.
- (B) weight divergence of layers 1–4 vs W1.
- (C) most non-iid and (D) most iid: federated model vs centralized baseline vs two extrapolation models (predicted solutions).

---

## Page 20 — 2D Helmholtz + Allen-Cahn Setup

### Example: 2D Helmholtz equation
- PDE: `−u_xx − u_yy − k₀²u = f`  on  Ω = [0,1]²
- Dirichlet BC: `u = 0` on ∂Ω; source `f = k₀² sin(k₀x) sin(k₀y)`; **wavenumber k₀ = 2πn with n = 2** (so k₀ = 4π).
- Data: 12 collocation points per wavelength per direction ⇒ 24 points each direction; xy-partition, n ∈ [2, 4, 6, 10, 12, 24].
- **Results:** higher W1 ⇒ larger L2 error (Fig. 8A). More subdomains ⇒ W1 ↓ ⇒ both federated and extrapolation improve. Weight divergence of layers 2 & 3 roughly positively correlated with W1 (layer 2 more sensitive) (Fig. 8B). Extrapolation clients are poor in regions they never saw (e.g., client 1 weak in bottom-right & upper-left); federated model aggregates the full-domain information (Figs. 8D/E).

### Example: Allen-Cahn equation (time-dependent)
- PDE: `∂u/∂t = d·∂²u/∂x² + 5(u − u³)`,  x ∈ [−1,1], t ∈ [0,1]
- Initial condition: `u(x,0) = x² cos(πx)`; boundary: `u(±1,t) = −1`; **d = 0.001**.
- Data: 8000 residual points in the domain, 400 boundary points, 800 initial-condition points.
- **Partition only the spatial domain** (time is fully shared): 1D method with n ∈ [2, 4, 6, 8, 10, 20, 32, 40]. Three-client case: n ∈ [3, 6, 9, 12, 15, 18, 30, 60], so all clients keep **complete temporal information** over their assigned spatial region.
- Training hyperparameters (epochs, architecture, activation, lr) identical across 2-client and 3-client runs.

---

## Page 21 — Figure 8 (Helmholtz)

**Figure 8 (described):**
- (A) L2 relative error vs # subdomains & W1.
- (B) weight divergence of hidden layers vs # subdomains & W1.
- (C) ground truth solution.
- (D) most non-iid and (E) most iid: federated model vs two extrapolation baselines. Black dashed line = xy-partition block assignment for the two clients.

---

## Page 22 — Allen-Cahn Results + Inverse Navier-Stokes Setup

### Allen-Cahn results
- L2 error increases with W1 in **both** 2-client (Fig. 9A) and multi-client (Fig. 9B, mean pairwise W1) settings.
- FedPINN beats the baseline PINN in the extrapolation scenario; both improve as iid level increases.
- Most non-iid (2-client): client 1 owns x ∈ [−1, 0], client 2 owns x ∈ [0, 1] (both over all t).
- Figs. 9F/G: absolute errors of the two extrapolation clients vs the federated server — **FL greatly improves the extrapolation clients regardless of heterogeneity level**.

### Example: inverse problem — 2D Navier-Stokes (flow past a cylinder, Re = 100)
- Equations (incompressible):
  - `u_t + C₁(u u_x + v u_y) = −p_x + C₂(u_xx + u_yy)`
  - `v_t + C₁(u v_x + v v_y) = −p_y + C₂(v_xx + v_yy)`
- u, v = x- and y-velocity components; p = pressure; **C₁, C₂ = unknown parameters to infer** (true values: C₁ = 1, C₂ = 0.01).
- Domain [1, 8] × [−2, 2], time [0, 1]. Data: 200 boundary points, 100 initial points, 700 in-domain mesh points.
- Partition the spatial domain with the **2D x-partition**, n ∈ [2, 4, 6, 8, 10, 20].
- **Special aggregation trick:** they average the *gradients* AND the *inferred parameter values* during aggregation → after a few rounds both clients agree on the parameters.

---

## Page 23 — Figure 9 (Allen-Cahn)

**Figure 9 (described):**
- (A) L2 error, 2-client. (B) L2 error, multi-client. (C) ground truth solution.
- (D/E) 2-client, most non-iid & most iid: federated vs two extrapolation baselines.
- (F/G) multi-client, most non-iid & most iid: federated vs extrapolation baselines.

---

## Page 24 — Inverse Navier-Stokes Results (Figure 10, Table 2)

### Table 2: computational cost (min epochs to reach an error threshold)
- Rows = error thresholds (10%, 5%, 1%, 0.5%). Columns = data heterogeneity (2 / 6 / 10 subdomains, and centralized), for C₁ and C₂.

| Threshold | 2 subdom (C1/C2) | 6 subdom (C1/C2) | 10 subdom (C1/C2) | Centralized (C1/C2) |
|---|---|---|---|---|
| 10% | 18k / 3.4k | 6.8k / 4.5k | 5.5k / 5.3k | 2.9k / 1k |
| 5% | 28k / 3.5k | 10.2k / 5.3k | 8.3k / 5.5k | 3.8k / 2.3k |
| 1% | — / — | 23.6k / 5.7k | 13k / 5.7k | 5.3k / 2.4k |
| 0.5% | — / — | 25.9k / 5.7k | 13.8k / 5.7k | 5.6k / 2.4k |

- Read: **fewer subdomains (more heterogeneous) ⇒ many more epochs needed**; with 2 subdomains the 1% and 0.5% thresholds are **never reached** (—).

### Conclusions
- Less heterogeneity ⇒ **faster convergence**; heterogeneity is **negatively correlated with convergence rate and performance**.
- The standard deviation of inferred C₁, C₂ is **much larger with 2 subdomains (Fig. 10A)** than with 6 or 10 (B, C) → heterogeneity also hurts **stability**; less heterogeneity ⇒ more stable federated model.

### Figure 10 (described)
- Predicted C₁ and C₂ converging to true values over training epochs for (A) 2, (B) 6, (C) 10 subdomains.

---

## Page 25 — Inverse Diffusion-Reaction + Operator Learning Intro

### Example: inverse problem — infer space-dependent reaction rate k(x)
- PDE: `λ u_xx − k(x) u = f`,  x ∈ [0,1]; zero BC u(0)=u(1)=0; **λ = 0.01**; source `f = sin(2πx)`.
- Objective: infer the **reaction-rate function** k(x) from measurements of the concentration u.
- True k: `k(x) = 0.1 + exp(−0.5((x−0.5)/0.15)²)` — a Gaussian bump around x = 0.5.
- Because k is a *function* (not a constant), they train a **second neural network** for k in addition to the one for u.
- Data: 24 observations of u + 10 PDE residual points. Federated: observation points partitioned by the 1D method with n ∈ [2, 4, 6, 8, 12, 24].

### Results
- L2 errors for both k(x) and u(x) **increase with W1** (Figs. 11A/B first column) — same trend as forward problems.
- **No clear trend** of weight divergence for the inferred k(x) (Fig. 11A second column).
- Last two columns: inferred k(x) and learned u(x) under high vs low heterogeneity.

### Section 6: Operator learning with FedDeepONet
- Problems: antiderivative (6.2), diffusion-reaction (6.3), Burgers (6.4), plus a study of local-iteration count (6.5).

---

## Page 26 — DeepONet Architecture + Antiderivative Operator

### DeepONet fundamentals
- Learn an **operator G** mapping function space V → function space U: `G: V → U, v ↦ u`.
  - v defined on domain D ⊂ ℝ^d; u defined on domain D′ ⊂ ℝ^{d′}.
- **Two subnetworks:**
  - **Branch network:** takes function values at m sensor points `[v(x₁), …, v(x_m)]` → outputs `[b₁(v), …, b_p(v)]` (p = # neurons).
  - **Trunk network:** takes a location ξ → outputs `[t₁(ξ), …, t_p(ξ)]`.
  - **Output = inner product of branch & trunk + bias:**
    - `G(v)(ξ) = Σ_{k=1..p} b_k(v)·t_k(ξ) + b₀`
  - Intuition: branch encodes *which* function; trunk encodes *where* to evaluate; their product gives the value.

### Example: Antiderivative operator
- ODE: `du/dx = v(x)`, x ∈ [0,1], u(0) = 0.
- Operator to learn: `G: v(x) ↦ u(x) = ∫₀ˣ v(τ) dτ`.
- Inputs v sampled from Chebyshev space (Sec. 2.2.4). Two clients: forward & inverse directions, n ∈ [1, 10].
- Data sizes: centralized & extrapolation = 200 training functions; federated = 200 total (100 per client). Test = 1000 functions from the full Chebyshev space (M=10).
- Reference solution via **Runge-Kutta (4,5)**.

### Results
- Fewer nonzero terms (more heterogeneity) ⇒ **higher L2 error** (Fig. 12A). FedDeepONet beats both extrapolation baselines and approaches the centralized model as heterogeneity shrinks.

---

## Page 27 — Antiderivative Results (Figure 12)

- **Notable jump:** error drops sharply (10⁻¹ → 5×10⁻²) when n goes from 5 → 6, because at n = 6 the two clients' combined basis functions **cover the full Chebyshev space** — all information is present in the federation.
- Weight divergence (trunk & branch layers) is **less sensitive to non-iid** than in function approximation / PINN cases (Fig. 12B).
- Extrapolation models are unsatisfactory; FedDeepONet leverages both clients' knowledge **without sharing raw data** (Figs. 12C/D).

### Figure 12 (described)
- (A) L2 relative error vs # nonzero Chebyshev terms. (B) weight divergence vs terms. (C) n = 6 (high heterogeneity), (D) n = 10 (low heterogeneity): federated vs two extrapolation baselines.

---

## Page 28 — Diffusion-Reaction & Burgers' Equation (operator learning)

### Example: Diffusion-reaction equation
- PDE: `∂u/∂t = D·∂²u/∂x² + k·u² + v(x)`,  x,t ∈ [0,1]; zero initial & boundary conditions; **k = D = 0.01**.
- Learn `G: v(x) ↦ u(x, t)` (source term → solution).
- Data: 500 training functions (250 per client), 1000 test functions. Reference solution by 2nd-order finite difference on a 101×101 mesh.
- Test inputs v sampled from Chebyshev space with n = 10.
- **Result (Fig. 13):** FedDeepONet outperforms extrapolation baselines. Client 2 (inverse-order terms) trains worse than client 1 → **the quality of a client's local data (how well it spans the learning space) strongly affects the federated model's final performance**.

### Example: Burgers' equation
- PDE: `∂u/∂t + u·∂u/∂x = ν·∂²u/∂x²`,  x,t ∈ [0,1]; periodic BC; initial condition u₀(x) = v(x); **ν = 0.1**.
- Learn `G: v(x) ↦ u(x, t)` (initial condition → solution). v from Chebyshev space, M = 10.
- Data: 200 training functions total, 500 test functions. Same DeepONet architecture as diffusion-reaction.
- **Results (Fig. 14A):** more nonzero terms (more iid) ⇒ lower L2 error. Client 2's extrapolation error drops sharply while client 1's is smoother (unlike diffusion-reaction) — because with **large viscosity the solution becomes very diffusive and decays fast**, weakening the effect of the different input functional spaces.

---

## Page 29 — Figure 13 (Diffusion-Reaction, operator learning)

**Figure 13 (described):**
- Top: one example testing function v(x) and its ground-truth solution u(x, t).
- Bottom (n = 10): absolute prediction errors — first column = FedDeepONet, remaining columns = the two extrapolation baselines.

---

## Page 30 — Figure 14 (Burgers' equation)

**Figure 14 (described):**
- (A) L2 error vs # nonzero Chebyshev terms, two clients. (B) L2 error, three clients.
- (C) federated model comparison: 2 vs 3 clients.
- (D) n = 5, (E) n = 10: ground truth (first column) + absolute errors of FedDeepONet and extrapolation baselines.

---

## Page 31 — Multi-Client Burgers + Communication-Frequency Insensitivity

### Multi-client Burgers
- Error trends match earlier experiments (Fig. 14B); clients 2 & 3 behave similarly and worse than client 1 — their local knowledge gaps hurt extrapolation but **federated learning can mediate this**.
- **3 clients beat 2 clients** (Fig. 14C) for the same # terms per client: the 3-client setup introduces the *middle* nonzero terms (more total information). As terms increase, both 2- and 3-client models converge to the centralized DeepONet.

### Sec. 6.5 — Insensitivity of accuracy to communication frequency
- Test on antiderivative & diffusion-reaction operators with **local iterations E ∈ [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]**, total iterations fixed at **50000**.
  - Larger E ⇒ fewer communication rounds. E = 1 ≈ centralized (sync every step); E = 1000 = very few syncs.
- **Result (Figs. 15A/B):** performance is nearly identical across all E — **even E = 1000**. Federated operator learning is **insensitive to communication frequency**.
- **Practical importance:** one communication round (aggregation + broadcast) is often a large share of the total cost. Because accuracy doesn't depend on sync frequency, FedDeepONet can be trained with **many local iterations and few communications** to get good accuracy cheaply.

---

## Page 32 — Conclusions + Appendices Start

### Conclusions (summary of the paper's message)
1. **First systematic study** of FedSciML for function approximation & PDE solving.
2. Data-generation methods create controllable non-iid; **W1 distance** quantifies it. In **all tasks**, smaller W1 (less non-iid) ⇒ smaller L2 relative error.
3. Federated models **beat local-only (extrapolation) baselines**; as W1 → ~0.005, federated **converges to the centralized baseline** (forward & inverse problems).
4. Introduced **generalized weight divergence** + proved an **upper bound**. Weight divergence tracks error in function approximation & FedPINN, **but not for FedDeepONet** (different setting & non-iid control method).
5. Multi-client (mean pairwise W1): same trend as 2-client, but **errors grow with the number of clients** — conjectured to be due to increasing divergence among clients.

### Future work
- Different aggregation weights for local models; adaptive updating algorithms; more theoretical analysis of FedSciML.

### Acknowledgments
- DOE grants DE-SC0025592, DE-SC0025593; NSF grant DMS-2347833.

### Appendices
- **A:** Abbreviations & notations (Table 3).
- **B:** Network architectures (Table 4).

---

## Page 33 — Appendix A: Abbreviations & Notations (Table 3)

| Symbol | Meaning |
|---|---|
| αᵢ | coefficient of the i-th term in Chebyshev polynomials |
| G | operator to learn |
| Ω | domain of the PDE |
| D | dataset of the federated model |
| D_k | dataset of the k-th client |
| E_WD^{l,E} | weight divergence at l-th global epoch and E-th local epoch |
| {x₁,…,x_m} | scattered sensor locations |
| [v(x₁),…,v(x_m)] | input of the branch network |
| [b₁(v),…,b_p(v)] | output of the branch network |
| ξ | input of the trunk network |
| [t₁(ξ),…,t_p(ξ)] | outputs of the trunk network (p = # neurons) |
| F | governing PDEs and/or physical constraints |
| L | loss of the federated model |
| L_k | loss of the k-th client |
| ℓ | pointwise squared-error loss |
| K | number of clients |
| E | number of local epochs |
| l | number of global epochs |
| N | total number of samples |
| N_k | number of samples for client k |
| n | # subdomains (function approx & PINNs) OR # nonzero Chebyshev terms (operator learning) |

---

## Page 34 — Appendix B: Network Architectures (Table 4) + Proof Start

### Table 4 — architectures & training settings
| Example | Width | Depth | Activation | Local epochs | Global epochs |
|---|---|---|---|---|---|
| Gramacy & Lee (Sec. 4.1) | 3 | 64 | tanh | 5 | 3000 |
| Schaffer (Sec. 4.2) | 3 | 64 | tanh | 5 | 3000 |
| 1D Poisson (Sec. 5.2) | 3 | 20 | tanh | 5 | 1000 |
| 2D Helmholtz (Sec. 5.3) | 3 | 64 | sine | 5 | 2000 |
| Allen-Cahn (Sec. 5.4) | 3 | 64 | sine | 5 | 10000 |
| Navier-Stokes inverse (Sec. 5.5) | 6 | 50 | tanh | 1 | 40000 |
| Diffusion-reaction inverse (Sec. 5.6) | 3 | 20 | tanh | 5 | 20000 |
| Antiderivative (Sec. 6.2) | 2 | 40 | ReLU | 5 | 10000 |
| Diffusion-reaction (Sec. 6.3) | 3 | 100 | ReLU | 5 | 10000 |
| Burgers (Sec. 6.4) | 2 | 64 | ReLU | 5 | 10000 |

- Notes: "Width = 3" means 3 hidden layers of size "Depth". DeepONet trunk & branch share hidden-layer sizes; the first layer's dimension depends on the input dimension and # sensors. Default optimizer = Adam.

### Appendix C — Proof of Theorem 3.1 (start)
- Notation: θⁱ = centralized model at epoch i; θ_kⁱ = client k's model at local epoch i. Same initialization ⇒ θ⁰ = θ_k⁰ for all k.
- **Centralized one-epoch update (Eq. 5):**
  - `θ^{j+1} = θʲ − η∇L(θʲ; D) = θʲ − (η/N) Σ_{dᵢ∈D} ∇ℓ(θʲ; dᵢ)`
- Summing over j = 0..E−1:
  - `θ^E = θ⁰ − (η/N) Σ_{dᵢ∈D} Σ_{j=0..E−1} ∇ℓ(θʲ; dᵢ)`
- **Client update** (same argument per client):
  - `θ_k^E = θ_k⁰ − (η/N_k) Σ_{dᵢ∈D_k} Σ_{j=0..E−1} ∇ℓ(θ_kʲ; dᵢ)`
- Subtract the two and use the triangle inequality (Eq. 6):
  - `||θ^E − θ_k^E|| ≤ η Σ_{j=0..E−1} || (1/N)Σ_{dᵢ∈D} ∇ℓ(θʲ;dᵢ) − (1/N_k)Σ_{dᵢ∈D_k} ∇ℓ(θ_kʲ;dᵢ) ||`

---

## Page 35 — Proof of Theorem 3.1 (Completed)

- **Bound the average gradients (Assumption 1):**
  - `||(1/N) Σ_{dᵢ∈D} ∇ℓ(θʲ; dᵢ)|| ≤ M`  …(Eq. 7)
  - `||(1/N_k) Σ_{dᵢ∈D_k} ∇ℓ(θ_kʲ; dᵢ)|| ≤ M`  …(Eq. 8)
- Substitute into Eq. 6 (each term bounded by M + M = 2M):
  - `||θ^E − θ_k^E|| ≤ η Σ_{j=0..E−1} 2M = 2ηME`
- Relate to weight divergence:
  - `E_WD^{1,E} = || θ^E − Σ_{k=1..K} (N_k/N) θ_k^E || ≤ Σ_{k=1..K} (N_k/N) ||θ^E − θ_k^E|| ≤ 2ηME`
- **Key remark:** the bound does **not** depend on the global epoch, as long as both models share the same initialization. So for any l:
  - `E_WD^{l,E} ≤ l · E_WD^{1,E} ≤ 2ηMEl`  ✓

---

## Pages 36–40 — References (annotated)

### Foundation / methods
- **[1]** Raissi, Perdikaris, Karniadakis — *Physics-informed neural networks* (JCP 2019). The original PINN paper.
- **[2]** Lu et al. — *DeepXDE* (SIAM Review 2021). The library used for all experiments here.
- **[3]** Karniadakis et al. — *Physics-informed machine learning* (Nat. Rev. Phys. 2021). Survey/review.
- **[4]** Psaros et al. — *Meta-learning PINN loss functions* (JCP 2022).
- **[5]** Yu et al. — *Gradient-enhanced PINNs* (CMAME 2022).
- **[6]** McClenny & Braga-Neto — *Self-adaptive PINNs* (2020).
- **[7]** Jin et al. — *NSFnets* (JCP 2021).
- **[8]** Wu et al. — *Adaptive sampling for PINNs* (CMAME 2023).
- **[9]** Wang, Wang, Perdikaris — *Fourier feature networks* (CMAME 2021).
- **[10]** Hao et al. — *PINNacle benchmark* (2023).

### DeepONet & operators
- **[11]/[12]** Lu et al. — *DeepONet* (arXiv 2019; Nat. Mach. Intell. 2021). The operator-learning framework FedDeepONet builds on.
- **[13]** Zhu et al. — *Reliable extrapolation of neural operators* (CMAME 2023).
- **[14]** Lin et al. — bubble dynamics; **[15]** Goswami et al. — fracture; **[16]** Osorio et al. — solar-thermal; **[17]** Cai et al. — DeepM&Mnet (electroconvection); **[18]** Yin et al. — multiscale FE coupling.
- **[19]** Lu et al. — POD-DeepONet vs FNO comparison; **[20]** Wang et al. — physics-informed DeepONet; **[21]** Lu et al. — multifidelity DeepONet; **[22]** Jin et al. — MIONet; **[23]** Zhu et al. — Fourier-DeepONet; **[24]** Jiang et al. — Fourier-MIONet; **[25]** Yang et al. — UQ for operators; **[26]** Zhang et al. — D2NO (distributed operators).

### Federated learning
- **[27]** McMahan et al. — *Communication-efficient learning of deep networks from decentralized data* (AISTATS 2017). **The FedAvg paper — the foundation of FL.**
- **[28]**–**[31]** FL applications: patient similarity, EHR predictive models, brain-tumour segmentation, patient clustering (healthcare).
- **[32]**–**[34]** FL in finance / clouds.
- **[35]** Nguyen et al. — *FL for IoT survey* (IEEE COMST 2021).
- **[36]** Wei et al. — *FL with differential privacy* (TIFS 2020); **[37]** Truex et al. — LDP-Fed; **[38]** Geyer et al. — DP at client level.
- **[39]** Hsu et al. — non-iid effects for federated visual classification.
- **[40]** Reddi et al. — *Adaptive federated optimization* (FedAdagrad/FedAdam/FedYogi) — the FedOPT paper.
- **[41]** Moya & Lin — *Fed-DeepONet* (Algorithms 2022). The prior FL+DeepONet work this paper extends (to PINNs & function approximation).
- **[42]** Kairouz et al. — *Advances and open problems in FL* (Foundations & Trends in ML 2021). Comprehensive survey.

### Theory & datasets used
- **[43]** Flamary et al. — *POT: Python optimal transport* (JMLR 2021). Used to compute W1.
- **[44]** Zhao et al. — *FL with non-iid data* (2018). **Origin of the weight-divergence concept.**
- **[45]** Li et al. — *FedProx* (2020). Heterogeneous-network FL with proximal term.
- **[46]** Li et al. — *Convergence of FedAvg on non-iid data* (ICLR 2020).
- **[47]** Shin — *Convergence of PINNs* (2020).
- **[48]** Deng et al. — *Convergence rate of DeepONets* (2021).
- **[49]** Gramacy & Lee — *Cases for the nugget* (Statistics & Computing 2012). Source of the 1D test function.
- **[50]** Schaffer — *Vector evaluated genetic algorithms* (1984). Source of the 2D Schaffer function.
- **[51]** Lu et al. — *PINNs with hard constraints for inverse design* (SIAM J. Sci. Comput. 2021). The hard-constraint method used in all PINN examples.
- **[52]** Jagtap et al. — *cPINN* (CMAME 2020); **[53]** Jagtap & Karniadakis — *XPINNs* (AAAI 2021); **[54]** Shukla et al. — *Parallel PINNs via domain decomposition* (JCP 2021). Domain-decomposition PINNs that FedPINN is contrasted with.

---

## Master Formula Sheet

**Federated objective:** `min_θ L(θ) = Σ_k (N_k/N) L_k(θ)`,  `L_k(θ) = (1/N_k) Σ_{dᵢ∈D_k} ℓ(θ; dᵢ)` (ℓ = squared error)

**FedAvg aggregation:** `θ^{l+1} ← θ^l − η Σ_k (N_k/N) Σ_{i=1..E} g_k^i`

**FedOPT:** Client-OPT locally, then `θ^{l+1} = Server-OPT(θ^l, −Σ_k (N_k/N) Δ_k^l)`

**FedAvg-Adam:** Client-OPT = Adam, Server-OPT = SGD; practical update `θ^{l+1} ← Σ_k (N_k/N) θ_k^{l,E}`

**W1 distance:** `W1(µ,ν) = inf_γ ∫ |x−y| dγ(x,y)` (= Earth Mover's Distance; computed via POT)

**Mean pairwise W1 (K ≥ 3):** `W1 = (1/((K−1)(K−2))) Σ_{i<j} W1(µ_i, µ_j)`

**Weight divergence:** `E_WD = θ_FedAvg − θ_SGD`;  **Theorem 3.1:** `E_WD^{l,E} ≤ 2ηMEl` (bounded gradients `||∇ℓ|| ≤ M`)

**DeepONet output:** `G(v)(ξ) = Σ_{k=1..p} b_k(v) t_k(ξ) + b₀`

**PINN residual loss:** `L_r(θ;x) = ||F[û(θ)](x) − f(x)||²₂`

**Chebyshev generating space:** `p(x) = Σ_{i=0..M} a_i T_i(x)`, a_i ∈ [−1,1], M = 10 (n nonzero terms control non-iid)

---

## Master Experiment Summary Table

| # | Task | Equation / function | Clients | Partition | Key result |
|---|---|---|---|---|---|
| 1 | 1D function approx | Gramacy & Lee | 2 | 1D, n 1→50 | W1 ↓ ⇒ error ↓; divergence ↑ with heterogeneity |
| 2 | 2D function approx | Schaffer | 2, then 3/4/5 | x-partition n 1→25 | Same trend; more clients ⇒ more error; federated beats extrapolation |
| 3 | 1D Poisson (fwd) | −Δu = Σ i·sin(ix)+8sin(8x) | 2 | 1D, n 6→32 | Federated > extrapolation; ≈ centralized when W1 small |
| 4 | 2D Helmholtz (fwd) | −u_xx−u_yy−k₀²u=f, k₀=4π | 2 | xy-partition | W1 ↑ ⇒ error ↑; layer-2 divergence most sensitive |
| 5 | Allen-Cahn (fwd, time-dep.) | u_t = d·u_xx + 5(u−u³) | 2 & 3 | x-only, n 2→40 | FL improves extrapolation regardless of heterogeneity |
| 6 | Inverse Navier-Stokes | 2D cylinder, Re=100 | 2 | 2D x-partition | Less heterogeneity ⇒ faster + more stable convergence |
| 7 | Inverse diffusion-reaction | infer k(x) in λu_xx−k(x)u=f | 2 | 1D, n 2→24 | k(x) & u(x) errors ↑ with W1; no divergence trend for k |
| 8 | Antiderivative (operator) | G: v ↦ ∫₀ˣ v dτ | 2 | Chebyshev fwd/inv | FedDeepONet > extrapolation; jump at n=6 |
| 9 | Diffusion-reaction (operator) | G: v(x) ↦ u(x,t) | 2 | Chebyshev fwd/inv | Client data quality matters; federated > baselines |
| 10 | Burgers (operator) | G: v(x) ↦ u(x,t), ν=0.1 | 2 & 3 | Chebyshev | 3 clients > 2; viscosity damps functional-space effect |

---

## Key Takeaways (for an exam / quick revision)

1. **FedSciML = FL + SciML.** Privacy-preserving way to solve PDE/operator problems on distributed, heterogeneous data.
2. **W1 distance** is the heterogeneity ruler: smaller W1 ⇔ more iid ⇔ smaller error.
3. **FedAvg-Adam** = local Adam updates + global averaging (special case of FedOPT).
4. **Weight divergence** `θ_FedAvg − θ_SGD` is bounded: `≤ 2ηMEl` — grows linearly with local epochs, independent of heterogeneity.
5. **Federated > extrapolation (local-only)** in every experiment; **federated ≈ centralized** as W1 → 0.
6. **FedDeepONet is insensitive to communication frequency** — big practical win (few syncs, many local steps).
7. **FedPINN ≠ domain decomposition** — clients never reveal where their domain is.
8. More clients ⇒ more divergence ⇒ larger errors (but 3 clients beat 2 for Burgers because more basis coverage).
9. Hard constraints, tanh/sine/ReLU activations, Adam — the practical recipe (Table 4).
