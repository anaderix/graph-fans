# Phase 5a Execution Plan: Independent Per-Mode Diffusion

## Objective
Test whether diffusing spectral coefficients independently per eigenmode (bypassing the GCN) can match or beat spatial-domain diffusion on spectral W1. Diagnostic: if per-mode independent diffusion improves W1, it validates spectral-domain approach for 5b/5c. Expected failure mode: spatial incoherence.

## Design Decisions
1. **Shared MLP** conditioned on (lambda_k, t, E_k): 3-layer MLP, input d+3=7, hidden 128, output d=4, SiLU + LayerNorm
2. **Per-mode schedule**: CosineScheduleSDE with mode-specific t_max(k) = T * (E_k/E_max)^0.5, clamped [0.1*T, T]
3. **Epsilon-prediction + DDIM** per mode (same as spatial baseline)
4. **Batching**: all 50 modes batched into single [n_modes, d+3] tensor for efficient MLP forward pass

## Files to Create

| File | Purpose | Lines (est.) |
|------|---------|-------------|
| graph_fans/phase5/__init__.py | Package init | 1 |
| graph_fans/phase5/spectral_score_network.py | Shared MLP for per-mode denoising | ~60 |
| graph_fans/phase5/mode_schedule.py | Per-mode schedule derivation from energy profile | ~120 |
| graph_fans/phase5/spectral_trainer.py | Training loop + generation for spectral diffusion | ~300 |
| graph_fans/phase5/spatial_coherence.py | Node-neighbor correlation, cross-mode correlation metrics | ~80 |
| scripts/test_phase5a.py | 3-way comparison experiment script | ~300 |
| tests/test_phase5.py | Unit + integration tests | ~200 |

## Implementation Details

### spectral_score_network.py
- SpectralScoreNetwork(n_features=4, hidden_dim=128, n_layers=3)
- forward(c_k_t, lambda_k, t, E_k) -> eps_pred
- Input: concat [c_k_t (d), lambda_k (1), t (1), E_k (1)] = 7 dims
- 3 layers: Linear+LayerNorm+SiLU, final Linear to d

```python
class SpectralScoreNetwork(nn.Module):
    """Shared MLP for per-mode epsilon prediction.
    
    Conditioned on eigenvalue lambda_k, diffusion time t, and mode energy E_k.
    All modes are batched together for efficient forward pass.
    """
    
    def __init__(self, n_features: int = 4, hidden_dim: int = 128, n_layers: int = 3):
        super().__init__()
        input_dim = n_features + 3  # c_k_t (d) + lambda_k (1) + t (1) + E_k (1)
        layers = []
        for i in range(n_layers):
            in_d = input_dim if i == 0 else hidden_dim
            layers.extend([
                nn.Linear(in_d, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            ])
        layers.append(nn.Linear(hidden_dim, n_features))
        self.net = nn.Sequential(*layers)

    def forward(self, c_k_t: Tensor, lambda_k: Tensor, t: Tensor, E_k: Tensor) -> Tensor:
        """Predict epsilon for all modes in batch.
        
        Args:
            c_k_t: [batch, d] noisy spectral coefficients
            lambda_k: [batch, 1] eigenvalues
            t: [batch, 1] diffusion times
            E_k: [batch, 1] mode energies (from training data)
        
        Returns:
            eps_pred: [batch, d] predicted noise
        """
        x = torch.cat([c_k_t, lambda_k, t, E_k], dim=-1)
        return self.net(x)
```

### mode_schedule.py
- ModeSchedule dataclass: mode_idx, eigenvalue, energy, t_max
- ModeScheduleSet: wraps CosineScheduleSDE with per-mode t_max(k)
  - t_max(k) = T * (E_k / E_max)^energy_exponent, clamped [0.1*T, T]
  - sample_t(mode_idx, rng) -> float
  - get_ddim_grid(mode_idx, n_steps) -> np.ndarray (t_max(k) to 0)
  - alpha_bar(mode_idx, t), perturb(), ddim_step() delegating to base SDE

```python
@dataclass
class ModeSchedule:
    """Schedule parameters for a single eigenmode."""
    mode_idx: int
    eigenvalue: float
    energy: float
    t_max: float


class ModeScheduleSet:
    """Per-mode schedule collection wrapping CosineScheduleSDE.
    
    High-energy modes (low frequency) get larger t_max (more noise needed
    to corrupt them). Low-energy modes (high frequency) get smaller t_max
    (already close to noise, less corruption needed).
    """
    
    def __init__(
        self,
        eigenvalues: np.ndarray,
        mode_energies: np.ndarray,
        base_sde: CosineScheduleSDE,
        energy_exponent: float = 0.5,
        t_max_floor: float = 0.1,
    ):
        T = base_sde.T
        E_max = mode_energies.max()
        self.base_sde = base_sde
        self.schedules = []
        for k in range(len(eigenvalues)):
            ratio = (mode_energies[k] / E_max) if E_max > 0 else 1.0
            t_max_k = T * ratio**energy_exponent
            t_max_k = float(np.clip(t_max_k, t_max_floor * T, T))
            self.schedules.append(ModeSchedule(
                mode_idx=k,
                eigenvalue=float(eigenvalues[k]),
                energy=float(mode_energies[k]),
                t_max=t_max_k,
            ))

    def sample_t(self, mode_idx: int, rng: np.random.Generator) -> float:
        """Sample uniform t in [eps, t_max(k)]."""
        eps = 1e-5
        t_max = self.schedules[mode_idx].t_max
        return float(rng.uniform(eps, t_max))

    def get_ddim_grid(self, mode_idx: int, n_steps: int) -> np.ndarray:
        """Return DDIM timestep grid from t_max(k) to ~0."""
        t_max = self.schedules[mode_idx].t_max
        return np.linspace(t_max, 1e-5, n_steps + 1)

    def alpha_bar(self, mode_idx: int, t: float) -> float:
        """Delegate to base SDE alpha_bar."""
        return self.base_sde.alpha_bar(t)

    def perturb(self, x_0, mode_idx: int, t: float) -> tuple:
        """Forward diffuse using mode-specific time."""
        return self.base_sde.perturb(x_0, t)

    def ddim_step(self, x_t, eps_pred, mode_idx: int, t_now: float, t_next: float):
        """DDIM deterministic step using base SDE alpha_bar."""
        return self.base_sde.ddim_step(x_t, eps_pred, t_now, t_next)
```

### spectral_trainer.py
- SpectralTrainConfig dataclass (n_epochs=500, lr=1e-3, hidden_dim=128, n_layers=3, energy_exponent=0.5, etc.)
- SpectralTrainer:
  - __init__: eigendecomp, project dataset to spectral domain [N, n_modes, d], compute per-mode energy from training data, build ModeScheduleSet, build MLP, optimizer, EMA, scheduler
  - train(): per epoch, sample feature realization, batch all modes, sample t_k per mode, forward diffuse, predict epsilon, MSE loss, backprop
  - generate(n_steps=200, n_samples=1): per-mode DDIM from t_max(k) to 0, reconstruct x = U @ c
  - sanity_check(): std ratio, spectral L2, spatial coherence

```python
@dataclass
class SpectralTrainConfig:
    n_epochs: int = 500
    lr: float = 1e-3
    hidden_dim: int = 128
    n_layers: int = 3
    n_features: int = 4
    energy_exponent: float = 0.5
    t_max_floor: float = 0.1
    n_gen_steps: int = 200
    use_ema: bool = True
    ema_decay: float = 0.999
    use_lr_scheduler: bool = True
    seed: int = 42
    device: str = "cpu"
    grad_clip: float = 1.0
    weight_decay: float = 1e-4


class SpectralTrainer:
    def __init__(
        self,
        graph: nx.Graph,
        features_list: list[np.ndarray],  # [N_train, n_nodes, n_features]
        config: SpectralTrainConfig,
    ):
        # Eigendecomposition (amortized, computed once)
        eigenvalues, eigenvectors = compute_laplacian_spectrum(graph)
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors  # [n_nodes, n_modes]
        self.n_modes = eigenvectors.shape[1]
        self.config = config
        self.device = torch.device(config.device)
        
        # Project all training features to spectral domain: [N, n_modes, d]
        U_T = eigenvectors.T  # [n_modes, n_nodes]
        self.spectral_data = np.stack([
            U_T @ feat for feat in features_list
        ])  # [N_train, n_modes, n_features]
        
        # Compute per-mode energy from training data
        self.mode_energies = (self.spectral_data ** 2).mean(axis=(0, 2))  # [n_modes]
        
        # Build per-mode schedule set
        base_sde = CosineScheduleSDE()
        self.schedule_set = ModeScheduleSet(
            eigenvalues, self.mode_energies, base_sde,
            energy_exponent=config.energy_exponent,
            t_max_floor=config.t_max_floor,
        )
        
        # Build shared MLP
        self.model = SpectralScoreNetwork(
            n_features=config.n_features,
            hidden_dim=config.hidden_dim,
            n_layers=config.n_layers,
        ).to(self.device)
        
        # Optimizer, EMA, scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        if config.use_ema:
            self.ema_model = copy.deepcopy(self.model)
        if config.use_lr_scheduler:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config.n_epochs)

    def train(self) -> dict:
        """Training loop: per epoch, sample one realization, batch all modes."""
        rng = np.random.default_rng(self.config.seed)
        losses = []
        
        for epoch in range(self.config.n_epochs):
            # Sample random training feature realization
            idx = rng.integers(len(self.spectral_data))
            c_all = self.spectral_data[idx]  # [n_modes, d]
            
            # Sample t_k per mode, batch forward diffusion
            t_per_mode = np.array([
                self.schedule_set.sample_t(k, rng) for k in range(self.n_modes)
            ])
            
            # Forward diffuse each mode independently
            c_0 = torch.tensor(c_all, dtype=torch.float32, device=self.device)
            t_tensor = torch.tensor(t_per_mode, dtype=torch.float32, device=self.device)
            noise = torch.randn_like(c_0)
            
            # alpha_bar for each mode's t
            alpha_bars = torch.tensor([
                self.schedule_set.alpha_bar(k, t_per_mode[k])
                for k in range(self.n_modes)
            ], dtype=torch.float32, device=self.device).unsqueeze(-1)  # [n_modes, 1]
            
            c_t = torch.sqrt(alpha_bars) * c_0 + torch.sqrt(1 - alpha_bars) * noise
            
            # Conditioning: eigenvalues and mode energies
            lambda_k = torch.tensor(
                self.eigenvalues, dtype=torch.float32, device=self.device
            ).unsqueeze(-1)  # [n_modes, 1]
            E_k = torch.tensor(
                self.mode_energies, dtype=torch.float32, device=self.device
            ).unsqueeze(-1)  # [n_modes, 1]
            t_cond = t_tensor.unsqueeze(-1)  # [n_modes, 1]
            
            # Forward pass (batched over all modes)
            eps_pred = self.model(c_t, lambda_k, t_cond, E_k)
            
            # MSE loss
            loss = ((eps_pred - noise) ** 2).mean()
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
            
            if self.config.use_ema:
                _ema_update(self.ema_model, self.model, self.config.ema_decay)
            if self.config.use_lr_scheduler:
                self.scheduler.step()
            
            losses.append(loss.item())
        
        return {"losses": losses, "final_loss": np.mean(losses[-50:])}

    @torch.no_grad()
    def generate(self, n_steps: int = 200, n_samples: int = 1) -> list[np.ndarray]:
        """Generate features via per-mode DDIM, reconstruct to spatial domain."""
        model = self.ema_model if self.config.use_ema else self.model
        model.eval()
        
        results = []
        for _ in range(n_samples):
            # Start from noise for each mode, scaled by t_max(k)
            c_gen = torch.randn(
                self.n_modes, self.config.n_features,
                device=self.device,
            )
            
            # Per-mode DDIM: each mode follows its own schedule grid
            for k in range(self.n_modes):
                ts = self.schedule_set.get_ddim_grid(k, n_steps)
                c_k = c_gen[k:k+1]  # [1, d]
                
                # Scale initial noise by mode's t_max
                alpha_bar_start = self.schedule_set.alpha_bar(k, ts[0])
                c_k = c_k * np.sqrt(1 - alpha_bar_start)
                
                lambda_k = torch.tensor(
                    [[self.eigenvalues[k]]], dtype=torch.float32, device=self.device)
                E_k = torch.tensor(
                    [[self.mode_energies[k]]], dtype=torch.float32, device=self.device)
                
                for i in range(len(ts) - 1):
                    t_now, t_next = ts[i], ts[i + 1]
                    t_cond = torch.tensor(
                        [[t_now]], dtype=torch.float32, device=self.device)
                    eps_pred = model(c_k, lambda_k, t_cond, E_k)
                    c_k = self.schedule_set.ddim_step(
                        c_k, eps_pred, k, t_now, t_next)
                
                c_gen[k] = c_k[0]
            
            # Reconstruct: x = U @ c
            c_np = c_gen.cpu().numpy()
            x_gen = self.eigenvectors @ c_np  # [n_nodes, d]
            results.append(x_gen)
        
        return results
    
    def sanity_check(self, ref_features: np.ndarray, gen_features: np.ndarray) -> dict:
        """Quick sanity check: std ratio, spectral L2."""
        std_ratio = gen_features.std() / ref_features.std()
        spectral_gen = self.eigenvectors.T @ gen_features
        spectral_ref = self.eigenvectors.T @ ref_features
        spectral_l2 = np.sqrt(((spectral_gen**2).mean(axis=1) -
                                (spectral_ref**2).mean(axis=1))**2).mean()
        return {"std_ratio": float(std_ratio), "spectral_l2": float(spectral_l2)}
```

### spatial_coherence.py
- node_neighbor_correlation(features, graph) -> float
- cross_mode_energy_correlation(features, eigenvectors) -> float
- spatial_coherence_summary(ref, gen, graph, eigenvectors) -> dict

```python
def node_neighbor_correlation(features: np.ndarray, graph: nx.Graph) -> float:
    """Average Pearson correlation of feature vectors between adjacent nodes.
    
    High values indicate spatially smooth features (typical of community structure).
    Random noise would give ~0.0; community features typically give 0.5-0.8.
    
    Args:
        features: [n_nodes, n_features]
        graph: nx.Graph
    
    Returns:
        Mean correlation across all edges.
    """
    correlations = []
    for u, v in graph.edges():
        fu, fv = features[u], features[v]
        # Pearson correlation between feature vectors
        if np.std(fu) > 1e-10 and np.std(fv) > 1e-10:
            corr = np.corrcoef(fu, fv)[0, 1]
            correlations.append(corr)
    return float(np.mean(correlations)) if correlations else 0.0


def cross_mode_energy_correlation(features: np.ndarray, eigenvectors: np.ndarray) -> float:
    """Correlation of per-mode energy profile with reference energy ordering.
    
    Checks whether the generated features' spectral energy profile follows
    the expected ordering (higher energy in low-frequency modes for community features).
    
    Args:
        features: [n_nodes, n_features]
        eigenvectors: [n_nodes, n_nodes]
    
    Returns:
        Spearman rank correlation of energy profile vs mode index.
    """
    coeffs = eigenvectors.T @ features  # [n_modes, d]
    mode_energy = (coeffs ** 2).sum(axis=1)  # [n_modes]
    # For community features, energy should decrease with mode index
    from scipy.stats import spearmanr
    ranks = np.arange(len(mode_energy))
    corr, _ = spearmanr(mode_energy, -ranks)  # negative: high energy at low index
    return float(corr)


def spatial_coherence_summary(
    ref_features: np.ndarray,
    gen_features: np.ndarray,
    graph: nx.Graph,
    eigenvectors: np.ndarray,
) -> dict:
    """Compute spatial coherence comparison between reference and generated features."""
    return {
        "ref_neighbor_corr": node_neighbor_correlation(ref_features, graph),
        "gen_neighbor_corr": node_neighbor_correlation(gen_features, graph),
        "ref_mode_energy_corr": cross_mode_energy_correlation(ref_features, eigenvectors),
        "gen_mode_energy_corr": cross_mode_energy_correlation(gen_features, eigenvectors),
    }
```

### test_phase5a.py (experiment script)
- 3-way comparison: uniform_spatial (Phase 2 baseline), band_spatial (Phase 2g best), spectral_5a
- Uses existing Trainer for spatial baselines, new SpectralTrainer for spectral
- Same datasets (reuse phase2f_small/datasets or phase4a/datasets), same W1 evaluation, same seeds
- Paired t-test with Bonferroni correction (4 families, alpha=0.0125)
- CLI: --families --n-nodes 50 --n-seeds 5 --n-epochs 500 --device cuda --output results/phase5a/phase5a_results.json

Script structure:
```python
def run_experiment(args):
    for family in families:
        for seed in range(args.n_seeds):
            graph, features_train, features_ref = load_or_generate_dataset(...)
            
            # Method 1: uniform_spatial (Phase 2 baseline)
            trainer_uniform = Trainer(graph, features_train,
                TrainConfig(noise_shaping="uniform", n_epochs=args.n_epochs, ...))
            trainer_uniform.train()
            gen_uniform = [trainer_uniform.generate() for _ in range(50)]
            w1_uniform = mean_w1(features_ref, gen_uniform, eigenvectors)
            
            # Method 2: band_spatial (Phase 2g best)
            importance_weights = compute_importance_weights(...)
            trainer_band = Trainer(graph, features_train,
                TrainConfig(noise_shaping="band", n_epochs=args.n_epochs, ...),
                importance_weights=importance_weights)
            trainer_band.train()
            gen_band = [trainer_band.generate() for _ in range(50)]
            w1_band = mean_w1(features_ref, gen_band, eigenvectors)
            
            # Method 3: spectral_5a (new)
            spec_trainer = SpectralTrainer(graph, features_train,
                SpectralTrainConfig(n_epochs=args.n_epochs, ...))
            spec_trainer.train()
            gen_spectral = spec_trainer.generate(n_samples=50)
            w1_spectral = mean_w1(features_ref, gen_spectral, eigenvectors)
            
            # Spatial coherence diagnostics
            coherence = spatial_coherence_summary(
                features_ref[0], gen_spectral[0], graph, eigenvectors)
            
            results.append({family, seed, w1_uniform, w1_band, w1_spectral, coherence})
    
    # Statistical analysis: paired t-test with Bonferroni correction
    for family in families:
        for pair in [("band", "spectral"), ("uniform", "spectral"), ("uniform", "band")]:
            t_stat, p_val = ttest_rel(w1_a, w1_b)
            significant = p_val < 0.0125  # Bonferroni: 0.05/4
    
    save_results(results, args.output)
```

### tests/test_phase5.py (~15 tests)
- TestSpectralScoreNetwork: output shape, conditioning produces different outputs, batch mode
- TestModeSchedule: t_max energy-proportional, clamping, ddim grid endpoints, perturb/ddim shapes
- TestSpectralTrainer: projection roundtrip, loss decreases, generation shape, finite values
- TestSpatialCoherence: neighbor correlation on community features > 0.3

```python
class TestSpectralScoreNetwork:
    def test_output_shape(self):
        """Output [batch, d] matches input feature dim."""
    
    def test_conditioning_changes_output(self):
        """Different lambda_k, t, E_k produce different predictions."""
    
    def test_batch_mode(self):
        """All 50 modes batched into single forward pass."""


class TestModeSchedule:
    def test_t_max_energy_proportional(self):
        """Higher energy modes get larger t_max."""
    
    def test_t_max_clamping(self):
        """t_max clamped to [0.1*T, T]."""
    
    def test_ddim_grid_endpoints(self):
        """Grid starts at t_max(k), ends near 0."""
    
    def test_perturb_shape(self):
        """perturb returns correct shape."""
    
    def test_ddim_step_shape(self):
        """ddim_step returns correct shape."""


class TestSpectralTrainer:
    def test_projection_roundtrip(self):
        """U @ U^T @ x == x for features in eigenbasis."""
    
    def test_loss_decreases(self):
        """Train 100 epochs on small graph, final loss < initial loss."""
    
    def test_generation_shape(self):
        """generate() returns [n_nodes, n_features]."""
    
    def test_finite_values(self):
        """Generated features contain no NaN or Inf."""


class TestSpatialCoherence:
    def test_neighbor_correlation_community(self):
        """Community features on SBM give neighbor correlation > 0.3."""
    
    def test_neighbor_correlation_noise(self):
        """Random noise gives neighbor correlation near 0."""
    
    def test_cross_mode_energy_correlation(self):
        """Community features have positive energy-mode correlation."""
```

## Data Requirements
- Families: SBM(q=0.05), SBM(q=0.1), BA(m=2), BA(m=5), n=50, d=4, community
- 100 train + 50 ref per seed, 5 seeds per family
- Reuse existing dataset cache: results/phase2f_small/datasets/ or results/phase4a/datasets/
- Total: 3 methods x 4 families x 5 seeds = 60 training runs

## Run Configuration
```bash
uv run python scripts/test_phase5a.py \
    --families "SBM(q=0.05),SBM(q=0.1),BA(m=2),BA(m=5)" \
    --n-nodes 50 --n-seeds 5 --n-epochs 500 \
    --device cuda --output results/phase5a/phase5a_results.json \
    --dataset-dir results/phase4a/datasets
```
Expected runtime: 30-60 min on L40S (MLP ~1000x cheaper than GCN per forward pass)

## Execution Sequence

| Day | Activity |
|-----|----------|
| 1 | Implement spectral_score_network.py, mode_schedule.py, spatial_coherence.py, write unit tests |
| 2 | Implement spectral_trainer.py, integration tests, CPU smoke test (n=10, 50 epochs) |
| 3 | Write test_phase5a.py experiment script, CPU dry run (1 seed, 50 epochs, 1 family) |
| 4 | Run Phase 5a on GPU (~30-60 min), collect results |
| 5 | Analyze results, spatial coherence diagnostics, write LOG.md entry |

## Success Criteria
- Primary: spectral_5a W1 < band_spatial W1 for >= 2/4 families (Bonferroni alpha=0.0125)
- Secondary: node-neighbor correlation >= 0.3 (vs ~0.5-0.8 for real community features)
- No-go: W1 worse than uniform_spatial, or loss doesn't drop below 0.5

## Analysis Plan
1. 3-way W1 table: uniform vs band vs spectral, paired t-test
2. Per-mode W1 profile: which modes improve most (low-freq vs high-freq)
3. Spatial coherence: neighbor correlation comparison across methods
4. Training stats: loss curves, wall-clock time, generation time per method
5. Diagnostic: spectral energy profile of generated features vs reference (does it match?)

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Cross-mode correlations lost | High | Major | Proceed to 5b/5c if W1 passes but coherence fails |
| Spatial incoherence in reconstructed features | Medium | Major | Spatial coherence metrics detect this; add smoothness regularization if marginal |
| Per-mode MLP too simple for d-dim denoising | Low | Minor | d=4 is trivially low-dimensional; increase hidden_dim if needed |
| Mode energy estimates noisy for small dataset | Low | Minor | Average over all 100 training samples; clip extreme ratios |
| Per-mode DDIM loop slow (50 sequential modes) | Medium | Moderate | Batch modes with same n_steps; vectorize alpha_bar computation |

## Connection to Roadmap

This implements Phase 5a from `plans/phase5.md`. It is the diagnostic gateway for spectral-domain diffusion: if independent per-mode beats spatial on W1 despite lacking cross-mode interaction, the spectral approach is validated and 5b/5c address the remaining coherence gap. If it fails on W1, spatial approaches (Phase 4 refinements or Phase 6 architecture changes) are the better path.

Key difference from Phase 3a (per-mode shaping, NO-GO): Phase 3a shaped noise in a spatial GCN pipeline, creating train/generate mismatch. Phase 5a diffuses entirely in spectral domain -- no GCN, no mismatch, each mode is an independent d-dimensional denoising problem.
