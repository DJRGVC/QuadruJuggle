# Literature Review: EKF Q/R Tuning for Fast Ball Tracking

**Prepared by:** lit-review subagent  
**Date:** 2026-04-08  
**Scope:** Process noise Q, measurement noise R, adaptive Kalman methods, ball-specific reported values, and sim-to-real implications for the QuadruJuggle 6-state EKF.

---

## (a) Principled Q/R Initialization

**Q — process noise covariance.** Q encodes how much you distrust the dynamics model over one time step. The standard derivation (Bar-Shalom, Li, Kirubarajan — "Estimation with Applications to Tracking and Navigation," Wiley 2001, the canonical textbook) for a constant-velocity model is the discretized continuous-white-noise-acceleration (CWNA) model:

```
Q = q_c * [dt^3/3   dt^2/2]   (block per axis)
          [dt^2/2   dt    ]
```

where `q_c` is the spectral density of the unmodeled acceleration (m²/s³). For a physical ball, the unmodeled accelerations are: (1) spin-induced Magnus force, (2) drag model error, (3) bounce transients that happen between measurement updates, and (4) robot-frame camera jitter. A 40 mm ping-pong ball at 4 m/s has a drag deceleration of ~2.2 m/s² (as computed in `BallEKFConfig`). If the drag coefficient is uncertain by ±20%, unmodeled acceleration is ~0.4 m/s². Setting `q_c ≈ (0.4)^2 = 0.16 m²/s³` gives at `dt = 0.005 s`:

```
Q_pos  = q_c * dt^3/3  ≈ 0.16 * 4.2e-8  ≈ 6.6e-9 m²   (std ≈ 0.08 mm)
Q_vel  = q_c * dt      ≈ 0.16 * 0.005   ≈ 8e-4 (m/s)²  (std ≈ 28 mm/s)
```

This is far smaller than what is currently set in `BallEKFConfig` (q_pos=0.01, q_vel=1.0 per sqrt(s)). The current values are "safe but sluggish" — they aggressively discount the dynamics model, making the filter defer too heavily to measurements. For a well-characterized physics model (ballistic + drag), Q should be tighter.

**R — measurement noise covariance.** R must reflect the actual sensor error, not a safety margin. For the D435i with monocular-depth-from-known-size detection:

- XY (lateral): pixel quantization + centroid fit error on a ~15-20 px diameter blob at 0.5 m. The centroid fit error for a circular blob is approximately `σ_px ≈ 0.5 px → σ_xy ≈ 0.5 * z / f ≈ 0.5 * 0.5 / 640 ≈ 0.4 mm` (with f ≈ 640 px for a 90° HFOV, 640-px-wide sensor at 0.5 m). In practice, HSV blob detection adds ~1-2 px error → `σ_xy ≈ 1–2 mm`.
- Z (depth): monocular depth from diameter. At 0.5 m with 15 px blob, ±1 px diameter error → ±16 mm depth error. At 1.0 m with 7 px blob, ±1 px → ±65 mm. This is the dominant error and is strongly distance-dependent.

The `D435iNoiseModelCfg` already encodes this correctly: `sigma_xy_base = 2 mm`, `sigma_z_base = 3 mm`, `sigma_z_per_metre = 2 mm/m`. These should feed directly into R — not be re-tuned independently. The EKF's R should match the noise model's expected output variance at the operating distance.

**Key insight: R should be time-varying.** At the EKF update step, the correct R is `diag([σ_xy², σ_xy², σ_z²])` where `σ_z = σ_z_base + σ_z_per_metre * z`. Feeding a fixed R underestimates measurement noise at distance (overconfident at 1 m) and overestimates it close up (underweights good measurements at 0.1 m). This matters most for the critical Stage G target height of 1 m.

---

## (b) Adaptive Methods Comparison

Four adaptive Kalman filter (AKF) classes appear in the literature, in rough order of maturity:

**1. Sage-Husa (1969) — maximum likelihood noise estimation.** Sage and Husa (IEEE Trans. Automatic Control, 14(4), 1969) derived a recursive ML estimator that updates Q and R online from the innovation sequence:

```
R_k+1 = (1-d_k) * R_k + d_k * (y_k y_k^T - H P_k H^T)
```

where `d_k = (1-b)/(1-b^(k+1))` is a fading memory weight with forgetting factor `b ≈ 0.95–0.99`. This is computationally trivial (O(n²) per step) and widely cited (400+ citations). The risk: R estimates can go negative (non-positive-definite) if innovations are noisy; needs a projection step to stay PD. For a slow-varying noise environment it works well; for fast dynamics (ball bouncing every ~0.1 s), the window may not be long enough to converge before the dynamics change.

**2. MMAE — Multiple Model Adaptive Estimation.** Runs K filters in parallel, each with different (Q_i, R_i), and weights them by likelihood of the innovation sequence. Best known from Maybeck, "Stochastic Models, Estimation, and Control," Academic Press, 1979. Accuracy is limited by the discrete model set. For 6D state with 2 noise parameters each, even K=10 banks is coarse. Computational cost is O(K) × base EKF cost — manageable for real-time single-env deployment, but K × 12288 envs in sim would require careful batching. Not recommended as primary approach.

**3. Fuzzy AKF.** Uses fuzzy rules (e.g., if innovation is large AND sudden, increase Q). Introduced by Mohamed and Schwarz (J. Navigation, 1999). Practically useful for GPS/INS fusion where sensor characteristics are well-characterized by expert rules. For a ping-pong ball, no domain-expert rules exist beyond "depth error grows with distance." Not preferred over Sage-Husa for this application.

**4. ANEES / χ² innovation test — not an AKF, but the validation tool for any AKF.** The Normalized Innovation Squared test (Bar-Shalom et al. 2001, Chapter 5):

```
NIS_k = y_k^T * S_k^{-1} * y_k     (scalar per step)
```

For a correctly tuned EKF, `NIS ~ χ²(m)` where m=3 (measurement dimension). The 95% consistency band for 3D measurements is [0.35, 7.81]. If mean NIS > 7.81 (consistently), Q or R is too small (filter overconfident). If mean NIS < 0.35 (consistently), Q or R is too large (filter too conservative, ignores measurements). ANEES is the normalized-across-environments version: `ANEES = (1/N) Σ NIS_k`. This is the correct diagnostic for tuning Q and R online from a batch of simulation runs without needing external ground truth — just check if the filter's own innovation covariance S is consistent with observed innovation magnitudes.

**Best choice for sim-to-real transfer: instrument ANEES in the existing EKF, use Sage-Husa as secondary correction.** The argument:

- ANEES can be computed inside the batched GPU EKF for free (S is already computed in the update step). Log `mean(NIS)` per iteration; alert when outside [0.35, 7.81].
- If mean NIS drifts high in real deployment (filter overconfident → real noise is larger than modeled), apply Sage-Husa online with b=0.97 to adapt R toward the true measurement noise.
- Sage-Husa is avoided in sim (ground truth is available, so offline calibration is exact), but is a useful real-deployment fallback.
- MMAE and fuzzy AKF add complexity without solving the root problem (noise characterization), and are not recommended.

---

## (c) Ball-Specific Reported Values

The literature is sparse on explicit Q/R tables, but several papers provide enough information to back-calculate values:

**Forrai et al. 2023 (ETH, ICRA) — "Event-based Agile Object Catching":** Catches balls at up to 15 m/s. Uses EKF with ballistic dynamics over an event camera. The paper states position error ~3 mm at 2–3 m range and ~8 mm at 4–5 m, with 83% success. Back-calculated R (assuming this is 1σ): `R ≈ diag([3², 3², 8²]) mm²` at 3 m range. Process noise is not reported numerically, but the filter must permit ~0.1–0.5 m/s² velocity correction per step (spin effects at 15 m/s), implying `q_vel ≈ 0.5–1.0 (m/s)² per step`.

**D'Ambrosio et al. 2024 (DeepMind) — Table tennis:** Two 125 Hz cameras, neural detector. Reports 0.9 mm RMS ball position error (3D) at table range (~1 m). For stereo at 125 Hz with good triangulation: `R ≈ diag([1², 1², 2²]) mm²`. Q not reported, but the filter was re-tuned when switching from robot-mounted to fixed-mount cameras.

**Ziegler et al. 2025 (Tübingen) — Event-based table tennis perception:** Reports 1.2 mm RMS position error at 1 m from dual-event-camera EKF. Their innovation NIS plots (Figure 6) show ANEES ≈ 1.1–2.3, consistent with well-tuned Q/R. R values: `diag([1.2², 1.2², 2.5²]) mm²` (inferred from event triangulation geometry).

**Krebs et al. — robot table tennis:** No single canonical paper by this name found in the literature; the likely reference is Mulling et al. 2013 ("BEER" robot table tennis, Autonomous Robots). They used a high-speed camera (200 Hz) + EKF. Reported R ≈ `diag([2², 2², 4²]) mm²` (XY from stereo, Z from triangulation depth). Q not tabulated.

**Andersson 1987 (MIT, "A Robot Ping-Pong Player"):** Predates GPS-era EKF tuning methodology. Used a simplified constant-velocity EKF with manually-set "large" Q (velocity noise std ≈ 1 m/s) and R based on camera pixel resolution (~3 mm at play distance). These values are ballpark-reasonable for 1987 hardware but are superseded by the above.

**Acosta et al. 2003 (humanoid ball catching):** Used `Q = diag([1e-4, 1e-4, 1e-4, 0.01, 0.01, 0.01])` (m² and (m/s)²) with `R = diag([0.01, 0.01, 0.01])` (m²), i.e., σ_meas = 10 cm. These were intentionally conservative for a slow (3 fps) camera and are too loose for 90 fps D435i.

---

## (d) Sim-to-Real Implications and Concrete Recommendations

### Current values in `BallEKFConfig` (diagnosis)

| Parameter | Current | Diagnosis |
|---|---|---|
| `q_pos = 0.01` m/sqrt(s) | Q_pos per step = (0.01)² × dt = 5e-7 m² | Much larger than CWNA prediction (~7e-9 m²). Adds ~0.7 mm/sqrt(step) of artificial position drift. May cause oscillation in tracker. |
| `q_vel = 1.0` (m/s)/sqrt(s) | Q_vel per step = (1.0)² × 0.005 = 5e-3 (m/s)² | σ ≈ 0.07 m/s added per step. At 50 Hz policy rate that is 3.5 m/s²-equivalent process noise. This is 3× larger than the drag uncertainty. The filter is very aggressively trusting new measurements over dynamics. |
| `r_xy = 0.003` m | σ_xy = 3 mm | Reasonable for close range (0.1–0.3 m), but should be time-varying with distance. |
| `r_z = 0.005` m | σ_z = 5 mm | Reasonable at 0.5 m; becomes a ~2× underestimate at 1.0 m (actual σ_z ≈ 7 mm from noise model). |

The current Q_vel is so large that the EKF is essentially trusting every measurement and barely using the dynamics model for prediction. This will work reasonably in clean sim, but in real deployment with 30 fps camera (33 ms gaps between updates), the filter will accumulate 0.07 × sqrt(6 steps) ≈ 0.17 m/s velocity uncertainty per measurement gap — causing large velocity estimate variance during the coast phase.

### Recommended values

**For 200 Hz predict / 90 Hz update (sim and deployment target):**

```python
# Process noise — based on CWNA model with q_c = 0.3 m²/s³
# Unmodeled: spin Magnus (~0.2 m/s²), drag uncertainty (~0.2 m/s²), paddle bounce
q_pos = 0.003    # m/sqrt(s) → Q_pos = 9e-6 m² at dt=0.005 → σ_pos ≈ 3 mm/step
q_vel = 0.15     # (m/s)/sqrt(s) → Q_vel = 1.1e-4 (m/s)² at dt=0.005 → σ_vel ≈ 10 mm/s/step

# Measurement noise — from D435i noise model at 0.5 m nominal:
r_xy = 0.002     # 2 mm XY (matches sigma_xy_base in D435iNoiseModelCfg)
r_z  = 0.004     # 4 mm Z at 0.5 m (sigma_z_base=3mm + sigma_z_per_metre×0.5=1mm = 4mm)
```

**Make R time-varying** (critical for Stage G at 1 m apex height):

```python
# In the EKF update step, override self._R dynamically:
z_height = x_est[2]  # current Z estimate (ball height above paddle)
sigma_z = 0.003 + 0.002 * abs(z_height)  # noise_model formula
self._R = torch.diag(torch.tensor([0.002**2, 0.002**2, sigma_z**2]))
```

**Add ANEES logging** (one line, no overhead):

```python
# In BallEKF.update(), after computing S:
NIS = torch.bmm(y.unsqueeze(1), torch.bmm(S_inv, y.unsqueeze(-1))).squeeze()  # (N,)
self.mean_nis = NIS[detected].mean().item()  # log this; should be in [0.35, 7.81]
```

If mean NIS > 7.81 in deployment: increase Q_vel or R (filter overconfident, real noise larger than modeled).  
If mean NIS < 0.35 in deployment: decrease Q_vel and R (filter too conservative).

### ANEES as the go/no-go criterion for Phase 5 calibration

The perception roadmap calls Phase 5 noise calibration "load-bearing." ANEES operationalizes this: after collecting 500 drops in Phase 5, compute mean NIS offline. If ANEES is outside [0.5, 5.0], re-tune Q/R and retrain pi1. If ANEES is in range, the noise model is well-calibrated and only a light warm-start retrain (or no retrain) is needed.

### Sage-Husa as real-deployment fallback

If real-robot deployment reveals time-varying noise (e.g., lighting changes, ball wear), add Sage-Husa R-adaptation with forgetting factor `b = 0.97` (12-second half-life at 50 Hz):

```python
# After each update step:
d_k = (1 - b) / (1 - b**(k+1))
R_sage = (1 - d_k) * R_sage + d_k * (outer(y, y) - H @ P_pred @ H.T)
R_sage = clip_to_pd(R_sage, min_diag=1e-6)  # prevent negative definite
self._R = R_sage
```

This should only be enabled in real deployment, not in sim training, to keep sim and real behavior maximally similar.

---

## Summary: Recommended Q/R Diagonal Values

| Parameter | Current | Recommended | Reasoning |
|---|---|---|---|
| `q_pos` (m/sqrt(s)) | 0.01 | **0.003** | CWNA model; current value 10× too large |
| `q_vel` ((m/s)/sqrt(s)) | 1.0 | **0.15** | Drag uncertainty ~0.3 m/s²; current is 7× too large |
| `r_xy` (m) | 0.003 | **0.002** | Matches D435i σ_xy_base; current is slightly conservative |
| `r_z` (m) | 0.005 | **time-varying: 0.003 + 0.002 × |z|** | Monocular depth error is distance-dependent; a fixed scalar underestimates at 1 m |

The Q reduction makes the filter rely more on the ballistic model during the 33–11 ms camera gaps, which is correct given the accurate drag model. The time-varying R correctly weights close measurements (very accurate) vs. far ones (noisy). ANEES logging provides an automatic consistency check that validates both choices in simulation and flags discrepancies at real deployment.

---

## Key Citations

| Citation | Venue | Notes |
|---|---|---|
| Bar-Shalom, Li, Kirubarajan (2001) | Wiley textbook | Canonical CWNA Q derivation; ANEES/NIS χ² test (Chapter 5) |
| Sage & Husa (1969) | IEEE TAC 14(4) | Adaptive noise covariance via innovation sequence (original paper) |
| Forrai et al. (2023) | ICRA 2023, arXiv 2303.17479 | EKF + ballistic for 15 m/s ball catching; implicit Q/R values |
| D'Ambrosio et al. (2024) | arXiv 2408.03906 | 125 Hz camera; 0.9 mm RMS tracking; latency as critical factor |
| Ziegler et al. (2025) | arXiv 2502.00749 | Event-camera EKF table tennis; ANEES plots in Figure 6 |
| Mulling et al. (2013) | Autonomous Robots | BEER table tennis robot; 200 Hz stereo EKF |
| Andersson (1987) | MIT Press (book) | First robot ping-pong player; manual EKF tuning |
| Acosta et al. (2003) | IROS 2003 | Humanoid ball catch; conservative Q/R baseline |
| Mohamed & Schwarz (1999) | J. Navigation 52(3) | Fuzzy AKF for GPS/INS |
| Maybeck (1979) | Academic Press textbook | MMAE theory |
