# Four-scheme independent code audit

Reviewed runner, models.py, layers_transformer.py, run_models.py and complete tracked diff on 2026-11-11 Asia/Shanghai. No repository code modified by this reviewer.

## Confirmed
- Scheme 1: temporal/spatial/productivity each independent history Transformer.
- Scheme 2: temporal/spatial Transformer; KappaNetwork 5 hidden layers x 32, tanh, NonNeg, softplus output.
- Scheme 3: temporal Transformer; spatial and Kappa original FCNN.
- Scheme 4: same temporal/spatial as 2; Kappa unconstrained GELU MLP 4 hidden layers x 64, softplus output. It is WIDER, not deeper than the 5-hidden-layer FCNN; both are multilayer perceptrons. Thus it is not a depth-only ablation.
- Transformer d_model=32, 8 heads, one layer, FFN128, pre-norm, GELU, dropout0, math SDPA; separate encoders per component; four mixture components.
- Fixed history context: age at previous event and magnitude; query t/r absent. Both integration endpoints use identical mixture parameters and H tokens (newest lower CDF is exactly zero).
- Temporal Omori CDF sum w*(1-(c/(u+c))**(p-1)); spatial ETAS radial CDF sum w*(1-(D/(u*u+D))**(q-1)); softmax weights, positive scales and exponents>1.
- Density in model forward is autograd derivative of CDF, with spatial 1/(2*pi*r) and second derivative r=0 limit. A closed-form plane-density helper exists but is not used in the likelihood forward.
- Upgraded schemes use lr1e-4 and mu0=train target count/(sum interevent intervals * area), here 6.251669e-4. Mu excluded from Adam; analytic stochastic-declustering update after each epoch. Uniform over space, variable across epochs.
- Finite SanJac polygon mass multiplies triggered compensator, using 128-angle polar quadrature. Full-plane intensity is not renormalized onto the polygon.
- NLL is unchanged: -mean(log(lambda_ts)-compensator). No PDF regression or auxiliary loss.
- Official SanJac jitter catalog, no extra training-time jitter; identical 11770/3449/4400 targets and H128 for every scheme.
- Default all-FCNN/ETAS retains lr0.005, mu0=1, no boundary correction; original default behavior is preserved.

## Important reporting caveats
1. fit_eval restores best snapshot ONLY on early stopping. At the 20-epoch budget it evaluates LAST epoch, even if best_epoch differs. Runner comments at 239-244 do NOT implement an unconditional best-weight restore. State explicitly which checkpoint is evaluated. Validation loss used for selection is before the analytic mu update; saved state is after mu update, following upstream callback order.
2. With boundary correction, l_t in models.py:568-570 still lacks the per-source mass factor while compensator includes it. Consequently LL_ts is the intended finite-region joint NLL estimate, but existing LL_t/LL_s are LEGACY algebraic diagnostic components, not a normalized finite-region time/conditional-space likelihood decomposition. For physical finite-region components, use lambda_t_R=mu*A+sum(kappa*h*mass). Do not conceal this caveat or claim a clean decomposition from the raw JSON.
3. Scheme 3 keeps FCNN spatial CDF, not an ETAS-mixture spatial CDF (required by its FCNN ablation). Boundary correction and mu/lr upgrades still apply. Original spatial epsilon Jacobian is retained in this path.
4. Baselines run by existing runner share official targets, H128 and 20 epoch cap, but retain lr0.005, mu0=1 and no boundary correction. These are default-path references, not fully controlled architecture-only ablations.
5. One seed, limited 20-epoch CPU budget and H128 are preliminary, not paper-scale converged results. Scheme4 also changes width, activation and weight constraints, not depth alone.

## Validation
pytest was initially absent; installed pytest into existing .venv without changing tracked dependency files. Single-thread nice10 run: 41 passed, 9 subtests passed, 7.43s. Log: four-schemes-pytest.log.
Full original fcnn-etas-fingerprint.json comparison additionally checks all eight forward tensors, scalar loss/mu, and two representative gradient tensors per model, bitwise equal (flattening scalar parameter shape only). Artifact: four-schemes-fingerprint-verification.json.
