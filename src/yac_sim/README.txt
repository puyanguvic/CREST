YAC / SCC simulation code (paper-aligned)

Run:
  python -m yac_scc --outdir result

Key updates vs previous:
- Bounded disturbances/noise (w_bar, v_bar) consistent with the paper assumptions.
- Observer update in paper form: xhat = xhat^- + L (y - C xhat^-), with configurable L_gain and C (full-state by default).
- Outputs include prediction error norm ||tilde x|| and innovation norm.
- Trade-off curves use delivered packets as the primary communication budget metric.
