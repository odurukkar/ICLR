# Figure Catalog

## `fig_external` — transfer-and-falsification dashboard

Purpose: show why a directionally favorable macro result still fails confirmation.

Source tables:

- `effects.csv`: policy-minus-MES per-series CSD and exclusion contrasts.
- `city_intervals.csv`: authenticated city-by-seed point estimates and paired-bootstrap limits.
- `diagnostics.json`: frozen primary seed and mechanical `falsified` classification.

Panel estimands:

- Panel (a): city mean endowment-minus-MES CSD difference with paired-bootstrap 95% interval.
- Panel (b): per-series endowment-minus-MES CSD difference for each frozen seed; beige is an exact tie and the adjacent strip encodes city.
- Panel (c): primary-seed per-series CSD difference versus exclusion difference; descriptive only.

Registered outputs:

- `figures/fig_external.pdf`: SHA-256 `2476eb4846982a49d7479960dddf2bcf1d391337e2d0acd3501bee5da0ed73a4`.
- `figures/fig_external.png`: SHA-256 `b53af50d3c4a817be5d07ca571efeaba0f372d63eeea6420943eab2d9bd0b20f`.
- Paper copies under `iclr_paper/tex/` are byte-identical to these registered outputs.
