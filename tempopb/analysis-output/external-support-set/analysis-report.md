# External Support-Set Analysis

## 1. Immutable protocol identity and support-set projection

Study: `fresh-city-support-set-confirmation-v3-unicode-amendment`.
Protocol lock SHA-256: `a6d7f945b6f6181f01ad30b9593d9d5f1cdbf1ec4cff977ad71488e76d17595b`.
Katowice projection: cumulative listed-project support; ignore point magnitude.
Krakow projection: ordinal listed-project support; ignore list order.

## 2. Authenticated corpus and demographic coverage

228 authenticated elections; 32 series; 96 scored-election demographic checks.
Observed minimum coverage: 0.9912280701754386.

## 3. City-macro differences and frozen city intervals

- Seed 1 city-macro CSD difference: -0.013137966195055278.
- Seed 2 city-macro CSD difference: -0.005357752651077373.
- Seed 42 city-macro CSD difference: -0.007153800547544324.
- Seed 1, Poland/Katowice: 95% interval [-0.03069432927693753, 0.0016126922938029373]; exact sign-flip p-value 0.25; Holm p-value not applicable.
- Seed 1, Poland/Krakow: 95% interval [-0.040688923656287554, 0.008176038678425564]; exact sign-flip p-value 0.35882568359375; Holm p-value not applicable.
- Seed 2, Poland/Katowice: 95% interval [-0.00803283252667272, 0.0]; exact sign-flip p-value 1.0; Holm p-value not applicable.
- Seed 2, Poland/Krakow: 95% interval [-0.027793162678913012, 0.005117572627737076]; exact sign-flip p-value 0.59375; Holm p-value not applicable.
- Seed 42, Poland/Katowice: 95% interval [-0.028176841645247863, 0.0]; exact sign-flip p-value 0.25; Holm p-value 0.5.
- Seed 42, Poland/Krakow: 95% interval [-0.025908745651570884, 0.0166219908152236]; exact sign-flip p-value 0.83984375; Holm p-value 0.83984375.

## 4. Descriptive series diagnostics

Primary seed 42 decomposition: 8 wins, 20 ties, 4 losses.
Across seeds: 4 always-wins, 13 all-ties, and 5 sign switches.
Top-three improvement concentration: 0.7889575669610045.

## 5. Frozen mechanical gate conditions

Frozen gates: 4 passed and 4 failed.
- endowment_actuation: passed (endowment >= 8 series and endowment > priority).
- exclusion_increase_all_cities_seeds: failed (exclusion increase <= 0.005 for both cities and every seed).
- macro_at_most_minus_0_005_all_seeds: passed (city macro <= -0.005 for every seed).
- macro_seed_spread: failed (city-macro seed spread <= 0.005).
- negative_each_city_all_seeds: passed (difference < 0 for both cities and every seed).
- primary_ci_excludes_zero_each_city: failed (95% interval upper endpoint < 0 in both cities).
- primary_holm_p_each_city: failed (Holm-adjusted p <= 0.05 in both cities).
- welfare_ratio_all_cities_seeds: passed (welfare ratio >= 0.98 for both cities and every seed).

## 6. Mechanical decision

Mechanical decision: falsified.
