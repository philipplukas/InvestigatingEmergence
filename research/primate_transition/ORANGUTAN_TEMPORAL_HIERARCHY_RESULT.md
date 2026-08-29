# Orangutan temporal hierarchy reanalysis — frozen result

## Scope

Target capability: **temporal hierarchical organization only**. Do not infer syntax, semantics, phonology, open-ended productivity, or human-scale recursive grammar from these data.

Primary public source: OSF project `w3ne5`, current file `ALL ELEMENTS Index_2 share.xlsx`.

## Public-data audit

Current OSF release exposes one tabular source with:

- 3,599 raw rows; 3,595 after whitespace/label cleanup;
- 19 distinct `Master File Name` values;
- 7 distinct `Individual` labels.

The version-of-record paper reports 8,993 vocal elements from 66 long-call recordings. Therefore the current public OSF table is treated as a **released subset**, not as the full published sample.

Frozen reproducibility label:

`PUBLIC_RAW_RELEASE_INCOMPLETE_RELATIVE_TO_PUBLISHED_SAMPLE`

## Two-stratum access

Direct interval-containment audit on the released table:

- 19/19 released calls contain lower-level elements temporally nested inside full-pulse intervals;
- 7/7 released individual labels show nested two-stratum organization;
- 94.811% of lower-level annotations begin inside a full-pulse interval;
- 63.210% of released full-pulse annotations contain at least one lower-level element;
- individual-cluster bootstrap 95% interval for the latter: [0.538, 0.734].

Frozen label:

`ORANGUTAN_TWO_STRATUM_TEMPORAL_HIERARCHY_SUPPORTED`

## Rhythm reconstruction from onset times

Inter-onset intervals were reconstructed from consecutive onsets, filtered with the paper's `0.025 < t_k < 5 s` gate. For lower levels, IOIs were calculated inside containing full-pulse intervals.

Reconstructed KDE modes of

`r_k = t_k / (t_k + t_{k+1})`

are:

- Full pulse: 0.5011
- Grumble sub-pulse: 0.5004
- Sub-pulse transitory element: 0.4951
- Bubble sub-pulse: 0.5026

These independently reproduce the near-0.5 isochrony geometry reported in the article.

Frozen label:

`ISOCHRONY_WITHIN_ISOCHRONY_REPRODUCED_ON_PUBLIC_SUBSET`

## Temporal-scale separation

Per released call, median full-pulse IOI divided by median nested lower-level IOI:

- Grumble: median 16.63x (min 6.78x)
- Transitory: median 14.56x (93.3% of estimable files >5x)
- Bubble: median 15.10x (min 10.36x)

Pooling corresponding nested lower/full intervals gives lower/full IOI ratio mean 0.0603 and median 0.0549.

Thus the nested level occupies a substantially faster temporal scale than the enclosing full-pulse rhythm.

## Identification matrix

- Access to two-stratum temporal nesting: **SUPPORTED on released subset**.
- Isochrony at both strata: **REPRODUCED on released subset**.
- Timescale separation: **QUANTIFIED on released subset**.
- Cross-call/individual stability: **SUPPORTED on released subset**.
- Acquisition / learning cost: **UNIDENTIFIED** (no controlled developmental/training trajectory).
- Generalization to unseen hierarchical depths: **UNRESOLVED**.
- Semantic hierarchy / syntax: **UNRESOLVED and not licensed by this evidence**.

Observed structural depth in this analysis is 2: full-pulse stratum -> sub-pulse stratum.

## Conservative verdict

`ORANGUTAN_TWO_STRATUM_TEMPORAL_HIERARCHY_SUPPORTED__ISOCHRONY_WITHIN_ISOCHRONY_REPRODUCED_ON_PUBLIC_SUBSET__PUBLIC_RAW_RELEASE_INCOMPLETE_RELATIVE_TO_PUBLISHED_SAMPLE__SEMANTIC_HIERARCHY_UNRESOLVED`
