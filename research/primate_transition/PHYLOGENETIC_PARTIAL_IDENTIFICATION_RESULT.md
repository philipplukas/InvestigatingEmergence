# Phylogenetic partial-identification result

## Scope

This is a sensitivity analysis over latent binary capability states on a coarse primate phylogeny. It is **not** a calibrated evolutionary clock and does not treat unresolved extant evidence as absence.

Evidence rule:

- `SUPPORTED` -> strong positive observation likelihood;
- `UNRESOLVED` -> no likelihood preference between latent presence/absence;
- no unresolved cell is encoded as `0`.

Six gain/loss regimes were compared:

1. moderate symmetric gain/loss;
2. slow symmetric / phylogenetic inertia;
3. rare gain + easier loss;
4. strong one-origin / homology-favoring;
5. convergence-friendly;
6. loss-friendly.

## Robust cross-model result

At the approximate human–Pan ancestral node (~6.5 Ma anchor), the **minimum posterior across all six models** is:

- finite sequencing: 0.9513;
- restricted semantic composition: 0.9365;
- plant/factorization precursor: 0.9391;
- learning/plasticity: 0.9513.

Thus these four precursor families receive robust ancestral support across very different evolutionary histories.

Temporal hierarchy does **not** meet the same criterion. At the Pan ancestral node (~1.8 Ma anchor), its cross-model posterior range is broad enough to remain model-sensitive.

Likewise, because the nonhuman cells remain unresolved rather than negative, the following higher-order capabilities remain strongly model-sensitive:

- semantic embedding;
- bracket-sensitive semantic hierarchy;
- novel-depth productive hierarchy.

For each of these three, the tested models yield the same posterior range at the Pan node because they currently share the same extant evidence pattern (human positive reference; nonhuman unresolved).

## Identification lesson

Evolutionary biology constrains **admissible histories**. It cannot manufacture a capability upper bound from an unresolved extant observation.

Therefore:

`PHYLOGENETIC_PRECURSOR_ANCESTRY_CAN_BE_ROBUST`

but

`ADVANCED_LANGUAGE_ANCESTRY_REMAINS_MODEL_SENSITIVE`

and

`EVOLUTIONARY_LAWS_CONSTRAIN_HISTORIES_NOT_MISSING_CAPABILITY_UPPER_BOUNDS`.

## Interpretation

The result shifts the plausible evolutionary frontier upward. Generic finite sequencing, restricted semantic composition, vocal-plant/factorization precursors and learning/plasticity are all compatible with—and under this sensitivity set robustly favor—presence near the human–Pan ancestral level. The unresolved frontier is instead the coupling/integration ladder:

`semantic embedding -> bracket-sensitive semantic hierarchy -> productive novel-depth generalization`.

Exact divergence-time uncertainty, trait-specific transition rates and explicit correlated-trait CTMCs should be layered in next; the present result is deliberately a robust sensitivity bound rather than a precise date estimate.
