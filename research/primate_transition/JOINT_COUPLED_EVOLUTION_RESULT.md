# Joint coupled evolutionary model — frozen result

## State space

Traits:
- `F_seq`: finite sequencing
- `S_sem`: restricted semantic composition
- `T_hier`: temporal hierarchy
- `E_embed`: semantic embedding
- `H_bracket`: bracket-sensitive semantic hierarchy
- `P_novel`: novel-depth productive hierarchy

Hard prerequisite lattice:

- `E_embed => F_seq & S_sem`
- `H_bracket => E_embed & T_hier`
- `P_novel => H_bracket`

Only 12 of the 64 nominal binary trait combinations are admissible under these prerequisites.

## Evidence rule

- `SUPPORTED` contributes a strong positive observation likelihood.
- `UNRESOLVED` contributes no directional likelihood.
- No unresolved nonhuman cell is encoded as absence.

## Evolutionary sensitivity models

Four joint transition models were compared:

- `J0_gradual_prerequisite`: hard prerequisites, mostly incremental gains/losses.
- `J1_global_integration_jump`: simultaneous gains in the advanced `E/H/P` bundle are globally easier.
- `J2_human_branch_integration`: advanced gains are specifically cheaper on the AfricanApe -> Human branch.
- `J3_convergence_friendly`: independent gains are comparatively easy.

The transition-rate choices are sensitivity settings, not fitted/calibrated evolutionary clocks. Relative model weights therefore indicate compatibility under the chosen regimes, not definitive Bayes factors for biological histories.

## Main result

Among the tested sensitivity settings, relative equal-prior weights were approximately:

- `J0_gradual_prerequisite`: 0.018
- `J1_global_integration_jump`: 0.248
- `J2_human_branch_integration`: 0.066
- `J3_convergence_friendly`: 0.668

This does **not** establish convergence as the true history; it shows that the current positive-only/unresolved comparative evidence does not require a uniquely human branch-localized jump.

For the AfricanApe -> Human branch, posterior probability of >=2 advanced gains was strongly model-dependent:

- gradual prerequisite: 0.031
- global integration jump: 0.045
- human-branch integration: 0.553
- convergence-friendly: 0.095

Probability that the full advanced `E+H+P` bundle was acquired on that branch was likewise model-sensitive:

- gradual prerequisite: 0.197
- global integration jump: 0.106
- human-branch integration: 0.783
- convergence-friendly: 0.257

The posterior probability of the full advanced state at the African-ape ancestral node ranged from about 0.215 to 0.891 across models; at the Pan ancestor it ranged from about 0.208 to 0.788. Thus the timing of the advanced bundle is not identified.

## Interpretation

The prerequisite lattice sharply constrains which combinations of capacities are biologically coherent, but it does not identify whether evolution proceeded by:

1. gradual accumulation of latent capacities;
2. one or more globally coupled integration jumps;
3. a final human-branch-localized integration event; or
4. convergence/repeated gains.

The data support precursor accumulation. They do not yet identify the topology or timing of a final language-specific integration event.

## Frozen verdict

`PREREQUISITE_LATTICE_SHARPENS_STATE_SPACE__PRECURSOR_ACCUMULATION_SUPPORTED__FINAL_INTEGRATION_EVENT_NOT_IDENTIFIED__HUMAN_BRANCH_JUMP_REMAINS_MODEL_SENSITIVE`
