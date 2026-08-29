# Wild Pan structure-from-content result

## Search target

Find existing chimpanzee or bonobo evidence where the same semantic/call material occurs under different structural organization, ideally approaching a bracket-sensitive contrast such as `[AB]C` versus `A[BC]`.

## Chimpanzee reversed bigrams

The 2025 Taï meaning dataset contains four call-pair reversals:

- `GR_HO` vs `HO_GR`
- `GR_PG` vs `PG_GR`
- `HO_PH` vs `PH_HO`
- `PG_PN` vs `PN_PG`

The paper identifies clear semantic/event-distribution ordering effects for:

- `GR_HO <-> HO_GR`
- `HO_PH <-> PH_HO`

A direct reanalysis of the public 2025 context table gives Jensen-Shannon context divergences:

- `GR_HO <-> HO_GR`: 0.1588 bits
- `GR_PG <-> PG_GR`: 0.0951 bits
- `HO_PH <-> PH_HO`: 0.1613 bits
- `PG_PN <-> PN_PG`: 0.1258 bits

## Cross-check against 2022 trigram embedding

Robust frequent bigrams embedded within trigrams in the 2022 structural study include:

`GR_PG, HO_PH, PH_PB, PG_GR, PH_PS, PG_PB, PB_PH, PG_PN, PB_PS, PS_SC, SC_PS, PB_BK, PS_PB`.

Crossing the two studies gives:

- `HO_PH <-> PH_HO`: clear semantic order effect, but only `HO_PH` is in the robust embedded set.
- `GR_HO <-> HO_GR`: clear semantic order effect, but neither orientation is in the robust embedded set.
- `GR_PG <-> PG_GR`: both orientations are robustly embedded, but this pair was not one of the two clear semantic order-effect pairs.
- `PG_PN <-> PN_PG`: only `PG_PN` robustly embedded; no clear semantic order effect.

Therefore:

`N(clear order-effect pairs AND both orientations robustly embedded) = 0`.

## Bonobo same-multiset playback controls

The 2011 bonobo food-playback table contains several natural same-multiset/different-order four-call stimuli:

- `P P PY P` vs `PY P P P`: both produced to kiwi.
- `Y P Y Y` vs `Y Y Y P`: both produced to apple.
- `PY PY Y Y` vs `PY Y PY Y`: both produced to apple.

These are descriptive rather than experimentally controlled permutation tests, but they show that coarse food-of-production labels can remain stable under reordering for some four-call sequences.

## Other wild Pan evidence

- 2023 chimpanzee `alarm-huu + waa-bark` playback establishes compositional-like processing of an artificial combination versus its components, but the paper explicitly proposes reversal as future work; no reversal condition was included.
- 2022 chimpanzee greeting-hoots show population-specific `PH+PG` versus `PG+PH` ordering, but no receiver-semantic playback contrast.
- 2021 wild bonobo gesture work reports no effect of sequence presence/position on gesture meaning; situational context did matter.
- 2025 bonobo vocal compositionality establishes restricted and nontrivial two-call composition, but no >=3-unit bracket contrast.

## Identification ladder

- Same components in different orders: **SUPPORTED**.
- Order-sensitive semantics at bigram level: **SUPPORTED**.
- Structural embedding of bigrams into trigrams: **SUPPORTED**.
- Semantically order-sensitive bigram embedded in at least one orientation: **SUPPORTED** (`HO_PH`).
- Both orientations of a semantically order-sensitive pair robustly embedded: **NOT ESTABLISHED**.
- Meaning of embedded `AB` preserved/transformed inside `ABC`: **UNIDENTIFIED**.
- Bracket-sensitive contrast `[AB]C` vs `A[BC]`: **UNRESOLVED**.

## Closest natural target

`HO_PH` is the strongest existing candidate because:

1. `HO_PH` and `PH_HO` have a clear order-associated semantic/context difference;
2. `HO_PH` is robustly reused inside trigrams (`HO_PH_HO`, `HO_PH_PB`, `HO_PH_PS` among high-support examples);
3. the missing datum is the context/meaning distribution of `HO_PH_X` and ideally matched `PH_HO_X` sequences.

A secondary target is `GR_PG <-> PG_GR`, because both orientations are robustly embedded, although no clear bigram-level semantic order effect was identified in 2025.

## Frozen verdict

`PAN_LINEAR_STRUCTURE_FROM_CONTENT_SUPPORTED__ORDER_SENSITIVE_SEMANTICS_SUPPORTED__STRUCTURAL_EMBEDDING_SUPPORTED__ORDER_SENSITIVE_BIDIRECTIONAL_EMBEDDING_NOT_ESTABLISHED__BRACKET_SENSITIVE_SEMANTICS_UNRESOLVED`
