# Pan trigram candidate refinement

## Updated empirical status

The developmental chimpanzee corpus changes one earlier conclusion.

For the semantically order-sensitive pair `HO_PH <-> PH_HO`:
- `HO_PH` has robust adult embedding evidence from the 2022 sequence study, including `HO_PH_HO`, `HO_PH_PB`, and `HO_PH_PS`.
- The reverse `PH_HO` orientation also occurs inside longer developmental sequences, including `PH_HO_PH_PB` and as a tail substring in `HO_PH_PB_PH_HO`.

Therefore bidirectional embedding is supported at the **occurrence level** for this order-sensitive pair. What remains unestablished is robust population-level embedding of the reverse `PH_HO` orientation.

For `GR_PG <-> PG_GR`:
- both orientations are robustly represented as embedded bigrams in the 2022 adult analysis;
- multiple longer developmental sequences contain these substrings;
- however the 2025 meaning-expansion study did not identify this reversed pair as one of the clearest semantic ordering-effect pairs.

## Complementary natural candidates

1. `HO_PH <-> PH_HO`: strongest semantic-order candidate, asymmetric robustness of embedding.
2. `GR_PG <-> PG_GR`: strongest structural bidirectional-embedding candidate, weaker standalone semantic-order evidence.

## Decisive missing object

No current public row-level table joins exact trigram identity to event/context. The highest-value estimand is therefore

`P(Y | HO_PH_X) vs P(Y | PH_HO_X)`

for matched third calls `X`, followed by an analogous `GR_PG_X` vs `PG_GR_X` test.

## Verdict

`BIDIRECTIONAL_EMBEDDING_OCCURRENCE_LEVEL_SUPPORTED_FOR_ORDER_SENSITIVE_HO_PH_PAIR__ROBUST_BIDIRECTIONAL_EMBEDDING_STILL_UNRESOLVED_FOR_THAT_PAIR__GR_PG_PAIR_STRUCTURALLY_ROBUST_BUT_SEMANTIC_ORDER_EFFECT_WEAK__TRIGRAM_CONTEXT_TABLE_REMAINS_DECISIVE_MISSING_OBJECT`
