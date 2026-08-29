# Chimpanzee semantic–structure coupling result

## Frozen verdict

`CHIMP_SAME_SPECIES_SEMANTIC_STRUCTURAL_COLOCATION_SUPPORTED`

`ALL_16_MEANING_TESTED_BIGRAM_TYPES_STRUCTURALLY_EMBEDDED`

`PREFERENTIAL_SEMANTIC_EMBEDDING_NOT_IDENTIFIED_AFTER_FREQUENCY_CONFOUND`

`SEMANTIC_PRESERVATION_UNDER_EMBEDDING_UNRESOLVED`

## Type-level join

The 2022 Taï structural source-data summaries contain 58 unique bigrams and 104 unique trigrams. The 2025 Taï meaning-expansion study analyzes 16 common bigrams. Joining the public type labels shows that all 16/16 meaning-tested bigram types occur contiguously inside at least one unique trigram. Across all 58 structural bigrams, 47/58 occur in at least one trigram.

The semantic-tested bigrams have between 2 and 14 distinct trigram extensions and occur in longer unique sequence types as well. Examples include PH_PB (14 unique trigram extensions), GR_PG (11), PH_PG (8), PH_PS (8), and HO_PH (7).

## Frequency confound

The 2025 semantic analysis deliberately selected bigrams produced by at least 10 chimpanzees. These are overwhelmingly high-frequency bigrams in the 2022 structural corpus. Bigram token frequency strongly predicts the number of distinct structural extensions (Spearman rho = 0.7497 for unique trigram extensions; rho = 0.6826 for all unique length>=3 extensions). Therefore the raw 16/16 embedding result is a lower bound on structural opportunity, not evidence that semantic bigrams are preferentially embedded because of their semantics.

A descriptive frequency-adjusted regression is not given a causal interpretation because semantic-test selection and frequency are strongly confounded.

## Semantic/structural decoupling falsifier

The two 2025 bigrams classified as showing no change in meaning, GR_PG and PH_PS, are among the most structurally embedded: 11 and 8 unique trigram extensions respectively. Among the 16 meaning-tested bigrams, the mean number of trigram extensions is actually lower for the 14 meaning-changing bigrams than for the two no-change bigrams (difference = -4.571). An exact permutation test for the directional hypothesis that meaning-changing bigrams are more embedded gives p = 0.9583. The corresponding difference for all unique sequences of length >=3 is -30.714, directional p = 0.95.

Thus structural embedding alone is not evidence that semantic contribution is preserved or compositionally transformed at the higher sequence level.

## Coupling ladder

- Gamma0 species co-location: SUPPORTED.
- Gamma1 meaning-tested type structurally embedded: STRONGLY SUPPORTED at type level.
- Gamma1b same-identified-individual co-location: UNIDENTIFIED from the released 2022 type-level source data.
- Gamma2 predictable semantic contribution under embedding: UNRESOLVED.
- Gamma3 generalization to unseen embeddings: UNRESOLVED.
- Gamma4 productive recursive semantic hierarchy: UNRESOLVED.

The high-value missing coordinate is therefore `Gamma_sem|embed`: whether a meaning-bearing bigram makes a predictable semantic contribution when it is embedded in a longer structured sequence.

## Comparative-boundary correction

The 2025 bonobo semantic-composition result should no longer be used as an uncontested semantic anchor. Wartel et al. (PeerJ, 2026) report high false-positive rates for the original MCA-based pipeline under randomized data and a dependence-aware full-permutation Monte Carlo p = 0.2597. Earlier caller/MCA robustness checks in this branch did not implement that null and therefore do not resolve the critique.

Chimpanzees provide the stronger current semantic anchor: the 2025 Taï event-distribution analysis reports multiple meaning-expansion mechanisms, and independent 2023 Budongo playback work supports a meaningful alarm-huu + waa-bark combination. However, neither establishes semantic composition of longer embedded Taï trigrams.

The earlier `restricted semantics + temporal hierarchy` separator may be retained only as a strict evidence-level separator under a strong definition of temporal hierarchy. Mechanistically, the sharper frontier is now cross-level semantic preservation/productivity under structural embedding, not simple same-species co-location of semantics and sequence structure.
