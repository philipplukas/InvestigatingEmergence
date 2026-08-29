# Chimpanzee semantic embedding: public-data identification result

## Frozen verdict

`COARSE_SEQUENCE_EVENT_COUPLING_SUPPORTED`

`TYPE_LEVEL_SEMANTIC_STRUCTURAL_COLOCATION_SUPPORTED`

`UNDERLYING_JOINT_UTTERANCE_CONTEXT_DATA_EXIST`

`PUBLIC_RELEASES_PARTITION_AWAY_FIELDS_NEEDED_FOR_GAMMA_EMBED`

`SEMANTIC_EMBEDDING_REMAINS_UNIDENTIFIED`

## Public coarse coupling

The public 2023 concomitant-events dataset has 9,391 utterance rows from 98 callers. It retains a binary vocal-sequence indicator and a binary single-vs-concomitant-event indicator, plus caller/group/season/age, but not exact sequence strings or exact event identities.

Raw counts imply:

- P(sequence | single event) = 0.2170088
- P(sequence | concomitant event) = 0.4394619
- risk ratio = 2.0250879
- caller-cluster bootstrap 95% interval for risk ratio = [1.8384, 2.2295]
- raw odds ratio = 2.8287568
- caller-cluster bootstrap 95% interval for raw OR = [2.4276, 3.2946]
- Mantel-Haenszel OR across caller × season strata = 2.7298115
- I(sequence; event-complexity) = 0.030817 bits

This supports a coarse relation between event complexity and structural sequence use. It does not establish semantic composition or semantic preservation under embedding.

## Data-lineage audit

### 2022 Communications Biology structural study

Exact trigram types and embedded-bigram frequencies are publicly visible at figure/supplement level. The paper states that row-level source data are available from the corresponding author on reasonable request. No public row-level trigram × event table was identified.

### 2023 Developmental Science ontogeny deposit (Figshare 19336853)

The public data file contains 144 caller-season aggregate rows with columns:

`caller.id, season, n.utterances, max.combi.length, n.pant.utt, n.adj.pants, n.unique.utterances, age, sex, rank`

Thus exact utterance strings and behavioral context have been aggregated away.

### 2023 iScience concomitant-events deposit (Figshare 22214188)

The public 9,391-row table contains:

`caller identity, vocal sequence y/n, combined event y/n, group, field season, age`

Thus it retains utterance-level event complexity but not exact utterance identity or exact event identity.

### 2024 Bortolato thesis

The thesis documents that detailed behavioral events were collected for vocal utterances and separately analyzes exact bigram/trigram recombination. This strongly supports existence of richer underlying observational data, but no public master table containing both projections was identified in a broad Figshare author/project search.

### 2025 meaning-expansion dataset

The public GitHub `calldata.csv` provides exact single/bigram call identity, context, caller, and date for 4,323 utterances, but the study is restricted to singles and bigrams and therefore has no trigram rows.

## Identification ladder

- Gamma_coarse: event complexity -> sequence-vs-single production: **SUPPORTED**.
- Gamma_type: semantically characterized AB types reused structurally inside ABC: **SUPPORTED** from the previous type-overlap audit.
- Gamma_embed: semantic/context contribution of AB when AB appears inside ABC: **UNIDENTIFIED FROM CURRENT PUBLIC DATA**.
- Gamma_bracket: semantic distinction between [AB]C and A[BC]: **UNRESOLVED**.
- Gamma_productive: novel-structure/depth semantic generalization: **UNRESOLVED**.

## Minimal data needed for Gamma_embed

A row-level table containing at least:

`caller, date/recording, utterance_string, A, B, C, event/context`

with sufficient standalone AB and embedded ABC observations from the same call families.

The strongest immediate target is therefore obtaining/recovering the existing joint annotation table, not designing a new primate experiment.
