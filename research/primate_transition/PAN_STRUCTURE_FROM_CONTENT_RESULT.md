# Wild Pan structure-from-content result

## Updated identification target

The current natural target is semantic order under embedding, not hierarchy simpliciter. For the semantically order-sensitive chimpanzee pair `HO_PH <-> PH_HO`, the 2022 adult structural supplement contains both orientations inside longer sequences and, importantly, matched continuation environments.

## Matched embedded environments

Across the adult exact-sequence inventory, `HO_PH` and `PH_HO` share following-call environments:

- `GR`
- `PG`
- `PN`

Thus exact structural counterparts of `HO_PH_X` and `PH_HO_X` exist for three matched `X` values.

The pair also shares preceding-call environments `GR`, `PB`, `PS`, and `WH`.

For the structurally stronger pair `GR_PG <-> PG_GR`, shared following-call environments are `BK`, `HO`, `PB`, `PH`, `PN`, and `SC`, with shared predecessors `HO`, `PB`, `PH`, and `PN`.

Therefore the structural matching problem is solved at the sequence-type level.

## 2025 standalone semantic data

The public repository `tozbu/Chimpanzee_bigram_meaning` contains 4,322 raw rows with columns `date`, `caller`, `call.name`, `combi.length`, `context`, `context.length`, and `demosubset`. It covers standalone single calls and bigrams, not the longer embedded sequences.

For the two candidate pairs, direct event-token reanalysis gives:

- `HO_PH`: 136 utterances, 41 callers, 194 event tokens.
- `PH_HO`: 23 utterances, 19 callers, 35 event tokens.
- `GR_PG`: 124 utterances, 38 callers, 188 event tokens.
- `PG_GR`: 27 utterances, 19 callers, 52 event tokens.

Standalone semantic divergence:

- `HO_PH <-> PH_HO`: JS = 0.161311 bits; TV = 0.330486.
- `GR_PG <-> PG_GR`: JS = 0.095108 bits; TV = 0.218085.

## Public-data audit

The 2022 public supplement contains rich exact sequence inventories, but no occurrence-level `context`, `event`, `caller`, or `date` fields that could link an exact embedded sequence occurrence to a semantic label.

The 2025 public raw file contains `date`, `caller`, and `context`, but only for standalone single calls and bigrams. Its analysis code explicitly models those standalone calls and splits dyadic calls only into `call1` and `call2`; there is no hidden third-call or longer-sequence field.

A repository-history audit found no earlier richer version of `calldata.csv`, and public code search for exact matched trigram strings did not expose an alternative row-level semantic table.

## Partial-identification result

For a matched continuation `X`, define

`Gamma_X = D(P(Y | HO_PH_X), P(Y | PH_HO_X))`.

Without an occurrence-level join between the structural sequence and event label, the two embedded context distributions are unrestricted points of the 22-event simplex under a nonparametric model.

Hence the identified set is the full product simplex, giving only trivial bounds:

- TV(`P1`,`P2`) in `[0,1]`
- JS_bits(`P1`,`P2`) in `[0,1]`

for every matched embedded environment.

So the public data are sufficient to establish:

- standalone semantic order effects;
- bidirectional occurrence-level embedding;
- existence of matched `ABX` / `BAX` structural environments;

but not semantic preservation or transformation under embedding.

## Updated identification ladder

- `E0`: `AB` and `BA` occur standalone — **SUPPORTED**.
- `E1`: both orientations occur embedded — **SUPPORTED at occurrence level**.
- `E2`: matched `ABX` and `BAX` structural environments exist — **SUPPORTED**.
- `E3`: standalone `AB` and `BA` differ semantically — **SUPPORTED for `HO_PH <-> PH_HO`**.
- `E4`: event/context labels available for matched embedded occurrences — **MISSING PUBLICLY**.
- `E5`: semantic value survives/transforms embedding — **UNIDENTIFIED**.
- `E6`: bracket-sensitive `[AB]C` vs `A[BC]` meaning — **UNRESOLVED**.

## Frozen verdict

`STANDALONE_SEMANTIC_ORDER_EFFECT_QUANTIFIED__BIDIRECTIONAL_EMBEDDING_OCCURRENCE_LEVEL_SUPPORTED__MATCHED_EMBEDDED_STRUCTURES_EXIST__PUBLIC_OCCURRENCE_CONTEXT_JOIN_ABSENT__EMBEDDED_SEMANTIC_EFFECT_HAS_ONLY_TRIVIAL_NONPARAMETRIC_BOUNDS__BRACKET_SENSITIVE_SEMANTICS_UNRESOLVED`
