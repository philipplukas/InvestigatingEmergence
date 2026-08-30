# Chimp recording-level bridge audit result

## Target
Recover occurrence-level identifiers for matched embedded candidate families, especially `HO_PH_{GR,PG,PN}` versus `PH_HO_{GR,PG,PN}`, so exact long-sequence occurrences can be joined to event/context labels without requesting new data.

## Sources audited

1. 2022 structural sequence supplement: rich exact sequence-type repertoire, but no occurrence-level date/caller/context/recording key.
2. 2025 meaning dataset (`tozbu/Chimpanzee_bigram_meaning/calldata.csv`): 4,322 standalone single-call/bigram rows with date, caller, call.name and event context, but no longer-sequence identity or third-call field.
3. Bortolato et al. ontogeny Figshare article 19336853, file 36254292 (`Development of vocal sequences in wild chimpanzees - Data.xls`).

## Figshare bridge audit

The Figshare release contains one sheet with 144 rows and columns:

`caller.id, season, n.utterances, max.combi.length, n.pant.utt, n.adj.pants, n.unique.utterances, age, sex, rank`.

Thus it is caller-season aggregate data, not an occurrence-level utterance table. It contains no exact sequence string, date, time, recording/file/audio ID, context/event, or utterance identifier. Searches for the six matched targets

- `HO_PH_GR`
- `PH_HO_GR`
- `HO_PH_PG`
- `PH_HO_PG`
- `HO_PH_PN`
- `PH_HO_PN`

returned zero exact hits.

## Identification consequence

The ontogeny release confirms overlap at the study/caller-season level but cannot bridge an exact structural occurrence to a semantic event. Caller-season aggregation is many-to-one and does not permit reconstruction of which unique sequence type occurred in which event.

Therefore the embedded semantic distributions remain nonparametrically unidentified from currently recovered public releases. The sharp no-assumption bounds remain the full simplex product, and TV/JS divergence bounds remain `[0,1]`.

## Remaining public-data target

The useful missing object is now specifically an occurrence-level recording/annotation catalogue preserving at least one of:

- exact long sequence + date + caller;
- exact long sequence + recording/file ID;
- recording/file ID + event/context;
- an unaggregated utterance annotation table from the 2019/2020 Taï corpus.

Another caller-season aggregate table cannot solve the join.

## Frozen verdict

`FIGSHARE_ONTOGENY_RELEASE_AGGREGATED__NO_EXACT_SEQUENCE_OR_RECORDING_KEY__NO_MATCHED_TARGET_HITS__RECORDING_LEVEL_BRIDGE_NOT_RECOVERED__EMBEDDED_SEMANTIC_EFFECT_REMAINS_UNIDENTIFIED`
