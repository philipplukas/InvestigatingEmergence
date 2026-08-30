# Chimp concomitant-event bridge audit

## Target
Recover occurrence-level event/context labels for matched embedded sequence families, especially `HO_PH_{GR,PG,PN}` versus `PH_HO_{GR,PG,PN}`.

## 2023 concomitant-events dataset
Bortolato et al. (iScience 2023) analysed 9,391 utterances from 98 Taï chimpanzees and explicitly coded whether utterances were vocal sequences and whether events were single or combined. Public Figshare article 22214188 contains four files.

The utterance-level public Dataset S1 has 9,391 rows but only these variables:
- caller identity
- vocal sequence y/n (binary)
- combined event y/n (binary)
- group
- field season
- age

It does **not** contain the exact utterance/call sequence, event identity, date/time, recording ID, or audio filename.

Dataset S2 is caller-season aggregated and likewise cannot recover exact utterances.

An exhaustive exact-string search for `HO_PH_GR`, `PH_HO_GR`, `HO_PH_PG`, `PH_HO_PG`, `HO_PH_PN`, `PH_HO_PN` returned zero hits. A component-wise candidate search also returned zero rows because call identities are absent from the public analysis tables.

## Identification consequence
The 2023 deposit confirms that the underlying research corpus did pair vocal utterances with production events, but the public release projects the row-level data onto binary sequence/event indicators before deposit. It therefore cannot bridge the exact structural strings from the 2022 sequence supplement to semantic event labels.

## Verdict
`UNDERLYING_UTTERANCE_EVENT_JOIN_CONFIRMED_TO_EXIST_IN_STUDY__PUBLIC_2023_RELEASE_PROJECTS_AWAY_EXACT_CALL_IDENTITY_AND_EVENT_IDENTITY__NO_RECORDING_LEVEL_BRIDGE_RECOVERED__EMBEDDED_SEMANTIC_EFFECT_REMAINS_UNIDENTIFIED`
