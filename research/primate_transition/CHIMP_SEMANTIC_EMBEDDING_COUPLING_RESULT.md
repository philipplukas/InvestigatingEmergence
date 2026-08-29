# Chimpanzee semantic–structural embedding coupling result

## Frozen verdict

- `CHIMP_SEMANTIC_AND_STRUCTURAL_RECOMBINATION_COLOCATED`
- `SAME_BIGRAM_TYPE_SEMANTIC_AND_EMBEDDABLE_SUPPORTED`
- `SEMANTIC_INHERITANCE_UNDER_EMBEDDING_UNIDENTIFIED`
- `HIERARCHICAL_SEMANTIC_COMPOSITION_UNRESOLVED`

## Public-data intersection

The 2022 structural study and the 2025 meaning study are both from the Taï Chimpanzee Project, the same three communities, and largely overlapping collection periods. The public 2025 CSV spans 2019-01-03 to 2020-04-17; the 2022 structural paper reports January–February 2019 and December 2019–March 2020.

The 2025 study semantically/event-characterized 16 bigrams:

`GR_BK, GR_HO, GR_PG, GR_PN, HO_GR, HO_PG, HO_PH, HO_PN, PG_GR, PG_PN, PH_HO, PH_PB, PH_PG, PH_PS, PH_SC, PN_PG`.

Using the public 2022 Supplementary Data 1 sequence-type table:

- 16/16 occur as independent bigram types in the 2022 repertoire.
- 16/16 occur contiguously inside at least one 2022 trigram type.
- each occurs inside 2–14 distinct trigram types.
- each occurs inside 2–81 distinct longer sequence types (length >=3).
- several persist inside sequence types up to length 7–10.

Thus semantic characterization and structural reusability are co-located at the same ordered call-pattern types.

## Strong structural core

The four bigrams explicitly tested in the 2022 head/tail trigram analysis are all among the 2025 semantic bigrams:

- `GR_PG`: 113 trigram tokens in the public 2022 Fig. 5 source table (105 head, 8 tail); 2025 no-change candidate, close to PG.
- `HO_PH`: 94 trigram tokens (94 head, 0 tail); 2025 compositional candidate (mechanisms 2+3) and order-sensitive relative to `PH_HO`.
- `PH_PB`: 87 trigram tokens (62 head, 25 tail); 2025 compositional disambiguation candidate (mechanism 2).
- `PH_PS`: 76 trigram tokens (36 head, 40 tail); 2025 no-change candidate.

Two meaning-changing/compositional candidates (`HO_PH`, `PH_PB`) are therefore robustly reusable structural units in trigrams.

## Seven strong independent bigrams

All seven 2022 bigrams that were above chance and produced by at least 10 individuals are included in the 2025 semantic analysis:

`GR_PG, GR_PN, HO_PG, HO_PH, PH_PB, PH_PS, PN_PG`.

Five of seven are 2025 meaning-change candidates under the paper's mechanism taxonomy (`GR_PN, HO_PG, HO_PH, PH_PB, PN_PG`); `GR_PG` and `PH_PS` are no-change candidates.

## Identification boundary

The public data do **not** identify the key semantic-embedding quantity

`M(AB | embedded in ABX or XAB)`.

The reason is structural:

- the public 2022 supplementary data provide sequence structure/type information but no event/context labels for individual trigram tokens and no full public caller-level source table;
- the public 2025 semantic dataset contains singles and bigrams only and explicitly excludes sequences longer than two call types from the semantic analysis.

Therefore structural reuse of a meaning-bearing bigram does not yet establish that its meaning is preserved, compositionally transformed, or scope-sensitive when embedded.

## Refined missing coupling

Define a missing coupling coordinate `Gamma_sem,embed`, e.g. as the predictive gain for the event/meaning profile of `ABX` or `XAB` obtained by including an independently estimated semantic representation of `AB`, beyond a model using the third call and surface sequence statistics alone.

A direct test would compare:

- H0: `context(ABX)` is explained by `X` plus surface sequence statistics;
- H1: the semantic state associated with independently emitted `AB` contributes systematically when `AB` is embedded.

This is stronger than semantic + structural co-location and weaker than claiming unbounded hierarchical syntax.

## Consequence for the comparative separator

The prior strict separator `{restricted semantic composition, temporal hierarchy}` is **not collapsed**, because chimpanzee sequential recombination is not equivalent to the orangutan two-stratum temporal-hierarchy coordinate.

But any weaker separator of the form `{restricted semantics, structural recombination}` **is collapsed** by the Taï chimpanzee intersection.

The comparative frontier should therefore move upward toward semantic inheritance/scope under embedding, productive depth generalization, and cross-capability coupling rather than mere coexistence of semantics and sequence structure.
