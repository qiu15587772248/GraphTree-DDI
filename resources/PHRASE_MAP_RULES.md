# Phrase → tier matching cascade

GraphTree-DDI assigns each DrugBank DDI description a severity tier in `{0,1,2,3,4}`.
Tier 0 is reserved for constructed negatives (no known interaction). Positive
descriptions are labelled by `graphtree_ddi.data.preprocess.map_severity`.

The 184-phrase table is `resources/phrase_map_184.csv` (columns: `phrase`, `tier`).
Keys are lower-cased symptom phrases extracted from DrugBank template A.

## Cascade (applied in order)

1. **Empty description → Tier 1.**
   If the description is missing or whitespace-only, return 1.
2. **Generic adverse wording → Tier 1.**
   If the text matches `the risk or severity of adverse (effects|events)`
   (case-insensitive), return 1. These statements do not name a specific symptom.
3. **No "the risk or severity of" clause → Tier 1.**
   PK/PD templates (serum concentration, metabolism, excretion, efficacy, …)
   without that clause are Tier 1.
4. **Extract and exact-lookup.**
   From `the risk or severity of (.+?) can be` (case-insensitive, non-greedy),
   take `group(1)`, strip, lower-case, and look up `resources/phrase_map_184.csv`.
   A hit returns the tabulated tier (2, 3, or 4).
5. **Unmapped phrase → Tier 3.**
   If the extraction succeeds but the phrase is not in the 184-key table, or
   extraction fails after the clause was detected, return 3 (conservative
   “serious / needs intervention”).

## Table summary

- Keys: **184**
- Tier 2 (moderate): **44**
- Tier 3 (serious): **78**
- Tier 4 (critical): **62**

The same dictionary lives in `graphtree_ddi/data/phrase_label_map.py` and is
imported/compared by `build_v2.py`.
