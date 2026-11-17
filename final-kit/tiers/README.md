# Frozen tier inventory

All source files, including fallback IFEval metadata, are copied byte-for-byte from day1-supp tiers. `SHA256SUMS` is the ORIGINAL tier-only seal; all entries were checked after copying. This README is an addition covered by the final kit manifest, not by that original seal.

Seed: `20261117-supplement2-v1`. Primary defaults: minerva_math500/n197.json and mmlu_pro/n197.json, R1. MATH uses subject×level strata; MMLU uses 14 categories. Indices in those lists are GLOBAL pinned test-row IDs. Upstream MMLU child indices are category-filtered row IDs; lib/select.jq converts using the unchanged strata.json. Never pass a global MMLU ID directly to a child. Explicit child selection JSON must name that child and supply local IDs.

Each n197 file contains population/sample stratum counts and effective sample size. Use full-population weights, sum_h(N_h/N)*mean_h, for completed-tier descriptive estimates. MATH N=500, MMLU N=12032. Summary and STATISTICS.md retain the source planning tables. IFEval N=541 is documented fallback metadata only and is refused by the day-1 runners.

197/R1 is approximately 10 pp 80%-power planning sensitivity under q=.25 and the source homogeneous-within-stratum assumptions; it is not a measured effect or guaranteed precision. Effective n: MATH 196.44, MMLU 196.94; approximate 95% half-widths 6.99/6.98 pp in that scenario. See PROTOCOLS.md for uncertainty and claim restrictions.

Alternatives, never automatic reruns: full 500/12032 R1, or R8 approximately 8 pp scenario using MATH n54 and MMLU n50. A 5 pp scenario uses MATH n127/MMLU n126 at R8; MMLU n785 at R1. R8 reduces within-item sampling noise only. Freeze any alternative before observing outcomes, preserve draw seeds/identities, and do not pool it with the default.
