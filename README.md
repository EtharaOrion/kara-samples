<p align="center"><img src="images/hero.svg" alt="KARA" width="720"></p>

# KARA: samples

<p align="center"><b>The samples lane: 2 of 30 Harbor task bundles.</b></p>

<p align="center"><a href="#summary"><img src="https://img.shields.io/badge/bundles-2_of_30-845EF7" alt="bundles: 2 of 30"></a> <a href="#evidence-status"><img src="https://img.shields.io/badge/sealed-0-845EF7" alt="sealed: 0"></a> <a href="#difficulty-tiers"><img src="https://img.shields.io/badge/difficulty-measured-845EF7" alt="difficulty: measured"></a></p>

<p align="center"><sub><a href="#abstract">Abstract</a> | <a href="#summary">Summary</a> | <a href="#difficulty-tiers">Difficulty tiers</a> | <a href="#evidence-status">Evidence status</a> | <a href="#coverage">Coverage</a> | <a href="#dataset-structure">Dataset structure</a> | <a href="#verification-and-quality-assurance">Verification and quality assurance</a> | <a href="#reproduction">Reproduction</a> | <a href="#cite">Cite</a> | <a href="#license">License</a> | <a href="#contact">Contact</a></sub></p>

## Abstract

The curated lane of this corpus: at most 30 Harbor task bundles, every anchor seated first and every other bundle in seal order. Bundles past that ceiling live in `delivery/`.

Each bundle is one Harbor task in a directory named by its content-derived uuid5, so the name doubles as an integrity check. This page is rendered from the resident bundles on every reconcile; a value a bundle does not carry reads `absent`, and the page never fills one in.

## Summary

| Property | Value |
| --- | --- |
| Bundles | 2 of 30 |
| Sealed | 0 of 2 |
| Registrations | unregistered (2) |
| Audit verdicts | none (2) |
| Harbor schema | 1.3 (1), absent (1) |

## Difficulty tiers

Difficulty is measured by the signed pilot alone and never declared in a bundle, so this page states no tier a bundle claims.

## Evidence status

The verdict is the audit outcome recorded at the sealed digest, and `none` means no verdict has landed for that digest yet. Only an authoring run removes a bundle from this lane, so removal waits for the next one.

| uuid | Verdict | Sealed digest | Stream |
| --- | --- | --- | --- |
| 21b44ef3-9bf0-5ac9-92ea-053f42aeef96 | none | absent | none |
| c456c486-f4f9-5c5c-b2cd-8a0f127e7462 | none | absent | none |

## Coverage

| Keyword | Bundles |
| --- | --- |
| arithmetic | 1 |
| bounded-continuous | 1 |
| machine-learning | 1 |
| optimization | 1 |
| rl | 1 |
| transformer | 1 |

## Dataset structure

| Bundle | Task | Keywords | Registration | Sealed at | Tier |
| --- | --- | --- | --- | --- | --- |
| [`21b44ef3-9bf0-5ac9-92ea-053f42aeef96`](21b44ef3-9bf0-5ac9-92ea-053f42aeef96/) | `arith/add2-8d-trained` | arithmetic, bounded-continuous, machine-learning, optimization, rl, transformer | unregistered | unsealed | - |
| [`c456c486-f4f9-5c5c-b2cd-8a0f127e7462`](c456c486-f4f9-5c5c-b2cd-8a0f127e7462/) | absent | absent | unregistered | unsealed | - |

```mermaid
%%{init: {"theme":"base","themeVariables":{"primaryColor":"#131a33","primaryTextColor":"#e8ecff","primaryBorderColor":"#845EF7","lineColor":"#22d3ee","fontFamily":"monospace"}}}%%
flowchart LR
    bundle["&lt;uuid&gt;/"]
    bundle --> n0["task.toml<br/>runner manifest"]
    bundle --> n1["instruction.md<br/>objective the agent reads"]
    bundle --> n2["environment/<br/>agent sandbox image"]
    bundle --> n3["tests/<br/>verifier, hidden from the agent"]
    bundle --> n4["solution/<br/>private oracle"]
    bundle --> n5["README.md"]
    bundle --> n6["dataset/"]
    bundle --> n7["inspector.html"]
    bundle --> n8["iterations.json"]
    bundle --> n9["trajectories/"]
```

```mermaid
%%{init: {"theme":"base","themeVariables":{"primaryColor":"#131a33","primaryTextColor":"#e8ecff","primaryBorderColor":"#845EF7","lineColor":"#22d3ee","fontFamily":"monospace"}}}%%
flowchart LR
    staging["staging/"]
    staging --> samples["samples/<br/>2 of 30"]
    samples --> delivery["delivery/<br/>0 of 10000"]
    delivery --> stock["stock/<br/>0 of 10000"]
```

## Verification and quality assurance

Each bundle's signed provenance carrier holds its screening roots, screening instant, and absolute expiry. Placement moves that carrier into the private derivation set, which this page never reads, so reconcile lists no root it cannot verify here.

## Reproduction

Each bundle directory is a complete Harbor task: `task.toml` is the runner manifest, `instruction.md` the objective, `environment/` the sandbox image, and `tests/` the verifier. Point a Harbor runner at one bundle directory to reproduce it; `solution/` stays private.

## Cite

```bibtex
@misc{kara2026,
  title  = {KARA: Open-Ended Arithmetic-Transformer RL Environments with Measured Difficulty},
  author = {Ethara.AI},
  year   = {2026},
  note   = {Harbor task bundles, bounded continuous rewards, and a signed-pilot difficulty discipline}
}
```

## License

MIT, per [`LICENSE`](LICENSE). Task bundles, datasets, and trajectories produced through the instruments keep their own licenses where declared.

## Contact

Ethara.AI on GitHub: <https://github.com/Ethara-Ai>. Benchmark-integrity and vulnerability disclosure: `security@ethara.ai`, per [`SECURITY.md`](SECURITY.md). Contributions: [`CONTRIBUTING.md`](CONTRIBUTING.md).

---

## Copyright

Copyright (c) 2026 Green Rider Technology LLP (Ethara AI). All rights reserved.
