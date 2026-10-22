<p align="center"><img src="images/hero.svg" alt="KARA" width="720"></p>

# KARA: samples

<p align="center"><b>The samples lane: 32 of 30 Harbor task bundles.</b></p>

<p align="center"><a href="#summary"><img src="https://img.shields.io/badge/bundles-32_of_30-845EF7" alt="bundles: 32 of 30"></a> <a href="#evidence-status"><img src="https://img.shields.io/badge/sealed-0-845EF7" alt="sealed: 0"></a> <a href="#difficulty-tiers"><img src="https://img.shields.io/badge/difficulty-measured-845EF7" alt="difficulty: measured"></a></p>

<p align="center"><sub><a href="#abstract">Abstract</a> | <a href="#summary">Summary</a> | <a href="#difficulty-tiers">Difficulty tiers</a> | <a href="#evidence-status">Evidence status</a> | <a href="#coverage">Coverage</a> | <a href="#dataset-structure">Dataset structure</a> | <a href="#verification-and-quality-assurance">Verification and quality assurance</a> | <a href="#reproduction">Reproduction</a> | <a href="#cite">Cite</a> | <a href="#license">License</a> | <a href="#contact">Contact</a></sub></p>

## Abstract

The curated lane of this corpus: at most 30 Harbor task bundles, every anchor seated first and every other bundle in seal order. Bundles past that ceiling live in `delivery/`.

Each bundle is one Harbor task in a directory named by its content-derived uuid5, so the name doubles as an integrity check. This page is rendered from the resident bundles on every reconcile; a value a bundle does not carry reads `absent`, and the page never fills one in.

## Summary

| Property | Value |
| --- | --- |
| Bundles | 32 of 30 |
| Sealed | 0 of 32 |
| Registrations | unregistered (32) |
| Audit verdicts | none (32) |
| Harbor schema | 1.3 (31), absent (1) |

## Difficulty tiers

Difficulty is measured by the signed pilot alone and never declared in a bundle, so this page states no tier a bundle claims.

## Evidence status

The verdict is the audit outcome recorded at the sealed digest, and `none` means no verdict has landed for that digest yet. Only an authoring run removes a bundle from this lane, so removal waits for the next one.

| uuid | Verdict | Sealed digest | Stream |
| --- | --- | --- | --- |
| 136ea5ed-f12a-579f-af26-820b73fd6027 | none | absent | none |
| 21b44ef3-9bf0-5ac9-92ea-053f42aeef96 | none | absent | none |
| 278efa13-6d0b-579f-9cbe-74b1f19aff1c | none | absent | none |
| 295860c2-353a-527a-ac59-4eb8d0fadb6c | none | absent | none |
| 2a018fb7-cc7b-5770-8db1-e2763318eb4b | none | absent | none |
| 30aed99e-1cf6-50be-b2ed-3318dc2ab5fd | none | absent | none |
| 35c2a508-91ce-5260-8861-36646463f79a | none | absent | none |
| 397137b7-7528-5947-be12-6c381483bdfe | none | absent | none |
| 3aad6878-cdfe-510c-a62e-33131253a733 | none | absent | none |
| 3ae7fe27-79ff-5434-bb18-468c2b2ae3fb | none | absent | none |
| 3b440802-5809-5048-b0e1-31187a7385e3 | none | absent | none |
| 481e6565-e793-51d2-af68-27ddbeacc7bb | none | absent | none |
| 4870c00f-0da9-5380-8fe6-f42654474d15 | none | absent | none |
| 5c5599da-c0a9-5bce-825f-49593a1642df | none | absent | none |
| 5fb301a1-8f98-575d-9a99-1a9a139d3607 | none | absent | none |
| 655b89d1-0644-57ba-9093-2a87ff1213a7 | none | absent | none |
| 68ed8188-2c68-5f0e-8af3-a7d6526acb30 | none | absent | none |
| 7763252e-be9c-5b93-b932-cdf75ce80918 | none | absent | none |
| 8c01b86c-5dfb-5c3f-8489-b8c3037b0f53 | none | absent | none |
| 9a1fb5da-7248-5c24-90f7-301f7e4273af | none | absent | none |
| a34fc715-ce32-5eb6-a2f2-a766fd2642c8 | none | absent | none |
| a439d32b-6867-5627-8b88-0f0627b5f901 | none | absent | none |
| c456c486-f4f9-5c5c-b2cd-8a0f127e7462 | none | absent | none |
| c8367b4b-f78b-513c-ab4d-b612c2c9bd76 | none | absent | none |
| e08b70dc-9503-54f1-bb34-8691bae476d0 | none | absent | none |
| e4f0dc7d-329a-5a4b-80c4-0b86d2ce63c6 | none | absent | none |
| e97c93c1-ab71-51a6-920c-979f58e611bb | none | absent | none |
| edf64f9c-8661-51b1-a18f-b05d5f044abb | none | absent | none |
| ef005d6a-44ee-5385-a2af-c70c937dd520 | none | absent | none |
| f5b49aa2-f183-5876-b288-d40bb2cd975b | none | absent | none |
| fa0960bc-9823-51cd-a430-02d95c8bf6ad | none | absent | none |
| fff48a04-0ee9-539f-a5f0-9e099050eb2e | none | absent | none |

## Coverage

| Keyword | Bundles |
| --- | --- |
| arithmetic | 31 |
| bounded-continuous | 31 |
| machine-learning | 31 |
| optimization | 31 |
| rl | 31 |
| transformer | 31 |

## Dataset structure

| Bundle | Task | Keywords | Registration | Sealed at | Tier |
| --- | --- | --- | --- | --- | --- |
| [`136ea5ed-f12a-579f-af26-820b73fd6027`](136ea5ed-f12a-579f-af26-820b73fd6027/) | `arith/sub2-12d-trained` | arithmetic, bounded-continuous, machine-learning, optimization, rl, transformer | unregistered | unsealed | - |
| [`21b44ef3-9bf0-5ac9-92ea-053f42aeef96`](21b44ef3-9bf0-5ac9-92ea-053f42aeef96/) | `arith/add2-8d-trained` | arithmetic, bounded-continuous, machine-learning, optimization, rl, transformer | unregistered | unsealed | - |
| [`278efa13-6d0b-579f-9cbe-74b1f19aff1c`](278efa13-6d0b-579f-9cbe-74b1f19aff1c/) | `arith/sub3-9d-trained` | arithmetic, bounded-continuous, machine-learning, optimization, rl, transformer | unregistered | unsealed | - |
| [`295860c2-353a-527a-ac59-4eb8d0fadb6c`](295860c2-353a-527a-ac59-4eb8d0fadb6c/) | `arith/sub2-9d-trained` | arithmetic, bounded-continuous, machine-learning, optimization, rl, transformer | unregistered | unsealed | - |
| [`2a018fb7-cc7b-5770-8db1-e2763318eb4b`](2a018fb7-cc7b-5770-8db1-e2763318eb4b/) | `arith/sub3-11d-trained` | arithmetic, bounded-continuous, machine-learning, optimization, rl, transformer | unregistered | unsealed | - |
| [`30aed99e-1cf6-50be-b2ed-3318dc2ab5fd`](30aed99e-1cf6-50be-b2ed-3318dc2ab5fd/) | `arith/sub2-11d-trained` | arithmetic, bounded-continuous, machine-learning, optimization, rl, transformer | unregistered | unsealed | - |
| [`35c2a508-91ce-5260-8861-36646463f79a`](35c2a508-91ce-5260-8861-36646463f79a/) | `arith/add3-9d-trained` | arithmetic, bounded-continuous, machine-learning, optimization, rl, transformer | unregistered | unsealed | - |
| [`397137b7-7528-5947-be12-6c381483bdfe`](397137b7-7528-5947-be12-6c381483bdfe/) | `arith/sub3-15d-trained` | arithmetic, bounded-continuous, machine-learning, optimization, rl, transformer | unregistered | unsealed | - |
| [`3aad6878-cdfe-510c-a62e-33131253a733`](3aad6878-cdfe-510c-a62e-33131253a733/) | `arith/sub2-15d-trained` | arithmetic, bounded-continuous, machine-learning, optimization, rl, transformer | unregistered | unsealed | - |
| [`3ae7fe27-79ff-5434-bb18-468c2b2ae3fb`](3ae7fe27-79ff-5434-bb18-468c2b2ae3fb/) | `arith/add3-8d-trained` | arithmetic, bounded-continuous, machine-learning, optimization, rl, transformer | unregistered | unsealed | - |
| [`3b440802-5809-5048-b0e1-31187a7385e3`](3b440802-5809-5048-b0e1-31187a7385e3/) | `arith/add3-14d-trained` | arithmetic, bounded-continuous, machine-learning, optimization, rl, transformer | unregistered | unsealed | - |
| [`481e6565-e793-51d2-af68-27ddbeacc7bb`](481e6565-e793-51d2-af68-27ddbeacc7bb/) | `arith/sub3-12d-trained` | arithmetic, bounded-continuous, machine-learning, optimization, rl, transformer | unregistered | unsealed | - |
| [`4870c00f-0da9-5380-8fe6-f42654474d15`](4870c00f-0da9-5380-8fe6-f42654474d15/) | `arith/add2-9d-trained` | arithmetic, bounded-continuous, machine-learning, optimization, rl, transformer | unregistered | unsealed | - |
| [`5c5599da-c0a9-5bce-825f-49593a1642df`](5c5599da-c0a9-5bce-825f-49593a1642df/) | `arith/add2-10d-trained` | arithmetic, bounded-continuous, machine-learning, optimization, rl, transformer | unregistered | unsealed | - |
| [`5fb301a1-8f98-575d-9a99-1a9a139d3607`](5fb301a1-8f98-575d-9a99-1a9a139d3607/) | `arith/sub3-10d-trained` | arithmetic, bounded-continuous, machine-learning, optimization, rl, transformer | unregistered | unsealed | - |
| [`655b89d1-0644-57ba-9093-2a87ff1213a7`](655b89d1-0644-57ba-9093-2a87ff1213a7/) | `arith/sub2-8d-trained` | arithmetic, bounded-continuous, machine-learning, optimization, rl, transformer | unregistered | unsealed | - |
| [`68ed8188-2c68-5f0e-8af3-a7d6526acb30`](68ed8188-2c68-5f0e-8af3-a7d6526acb30/) | `arith/add3-10d-trained` | arithmetic, bounded-continuous, machine-learning, optimization, rl, transformer | unregistered | unsealed | - |
| [`7763252e-be9c-5b93-b932-cdf75ce80918`](7763252e-be9c-5b93-b932-cdf75ce80918/) | `arith/sub2-10d-trained` | arithmetic, bounded-continuous, machine-learning, optimization, rl, transformer | unregistered | unsealed | - |
| [`8c01b86c-5dfb-5c3f-8489-b8c3037b0f53`](8c01b86c-5dfb-5c3f-8489-b8c3037b0f53/) | `arith/add2-13d-trained` | arithmetic, bounded-continuous, machine-learning, optimization, rl, transformer | unregistered | unsealed | - |
| [`9a1fb5da-7248-5c24-90f7-301f7e4273af`](9a1fb5da-7248-5c24-90f7-301f7e4273af/) | `arith/sub2-13d-trained` | arithmetic, bounded-continuous, machine-learning, optimization, rl, transformer | unregistered | unsealed | - |
| [`a34fc715-ce32-5eb6-a2f2-a766fd2642c8`](a34fc715-ce32-5eb6-a2f2-a766fd2642c8/) | `arith/add3-15d-trained` | arithmetic, bounded-continuous, machine-learning, optimization, rl, transformer | unregistered | unsealed | - |
| [`a439d32b-6867-5627-8b88-0f0627b5f901`](a439d32b-6867-5627-8b88-0f0627b5f901/) | `arith/add3-13d-trained` | arithmetic, bounded-continuous, machine-learning, optimization, rl, transformer | unregistered | unsealed | - |
| [`c456c486-f4f9-5c5c-b2cd-8a0f127e7462`](c456c486-f4f9-5c5c-b2cd-8a0f127e7462/) | absent | absent | unregistered | unsealed | - |
| [`c8367b4b-f78b-513c-ab4d-b612c2c9bd76`](c8367b4b-f78b-513c-ab4d-b612c2c9bd76/) | `arith/sub3-13d-trained` | arithmetic, bounded-continuous, machine-learning, optimization, rl, transformer | unregistered | unsealed | - |
| [`e08b70dc-9503-54f1-bb34-8691bae476d0`](e08b70dc-9503-54f1-bb34-8691bae476d0/) | `arith/add3-11d-trained` | arithmetic, bounded-continuous, machine-learning, optimization, rl, transformer | unregistered | unsealed | - |
| [`e4f0dc7d-329a-5a4b-80c4-0b86d2ce63c6`](e4f0dc7d-329a-5a4b-80c4-0b86d2ce63c6/) | `arith/add2-12d-trained` | arithmetic, bounded-continuous, machine-learning, optimization, rl, transformer | unregistered | unsealed | - |
| [`e97c93c1-ab71-51a6-920c-979f58e611bb`](e97c93c1-ab71-51a6-920c-979f58e611bb/) | `arith/add2-15d-trained` | arithmetic, bounded-continuous, machine-learning, optimization, rl, transformer | unregistered | unsealed | - |
| [`edf64f9c-8661-51b1-a18f-b05d5f044abb`](edf64f9c-8661-51b1-a18f-b05d5f044abb/) | `arith/sub3-14d-trained` | arithmetic, bounded-continuous, machine-learning, optimization, rl, transformer | unregistered | unsealed | - |
| [`ef005d6a-44ee-5385-a2af-c70c937dd520`](ef005d6a-44ee-5385-a2af-c70c937dd520/) | `arith/sub2-14d-trained` | arithmetic, bounded-continuous, machine-learning, optimization, rl, transformer | unregistered | unsealed | - |
| [`f5b49aa2-f183-5876-b288-d40bb2cd975b`](f5b49aa2-f183-5876-b288-d40bb2cd975b/) | `arith/add3-12d-trained` | arithmetic, bounded-continuous, machine-learning, optimization, rl, transformer | unregistered | unsealed | - |
| [`fa0960bc-9823-51cd-a430-02d95c8bf6ad`](fa0960bc-9823-51cd-a430-02d95c8bf6ad/) | `arith/sub3-8d-trained` | arithmetic, bounded-continuous, machine-learning, optimization, rl, transformer | unregistered | unsealed | - |
| [`fff48a04-0ee9-539f-a5f0-9e099050eb2e`](fff48a04-0ee9-539f-a5f0-9e099050eb2e/) | `arith/add2-11d-trained` | arithmetic, bounded-continuous, machine-learning, optimization, rl, transformer | unregistered | unsealed | - |

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
    staging --> samples["samples/<br/>32 of 30"]
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
