---
name: adder-certificate-needs-exhaustive-saturation
description: "Sampled clamp-saturation penalties leave rare digit pairs unsaturated, which breaks the whole-domain certificate even though sampled accuracy is 1.0."
metadata: 
  node_type: memory
  type: project
  originSessionId: ce355b87-5968-48c0-a337-9a2c0b5863db
  modified: 2026-09-07T04:28:23.444Z
---

In the `/workspace` addition model, `finetune.py`'s saturation penalty originally
averaged over *sampled* tokens. On 2026-09-07 that let the pair (0,9) land inside
the clamp's linear region (gate 0.732 instead of 0/1) while held-out exact match
still read 1.0000 and 40k spot checks found no error — the leak only cost about
11 logits of attention margin, invisible at n=8 but fatal at larger widths and
fatal to the proof.

**Why:** the certificate's first step is a *sufficient* condition — exact gate
saturation is what makes the key and value constant within a carry class, which is
what makes the `3^n` enumeration cover the domain. Sampling can satisfy accuracy
without satisfying the premise the proof rests on.

**How to apply:** compute bank slack over all 100 digit pairs plus the (0,0) pad
directly from the weights (`finetune.bank_slack`) — it is a pure function of the
parameters, needs no data, and costs nothing. Use it both as the training penalty
and as a hard gate when selecting members, and rank export candidates by
*certified* margin rather than by read-out margin alone. Related:
[[adder-two-phase-curriculum]], [[adder-stop-at-twelve-parameters]].
