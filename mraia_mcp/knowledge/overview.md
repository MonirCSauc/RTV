---
title: Project overview
summary: What MRAIA is, the research question, team, hardware, status and next steps. Read this first.
---

# MRAIA: real-time virtual try-on smart mirror

## What it is

MRAIA is a smart mirror for clothing stores. A customer stands in front of a camera in their own clothes, picks a
garment from the store's catalog, and sees their live reflection wearing it. It is the graduation project (Thesis 1
and 2, AUC, Fall 2026 to May 2027) of a three-person Data Science team and the basis of a startup.

- Team: Ahmed Ayman, Ahmed Monir, Beshoy Botros. Supervisor: Dr. Noha.
- Starting model: Wu, Shen and Igarashi (2025), "Real-Time Per-Garment Virtual Try-On with Temporal Consistency for
  Loose-Fitting Garments", arXiv 2506.12348, code at github.com/ZaiqiangWu/RTV ("RTV", "Wu et al.").
- Our repository is a fork of RTV. The tutoring notes follow upstream commit `dd165d5`; see topic `fork_changes`
  for where the fork departs.

## Research question

How can a real-time per-garment try-on mirror generalise across customers and garments: customers who arrive in
their own, often loose, clothing and stand in a busy store, and a whole catalog of garments, without adding
computational cost at inference?

1. The customer's clothing: how can the body be represented so loose clothing does not break it? (M1, M2)
2. The store: how can the system isolate the customer from bystanders and clutter, live? (M3)
3. The catalog: how can one model serve many garments and body shapes instead of one full network per garment? (M5)

The proposal suggested five modifications, M1 to M5 (topic `proposals`). **They are recommendations from the
proposal stage, not a fixed plan.** The goal is the research question above, not these particular methods. If a
different modification would answer the question better (more accurate, faster, simpler, or more feasible on our
hardware), propose it with reasons and evidence, and record it with the `suggest_modification` tool so the team can
review it.

## The mirror's five requirements

Body (find the body under loose clothes), Background (ignore clutter and other shoppers), Fidelity (keep logos,
textures, shape), Stability (consistent frame to frame), Speed (real time).

## Hardware and resources

- Today: one NVIDIA RTX 4080, 16 GB, in the AUC Machine Learning lab (Windows).
- Requested (Nov 2026): cloud GPU subscription, NVIDIA A100 80 GB class, Dec 2026 to May 2027, through Dr. Noha to
  the Office of the Dean. Not granted yet. See topic `compute`.
- Wu et al. report 10.5 FPS on an RTX 4090; expect lower FPS and longer training on the 4080.

## Status (as of 21 Nov 2026)

- Literature review submitted; proposal presented (went well).
- Repo tutoring: audit done; lessons 1 to 11 taught (notes in `TUTORING.md` in the repo root).
- Fork already contains several baseline fixes (M4 quick wins); see `fork_changes`.
- Next step: build the missing training pipeline pieces and start training the proposed architecture. Large
  training runs (M5 backbone, recurrent clips, M2) wait for the cloud GPU.

## Where things are

- Proposal deck: `MRAIA_Proposal_Deck_8.pptx` (latest version).
- Tutoring and audit notes: `TUTORING.md`.
- This knowledge base: `mraia_mcp/knowledge/` (edit the markdown files directly; the server re-reads them on every
  call).
