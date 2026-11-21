---
title: Decisions log
summary: Decisions the team has made and why, so they are not reopened by accident. New decisions are appended at the end.
---

# Decisions

Each entry: the decision, the reason, and alternatives considered. A decision can be revisited, but say so
explicitly and record the new decision.

- **2026-10 · Start from Wu et al. (RTV).** Only reviewed method that already runs in real time with a small network
  and fits a 16 GB GPU; code and weights public. LiveVVT rejected as a base: about 20 GB peak memory.
- **2026-10-28 · Learn from upstream, work in the fork.** Tutoring notes describe upstream commit `dd165d5`; our
  changes are tracked separately so they are never confused with the original.
- **2026-11 · Framing: generalise across customers and across garments.** Body-map and shared-backbone directions are
  both main contributions; isolation and baseline fixes support them. The specific methods (M1 to M5) are
  recommendations, open to better alternatives.
- **2026-11 · Speed is a constraint, not a gap.** Any change must add little or no inference cost.
- **2026-11 · Face recognition rejected for per-frame person selection.** Face hidden when turning, extra cost,
  biometric privacy. Possible only as a one-time session login.
- **2026-11 · Record our own data.** No public dataset has customers in their own loose clothing or store scenes.
- **2026-11-19 · Cloud GPU requested.** A100 80 GB class through Dr. Noha to the Office of the Dean, no price or hours
  stated. Large training waits for it; local 16 GB GPU used for everything else meanwhile.

## Log

<!-- record_decision appends dated entries below this line -->
