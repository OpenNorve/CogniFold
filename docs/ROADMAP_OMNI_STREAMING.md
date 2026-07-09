# Roadmap — omni-modal, always-on agent

CogniFold is extending toward an **omni-modal, always-on agent**: a system that
continuously watches an audio + video stream and **proactively decides when to act
and what to say**, instead of only responding when it is explicitly asked.

## What we're building
- **Omni perception** — jointly attend to vision, speech, and **non-speech sound**
  (environmental audio), since real-world cues are inherently multi-modal.
- **Proactive & streaming** — choose the moment to respond with no polling or fixed
  schedule, and hold up over **long-horizon** (minutes-scale) streams.
- **In-model working memory** — keep and consolidate state online, extending
  CogniFold's memory story from discrete events to a continuous audio-visual stream.

## Primary benchmark
**OmniPro — A Comprehensive Benchmark for Omni-Proactive Streaming Video
Understanding** (Zhao et al., 2026; [arXiv:2605.18577](https://arxiv.org/abs/2605.18577)):
2,700 human-verified samples, 9 sub-tasks across 3 cognitive levels, 84%
audio-dependent, evaluated under a dual **Probe** / **Online** protocol. We also
track transfer to OVO-Bench and StreamingBench.

## Status
Early — this note states the **direction**. Design details and results land
incrementally.
