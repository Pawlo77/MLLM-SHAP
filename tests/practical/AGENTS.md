# Practical tests agents

Procedure: skill `practical`; scenarios `scenarios.yaml`, models `models.yaml`; not part of `make check` or CI.

- Marker `practical(<feature>)`. The quick tier runs pinned `Qwen2.5-1.5B-Instruct` (generator and judge) and `SmolLM2-360M-Instruct` on a GPU; skill `ship` runs the features `scripts/practical_features.py` maps from the diff. Rubric checks assert; a logit judge (YES-minus-NO margin of at least 1 nat in two of three paraphrases) answers semantic questions only. Mock and random-weight models cannot show whether attributions make sense; a generation cache makes warm runs take seconds. Rejected: a hosted judge (keys, cost).
- The deep tier (`deep.*`, run by skill `release`) uses `qwen2.5-3b-instruct` through LM Studio's `api` backend, judged by `gemma-3-4b-it` (the 3B misjudged). Only `deep.runtime.kv_cache` loads `Qwen2.5-3B-Instruct` in-process, since LM Studio has no prefix cache. A cached 10-player exact reference feeds every sampled run.
- The in-process 3B needed 7.4 GB and 28 minutes cold; LM Studio takes 13 minutes cold and 4 warm (M4 Pro). Its output changes under parallel load, so keep LM Studio otherwise idle and clear the generation cache after a run that overlapped other use.
- Thresholds carry the numbers they were measured with; never loosen one without them.
