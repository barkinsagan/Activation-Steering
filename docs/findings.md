# Key Findings

## Finding 1: ML-domain steering improves ML task performance
**Supported by:** (add experiment IDs as you run them)
**Summary:** Applying difference-in-means activation steering derived from ML-domain prompts improves the model's performance on ML completion/multiple-choice tasks. Steering in the target domain generalizes to the task.
**Confidence:** medium (initial results, needs more runs to confirm across layers/coefs)
**Caveats:** Tested on a custom ML textbook completion dataset — generalization to other ML benchmarks unknown.

## Finding 2: CS-domain steering also works
**Supported by:** (add experiment IDs)
**Summary:** Steering with CS-domain prompts similarly improves performance on CS tasks, suggesting the effect is not specific to ML and may be a general property of domain-specific activation steering.
**Confidence:** low (early result, needs replication)
**Caveats:** Need to verify whether CS steering bleeds into ML performance or vice versa (cross-domain contamination check pending).

## Open Questions
- Does steering in domain A hurt performance on domain B?
- Which layers are most sensitive to domain steering?
- Does the effect scale with coefficient magnitude linearly, or is there a sweet spot?
- Does open-completion vs. multiple-choice formatting affect which layers respond best?
