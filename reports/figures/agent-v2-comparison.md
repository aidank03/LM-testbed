# Explicit motion-task comparison

![Exact local-agent and conventional decision counts](agent-v2-comparison.png)

**Figure.** Frozen v2 development comparison of two conventional methods and two local models, with and without numerical tools. Each model/workflow was run twice on the same 18 observations in 12 independent experiment groups. Points show exact counts, and repeat 1 and 2 remain separate. Required-ambiguity recovery has five eligible observations; the other panels use all 18 assigned cases. All repetitions had identical final decisions and execution statuses.

Contract correctness means agreement with the explicitly supplied diagnostic assumptions. Physical wrong-decisive counts compare emitted answers against simulator motion; failures and abstention can lower those counts, so completion and required ambiguity must also be considered. No sampling confidence intervals or pooled independent-sample claims are shown.

Direct mode has no structured numerical-result field, so this figure compares decisions rather than direct numerical reconstruction. Qwen never invoked numerical tools in the tool-available arm. Nemotron used them, but failures to cite valid artifact identifiers prevented some workflows from completing. No local workflow exceeded the bounded conventional estimator's 18/18 contractual decisions, and the bounded estimator itself failed physically in all four deliberately hidden-mismatch cases.

[Detailed report](../AGENT_V2_COMPARISON.md) · [PDF](agent-v2-comparison.pdf) · [Scored data](../agent-v2-comparison/comparison.json) · [Trace audit](../agent-v2-trace-audit.json)

Reproduce with `python experiments/summarize_agent_evaluation.py`. The figure was visually checked for sample labels, overlapping marks, missing-result interpretation, and clipping. The PDF uses the same Matplotlib figure.
