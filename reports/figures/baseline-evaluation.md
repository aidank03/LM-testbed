# Conventional motion–drift benchmark

![Coverage, wrong decisive answers, and abstention](baseline-evaluation.png)

**Figure.** Results from 216 synthetic observations representing 144 independent experiment groups. Orange circles show the naive estimator; blue squares show the estimator using supplied nuisance bounds and reference uncertainty. Panel A reports how often a finite interval contains the simulator's material displacement. Panel B reports physically wrong decisive answers divided by all assigned cases. Panel C shows scientific abstention, which must be considered alongside false confidence. Counts are printed next to each point.

The bounded estimator appropriately returns no finite interval for all 48 structurally ambiguous cases; its missing coverage is not plotted as 0%. A reference helps under the assumed calibration/transfer model. Both estimators miss all true displacements under unseen drift, and the bounded estimator also misses all under reference mismatch. Correct handling of advertised uncertainty therefore does not protect against every omitted diagnostic effect.

The dashed 95% guide refers to the Gaussian noise component, conditional on supplied assumptions; nuisance bounds can enlarge the intervals. Points are descriptive proportions, with no sampling confidence intervals. Ambiguous counterfactual pairs and their corresponding reference variants share experiment groups and must not be treated as independent observations. “Calibrated” refers to a synthetic assumption, not an independently validated real reference diagnostic. These results establish neither melting nor an instability mechanism.

[PDF](baseline-evaluation.pdf) · [Source and version metadata](baseline-evaluation.json) · [Evaluation data](../motion-development/comparison.json)

Reproduce from the repository root:

```sh
python experiments/plot_baseline_evaluation.py
```

The script reads the saved comparison only; it does not generate cases, run candidate methods, or rescore the benchmark. The PNG was visually inspected for clipping, labels, sample counts, and missing-value treatment; the PDF uses the same Matplotlib figure.
