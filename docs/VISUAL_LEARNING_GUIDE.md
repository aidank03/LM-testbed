# Visual learning guide

The ten world-model notebooks now include 27 additional visual labs. Look for **“Visual lab”** headings within each lesson. Every lab explains what the picture shows and gives you something to predict, notice or question.

| Lesson | New visual explanations | Question to keep in mind |
|---|---|---|
| 05 | Configuration ownership; shot × time tensor heatmaps | Does this setting change the physical world, the observation, or our interpretation? |
| 06 | Circuit–liner feedback diagram; linked shot playback; parameter-sensitivity bars | How can motion change the current that caused it? |
| 07 | The same edge under three blur settings; sampling, exposure and clock timeline | Is a weak feature absent, or simply hard to observe? |
| 08 | Mass posterior marginals; indistinguishable mean-state worlds; calibration-bias comparison | What additional evidence distinguishes the remaining explanations? |
| 09 | Calibration-score distributions; individual intervals under matched and wrong physics | Do narrow error bars actually cover the answer? |
| 10 | Teacher forcing versus free rollout; per-shot forecast-error heatmaps | How can small one-step errors become a poor long-term prediction? |
| 11 | Diagnostic-token and missingness maps; measured attention weights; parameter-specific ablation heatmaps | Which information is missing, and which parameter needs it? |
| 12 | Input/target token alignment; causal mask; step-by-step generation playback | What can the model see, what is scored, and what token does it choose next? |
| 13 | Observation → LM → runtime → tool diagram; action/outcome comparison; actual episode trace | Did the model choose correctly, and did the runtime have enough evidence to act? |
| 14 | Information gain and cost; sequential posterior updates; held-back residuals; full inference cycle | Did the new measurement resolve uncertainty, and do independent predictions agree? |

## Use the animations

Notebook **06** includes a linked view of the evolving current, radius and an end-on radius sketch. Notebook **12** steps through the tiny LM's generated JSON one token at a time, showing the top next-token probabilities.

The play/pause and frame controls are embedded in the saved notebook outputs. A static fallback figure appears before each animation. If your notebook viewer suppresses JavaScript outputs, the static figures still work; running the cell in your normal trusted Jupyter environment can enable playback. No network service or new dependency is required.

## A useful review rhythm

1. Read the “Predict,” “Notice,” or “Ask” prompt before studying the figure.
2. Say what you expect to change and why.
3. Inspect the figure and compare it with the existing numeric result.
4. Change one parameter in the visible code and run again.
5. Keep the result even when it contradicts your prediction.

The new plots use existing simulations, model outputs and held-out examples. Attention weights and token probabilities are labeled as model computations; they are not physical causal explanations or calibrated scientific confidence. Diagram colors group roles rather than encode measured quantities. Measured heatmaps specify their scales, including when panels share a scale.
