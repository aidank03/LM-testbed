# Video workflow notes for Factor

Status: source captions retrieved; selected passages inspected; proposed design implications, not an evaluation of the speaker's claims. Prepared 2026-09-05.

## Source and access record

- [Ex-NASA dev reveals his Agentic Engineering Workflow](https://www.youtube.com/watch?v=xgkjtF89-44), published by David Ondrej, 2026-08-07, duration 58:37. These fields were read directly from public YouTube metadata.
- YouTube's original English automatic captions were retrieved using `yt-dlp` 2026.8.19. The listing contained automatic captions but no manually provided subtitle track. No video or audio was downloaded, and no account credentials were used.
- The caption source contains 3,959 timestamped events; the last starts at 3,515.92 seconds. Source SHA-256: `fcdb0747cdf78ee84393b5b03c3cbafec82bba89f3abcfbcc74b96696d12ffc6`.
- Inspected caption windows: 00:00–01:15, 02:50–04:40, 06:45–08:55, 11:25–13:55, 14:55–20:50, and 44:30–47:20. This is a focused program-design reading, not an audio-verified review of the complete interview. Automatic captions can misrecognize names and technical terms.
- The ordinary web reader returned a minimal YouTube page; direct timed-text and oEmbed requests failed. No browser was available through computer use. A temporary caption-reading environment with permitted public network access succeeded. Secondary summaries helped locate passages; the notes below were then checked against the retrieved YouTube captions.
- Full captions remain a temporary research source, outside the project deliverable. No permissive redistribution license was established. This document contains paraphrases and links, not a reproduced transcript.

## Selected passages

| Video location | Paraphrased discussion |
| --- | --- |
| [03:08–04:18](https://www.youtube.com/watch?v=xgkjtF89-44&t=188s) | Upfront agreement can reduce review work. Automated review and application testing help, while maintaining confidence remains difficult. Incidents can enter the development queue. |
| [07:08–08:49](https://www.youtube.com/watch?v=xgkjtF89-44&t=428s) | The speakers discuss lighter review for familiar changes, stronger checks for uncertain ones, and the risk of losing understanding of the codebase. |
| [11:52–13:43](https://www.youtube.com/watch?v=xgkjtF89-44&t=712s) | Start from a user problem and measurable outcome. External measurements can give an agent more useful feedback than open-ended instructions alone. |
| [15:19–19:21](https://www.youtube.com/watch?v=xgkjtF89-44&t=919s) | Separate component architecture from program design: call paths, file locations, types, and function signatures are decisions to inspect before implementation. |
| [19:24–20:42](https://www.youtube.com/watch?v=xgkjtF89-44&t=1164s) | Build a narrow, runnable path through the system, then add behavior in increments that can be checked along the way. |
| [46:43–47:04](https://www.youtube.com/watch?v=xgkjtF89-44&t=2803s) | Increasing agent throughput does not remove a review bottleneck; judge the complete delivery process. |

## Factor design decisions to test

The following are our proposed experiments and contracts. The interview motivates them but does not validate them for Factor.

1. **Define one complete user outcome.** Start with a callable motion-versus-drift analysis that returns a quantity, units, uncertainty, assumptions, evidence references, and an explicit unresolved result when appropriate. Its usefulness depends on diagnostic information and decision tolerance, not whether an LLM produced fluent text.
2. **Specify the call path before expanding implementation.** Review input validation → information/identifiability check → permitted execution → reconstruction → result validation → provenance. Keep evaluator-only truth outside this path. Local and cloud implementations must receive the same permitted information in an algorithm comparison.
3. **Make a small slice reviewable.** First run one input through the public interface to a structured result with a conventional estimator. Next add ambiguity handling, then a calibrated reference track. Add cloud execution only after local behavior and data-release policy are testable.
4. **Close the feedback loop with bounded authority.** A failed development case produces a reproducible finding and proposed change. Checks decide whether the proposal qualifies for review; they do not grant permission to change the benchmark's acceptance criteria, access locked answers, operate equipment, or publish results.
5. **Test that feedback cannot be gamed.** Include a candidate that returns narrow but wrong intervals, a candidate that always abstains, and a candidate that produces malformed units. Check false confidence and useful answer rate separately. A passing test suite establishes implementation properties, not physical validity.
6. **Measure the actual bottleneck.** For several comparable feature slices, record implementation time, expert review time, escaped defects, rework, monetary cost, and time to make a later specified change. Keep the information and compute budgets explicit. This pilot can reveal problems; a small uncontrolled comparison cannot establish a general productivity advantage.

The first closed loop should improve one known workflow under a fixed evaluation contract. Autonomous generation of ever more features is not a success criterion.

## Related primary writing

HumanLayer's own [Why Software Factories Fail](https://github.com/humanlayer/advanced-context-engineering-for-coding-agents/blob/main/wsff.md) describes the same four stages and reports the author's experience with removing code review. Its [workflow documentation](https://docs.humanlayer.com/explanation/workflow-phases) distinguishes existing-system research, product decisions, technical design, and checkable implementation slices. These pages were opened and inspected separately. They are related primary writing, not a transcript of this interview, and their workflow rules do not constitute instructions for Factor.

## Claims left unvalidated

This reading does not establish universal limits of coding models, a fixed context-length failure threshold, a productivity multiplier, the comparative superiority of any provider, or the safety of unattended deployment. The host's biographical and priority claims were not independently checked. The proposed review process also needs evaluation: another model's agreement is not independent proof, and deterministic checks can faithfully score the wrong objective.

Plain-language takeaway: build Factor so each small result can be checked, its uncertainty can be understood, and a failed result teaches us what to change next.
