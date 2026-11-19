# Experiment Results Summary

## Scope

This summary covers the ten questions Q001–Q010. The experiment used three sequential calls for each condition, for six calls per question in total. Quality scores are the overall scores reported by the user for each answer; a per-criterion evidence breakdown and independent second-rater scores were not supplied.

## Aggregate results

| Condition | Questions | Mean score (out of 7) | Mean score (%) | Mean total tokens per question | Mean summed latency per question |
|---|---:|---:|---:|---:|---:|
| Multi-LLM | 10 | 6.0 | 85.7% | 27,377.6 | 13.806 s |
| Single-LLM baseline | 10 | 6.0 | 85.7% | 26,872.3 | 12.704 s |

Scores were identical between conditions for each of the ten questions. In these runs, the multi-LLM condition used an average of 505.3 more total tokens and took an average of 1.102 seconds longer per question than the baseline. The result does not show a score advantage for multi-LLM collaboration on this small question set.

## Limits on interpretation

These are descriptive results from ten questions, not a statistical significance test. Scores were supplied as overall ratings without criterion-level detail or an independent second rater. Model-generated token counts and latency are recorded in `run_log.csv`; cost was not available in the run metadata and is therefore not reported. The findings should be described as preliminary and should not be generalized beyond this dataset and configuration.
