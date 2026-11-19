# Multi-LLM Collaboration Experiment and Trained Question Router

This repository contains the team's three-call text-level collaboration experiment and a separate, small question-routing model trained from the experiment's ten curated questions.

## What the trained model does

`models/question_router.json` is a Multinomial Naive Bayes text classifier trained from scratch with Python's standard library. It maps a question to one of three suggested workflow roles: `evidence_extractor`, `method_analyst`, or `cross_paper_synthesizer`. It is a small course-project demonstration of a trainable component related to the multi-LLM workflow.

The ten questions are too few to establish useful generalization. The role labels were manually grouped for this prototype from the question wording and the project's existing question categories. The model has no held-out evaluation and was not used to produce or score the report's ten-question experiment. Do not claim that it improves answer quality or that its output is reliable for new domains. Add more independently labeled questions and a held-out evaluation before making performance claims.

## Train and run the router

Requires Python 3.10 or newer; no third-party packages are needed for the router.

```powershell
python src/train_router.py --data data/router_training.csv --output models/question_router.json
python src/predict_router.py --model models/question_router.json --question "How do the two papers' methods differ?"
```

The predictor returns a route and normalized model scores. Those scores are not calibrated probabilities.

## Original experiment

`app.py` implements the inference-time comparison described in the report:

- Three distinct model calls for evidence extraction, analysis, and answer synthesis.
- Three sequential calls to one model for draft, self-review, and revision.
- The same question and fixed source packet are passed to both conditions.

This original experiment does not train model weights. The trained router is a separate extension and should be described separately in the paper and presentation.

To run the original app, install `requirements.txt`, copy `.env.example` to `.env`, enter your own provider settings, and supply the authorized source packets. Never commit `.env` or API credentials. The source PDFs from the experiment ZIP were omitted from this cleaned repository; add only documents you may redistribute and retain their source citations and licenses.

After adding the source packets at the paths listed in the CSV, a one-item run can be started with:

```powershell
python app.py --dataset-csv data/experiment_dataset.csv --limit 1 --output-dir results
```

## Included project materials

- `data/experiment_dataset.csv`: the team's ten-question dataset from the supplied experiment package.
- `data/source_manifest.csv`: source titles, URLs, passage identifiers, and reuse/license notes from the package.
- `data/router_training.csv`: only the question text and the manually assigned router role labels; reference answers and evaluation data are not used to train the router.
- `data/evaluation_log.csv`, `data/run_log.csv`, `data/results_summary.*`, and `data/scoring_rubric.txt`: the supplied experiment's evaluation and aggregate records.
- `src/train_router.py`, `src/predict_router.py`, and `models/question_router.json`: training code, inference code, and the trained model artifact.
- `sources/README.md`: source-paper provenance and instructions for adding authorized local source files.

The report describes user-supplied overall scores, not independent second-rater scores. The results are a small descriptive pilot and should not be presented as proof that collaboration is better or worse in general.
