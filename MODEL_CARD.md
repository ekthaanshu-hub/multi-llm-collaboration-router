# Model card: question-role router

## Intended use

Suggest one of three workflow roles for a research-paper question: evidence extraction, method analysis, or cross-paper synthesis. The output can help demonstrate a trainable component for the project. It is not an answer generator and does not retrieve or verify evidence.

## Training data

Ten question prompts (Q001-Q010) from the team's supplied experiment dataset. The three role labels were manually grouped for this prototype from the question topics. Class counts are 3 evidence-extractor, 5 method-analyst, and 2 cross-paper-synthesizer questions.

## Method

Multinomial Naive Bayes trained from scratch with unigram and bigram token counts and Laplace smoothing. The implementation uses Python's standard library. `models/question_router.json` contains the learned vocabulary and log probabilities.

## Evaluation and limits

No held-out evaluation was run. Ten examples are too few to support a generalization or accuracy claim, and the manual grouping may be subjective. The relative scores returned by the predictor are not calibrated probabilities. Do not use this router to make decisions without human review. It was not part of the ten-question comparison reported in the paper.

## Data privacy and sources

The router-training CSV contains question text and role labels only. It excludes API credentials, model outputs, reference answers, and source PDFs. The question text originates in the team's experiment package; the relevant source papers are identified in the report and dataset manifest.
