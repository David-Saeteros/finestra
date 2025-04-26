# FINESTRA (Framework for Interpretable Neural Explanation of Semantic Trait Recognition Analysis)
## Personality Trait Classification with Explainable AI

This repository contains experiments for fine-tuning language models to predict personality traits from text, along with explainability analysis to understand model decisions.

## Associated Publication
This code repository accompanies our paper:
Title: Text speaks louder: Insights into Personality from Natural Language Processing
Authors: Saeteros, D., Gallardo-Pujol, D., & Ortiz-Martínez, D.
Journal: PLOS ONE
Status: In press
Year: 2025
If you use this code or the findings from our research, please cite our paper:
bibtex@article{Saeteros2025,
  title={Text speaks louder: Insights into Personality from Natural Language Processing},
  author={Saeteros, D. and Gallardo-Pujol, D. and Ortiz-Martínez, D.},
  journal={PLOS ONE},
  year={2025},
  note={In press}
}

## Project Structure

The repository is organized as follows:

```
├── explainability/
│   ├── big_five/
│   │   ├── explainable_ai_expers_essays_agr.ipynb
│   │   ├── explainable_ai_expers_essays_con.ipynb
│   │   ├── explainable_ai_expers_essays_ext.ipynb
│   │   ├── explainable_ai_expers_essays_neu.ipynb
│   │   ├── explainable_ai_expers_essays_opn.ipynb
│   │   └── utils.py
│   └── mbti/
│       ├── masked/
│       │   ├── explainable_ai_expers_masked_mbti_ft_bert.ipynb
│       │   ├── explainable_ai_expers_masked_mbti_ie_bert.ipynb
│       │   ├── explainable_ai_expers_masked_mbti_jp_bert.ipynb
│       │   ├── explainable_ai_expers_masked_mbti_ns_bert.ipynb
│       │   └── utils.py
│       └── unmasked/
│           ├── explainable_ai_expers_mbti_ft_bert.ipynb 
│           ├── explainable_ai_expers_mbti_ie_bert.ipynb
│           ├── explainable_ai_expers_mbti_jp_bert.ipynb
│           ├── explainable_ai_expers_mbti_ns_bert.ipynb
│           └── utils.py
├── fine_tuning/
│   ├── fine_tuning_expers_essays.ipynb
│   └── fine_tuning_expers_mbti.ipynb
```

## Datasets

Two main datasets are used in this project:
1. **Essays dataset** (Big 5 personality traits: Extraversion, Neuroticism, Agreeableness, Conscientiousness, Openness)
2. **MBTI dataset** (Myers-Briggs Type Indicator dimensions: I/E, N/S, F/T, J/P)

YOU CAN GET THEM UPON REQUEST TO THE AUTHORS

## Setup

### Requirements

Install the required packages:

```bash
pip install torch transformers transformers-interpret memory_profiler datasets accelerate nltk tweet-preprocessor pandas matplotlib seaborn wordcloud
```

### Data Preparation

The data is already preprocessed and available (you can get them upon request):
* Essays dataset: `df_essays_fragm_train.csv`, `df_essays_fragm_valid.csv`, and `df_essays_fragm_test.csv`
* MBTI dataset: `df_mbti_fragm_train.csv`, `df_mbti_fragm_valid.csv`, and `df_mbti_fragm_test.csv`

## Fine-Tuning

The `fine_tuning/` directory contains scripts for fine-tuning BERT and RoBERTa models on both datasets:

* `fine_tuning_expers_essays.ipynb`: Fine-tunes models on the Big Five personality traits
* `fine_tuning_expers_mbti.ipynb`: Fine-tunes models on the MBTI dimensions with options for masking MBTI-related terms

Fine-tuned models are saved to the `./fine_tuning_models/` directory.

## Explainability Analysis

The `explainability/` directory contains notebooks for analyzing and visualizing what influences model predictions:

### Big Five Personality Traits

Separate notebooks for each of the Big Five traits:
* `explainable_ai_expers_essays_agr.ipynb`: Agreeableness
* `explainable_ai_expers_essays_con.ipynb`: Conscientiousness
* `explainable_ai_expers_essays_ext.ipynb`: Extraversion
* `explainable_ai_expers_essays_neu.ipynb`: Neuroticism
* `explainable_ai_expers_essays_opn.ipynb`: Openness

### MBTI Dimensions

For MBTI, both masked and unmasked versions are provided:

#### Unmasked MBTI
* `explainable_ai_expers_mbti_ie_bert.ipynb`: Introversion/Extraversion
* `explainable_ai_expers_mbti_ns_bert.ipynb`: Intuition/Sensing
* `explainable_ai_expers_mbti_ft_bert.ipynb`: Feeling/Thinking
* `explainable_ai_expers_mbti_jp_bert.ipynb`: Judging/Perceiving

#### Masked MBTI (MBTI terms removed to avoid data leakage)
* `explainable_ai_expers_masked_mbti_ie_bert.ipynb`: Introversion/Extraversion
* `explainable_ai_expers_masked_mbti_ns_bert.ipynb`: Intuition/Sensing
* `explainable_ai_expers_masked_mbti_ft_bert.ipynb`: Feeling/Thinking
* `explainable_ai_expers_masked_mbti_jp_bert.ipynb`: Judging/Perceiving

## Features

### Fine-tuning
* Data preprocessing and fragmentation
* Text tokenization and model fine-tuning
* ROC AUC and accuracy evaluation metrics
* Support for both BERT and RoBERTa models

### Explainability Analysis
* Attribution analysis using `transformers-interpret`
* Word-level attribution visualization
* Word clouds and bar plots for influential words
* Comparison between correct and incorrect predictions
* Attribution summaries using multiple aggregation techniques

## Utilities

The `utils.py` files provide common functions for:
* Creating datasets and dataloaders
* Computing evaluation metrics
* Text preprocessing and tokenization
* Attribution filtering and visualization
* Word frequency analysis

## Output

The explainability notebooks generate:
* Word clouds showing the most influential words for each class
* Bar plots ranking words by attribution scores
* Histograms of attribution scores
* HTML visualizations of attributions in specific texts
* Pickle files containing attribution data for further analysis
