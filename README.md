# Effect of Chi-Square Feature Selection on Traditional Machine Learning Models for Fake News Classification

## Overview

This project studies how Chi-Square feature selection affects the traditional machine learning models used to classify news articles as fake or real. It compares models trained with the full TF-IDF feature set against models trained with a reduced feature set.

The experiments focus on classification performance, feature dimensionality, and training time. The results show that the effect of feature selection varies by classifier: it slightly improves Multinomial Naive Bayes accuracy, while Logistic Regression and Linear SVM perform marginally better without selection.

## Objectives

- Classify news articles as fake or real using traditional machine learning.
- Represent article text with TF-IDF unigram and bigram features.
- Use Chi-Square to select a smaller set of features.
- Compare Logistic Regression, Multinomial Naive Bayes, and Linear SVM with and without selection.
- Evaluate accuracy, macro precision, macro recall, macro F1-score, and training time.

## Dataset

The dataset is formed by combining `Fake.csv` and `True.csv`:

- Fake articles receive label `0`.
- Real articles receive label `1`.
- Each article's title and text are combined for feature extraction.
- Missing or empty content is handled, and duplicate articles are removed.
- The prepared dataset contains 39,103 articles.

The CSV files are in the `dataset/` directory. The experiment notebook is `CCA_Misinformation_Pattern_Recognition (1).ipynb`.

## Methodology

1. Combine and clean the fake and real article data.
2. Split the data into training and test sets using an 80/20 stratified split.
3. Convert text to TF-IDF features using unigrams and bigrams, with up to 5,000 features.
4. For the feature-selected configuration, apply Chi-Square and retain the top 3,000 features.
5. Train and evaluate each classifier under both configurations.

This produces six configurations: three classifiers, each evaluated with and without Chi-Square feature selection.

## Results

| Classifier | Without Chi-Square | With Chi-Square |
|---|---:|---:|
| Logistic Regression | 99.25% | 99.17% |
| Multinomial Naive Bayes | 95.19% | 95.40% |
| Linear SVM | **99.67%** | 99.62% |

Accuracy values are from the recorded experiments. The results show that Linear SVM achieved the highest accuracy in both configurations. Chi-Square selection slightly improved Multinomial Naive Bayes accuracy and slightly reduced accuracy for Logistic Regression and Linear SVM. The reduced feature set also reduced training time for all three models in the recorded experiments.

Detailed results are available in `cca_results/classification_reports.csv` and `cca_results/model_comparison.csv`.

## Conclusion

Chi-Square feature selection reduces the TF-IDF feature count from 5,000 to 3,000 and reduced training time in these experiments, but it does not improve accuracy for every classifier. Its effect should be evaluated for each model and dataset. Linear SVM delivered the strongest classification accuracy overall.

## Run the notebook

Install Python and Jupyter, along with the libraries used by the notebook: pandas, NumPy, scikit-learn, Matplotlib, and Seaborn. Then open `CCA_Misinformation_Pattern_Recognition (1).ipynb` in Jupyter or VS Code and run its cells. Keep the dataset files in the repository's `dataset/` directory so the notebook can load them.

## Repository contents

```text
.
|-- CCA_Misinformation_Pattern_Recognition (1).ipynb
|-- cca_results/
|   |-- classification_reports.csv
|   `-- model_comparison.csv
`-- dataset/
    |-- Fake.csv
    `-- True.csv
```

## Technologies

Python, Jupyter Notebook, Pandas, NumPy, scikit-learn, TF-IDF, Chi-Square feature selection, Matplotlib, and Seaborn.
