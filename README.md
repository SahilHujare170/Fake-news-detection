Effect of Chi-Square Feature Selection on Traditional Machine Learning Models for Fake News Classification

1. Project Overview

This project investigates the effect of Chi-Square feature selection on traditional machine learning models for fake news classification. The main objective is to compare the performance of three classifiers under two configurations: using the complete TF-IDF feature set and using a reduced feature set selected through the Chi-Square statistical method.

The project focuses on understanding whether reducing the number of text features improves classification performance, reduces computational requirements, or affects different classifiers in different ways.

2. Objectives

Develop a machine learning-based approach for classifying news articles as Fake or Real.

Extract meaningful textual features using TF-IDF.

Apply Chi-Square feature selection to reduce the feature space.

Compare Logistic Regression, Multinomial Naive Bayes, and Linear SVM.

Evaluate the impact of feature selection using standard classification metrics.

3. Dataset

The project uses two datasets:

Fake.csv – Contains fake news articles.

True.csv – Contains real news articles.

The datasets are combined to create a binary classification dataset.

Dataset preparation:

Fake news is assigned label 0.

Real news is assigned label 1.

Article titles and text are combined for feature extraction.

Missing and empty content is handled.

Duplicate articles are removed.

After preprocessing, the dataset contains 39,103 articles.

4. Methodology

The implementation follows these steps:

Text Preprocessing: Combine article titles and content, handle missing values, and remove duplicate entries.

TF-IDF Feature Extraction: Convert text into numerical representations using unigram and bigram features.

Chi-Square Feature Selection: Select the top 3,000 features from the original 5,000 TF-IDF features.

Model Training: Train and evaluate three traditional machine learning classifiers using both feature configurations.

Performance Evaluation: Compare accuracy, macro precision, macro recall, macro F1-score, and training time.

The dataset is divided into 80% training data and 20% testing data using stratified splitting.

5. Machine Learning Models

Each classifier is evaluated in two configurations:

Classifier

Configuration 1

Configuration 2

Logistic Regression

Without Chi-Square

With Chi-Square

Multinomial Naive Bayes

Without Chi-Square

With Chi-Square

Linear SVM

Without Chi-Square

With Chi-Square

This results in six experimental configurations.

6. Experimental Results

The following table presents the observed classification accuracy:

Classifier

Without Chi-Square

With Chi-Square

Logistic Regression

99.25%

99.17%

Multinomial Naive Bayes

95.19%

95.40%

Linear SVM

99.67%

99.62%

7. Key Observations

Linear SVM achieved the highest accuracy in both configurations, reaching 99.67% without Chi-Square feature selection.

Multinomial Naive Bayes showed a slight improvement in accuracy, from 95.19% to 95.40%, after applying Chi-Square selection.

Logistic Regression experienced a small decrease in accuracy, from 99.25% to 99.17%.

Linear SVM also showed a marginal decrease, from 99.67% to 99.62%.

Chi-Square feature selection reduced the number of features from 5,000 to 3,000 and reduced training time across all three classifiers in the recorded experiments.

8. Conclusion

The experimental results indicate that the effectiveness of Chi-Square feature selection depends on the classifier being used. Although feature selection reduces dimensionality and training time, it does not necessarily improve classification accuracy.

Multinomial Naive Bayes benefited slightly from the reduced feature set, whereas Logistic Regression and Linear SVM performed marginally better without feature selection. Linear SVM delivered the strongest overall classification performance among the evaluated models.

Therefore, feature selection should be evaluated in the context of the individual classifier rather than assumed to improve every machine learning model.

9. Technologies Used

Python

Pandas and NumPy

Scikit-learn

TF-IDF Vectorization

Chi-Square Feature Selection

Matplotlib and Seaborn

Jupyter Notebook
