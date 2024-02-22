# Dynamics of Speed Dating A Machine Learning Analysis of Match-Patterns

## Introduction
### Overview
The Speed Dating dataset, compiled by Sheena S. Iyengar, serves as the foundation for a final project exploring decision-making in dating. Collected during a speed dating experiment conducted by professors from Columbia University, Harvard University, and Stanford University in New York City, the dataset aims to observe how decision-making in dating varies by gender and race. The project's objective is to build classification models determining whether two people are a match based on their characteristics, offering potential applications in recommending highly compatible matches in the online dating industry.

## Data Pre-Processing
The raw dataset, consisting of data from 21 controlled speed dating sessions with 8378 sample units and 195 columns, undergoes pre-processing to create probabilities of a match. Parameters specific to the experiment or created by the researcher are removed. The dataset is modified to represent each date by a single row, eliminating duplicates. Missing values are handled, and categorical variables for career and career fields are categorized as "other" for missing values. Binary variables for the same career and working in the same field are created for effectiveness. Certain waves of the experiment and redundant columns are removed.

## Data Analysis
Table 1 presents a summary of statistics for all variables within the speed dating set, providing descriptions, means, standard deviations, and minimum and maximum values.

### Exploratory Visuals
Figure 1 illustrates the distribution of the binary response variable, 'match,' where 1 represents a match and 0 represents not a match. Figures 2 and 3 visualize the age distribution of people in the dating set. Figure 4 represents the change in match rate over the order of dates. Figures 5 and 6 visualize the effects of having the same race and working in the same career field on matched couples.

## Model Implementation
### Metrics
The metric used for evaluating model performance is test set accuracy.

### Tree-Based Classification Methods
Initial tree-based methods include a single decision tree, bagging, random forest, and boosted decision tree. The test set accuracy for the single decision tree is 86.3%. Bagging results in an accuracy of 88.82%, while the random forest achieves 88.5%. The boosted decision tree has a test set accuracy of 87.4%.

### Linear-Based Classification Methods
Linear models such as logistic regression, linear-discriminant analysis (LDA), and quadratic-discriminant analysis (QDA) are implemented. Ridge and lasso regressions are applied to a generalized linear model.

#### Variable Selection
Backwards variable selection is employed to choose predictors for linear models.

#### Results
Table 4 summarizes the test set accuracy of various models, with the most accurate models being the random forest and bagged decision tree, both achieving 88.82% accuracy.

### Support Vector Machines
Linear, radial, and polynomial support vector machines (SVMs) are tested, with the polynomial SVM achieving the highest accuracy of 88.24%.

## Conclusion
The dataset, originally aimed at observing individual decisions in dating, has potential applications beyond its initial scope. Models developed using this dataset could prove valuable in the online dating industry. However, limitations such as the dataset's focus on heterosexual and cisgender individuals and its location-specific nature should be considered. The study's findings, including insights into gender differences and partner preferences, provide valuable perspectives for future research.
