# Dynamics of Speed Dating: A Machine Learning Analysis of Match-Patterns


## Introduction
### Overview
The Speed Dating dataset, originally compiled by Sheena S. Iyengar, is being utilized for a final project to investigate decision-making in dating. This dataset was collected during a speed dating experiment conducted by professors from Columbia University, Harvard University, and Stanford University in New York City. The study aimed to observe how decision-making when dating varies between genders and races. This project's objective is to build classification models to determine whether two people are a match based on their characteristics. The binary response variable within the dataset is 'match,' equal to one when both parties like each other. This model has the potential to assist data scientists in the online dating industry by recommending highly compatible matches to users.

## Data Pre-Processing
The raw dataset consists of data from 21 controlled speed dating sessions, with 8378 different sample units and 195 columns of data. The goal for my modeling of the data is to create probabilities of a match based on the characteristics of a person and their potential partner. Therefore, I will be eliminating all the parameters that were specific to the experiment or created by the researcher. For example, there are many columns in the dataset based on participants' feedback during and after the experiment, and I am just going to focus on what they thought before. In addition, I will be taking out columns that pertain to the position of the participant during the experiment. Waves 6-9 of the experiment measured values differently than the other waves, so they were removed from the dataset.

In the dataset, each round of speed dating between two partners is represented by two rows: one represents the male, and the other represents the female. Therefore, we have double the rows needed and need to merge them so that each date is represented by only one row. The females were used as the principal, and their rows were merged together with the male data and renamed the columns to reflect this change. Data consistent within both datasets, such as interest correlation (intr_corr), was taken out of the male dataset so it was not represented in two columns.

To eliminate some missing values within the dataset, a category was made for the categorical variables for career and career field to represent a missing value as “another” career or career field. To make these variables more effective, instead of factoring the categorical variables, two binary variables were created to reflect if the couple had the same career or worked in the same field. Finally, decision and gender variables were removed due to perfect multicollinearity with the response.

## Data Analysis
Table 1 shows a summary of statistics for all the variables within the speed dating set used to build the models. Each variable has a corresponding description, mean, standard deviation, minimum value, and maximum value.

![Table 1](Visualizations/Table1.png)

### Exploratory Visuals
Figure 1 shows the distribution of the binary response variable, match, with 1 representing a match and 0 representing not a match.

![Figure 1](Visualizations/Figure1.jpg)


With approximately 1 out of 5 sample units being a match, 1 is the underrepresented group; however, the ratio, while not ideal, is sufficiently workable and within an acceptable range to proceed with classification. Figures 2 and 3 visualize the age distribution of people in the dating set.

![Figure 2](Visualizations/Figure%202.jpg)
![Figure 3](Visualizations/Figure3.jpg)

Through creating a normal curve, in Figure 2 the average female age from the study is 26, and in Figure 3 the average male age is also around 26. While the average age of the men is slightly more than women, there is more variance in the age of men; however, the maximum male age is only 42 while the maximum female age in the dataset is 55.

Figure 4 represents the change in match rate over order, or the number of dates they have met with that day.

![Figure 4](Visualizations/Figure4.jpg)


Each day of the study included meeting 22 people, so the order ranged from 1 – 22 in every sample unit. We see a decrease in the match rate from the first person they met until around 14, then we see a significant increase towards the end of the study. In a real-world setting, this could represent the number of profiles already seen by a user. Figure 5 and Figure 6 visualize the effects of having the same race and working in the same career field on matched couples.

Figure 5 shows that having the same career field has a minimal effect on matching with a partner; however, in Figure 6, around 40% of matches and non-matches were the same race.

![Figure 5](Visualizations/Figure5.jpg)
![Figure 6](Visualizations/Figure6.jpg)


## Model Implementation
### Metrics
The metric used to measure the goodness-of-fit of each model explored within the project is test set accuracy. This yields an unbiased measure of model performance by using the model to predict 20% of the dataset not used in training the model.

The accuracy is calculated using the formula: **Add formula**


### Tree-Based Classification Methods
The initial tree-based method involved creating a single decision tree, where the response variable was "match," and all other variables were used as predictors. The resulting tree is illustrated in Figure 07 and was formed using a Cp value of 0.0095969 and training using the training set consisting of 80% of pre-processed sample units.

![Figure 7](Visualizations/Figure7.jpg)

To optimize the tree, the original tree was first generated using the default Cp value. The package then generated a list of Cp values along with their corresponding cross-validation errors. The Cp value with the lowest cross-validation error was then selected, and the tree was re-run. This process was repeated with a list of Cp values with a smaller range until the optimal Cp value from the list matched the one used to create the original tree. Figure 08 displays the list of possible Cp values with their corresponding cross-validation errors and tree sizes.

![Figure 8](Visualizations/Figure8.jpg)

The test set accuracy of the single decision tree, depicted in Figure 07, is 86.3%. Figure 08	
To improve the single-tree method above, multi-tree methods were used to improve the accuracy of the tree model. First, a bagging method was used to create a model from multiple decision trees. For bagging, all 35 features were used to create each tree. The bagged decision tree had a test-set accuracy of 88.82%. From the model, the importance of each predictor contributing to the response was visualized in Figure 09. It is important to note the features deemed the most important through the bagged decision tree are included in the single, pruned decision tree.

![Figure 9](Visualizations/Figure9.jpg)

After a bagged decision tree, a random forest was made where only a third of the features were used to create each tree. The accuracy of this method was 88.5%, a slight decrease from the results from bagging. The advantage of using a random forest over bagging is the added variation through only using one-third of the predictors in each tree to prevent strong predictors in each tree. Therefore, since test set accuracy did not increase, there are no predictors that overpowered other predictors. However, as seen in Figure 10, predictors with less importance from the bagged decision tree have more importance in the random forest since they were used in more trees due to only using a third of the predictors.

![Figure 10](Visualizations/Figure10.jpg)

The final tree method used for classification was a boosted decision tree. For this method, 5000 trees were formed at the optimal depth of 4. The boosted tree had a test set accuracy of 87.4% which is the least accurate from the multi-tree methods but more than the single tree.  
In Figure 11, the importance of each variable in the boosted tree is very similar to those in the random forest in Figure 10. 

![Figure 11](Visualizations/Figure11.jpg)

One of the main distinctions between the boosted tree and other tree models is the significance of int_corr and order. These variables were deemed insignificant in the single decision tree, random forest, and bagged tree models. However, in the boosted tree model, formed using residuals, int_corr and order were the second and fourth most important variables, respectively.

Table 2 compares the importance of variables within the dataset across different tree techniques. In the random forest and bagged tree models, the mean decrease Gini index was utilized to determine the importance level of each variable. The mean decrease Gini index assesses the extent to which each variable contributes to the homogeneity of the nodes and leaves. The Gini index plays a crucial role in the formation of splits in a random forest and bagged tree. Therefore, the greater the Gini index, the more important the variable.

![Table 2](Visualizations/Table%202.png)

However, in the boosted decision tree models, the Gini index cannot be employed to measure importance since the trees
are formed through residuals. Instead, the relative influence of each variable is used to determine its importance. The higher the relative influence, the more significant the variable.

## Linear-Based Classification Methods

Here is the revised version with corrected grammar:

In addition to tree-based classification methods, several linear-based classification models, including logistic regression, linear-discriminant analysis (LDA), and quadratic-discriminant analysis (QDA), were utilized. Furthermore, ridge and lasso regressions were performed on a generalized linear model, and the L1 and L2 coefficient penalties were used to explore if other models were overfitting the dataset.

### Variable Selection

Backward variable selection was employed to determine which variables would offer the most predictive power without overfitting the dataset, thus avoiding poor test-set performance.

### Logistic Regression 

A logistic regression model was trained using the training set and the variables from backward selection. The model was then used to predict matches on the test set, returning a probability. Therefore, a threshold must be used to create the binary classification. The optimal threshold was determined to be 0.63 for the model, producing an 85% test set accuracy.

### Linear-Discriminant Analysis 

An LDA model, or linear-discriminant analysis, was trained using the same training set. Additionally, the same predictors were used within this model as in the logistic regression model. While testing additional predictors within the model, it was observed that the model overfit the dataset, leading to a lower test set accuracy. Similarly, to the logistic regression, a 0.63 threshold was used to classify the output of predictions on the test set. This model produced a test set accuracy of 86%. 

### Quadratic-Discriminant Analysis 

A QDA, or quadratic-discriminant analysis, model was trained using the same test set, predictors, and threshold as the previous two models. Being the most complex, this model proved to be the most accurate with an 87% test set accuracy.

### Ridge and Lasso Regression 

The L1 and L2 penalties from the Ridge and Lasso regressions were applied to a generalized linear model (GLM) to explore if a coefficient penalty benefits the model by making it more generalized to avoid overfitting. A 5-fold cross-validation yielded an optimal lambda of 0.002140288 for the ridge regression and 0.01545461 for the lasso regression.

Figure 12 shows that the optimal lambda for the ridge regression is where the binomial deviance is the smallest. Deviance is the opposite measure of log-likelihood on the validation set. Therefore, since we want to maximize the log-likelihood with lambda, we want to minimize the deviance. Figure 13 models the same as Figure 12 but for the ridge regression and its lambda. The first dotted, vertical line represents the optimal lambda where the deviance is minimized, and the second represents one standard error away from the optimal lambda. Since a small lambda corresponds with a negative log(lambda), we can see in both figures that our lambda, or penalty, is small.

![Figure 12](Visualizations/Figure%2012.jpg)
![Figure 13](Visualizations/Figure13.jpg)

Additionally, in Figure 12, we observe the elimination of variables as the penalty term becomes larger. Figure 13 has a constant 34 variables since the L2 penalty can only minimize coefficients and not reach zero to eliminate the predictor. Figure 12 shows that the optimal lambda uses 29 predictors instead of the 34 in the dataset.

Table 03 shows the coefficients after the L1 and L2 penalties were applied to them using ridge and lasso regression. In the lasso column, the variables eliminated were career_c_F, career_c_M, date_F, intel_F, and order. These variables are consistent with variables with a less importance in the bagged decision tree and random forest. 

![Table 3](Visualizations/Table03.png)


### Support Vector Machines

The final classification model used on the dataset is a support vector machine. Three support vector machines were tested: linear, radial, and polynomial. Each model used a 5-fold cross-validation method to train the model. This validation method allows us to find the optimal cost parameter for the linear model, as well as the gamma for the radial and polynomial models.

The linear model, with a cost parameter of 1, produced a test-set accuracy of 87.94%. The polynomial model, with a cost parameter of 1 and gamma of 0.01677244, produced a test-set accuracy of 88.24%. Finally, the radial model, with a cost parameter of 1 and a gamma of 0.01, produced a test-set accuracy of 88.04%. Therefore, the polynomial model produces the best test-set accuracy, indicating that the dataset must have a non-linear shape that can be separated.

## Results


 After performing all the models, the most accurate models were the random forest and bagged decision tree while the least accurate were the Ridge and Lasso regression model, as shown in Figure 14.
 
![Figure 14](Visualizations/Figure14.jpg)

The models that exhibited the best performance were the complex and flexible models, which typically have low bias and high variance. These types of models tend to perform better when dealing with datasets that have many predictors and fewer sample units. Although these models can fit the training set very well, they have a greater potential to overfit the test set, resulting in a low test-set accuracy. However, in this dataset, there was no evidence of overfitting in the complex models since ridge and lasso regression, which are used to prevent overfitting, did not produce better results.

To increase the number of sample units and balance the dataset, the response was oversampled, resulting in a training set with 40% matches. However, when trained on the same models as the original, the oversampled training set did not perform better.

When examining the significance of variables in the dataset across all models except the boosted decision tree, the same variables were consistently identified as significant. Additionally, in the boosted tree, 'like_F' was the most significant predictor for determining whether a couple would be a match.


## Conclusion

In addition to the insights gained from observing individual decisions and gender differences in partner selection, the data collected in this study holds potential for various real-world applications and implications. Understanding the factors that influence dating matches can have significant impacts on several aspects of society and individuals:

1. **Online Dating Platforms:** The findings can be utilized to enhance online dating platforms by tailoring recommendations based on individuals' interests, career fields, and other significant attributes. This could lead to more accurate and personalized matchmaking, improving the overall user experience.

2. **Promoting Inclusivity:** Recognizing the limitations of the dataset, efforts can be made to expand research to include a more diverse and representative sample. This would help ensure that dating algorithms and models are inclusive and considerate of various demographics, orientations, and gender identities.

3. **Addressing Gender Biases:** Insights into how gender biases affect partner selection could be applied to develop interventions or educational programs aimed at promoting more equitable and respectful dating practices. This could contribute to fostering healthier relationships and breaking down stereotypes.

4. **Cultural Sensitivity:** The study's focus on racial differences in dating opens the door to exploring cultural nuances and preferences. This knowledge could be used to develop culturally sensitive matchmaking algorithms that account for diverse cultural backgrounds, promoting understanding and compatibility.

5. **Ethical Considerations in Dating Apps:** The ethical implications raised regarding dating apps prioritizing user satisfaction over successful matches prompt a critical examination of the ethical standards and practices within the online dating industry. This could lead to the development of guidelines or regulations to ensure transparency and fairness in matchmaking algorithms.

6. **Impact on Society:** Understanding how intelligence and ambition influence partner selection sheds light on societal attitudes and expectations. This knowledge can be leveraged to challenge and reshape societal norms, fostering a more inclusive and supportive environment for individuals of all backgrounds and aspirations.

**Acknowledging Limitations:** It is important to note that the dataset had limitations in terms of inclusivity and representation. The majority of participants were white, and the study focused solely on heterosexual individuals. Future research endeavors should strive for a more diverse and comprehensive dataset to ensure broader applicability and relevance to a wider range of individuals.

In conclusion, while the study's primary focus was on building models for dating predictions, the implications of the findings extend beyond individual match outcomes. The data has the potential to shape the future of dating platforms, promote inclusivity, address biases, and contribute to a more thoughtful and ethical approach to online matchmaking. The responsibility lies not only with researchers but also with industry stakeholders to apply these insights ethically and responsibly for the benefit of users and society as a whole.



