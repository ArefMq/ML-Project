# Classification
This part, contains project sample for Classification in Python using Sci-Kit tools.

## Run Regression
In order to run the project please follow <a href="https://github.com/ArefMq/DataMiningProject/blob/master/README.md#Runing Project">
this link</a>.

 
## Results

The goal in this part was to train a Tree-Based Regression in order to predict the quality of the wine.
To achieve this goal, first a proper value, *k* should be determined, representing the depth of the tree. This value
has been found through training trees with different depth (from 2 to 20) then, select the one with lowest error (here
we use R-Squared). However, the best result appeared when using the **5-Fold cross validation** to calculate R-Squared 
values, and then use the average of those errors for the correspondent K-Depth trees.

The best found value for *k* is 15 in this example. And here are the result for a Tree-Based Regression with depth of 
15.

| Method                    | Data        | RSS     | TSS      | R^2   |
|---------------------------|-------------|---------|----------|-------|
| Tree-Based Regression     | Train Error | 273.908 | 2865.883 | 0.904 |
| Tree-Based Regression     | Test Error  | 782.631 | 975.073  | 0.197 |


In this part, tree different method (**Random Forest Classifier**, **Decision Tree Classifier**, and **Support Vector 
Classifier**) are compared together in order to find the best classifier. Here are the results for this comparision.
In this example **Decision Tree Classifier** has the highest F-Score and R-Squared value.

| Method                    | Data        | RSS     | TSS     | R^2   | Precision | Recall | F-Score |
|---------------------------|-------------|---------|---------|-------|-----------|--------|---------|
| Random Forest Classifier  | Train Error | 120.000 | 841.074 | 0.857 | 0.949     | 0.980  | 0.963   | 
| Random Forest Classifier  | Test Error  | 52.000  | 364.372 | 0.857 | 0.948     | 0.980  | 0.963   | 
| Decision Tree Classifier  | Train Error | 13.000  | 841.074 | 0.985 | 0.996     | 0.996  | 0.996   |
| Decision Tree Classifier  | Test Error  | 33.000  | 364.372 | 0.909 | 0.978     | 0.976  | 0.977   |
| Support Vector Classifier | Train Error | 263.000 | 841.074 | 0.687 | 0.898     | 0.943  | 0.918   |
| Support Vector Classifier | Test Error  | 118.000 | 364.372 | 0.676 | 0.903     | 0.932  | 0.916   |

Here is the ROC Curve of the mentioned methods:
![alt text][roc]

[roc]: https://github.com/ArefMq/DataMiningProject/blob/master/svm/roccurve.png "ROC"


