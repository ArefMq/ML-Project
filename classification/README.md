# Classification
This part, contains project sample for Classification in Python using Sci-Kit tools.

## Run Regression
In order to run the project please follow <a href="https://github.com/ArefMq/DataMiningProject/blob/master/README.md#Runing Project">
this link</a>.

 
## Results
There are two classes in this dataset, The *red wine* class and *white wine* class. These data has been combined together
and then, using a Classification method, we try to separate them. The following method has been applied on the data and
the Result can be viewed as below. Obviously, **Linear Discriminant Analysis (LDA)** worked best on these data.

| Method                              |  Data         |  RSS    |  TSS      |  R2     |  Precision |  Recall | F-Score |
|-------------------------------------|---------------|---------|-----------|---------|------------|---------|---------|
| Logistic Regression                 |  Train Error  |  90.00  |  832.964  |  0.892  |  0.970     |  0.976  |  0.973  |
|                                     |  Test Error   |  32.00  |  359.303  |  0.911  |  0.976     |  0.979  |  0.978  |
| **Linear Discriminant Analysis**    |**Train Error**|**26.00**|**843.111**|**0.969**|**0.992**   |**0.993**|**0.992**|
|                                     |**Test Error** |**7.00** |**359.812**|**0.981**|**0.994**   |**0.996**|**0.995**|
| Quadratic Discriminant Analysis     |  Train Error  |  67.00  |  864.928  |  0.923  |  0.986     |  0.975  |  0.980  |
|                                     |  Test Error   |  20.00  |  367.443  |  0.946  |  0.990     |  0.982  |  0.986  |
| Gaussian Naive Bayes                |  Train Error  |  140.0  |  876.597  |  0.840  |  0.968     |  0.951  |  0.959  |
|                                     |  Test Error   |  45.00  |  374.056  |  0.880  |  0.978     |  0.962  |  0.969  |
| Linear Regression                   |  Train Error  | 120.097 |  724.029  |  0.834  |            |         |         |
|                                     |  Test Error   |  45.993 |  316.009  |  0.854  |            |         |         |


Here is the ROC Curve of the mentioned methods using 5-Fold Cross Validation:
![alt text][roc5fold]

Here is the ROC Curve of the mentioned methods using Leave-One-Out Cross Validation:
![alt text][rocloocv]


[roc5fold]: https://github.com/ArefMq/DataMiningProject/blob/master/classification/roc5fold.png "ROC using 5-Fold CV"
[rocloocv]: https://github.com/ArefMq/DataMiningProject/blob/master/classification/rocloocv.png "ROC using LOOCV"
