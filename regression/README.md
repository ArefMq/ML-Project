# Regression
This part, contains project sample for Regression in Python using Sci-Kit tools.

## Run Regression
In order to run the project please follow <a href="https://github.com/ArefMq/DataMiningProject/blob/master/README.md#Runing Project">
this link</a>.

## Results
### Feature Selection
After searching the entire features power set, each subset is given to a Linear Regressor, and the
error of the model is calculated. Then, based on R-Squared of the result, these model are sorted.
Top 10 feature subset is demonstrated as below:

| #  | Subset                                                 | R-Squared |
|----|--------------------------------------------------------|-----------|
|  1 | residual sugar, pH                                     | 0.001     |
|  2 | pH                                                     | 0.002     |
|  3 | fixed acidity, residual sugar, free sulfur dioxide, pH | 0.002     |
|  4 | fixed acidity, free sulfur dioxide, pH                 | 0.002     |
|  5 | fixed acidity, free sulfur dioxide                     | 0.005     |
|  6 | fixed acidity, residual sugar, free sulfur dioxide     | 0.005     |
|  7 | fixed acidity, residual sugar, pH                      | 0.008     |
|  8 | chlorides, free sulfur dioxide                         | 0.008     |
|  9 | residual sugar, chlorides, free sulfur dioxide         | 0.009     |
| 10 | fixed acidity, pH                                      | 0.010     |


### Linear Regression
The Linear Regression model is selected for this part. Then, on 1000 iterations, the a random bootstrap of the data
is fed to the model and result (coefficient of the model) collected. Then, by variant of these coefficients the p-value
of each parameter is calculated. This task is done two times, one with full features and on with a subset of features.
The results are demonstrated below:

*Full Feature Model:*

| Field                      | SE         | t-Statistics  | P-value   |
|----------------------------|------------|---------------|-----------|
| fixed acidity              | 0.0015     |  54.9317      | 0.3971    |
| volatile acidity           | 0.0046     | -407.9326     | nan       |
| citric acid                | 0.0034     |  3.6015       | 0.3725    |
| residual sugar             | 0.0006     |  142.5016     | 0.3982    |
| chlorides                  | 0.0193     | -9.3186       | nan       |
| free sulfur dioxide        | 0.0000     |  73.5099      | 0.3976    |
| total sulfur dioxide       | 0.0000     | -9.8530       | nan       |
| density                    | 1.8038     | -93.8771      | nan       |
| pH                         | 0.0064     |  117.4798     | 0.3981    |
| sulphates                  | 0.0044     |  148.3923     | 0.3983    |
| alcohol                    | 0.0022     |  77.9685      | 0.3977    |


|             | RSS      | TSS      | R2    |
|-------------|----------|----------|-------|
| Train Error | 1083.854 | 1517.287 | 0.286 |
| Test Error  | 1700.998 | 2323.670 | 0.268 |
  


*Selected Feature Model:*

| Field                      | SE         | t-Statistics  | P-value  |
|----------------------------|------------|---------------|----------|
| fixed acidity              | 0.0007     | -129.3685     | nan      |
| residual sugar             | 0.0001     | -156.0511     | nan      |
| free sulfur dioxide        | 0.0000     |  41.5281      | 0.3965   | 
| pH                         | 0.0036     |  74.7831      | 0.3976   |


|             | RSS      | TSS      | R2    |
|-------------|----------|----------|-------|
| Train Error | 1483.573 | 1523.067 | 0.026 |
| Test Error  | 2270.294 | 2317.736 | 0.020 |

### Model Bench mark 
The **LinearRegression**, **Ridge**, **Lasso**, **ElasticNet** models has been compared together. The result are showed
below. Obviously, the Elastic-Net model worked best according to the R-Squared metric.


| Method               | Data            |   RSS        |   TSS        |   R2      |
|----------------------|-----------------|--------------|--------------|-----------|
| Logistic Regression  |   Train Error   |   1991.143   |   2811.063   |   0.292   |
|                      |   Test Error    |   775.305    |   1025.265   |   0.244   |
| Ridge Regression     |   Train Error   |   2021.061   |   2811.063   |   0.281   |
|                      |   Test Error    |   776.245    |   1025.265   |   0.243   |
| Lasso Regression     |   Train Error   |   2210.388   |   2811.063   |   0.214   |
|                      |   Test Error    |   851.525    |   1025.265   |   0.169   |
| **Elastic Net**      | **Train Error** | **2653.042** | **2811.063** | **0.056** |
|                      | **Test Error**  | **994.469**  | **1025.265** | **0.030** |
