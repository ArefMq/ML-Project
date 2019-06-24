# Neural Network
This part, contains project sample for Neural Network in Python.

## Run Regression
In order to run the project please follow <a href="https://github.com/ArefMq/DataMiningProject/blob/master/README.md#Runing Project">
this link</a>.

 
## Results
In order to achieve a clustering method using Neural Network approaches, we have to come up with a Network design.
In this problem, I have employed 5 different network structure, and then test via 5-Fold Cross validation. The result
is depicted below as well as the network structures.

![alt text][NetworkStructure]

| Model   | Data        | RSS    | TSS     | R^2   | Precision | Recall | F-Score |
|---------|-------------|--------|---------|-------|-----------|--------|---------|
| model#1 | Train Error | 80.980 | 839.030 | 0.903 | 0.970     | 0.972  | 0.971   |
|         | Test Error  | 36.507 | 366.374 | 0.900 | 0.967     | 0.977  | 0.972   |
| model#2 | Train Error | 110.708| 839.030 | 0.868 | 0.939     | 0.974  | 0.955   |
|         | Test Error  | 56.772 | 366.374 | 0.845 | 0.928     | 0.971  | 0.947   |
| model#3 | Train Error | 95.325 | 839.030 | 0.886 | 0.953     | 0.975  | 0.963   |
|         | Test Error  | 47.468 | 366.374 | 0.870 | 0.949     | 0.973  | 0.960   |
| model#4 | Train Error | 79.120 | 839.030 | 0.906 | 0.968     | 0.970  | 0.969   |
|         | Test Error  | 37.372 | 366.374 | 0.898 | 0.965     | 0.973  | 0.969   |
| model#5 | Train Error | 73.548 | 839.030 | 0.912 | 0.969     | 0.977  | 0.973   |
|         | Test Error  | 35.177 | 366.374 | 0.904 | 0.966     | 0.980  | 0.973   |
 
As you can see, these models has very small differences that can be neglected.


This is the ROC Curve of Models:
![alt text][roc5fold]
![alt text][roc5foldzoom]


### Comparing to the <a href="https://github.com/ArefMq/DataMiningProject/blob/master/classification/README.md">Second Project</a>
In the second project, the LDA classifier achieved the best results with F-Score of 0.995 on the test data. And as, we
can see that designed Neural Networks has relatively lower F-Score than the LDA. However, this does not mean in general 
LDA works better than every Neural Network in this problem. But there could be a Network that has better outcome than
my designs. Hence, we can conclude, on this dataset and with my knowledge and my effort, the LDA has shown a better result.


[NetworkStructure]: https://github.com/ArefMq/DataMiningProject/blob/master/nn/network_structure.png "Network Structure"
[roc5fold]: https://github.com/ArefMq/DataMiningProject/blob/master/nn/roc5fold.png "ROC"
[roc5foldzoom]: https://github.com/ArefMq/DataMiningProject/blob/master/nn/roc5fold_zoom.png "ROC Zoom"
