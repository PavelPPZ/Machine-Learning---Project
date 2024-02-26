---
title: "Machine Learning Project"
author: "Pavel"
date: "2024-02-26"
output: html_document
keep_md: true
---




# Summary
The use of wearable devices made it possible to collect large amount of data about personal activity relatively inexpensively. These devices are used by a group of enthusiasts who take measurements about themselves regularly to improve their health. In this project the goal is to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. 

The goal of the project is to predict the manner in which they did the exercise. This is the "classe" variable in the data set. 

Provided report describes and explains how the model was built, cross validated, what is the expected out of sample error. The selected prediction model is used to predict 20 different test cases. 

### Data
The training data for this project are available here: 

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

### Approach

The extensive data set of almost 2000 observations was used for model construction and testing. 

For this data set, “participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in 5 different fashions: - exactly according to the specification (Class A), - throwing the elbows to the front (Class B), - lifting the dumbbell only halfway (Class C), - lowering the dumbbell only halfway (Class D), - throwing the hips to the front (Class E)." The outcome variable is "classe". 

Two models were built and tested: Decision Tree and Random Forest. The model with the highest accuracy (Random Forest) was chosen as the final model. Prediction for the testing data set was done and accuracy was assessed.

### Cross validation

Cross-validation was done by sub-setting the training data set into 2 sub-samples: trainingNew data set (70%) and validationNew data set (30%). Models were trained on the trainingNew data set, and validated on the validationNew data set. The most accurate model  was tested on the original Testing data set.

### Expected out-of-sample error

The expected out-of-sample error can be assessed as 1 - accuracy on the validation data set. Accuracy is the proportion of correctly classified observation over the total sample in the validation data set. 

Expected accuracy is the expected accuracy in the out-of-sample data set (i.e. original testing data set). Thus, the expected value of the out-of-sample error should correspond to the expected percentage of mis-classified observations in the Test data set.

### Used Libraries

```r
library(caret); library(randomForest); library(rpart); library(RColorBrewer); library(rattle); library(ggplot2);library(lattice)
```

### Getting and cleaning the data
Data was downloaded from the links above and uploaded into R. Columns with NA data and columns not related to the model construction were removed. Variable "classe" was converted into factor variable.


```r
## reading data, adding NA for missing info
training <- read.csv("pml-training.csv", na.strings=c("NA","#DIV/0!",""))
testing <- read.csv("pml-testing.csv", na.strings=c("NA","#DIV/0!",""))

## removing the columns with NAs 
training <- training[,colSums(is.na(training))==0]
testing <- testing[,colSums(is.na(testing))==0]

## removing the columns 1-7 with data not used
training <- training[,-c(1:7)]
testing <- testing[,-c(1:7)]

## making "classe" the factor
training$classe <- as.factor(training$classe)
```

### Cross validation

Cross-validation was done by sub-setting the training data set into 2 sub-samples: trainingNew data set (70%) and validationNew data set (30%). 


```r
set.seed(12345)
## split training data into training and validation parts
trainingset <- createDataPartition(y=training$classe, p=0.7, list=FALSE)
trainingNew <- training[trainingset, ]
validationNew <- training[-trainingset, ]
```

### Data outlook


```r
plot(as.factor(trainingNew$classe), col="lightblue", main = "Classe distribution in training data set", xlab="classe", ylab="Frequency")
```

<div class="figure">
<img src="figure/unnamed-chunk-4-1.png" alt="plot of chunk unnamed-chunk-4" width="50%" />
<p class="caption">plot of chunk unnamed-chunk-4</p>
</div>

Data is distributed across all values of "classe" variable.

### Model 1: Decision Tree
Model construction on training data sub-set. Prediction calculation and accuracy assessment on validation data sub-set.


```r
model1 <- rpart(classe ~ ., data=trainingNew, method="class")
prediction1 <- predict(model1, validationNew, type = "class")

cm1 <- confusionMatrix(prediction1, validationNew$classe)
```

We see that the accuracy of the model is not high

```
##           Reference
## Prediction    A    B    C    D    E
##          A 1532  176   28   48   41
##          B   54  585   57   64   76
##          C   35  154  819  134  126
##          D   25   76   58  631   56
##          E   28  148   64   87  783
```

```
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull AccuracyPValue 
##   7.391674e-01   6.692057e-01   7.277473e-01   7.503499e-01   2.844520e-01   0.000000e+00 
##  McnemarPValue 
##   1.005297e-37
```

### Model 2: Random Forest
Model construction on training data sub-set. Prediction calculation and accuracy assessment on validation data sub-set.


```r
model2 <- randomForest(classe ~. , data=trainingNew, method="class")
prediction2 <- predict(model2, validationNew, type = "class")
cm2 <- confusionMatrix(prediction2, validationNew$classe)
```

Calculation takes more time but accuracy is much higher.

```
##           Reference
## Prediction    A    B    C    D    E
##          A 1673    3    0    0    0
##          B    1 1135    4    0    0
##          C    0    1 1022   10    0
##          D    0    0    0  954    1
##          E    0    0    0    0 1081
```

```
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull AccuracyPValue 
##      0.9966015      0.9957011      0.9947562      0.9979229      0.2844520      0.0000000 
##  McnemarPValue 
##            NaN
```

## Conclusion 

Random Forest algorithm performed much better than Decision Trees. Accuracy for Random Forest model was 0.996 compared to 0.74 for Decision Tree model. The Random Forest model was selected. 

The expected out-of-sample error is estimated as 0.4%. 

The Test data set comprises 20 cases. With such error we can expect that very few of the test samples will be mis-classified. Prediction results are presented below.


```r
finalpredict <- predict(model2, testing, type="class")
finalpredict
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

