
<h2> Practical Machine Learning Project </h2> <br>
<h3> by Dymytr Yovchev </h3>

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

Data

The training data for this project are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv


The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har. If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment.

#Setup working environment
(I used Azure ML Cloud)


    set.seed(19)
    options(warn=-1)


    install.packages("caret")
    install.packages("randomForest")
    install.packages("e1071")
    install.packages("rpart")
    install.packages("rpart.plot")
    install.packages("gbm")
    install.packages("kernlab")
    install.packages("pROC")

    Installing package into '/home/nbcommon/R'
    (as 'lib' is unspecified)


    
    The downloaded source packages are in
    	'/tmp/RtmpGoPJQ9/downloaded_packages'


    Installing package into '/home/nbcommon/R'
    (as 'lib' is unspecified)


    
    The downloaded source packages are in
    	'/tmp/RtmpGoPJQ9/downloaded_packages'


    Installing package into '/home/nbcommon/R'
    (as 'lib' is unspecified)


    
    The downloaded source packages are in
    	'/tmp/RtmpGoPJQ9/downloaded_packages'


    Installing package into '/home/nbcommon/R'
    (as 'lib' is unspecified)


    
    The downloaded source packages are in
    	'/tmp/RtmpGoPJQ9/downloaded_packages'


    Installing package into '/home/nbcommon/R'
    (as 'lib' is unspecified)


    
    The downloaded source packages are in
    	'/tmp/RtmpGoPJQ9/downloaded_packages'


    Installing package into '/home/nbcommon/R'
    (as 'lib' is unspecified)


    
    The downloaded source packages are in
    	'/tmp/RtmpGoPJQ9/downloaded_packages'


    Installing package into '/home/nbcommon/R'
    (as 'lib' is unspecified)


    
    The downloaded source packages are in
    	'/tmp/RtmpGoPJQ9/downloaded_packages'



    library(kernlab)
    library(caret)
    library(randomForest)
    library(e1071)
    library(rpart)
    library(rpart.plot)
    library(gbm)
    library(doParallel)
    library(pROC)


    # load the library
    library(AzureML)
    
    if(file.exists("~/.azureml/settings.json")){
        ws <- workspace()
    } else {
        # workspace_id <- ""
        # authorization_token <- ""
        ws <- workspace(workspace_id, authorization_token)
    }

#Read Data


    trainHAR <- download.datasets(ws, name = "pml-training.csv")
    testHAR <- download.datasets(ws, name = "pml-testing.csv")


    dim(trainHAR)
    dim(testHAR)


<ol class=list-inline>
	<li>19622</li>
	<li>160</li>
</ol>




<ol class=list-inline>
	<li>20</li>
	<li>160</li>
</ol>



#Clean data
- We replace all "#DIV/0!" or Inf to NA.
- Delete all columns that with NA values.
- Also delete all columns with metadata like timestamp or id.


    t2 <- trainHAR
    t2 <- do.call(data.frame,lapply(t2, function(x) replace(x, is.infinite(x),NA)))
    t2 <- do.call(data.frame,lapply(t2, function(x) replace(x, x == "#DIV/0!",NA)))
    classe <- trainHAR$classe
    
    t2 <- t2[, colSums(is.na(t2)) == 0]
    t2 <- t2[complete.cases(t2),]
    classe <- t2$classe
    t2 <- t2[, sapply(t2, is.numeric)]
    t2$class <- classe
        
    drops <- c("X","timestamp","window","kurtosis_roll_belt","total_accel_belt","skewness_pitch_dumbbell","skewness_roll_dumbbell",
                                  "skewness_yaw_dumbbell","kurtosis_roll_dumbbell","kurtosis_picth_dumbbell",
                                  "amplitude_yaw_forearm","max_yaw_forearm","raw_timestamp_part_1","raw_timestamp_part_2",
                                    "num_window","user_name","cvtd_timestamp","new_window")
    
    trainHARCleaned <- t2[ , !(names(t2) %in% drops)]
    
    print("So after all tranformations we receive:")
    dim(trainHARCleaned)

    [1] "So after all tranformations we receive:"



<ol class=list-inline>
	<li>19622</li>
	<li>52</li>
</ol>



#Create test and train sets


    inTrain <- createDataPartition(trainHARCleaned$class, p=0.70, list=F)
    trainData <- trainHARCleaned[inTrain, ]
    testData <- trainHARCleaned[-inTrain, ]

#Train Models
I choose LinearSvm and RandomForest models for training.
- LinearSvm - fast training, but average results.
- RandomForest - slow training and good results.
- Parallel RandomForest fast and accurate.
- Also I try gbm, but it is so slow for training.
If you have suggestion for accelerating training contact me.


    system.time(model.svm <- train(class ~., data=trainData , method="svmLinear"))
    model.svm


        user   system  elapsed 
    1149.368   46.436  182.582 



    Support Vector Machines with Linear Kernel 
    
    13737 samples
       51 predictor
        5 classes: 'A', 'B', 'C', 'D', 'E' 
    
    No pre-processing
    Resampling: Bootstrapped (25 reps) 
    
    Summary of sample sizes: 13737, 13737, 13737, 13737, 13737, 13737, ... 
    
    Resampling results
    
      Accuracy   Kappa     Accuracy SD  Kappa SD   
      0.7808382  0.721525  0.005551381  0.007088395
    
    Tuning parameter 'C' was held constant at a value of 1
     



    registerDoParallel()
    x <- trainData[-ncol(trainData)]
    y <- trainData$class
    
    system.time(
    model.rf <- foreach(ntree=rep(150, 6), .combine=randomForest::combine, .packages='randomForest') %dopar% {
        randomForest(x, y, ntree=ntree) 
    }
        )
    model.rf


       user  system elapsed 
    474.884  17.060  28.443 



    
    Call:
     randomForest(x = x, y = y, ntree = ntree) 
                   Type of random forest: classification
                         Number of trees: 900
    No. of variables tried at each split: 7




    system.time(
    model.rf2 <- train(class ~ ., method = "rf", data = trainData, importance = T,
                                trControl = trainControl(method = "cv", number = 10))
                )


        user   system  elapsed 
    3007.100   91.300  643.107 



    #train RF model on all cleaned train set
    registerDoParallel()
    x <- trainHARCleaned[-ncol(trainHARCleaned)]
    y <- trainHARCleaned$class
    
    system.time(
    model.rfall <- foreach(ntree=rep(150, 6), .combine=randomForest::combine, .packages='randomForest') %dopar% {
        randomForest(x, y, ntree=ntree) 
    }
    )


       user  system elapsed 
    176.656   5.084  29.364 



    
    Call:
     randomForest(x = x, y = y, ntree = ntree) 
                   Type of random forest: classification
                         Number of trees: 900
    No. of variables tried at each split: 7



#Calculate features importance


    imp.rf <- varImp(model.rf)
    imp.svm <- varImp(model.svm)
    imp.rf2 <- varImp(model.rf2)


    print("RF")
    imp.rf
    print("RF2")
    imp.rf2
    print("SVM")
    imp.svm

    [1] "RF"



<table>
<thead><tr><th></th><th scope=col>Overall</th></tr></thead>
<tbody>
	<tr><th scope=row>roll_belt</th><td>877.52</td></tr>
	<tr><th scope=row>pitch_belt</th><td>489.6506</td></tr>
	<tr><th scope=row>yaw_belt</th><td>643.1334</td></tr>
	<tr><th scope=row>gyros_belt_x</th><td>71.0621</td></tr>
	<tr><th scope=row>gyros_belt_y</th><td>82.09698</td></tr>
	<tr><th scope=row>gyros_belt_z</th><td>236.9644</td></tr>
	<tr><th scope=row>accel_belt_x</th><td>83.05175</td></tr>
	<tr><th scope=row>accel_belt_y</th><td>94.28738</td></tr>
	<tr><th scope=row>accel_belt_z</th><td>304.8276</td></tr>
	<tr><th scope=row>magnet_belt_x</th><td>173.6262</td></tr>
	<tr><th scope=row>magnet_belt_y</th><td>292.618</td></tr>
	<tr><th scope=row>magnet_belt_z</th><td>293.3654</td></tr>
	<tr><th scope=row>roll_arm</th><td>219.5392</td></tr>
	<tr><th scope=row>pitch_arm</th><td>126.825</td></tr>
	<tr><th scope=row>yaw_arm</th><td>169.9242</td></tr>
	<tr><th scope=row>total_accel_arm</th><td>71.738</td></tr>
	<tr><th scope=row>gyros_arm_x</th><td>91.51551</td></tr>
	<tr><th scope=row>gyros_arm_y</th><td>91.81628</td></tr>
	<tr><th scope=row>gyros_arm_z</th><td>41.88469</td></tr>
	<tr><th scope=row>accel_arm_x</th><td>173.7418</td></tr>
	<tr><th scope=row>accel_arm_y</th><td>111.7202</td></tr>
	<tr><th scope=row>accel_arm_z</th><td>93.4406</td></tr>
	<tr><th scope=row>magnet_arm_x</th><td>186.805</td></tr>
	<tr><th scope=row>magnet_arm_y</th><td>162.5706</td></tr>
	<tr><th scope=row>magnet_arm_z</th><td>133.3632</td></tr>
	<tr><th scope=row>roll_dumbbell</th><td>300.8371</td></tr>
	<tr><th scope=row>pitch_dumbbell</th><td>124.9325</td></tr>
	<tr><th scope=row>yaw_dumbbell</th><td>180.3091</td></tr>
	<tr><th scope=row>total_accel_dumbbell</th><td>188.7888</td></tr>
	<tr><th scope=row>gyros_dumbbell_x</th><td>88.11717</td></tr>
	<tr><th scope=row>gyros_dumbbell_y</th><td>180.691</td></tr>
	<tr><th scope=row>gyros_dumbbell_z</th><td>57.60932</td></tr>
	<tr><th scope=row>accel_dumbbell_x</th><td>174.9807</td></tr>
	<tr><th scope=row>accel_dumbbell_y</th><td>286.2442</td></tr>
	<tr><th scope=row>accel_dumbbell_z</th><td>234.6577</td></tr>
	<tr><th scope=row>magnet_dumbbell_x</th><td>350.5606</td></tr>
	<tr><th scope=row>magnet_dumbbell_y</th><td>467.9345</td></tr>
	<tr><th scope=row>magnet_dumbbell_z</th><td>541.2089</td></tr>
	<tr><th scope=row>roll_forearm</th><td>411.3742</td></tr>
	<tr><th scope=row>pitch_forearm</th><td>557.794</td></tr>
	<tr><th scope=row>yaw_forearm</th><td>116.9158</td></tr>
	<tr><th scope=row>total_accel_forearm</th><td>83.72187</td></tr>
	<tr><th scope=row>gyros_forearm_x</th><td>53.16946</td></tr>
	<tr><th scope=row>gyros_forearm_y</th><td>89.23285</td></tr>
	<tr><th scope=row>gyros_forearm_z</th><td>58.84906</td></tr>
	<tr><th scope=row>accel_forearm_x</th><td>222.0104</td></tr>
	<tr><th scope=row>accel_forearm_y</th><td>98.82626</td></tr>
	<tr><th scope=row>accel_forearm_z</th><td>172.5112</td></tr>
	<tr><th scope=row>magnet_forearm_x</th><td>150.5126</td></tr>
	<tr><th scope=row>magnet_forearm_y</th><td>155.6331</td></tr>
	<tr><th scope=row>magnet_forearm_z</th><td>195.6439</td></tr>
</tbody>
</table>



    [1] "RF2"



    rf variable importance
    
      variables are sorted by maximum importance across the classes
      only 20 most important variables shown (out of 51)
    
                          A      B     C     D     E
    pitch_belt        70.09 100.00 62.14 74.36 67.06
    roll_belt         62.08  88.77 96.09 89.06 53.79
    yaw_belt          86.98  86.43 67.65 85.87 51.32
    magnet_dumbbell_z 79.91  78.90 83.73 69.18 64.30
    magnet_dumbbell_y 49.47  50.54 80.66 48.72 34.23
    yaw_arm           32.93  71.79 54.78 65.07 43.74
    roll_arm          39.06  71.07 55.18 53.89 36.75
    accel_belt_z      34.75  66.42 56.25 54.02 39.46
    magnet_belt_z     47.94  57.34 60.67 65.46 55.02
    gyros_arm_y       32.21  64.96 47.40 50.21 38.62
    magnet_dumbbell_x 37.33  36.15 63.93 37.97 21.80
    magnet_belt_y     46.99  53.54 62.27 54.61 44.08
    accel_dumbbell_y  36.18  58.60 57.56 52.23 53.29
    pitch_forearm     43.80  55.57 54.28 58.37 54.66
    magnet_forearm_z  34.97  57.34 50.59 53.04 49.87
    accel_dumbbell_z  40.77  56.02 56.44 51.42 51.21
    accel_arm_y       23.82  56.42 41.88 45.35 42.80
    accel_forearm_z   29.04  53.05 56.38 39.57 46.70
    roll_dumbbell     21.99  41.16 56.23 48.23 34.75
    gyros_forearm_y   19.73  53.71 45.71 44.29 27.31


    [1] "SVM"



    ROC curve variable importance
    
      variables are sorted by maximum importance across the classes
      only 20 most important variables shown (out of 51)
    
                           A     B     C      D     E
    pitch_forearm     100.00 65.10 69.75 100.00 69.09
    roll_dumbbell      54.91 62.84 85.42  85.42 58.42
    accel_forearm_x    83.20 52.16 63.49  83.20 46.86
    magnet_arm_x       78.94 54.83 56.93  78.94 67.11
    magnet_arm_y       78.15 42.36 55.04  78.15 70.23
    magnet_forearm_x   74.24 52.33 42.77  74.24 43.61
    accel_arm_x        73.99 53.34 49.47  73.99 63.25
    pitch_dumbbell     54.43 73.54 73.54  63.87 49.04
    magnet_belt_y      67.69 60.29 62.47  62.11 67.69
    magnet_dumbbell_x  65.76 66.13 66.13  52.00 52.71
    magnet_dumbbell_y  47.78 65.14 65.14  49.31 52.81
    accel_dumbbell_x   59.14 60.43 60.43  49.74 41.25
    magnet_dumbbell_z  56.50 26.13 56.50  37.90 54.63
    magnet_arm_z       53.78 53.78 39.27  43.03 50.55
    magnet_belt_z      50.39 49.32 50.46  51.18 51.18
    pitch_arm          50.10 28.98 39.61  43.11 50.10
    roll_belt          46.52 42.03 43.29  48.34 48.34
    magnet_forearm_y   38.24 26.86 45.56  45.56 36.35
    total_accel_arm    44.99 29.47 34.64  44.99 35.73
    accel_dumbbell_z   44.77 44.77 43.14  25.40 27.92


#Calculate confusion matrix


    pred.rfpar <- predict(model.rf, newdata=testData)
    confusionMatrix(pred.rfpar,testData$class)


    Confusion Matrix and Statistics
    
              Reference
    Prediction    A    B    C    D    E
             A 1673    9    0    0    0
             B    1 1129   11    0    0
             C    0    1 1014    9    0
             D    0    0    1  955    2
             E    0    0    0    0 1080
    
    Overall Statistics
                                             
                   Accuracy : 0.9942         
                     95% CI : (0.9919, 0.996)
        No Information Rate : 0.2845         
        P-Value [Acc > NIR] : < 2.2e-16      
                                             
                      Kappa : 0.9927         
     Mcnemar's Test P-Value : NA             
    
    Statistics by Class:
    
                         Class: A Class: B Class: C Class: D Class: E
    Sensitivity            0.9994   0.9912   0.9883   0.9907   0.9982
    Specificity            0.9979   0.9975   0.9979   0.9994   1.0000
    Pos Pred Value         0.9946   0.9895   0.9902   0.9969   1.0000
    Neg Pred Value         0.9998   0.9979   0.9975   0.9982   0.9996
    Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    Detection Rate         0.2843   0.1918   0.1723   0.1623   0.1835
    Detection Prevalence   0.2858   0.1939   0.1740   0.1628   0.1835
    Balanced Accuracy      0.9986   0.9943   0.9931   0.9950   0.9991



    pred.rf <- predict(model.rf2, newdata=testData)
    confusionMatrix(pred.rf,testData$class)


    Confusion Matrix and Statistics
    
              Reference
    Prediction    A    B    C    D    E
             A 1673    9    0    0    0
             B    1 1129   12    0    0
             C    0    1 1014   12    0
             D    0    0    0  952    1
             E    0    0    0    0 1081
    
    Overall Statistics
                                              
                   Accuracy : 0.9939          
                     95% CI : (0.9915, 0.9957)
        No Information Rate : 0.2845          
        P-Value [Acc > NIR] : < 2.2e-16       
                                              
                      Kappa : 0.9923          
     Mcnemar's Test P-Value : NA              
    
    Statistics by Class:
    
                         Class: A Class: B Class: C Class: D Class: E
    Sensitivity            0.9994   0.9912   0.9883   0.9876   0.9991
    Specificity            0.9979   0.9973   0.9973   0.9998   1.0000
    Pos Pred Value         0.9946   0.9886   0.9873   0.9990   1.0000
    Neg Pred Value         0.9998   0.9979   0.9975   0.9976   0.9998
    Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    Detection Rate         0.2843   0.1918   0.1723   0.1618   0.1837
    Detection Prevalence   0.2858   0.1941   0.1745   0.1619   0.1837
    Balanced Accuracy      0.9986   0.9942   0.9928   0.9937   0.9995



    pred.svm <- predict(model.svm, newdata=testData)
    confusionMatrix(pred.svm,testData$class)


    Confusion Matrix and Statistics
    
              Reference
    Prediction    A    B    C    D    E
             A 1535  168  117   68   60
             B   32  804   93   44  146
             C   49   58  771  107   56
             D   53   22   30  713   58
             E    5   87   15   32  762
    
    Overall Statistics
                                              
                   Accuracy : 0.7791          
                     95% CI : (0.7683, 0.7896)
        No Information Rate : 0.2845          
        P-Value [Acc > NIR] : < 2.2e-16       
                                              
                      Kappa : 0.7188          
     Mcnemar's Test P-Value : < 2.2e-16       
    
    Statistics by Class:
    
                         Class: A Class: B Class: C Class: D Class: E
    Sensitivity            0.9170   0.7059   0.7515   0.7396   0.7043
    Specificity            0.9019   0.9336   0.9444   0.9669   0.9711
    Pos Pred Value         0.7880   0.7185   0.7406   0.8139   0.8457
    Neg Pred Value         0.9647   0.9297   0.9474   0.9499   0.9358
    Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    Detection Rate         0.2608   0.1366   0.1310   0.1212   0.1295
    Detection Prevalence   0.3310   0.1901   0.1769   0.1489   0.1531
    Balanced Accuracy      0.9094   0.8198   0.8479   0.8533   0.8377



    pred.rfall <- predict(model.rfall, newdata=testData)
    confusionMatrix(pred.rfall,testData$class)


    Confusion Matrix and Statistics
    
              Reference
    Prediction    A    B    C    D    E
             A 1674    0    0    0    0
             B    0 1139    0    0    0
             C    0    0 1026    0    0
             D    0    0    0  964    0
             E    0    0    0    0 1082
    
    Overall Statistics
                                         
                   Accuracy : 1          
                     95% CI : (0.9994, 1)
        No Information Rate : 0.2845     
        P-Value [Acc > NIR] : < 2.2e-16  
                                         
                      Kappa : 1          
     Mcnemar's Test P-Value : NA         
    
    Statistics by Class:
    
                         Class: A Class: B Class: C Class: D Class: E
    Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
    Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
    Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
    Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
    Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    Detection Rate         0.2845   0.1935   0.1743   0.1638   0.1839
    Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
    Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000


#Predict test data values


    testHAR$class <- 0
    tt <- testHAR[,names(testData)]
    
    prediction <- as.character(predict(model.svm, newdata=tt))
    print("SVM")
    prediction
    prediction <- as.character(predict(model.rf, newdata=tt))
    print("RF parallel")
    prediction
    prediction <- as.character(predict(model.rf2, newdata=tt))
    print("RF")
    prediction
    prediction <- as.character(predict(model.rfall, newdata=tt))
    print("RF parallel All")
    prediction

    [1] "SVM"



<ol class=list-inline>
	<li>'C'</li>
	<li>'A'</li>
	<li>'B'</li>
	<li>'C'</li>
	<li>'A'</li>
	<li>'E'</li>
	<li>'D'</li>
	<li>'D'</li>
	<li>'A'</li>
	<li>'A'</li>
	<li>'C'</li>
	<li>'A'</li>
	<li>'B'</li>
	<li>'A'</li>
	<li>'E'</li>
	<li>'E'</li>
	<li>'A'</li>
	<li>'B'</li>
	<li>'B'</li>
	<li>'B'</li>
</ol>



    [1] "RF parallel"



<ol class=list-inline>
	<li>'B'</li>
	<li>'A'</li>
	<li>'B'</li>
	<li>'A'</li>
	<li>'A'</li>
	<li>'E'</li>
	<li>'D'</li>
	<li>'B'</li>
	<li>'A'</li>
	<li>'A'</li>
	<li>'B'</li>
	<li>'C'</li>
	<li>'B'</li>
	<li>'A'</li>
	<li>'E'</li>
	<li>'E'</li>
	<li>'A'</li>
	<li>'B'</li>
	<li>'B'</li>
	<li>'B'</li>
</ol>



    [1] "RF"



<ol class=list-inline>
	<li>'B'</li>
	<li>'A'</li>
	<li>'B'</li>
	<li>'A'</li>
	<li>'A'</li>
	<li>'E'</li>
	<li>'D'</li>
	<li>'B'</li>
	<li>'A'</li>
	<li>'A'</li>
	<li>'B'</li>
	<li>'C'</li>
	<li>'B'</li>
	<li>'A'</li>
	<li>'E'</li>
	<li>'E'</li>
	<li>'A'</li>
	<li>'B'</li>
	<li>'B'</li>
	<li>'B'</li>
</ol>



    [1] "RF parallel All"



<ol class=list-inline>
	<li>'B'</li>
	<li>'A'</li>
	<li>'B'</li>
	<li>'A'</li>
	<li>'A'</li>
	<li>'E'</li>
	<li>'D'</li>
	<li>'B'</li>
	<li>'A'</li>
	<li>'A'</li>
	<li>'B'</li>
	<li>'C'</li>
	<li>'B'</li>
	<li>'A'</li>
	<li>'E'</li>
	<li>'E'</li>
	<li>'A'</li>
	<li>'B'</li>
	<li>'B'</li>
	<li>'B'</li>
</ol>



#Conclusions
- As we see the best solutions is RF Parallel(by time and precision).
- Usual Caret RF by accuracy is very near, but is much slower in training.
- There is no difference in predicting 20 final tests when you train model on 0.7 of train set or on all dataset
- SVM accuracy on train/test is about 75 percent where random forest give near to 99 percent precision and in final test we had big difference between RF and SVM


    models <- c("RF","RF Parallel","SVM", "RF Parallel All*")
    accuracy <- c(0.9939,0.9942,0.7791,1.00)
    elapsed_time <- c(643.107,28.443 ,182.582, 29.364 )
    conc <- data.frame(models,accuracy,elapsed_time)
                       
    print("trained on train dataset that is 0.7 from train dataset with dim:")
    dim(trainData)
    print("tested on test dataset that is 0.3 from train dataset with dim:")
    dim(testData)
    print("*trained on all train dataset with dim:")
    dim(trainHARCleaned)
    conc

    [1] "trained on train dataset that is 0.7 from train dataset with dim:"



<ol class=list-inline>
	<li>13737</li>
	<li>52</li>
</ol>



    [1] "tested on test dataset that is 0.3 from train dataset with dim:"



<ol class=list-inline>
	<li>5885</li>
	<li>52</li>
</ol>



    [1] "*trained on all train dataset with dim:"



<ol class=list-inline>
	<li>19622</li>
	<li>52</li>
</ol>




<table>
<thead><tr><th></th><th scope=col>models</th><th scope=col>accuracy</th><th scope=col>elapsed_time</th></tr></thead>
<tbody>
	<tr><th scope=row>1</th><td>RF</td><td>0.9939</td><td>643.107</td></tr>
	<tr><th scope=row>2</th><td>RF Parallel</td><td>0.9942</td><td>28.443</td></tr>
	<tr><th scope=row>3</th><td>SVM</td><td>0.7791</td><td>182.582</td></tr>
	<tr><th scope=row>4</th><td>RF Parallel All*</td><td>1</td><td>29.364</td></tr>
</tbody>
</table>


