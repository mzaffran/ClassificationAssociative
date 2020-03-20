setwd('/Users/margauxzaffran/Documents/ENSTA/3A/SOD322/ClassificationAssociative')

precision <- function(y_true, y_pred, class){
  VP <- sum(y_pred[y_true == class] == class)
  FP <- sum(y_pred[y_true != class] == class)
  return(VP / (VP+FP))
}

recall <- function(y_true, y_pred, class){
  VP <- sum(y_pred[y_true == class] == class)
  FN <- sum(y_pred[y_true == class] != class)
  return(VP / (VP+FN))
}

library('tidyverse')
library(rpart)

# Import des échantillons de d'apprentissage et de test

train <- read_csv('data/kidney_train.csv')
test <- read_csv('data/kidney_test.csv')
train$Class = as.factor(train$Class)
test$Class = as.factor(test$Class)
names(train) = make.names(names(train))
names(test) = make.names(names(test))

# Arbre maximal

init_tree = rpart(Class ~ ., data = train, method = "class")
plot(init_tree)
text(init_tree,cex=1)
print(init_tree)

max_tree = rpart(Class ~ ., data = train, method = "class", control = rpart.control(cp = 0))
plot(max_tree)
text(max_tree,cex=.7)
print(max_tree)

class_train_tree = predict(max_tree, type = "class")
class_test_tree = predict(max_tree, newdata = test, type = "class")

precision(train$Class, class_train_tree, 0)
recall(train$Class, class_train_tree, 0)

precision(train$Class, class_train_tree, 1)
recall(train$Class, class_train_tree, 1)

precision(test$Class, class_test_tree, 0)
recall(test$Class, class_test_tree, 0)

precision(test$Class, class_test_tree, 1)
recall(test$Class, class_test_tree, 1)

# Forêt aléatoire

library(randomForest)

rf = randomForest(Class ~ ., data = train, importance = TRUE)
class_train_rf = predict(rf, type = "class")
class_test_rf = predict(rf, newdata = test, type = "class")

precision(train$Class, class_train_rf, 0)
recall(train$Class, class_train_rf, 0)

precision(train$Class, class_train_rf, 1)
recall(train$Class, class_train_rf, 1)

precision(test$Class, class_test_rf, 0)
recall(test$Class, class_test_rf, 0)

precision(test$Class, class_test_rf, 1)
recall(test$Class, class_test_rf, 1)

varImpPlot(rf)

# RF toutes les données

data <- read_csv('data/kidney.csv')
summary(data)

# pc abnormal = 1
data$pc[data$pc == "abnormal"] = 1
data$pc[data$pc == "normal"] = 0
data$pc <- as.numeric(as.character(data$pc))
# pcc, ba present = 1 
data$pcc[data$pcc == "present"] = 1
data$pcc[data$pcc == "notpresent"] = 0
data$pcc <- as.numeric(as.character(data$pcc))
data$ba[data$ba == "present"] = 1
data$ba[data$ba == "notpresent"] = 0
data$ba <- as.numeric(as.character(data$ba))
# htn, dm, cad, pe, ane yes = 1
data$htn[data$htn == "yes"] = 1
data$htn[data$htn == "no"] = 0
data$htn <- as.numeric(as.character(data$htn))
data$dm[data$dm == "yes"] = 1
data$dm[data$dm == "no"] = 0
data$dm <- as.numeric(as.character(data$dm))
data$cad[data$cad == "yes"] = 1
data$cad[data$cad == "no"] = 0
data$cad <- as.numeric(as.character(data$cad))
data$pe[data$pe == "yes"] = 1
data$pe[data$pe == "no"] = 0
data$pe <- as.numeric(as.character(data$pe))
data$ane[data$ane == "yes"] = 1
data$ane[data$ane == "no"] = 0
data$ane <- as.numeric(as.character(data$ane))
# appet good = 0
data$appet[data$appet == "good"] = 0
data$appet[data$appet == "poor"] = 1
data$appet <- as.numeric(as.character(data$appet))
# class ckd = 1
data$class[data$class == "ckd"] = 1
data$class[data$class == "notckd"] = 0
data$class <- as.numeric(as.character(data$class))
data$class = as.factor(data$class)

niter = 10 

precision_train_0 = rep(NA,niter)
recall_train_0 = rep(NA,niter)
precision_train_1 = rep(NA,niter)
recall_train_1 = rep(NA,niter)
precision_test_0 = rep(NA,niter)
recall_test_0 = rep(NA,niter)
precision_test_1 = rep(NA,niter)
recall_test_1 = rep(NA,niter)

for (i in 1:niter){
  test = sample(1:length(data$class),63)
  train = -test
  train = data[train,]
  test = data[test,]
  
  rf = randomForest(class ~ ., data = train)
  class_train_rf = predict(rf, type = "class")
  class_test_rf = predict(rf, newdata = test, type = "class")
  
  precision_train_0[i] = precision(train$class, class_train_rf, 0)
  recall_train_0[i] =  recall(train$class, class_train_rf, 0)
  
  precision_train_1[i] = precision(train$class, class_train_rf, 1)
  recall_train_1[i] =  recall(train$class, class_train_rf, 1)
  
  precision_test_0[i] = precision(test$class, class_test_rf, 0)
  recall_test_0[i] = recall(test$class, class_test_rf, 0)
  
  precision_test_1[i] = precision(test$class, class_test_rf, 1)
  recall_test_1[i] = recall(test$class, class_test_rf, 1)
}

mean(precision_train_0)
sqrt(var(precision_train_0))
mean(recall_train_0)
sqrt(var(recall_train_0))
mean(precision_train_1)
sqrt(var(precision_train_1))
mean(recall_train_1)
sqrt(var(recall_train_1))

mean(precision_test_0)
sqrt(var(precision_test_0))
mean(recall_test_0)
sqrt(var(recall_test_0))
mean(precision_test_1)
sqrt(var(precision_test_1))
mean(recall_test_1)
sqrt(var(recall_test_1))
