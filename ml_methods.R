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

data <- read_csv('data/kidney.csv')
data$class = as.factor(data$class)
summary(data)

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
