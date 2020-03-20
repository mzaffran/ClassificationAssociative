rm(list=objects())
graphics.off()
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

library(rpart)

# Import des échantillons de d'apprentissage et de test

train <- read_csv('data/kidney_train.csv')
test <- read_csv('data/kidney_test.csv')
train$Class = as.factor(train$Class)
test$Class = as.factor(test$Class)
names(train) = make.names(names(train))
names(test) = make.names(names(test))

# Arbre CART

t1 = Sys.time()
max_tree = rpart(Class ~ ., data = train, method = "class", control = rpart.control(cp = 0))
tree_elag = prune(max_tree, cp = max_tree$cptable[which.min(max_tree$cptable[,4]),1])
t2 = Sys.time()
print(t2-t1)

class_train_tree = predict(tree_elag, type = "class")
class_test_tree = predict(tree_elag, newdata = test, type = "class")

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

t1 = Sys.time()
rf = randomForest(Class ~ ., data = train, importance = TRUE)
t2 = Sys.time()
print(t2-t1)

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

#varImpPlot(rf)