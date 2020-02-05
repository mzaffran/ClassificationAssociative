setwd('/Users/margauxzaffran/Documents/ENSTA/3A/SOD322/ClassificationAssociative')

library('tidyverse')

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

library(rpart)

# Création des échantillons de d'apprentissage et de test

test = sample(1:length(data$class),63)
train = -test
train = data[train,]
test = data[test,]

# Arbre maximal

init_tree = rpart(class ~ ., data = train)
plot(init_tree)
text(init_tree,cex=.8)
print(init_tree)

max_tree = rpart(class ~ ., data = train, control = rpart.control(cp = 0))
plot(max_tree)
text(max_tree,cex=.7)
print(max_tree)

# Analyse descriptive

library(corrplot)

corrplot(cor(data), method="circle")
