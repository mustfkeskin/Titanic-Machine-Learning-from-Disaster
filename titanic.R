rm(list = ls())
library(data.table)
library(Matrix)
library(xgboost)
library(caret)
library(dplyr)


cat("Read data")
df_train <- fread('/home/mustafa/Desktop/titanic/train.csv', sep=",", na.strings = "NA")
df_test  <- fread("/home/mustafa/Desktop/titanic/test.csv" , sep=",", na.strings = "NA")


df_test %>% summarise_each(funs(sum(is.na(.))))
df_train %>% summarise_each(funs(sum(is.na(.))))
df_test[is.na(df_test$Age),"Age"] <- mean(df_test$Age, na.rm = TRUE)
df_train[is.na(df_train$Age),"Age"] <- mean(df_train$Age, na.rm = TRUE)



data = rbind(df_train,df_test,fill=T)
data$Title <- gsub('(.*, )|(\\..*)', '', data$Name)

rare_title <- c('Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 
                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer')

# Also reassign mlle, ms, and mme accordingly
data$Title[data$Title == 'Mlle']        <- 'Miss' 
data$Title[data$Title == 'Ms']          <- 'Miss'
data$Title[data$Title == 'Mme']         <- 'Mrs' 
data$Title[data$Title %in% rare_title]  <- 'Rare Title'

data$Surname <- sapply(data$Name,  
                       function(x) strsplit(x, split = '[,.]')[[1]][1])

# Create a family size variable including the passenger themselves
data$Fsize <- data$SibSp + data$Parch + 1

data$FsizeD[data$Fsize == 1] <- 'singleton'
data$FsizeD[data$Fsize < 5 & data$Fsize > 1] <- 'small'
data$FsizeD[data$Fsize > 4] <- 'large'

data$isAlone <- 0
data[data$Fsize == 1,"isAlone"] <- 1

data$Child[data$Age < 18] <- 1
data$Child[data$Age >= 18] <- 0

data$Mother <- 0
data$Mother[data$Sex == 'female' & data$Parch > 0 & data$Age > 18 & data$Title != 'Miss'] <- 1


data$Deck <- factor(sapply(data$Cabin, function(x) unlist(strsplit(x, NULL)[[1]][1])))

data <- data[,-c("Ticket","Name","Surname", "Cabin", "Deck")]
ohe_feats = c('Sex', 'Embarked', 'Title', 'FsizeD')


for (f in ohe_feats){
    levels = unique(data[[f]])
    data[[f]] = factor(data[[f]], level = levels)
}

# one-hot-encoding features
#data = as.data.frame(data)
#dummies = dummyVars(~ Survived, Pclass, Sex, SibSp ,Parch, Embarked, Title, FsizeD, isAlone, Fsize, data = data)
#df_all_ohe <- as.data.frame(predict(dummies, newdata = data))
#df_all_combined <- cbind(data[,-c(which(colnames(data) %in% ohe_feats))],df_all_ohe)

#data = as.data.table(df_all_combined)

train = data[data$PassengerId %in% df_train$PassengerId,]
y_train <- train[!is.na(Survived),Survived]
train = train[,Survived:=NULL]
train = train[,PassengerId:=NULL]
train_sparse <- data.matrix(train)

test = data[data$PassengerId  %in% df_test$PassengerId,]
test_ids <- test[,PassengerId]
test[,Survived:=NULL]
test[,PassengerId:=NULL]
test_sparse <- data.matrix(test)

dtrain <- xgb.DMatrix(data=train_sparse, label=y_train)
dtest <- xgb.DMatrix(data=test_sparse);

gc()

# Params for xgboost
param <- list(booster = "gbtree",
              eval_metric = "auc", 
              objective = "binary:logistic",
              eta = .11,
              gamma = 1,
              max_depth = 6,
              min_child_weight = 1,
              subsample = .7,
              colsample_bytree = .7)


rounds = 72
mpreds = data.table(id=test_ids)

for(random.seed.num in 1:10) {
  print(paste("[", random.seed.num , "] training xgboost begin ",sep=""," : ",Sys.time()))
  set.seed(random.seed.num)
  xgb_model <- xgb.train(data = dtrain,
                         params = param,
                         watchlist = list(train = dtrain),
                         nrounds = rounds,
                         verbose = 1,
                         print_every_n = 5)
  
  vpreds = predict(xgb_model,dtest) 
  mpreds = cbind(mpreds, vpreds)    
  colnames(mpreds)[random.seed.num+1] = paste("pred_seed_", random.seed.num, sep="")
}

mpreds_2 = mpreds[, id:= NULL]
mpreds_2 = mpreds_2[, y := rowMeans(.SD)]

mpreds_2[mpreds_2$y <= 0.5,"x"] <- 0
mpreds_2[mpreds_2$y > 0.5,"x"] <- 1


submission = data.table(PassengerId=test_ids, Survived=mpreds_2$x)
write.table(submission, "titanic_xgboost.csv", sep=",", dec=".", quote=FALSE, row.names=FALSE)



library(randomForest)
fit <- randomForest(as.factor(y_train) ~ .,
                    data=train, 
                    importance=TRUE,
                    do.trace=50, 
                    ntree=200)

pred_test <- predict(fit, test)

submission_rf = data.table(PassengerId=test_ids, Survived=pred_test)
write.table(submission_rf, "titanic_rf.csv", sep=",", dec=".", quote=FALSE, row.names=FALSE)


