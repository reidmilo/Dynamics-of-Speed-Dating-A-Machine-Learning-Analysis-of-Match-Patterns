#FINAL PROJECT
library(dplyr)
library(caret)
library(tree)
library(tidyr)
library(gbm)
library(MASS)
library(randomForest)
library(pROC)
library(ggplot2)
library(ISLR)
library(rpart)
library(rpart.plot)
library(e1071)
library(ROCR)
library(glmnet)
library(writexl)

rawData <-read.csv('/Users/oliverreidmiller/Desktop/DATA 300/FINAL PROJECT/Speed Dating Data.csv')
# datingData <- rawData 

head(datingData)

# Data Cleaning -----------------------------------------------------------
predictors <- c("iid", "gender", "order", "pid", "match", "int_corr", "samerace",
                "age", "age_o", "field_cd", "imprace", "imprelig", "date",
                "go_out", "career_c", "exphappy", "attr", "attr_o", "sinc",
                "sinc_o", "intel", "intel_o", "fun", "fun_o", 
                "like", "like_o", "prob", "prob_o", "met")
#Took out Decisions - correlated 
datingData <- subset(datingData, select = predictors)
date_columns <- c("iid", "gender", "field_cd", "imprace", "imprelig", 
                  "date", "go_out", "career_c", "exphappy")
date_info <- rawData
date_info <- subset(date_info, select = date_columns)

names(date_info) <- c("pid", "gender_o", "field_cd_o", "imprace_o", "imprelig_o", "date_o",
                      "go_out_o", "career_c_o", "exphappy_o")
date_info <- subset(date_info, !duplicated(date_info[,1])) 
datingData <- merge(datingData, date_info, by = "pid")
datingData <- subset(datingData, gender==0)

newnames <- c("id_M", "id_F", "gender", "order", "match", "int_corr", "samerace", "age_F",
              "age_M", "field_cd_F", "imprace_F", "imprelig_F", "date_F", "go_out_F", "career_c_F",
              "exphappy_F", "attr_F", "attr_M", "sinc_F", "sinc_M", "intel_F", "intel_M",
              "fun_F", "fun_M", "like_F", "like_M", "prob_F", "prob_M", "met",
              "gender_M", "field_cd_M", "imprace_M", "imprelig_M", "date_M", "go_out_M",
              "career_c_M", "exphappy_M")
names(datingData) <- newnames

datingData$field_cd_F[is.na(datingData$field_cd_F)]<- 18
datingData$career_c_F[is.na(datingData$career_c_F)]<- 15
datingData$field_cd_M[is.na(datingData$field_cd_M)] <- 18
datingData$career_c_M[is.na(datingData$career_c_M)] <- 15

#datingData <- datingData %>% mutate(across(where(is.numeric), ~replace_na(., median(., na.rm=TRUE))))
datingData <-na.omit(datingData)

sum(complete.cases(datingData))


datingData$career_same <- datingData$career_c_F == datingData$career_c_M
datingData$field_same <- datingData$field_cd_F == datingData$field_cd_M
datingData$match <- as.factor(datingData$match)
datingData$field_same <- as.factor(datingData$field_same)
datingData$career_same <- as.factor(datingData$career_same)

datingData <- datingData[-c(1,2,3,30)]#TAKE OUT GENDER - perfect multicollinearity
datingData$met[datingData$met ==0] <-1
datingData$met[datingData$met ==3] <-2
datingData$met[datingData$met ==7] <-2
datingData$met[datingData$met ==8] <-2

sample <- sample(c(TRUE, FALSE), nrow(datingData), replace=TRUE, prob=c(0.8,0.2))
trainingSet  <- datingData[sample, ]
testSet   <- datingData[!sample, ]
# Data Visual Analysis ----------------------------------------------------
table(datingData$match)
print(622/(622+2896)*100)

matchPercent <- datingData %>%
  count(match) %>%
  mutate(pct = n / sum(n) ) %>%
  print()

ggplot(matchPercent, 
       aes(x = match, fill = pct)) + 
  geom_bar(position = "stack")+ scale_y_continuous(labels = scales::percent)

ggplot(data=matchPercent, aes(x=match, y=pct)) +
  geom_bar(stat = "identity")+ scale_y_continuous(labels = scales::percent_format(accuracy = 1))+
  xlab('Matches')+ylab('Percentage')+labs(title='Distribution of Response',caption= 'Figure 1')


x <- datingData$age_F 
h<-hist(x, breaks=20, col="red", xlab="Age of Female", 
        main="Female Age with Normal Curve", sub = 'Figure 2') 
xfit<-seq(min(x),max(x),length=40) 
yfit<-dnorm(xfit,mean=mean(x),sd=sd(x)) 
yfit <- yfit*diff(h$mids[1:2])*length(x) 
lines(xfit, yfit, col="blue", lwd=2)

summary
x <- datingData$age_M 
h<-hist(x, breaks=20, col="red", xlab="Age of Male", 
        main="Male Age with Normal Curve", sub = 'Figure 3') 
xfit<-seq(min(x),max(x),length=40) 
yfit<-dnorm(xfit,mean=mean(x),sd=sd(x)) 
yfit <- yfit*diff(h$mids[1:2])*length(x) 
lines(xfit, yfit, col="blue", lwd=2)

raceData <- datingData
raceData$samerace<-ifelse(raceData$samerace==1,TRUE,FALSE)


ggplot() + 
  geom_bar(data = raceData, aes(x = factor(match),fill = factor(samerace)),position = "fill")+
             xlab('Match')+ylab('Percent Same Race')+ scale_y_continuous(labels = scales::percent)+
  labs(title = 'Same Race Effects on Matches', caption = 'Figure 6')+ 
  guides(fill=guide_legend(title="Same Race"))

ggplot() + 
  geom_bar(data = datingData, aes(x = factor(match),fill = factor(field_same)),position = "fill")+
  xlab('Match')+ylab('Percent Same Career Field ')+ scale_y_continuous(labels = scales::percent)+
  labs(title = 'Same Career Effects on Matches', caption = 'Figure 5')+ 
  guides(fill=guide_legend(title="Same Career Field"))

orderData <- datingData
orderData$match <- as.numeric(orderData$match)
orderData$match <- orderData$match-1
orderData <- group_by(orderData, order) %>%
  summarize(avg_match = mean(match, na.rm = TRUE))

ggplot(data=orderData, aes(x=order, y=avg_match))+
  geom_smooth(color = 'red')+ geom_point()+
  xlab('Order')+ylab('Average Match Rate')+
  scale_x_continuous(breaks=orderData$order) +
  theme(axis.text.x=element_text(hjust=0.95,vjust=0.2))+ 
  scale_y_continuous(labels = scales::percent)+
  labs(title='Match Rate over Order',caption= 'Figure 4')




# Lasso/Ridge -------------------------------------------------------------
lassoData <- trainingSet
lassoData$match <- as.numeric(lassoData$match)
lassoTest <- testSet
lassoTest$match <- as.numeric(lassoTest$match)

x <- model.matrix(match~., data = lassoData)[, -1]
xTest<-model.matrix(match~., data = lassoData)[, -1]
y <- lassoData$match

grid <- seq(0.01, 10, length = 10)
ridge_mod <- glmnet(x, y, alpha = 0, lambda = grid)

predict(ridge_mod, s = 0.01, type ="coefficients")[1:17, ]
predict(ridge_mod, s = 10, type ="coefficients")[1:17, ]

cv_out <- cv.glmnet(x, y, alpha = 0)
optimalLambda <- cv_out$lambda.min

ridge_model <- glmnet(x, y, alpha = 0, lambda = NULL)

x.test <- model.matrix(match~., lassoTest)[,-1]
probabilities <- ridge_model %>% predict(newx = x.test)
pred_class <- ifelse(probabilities > 0.5, 1, 0)
class_pred <- lassoTest$match
mean(pred_class == class_pred)
#0.8514706
cv_ridge <- cv.glmnet(x, y, alpha = 0, family = "binomial")
plot(cv_ridge)
cv_ridge$lambda.min
cv_ridge$lambda.1se

ridge_model <- glmnet(x, y, alpha = 0, family = "binomial",
                      lambda = cv_ridge$lambda.min)

probabilities <- ridge_model %>% predict(newx = x.test)
predicted.classes <- ifelse(probabilities > -6.2, 1, 0)
mean(predicted.classes == lassoTest$match)



lasso_model <- glmnet(x, y, alpha = 1, lambda = optimalLambda)
probabilities <- lasso_model %>% predict(newx = x.test)
pred_class <- ifelse(probabilities > 0.5, 1, 0)
class_pred <- lassoTest$match
mean(pred_class == class_pred)

cv_lasso <- cv.glmnet(x, y, alpha = 1, family = "binomial")
plot(cv_lasso)
cv_lasso$lambda.min
cv_lasso$lambda.1se
log(cv_lasso$lambda.1se)

lasso_model <- glmnet(x, y, alpha = 1, family = "binomial",
                      lambda = cv_lasso$lambda.min)

probabilities <- ridge_model %>% predict(newx = x.test)
predicted.classes <- ifelse(probabilities > -6.2, 1, 0)
mean(predicted.classes == lassoTest$match)

  

# Tree --------------------------------------------------------------------
# datingTree <- tree(match ~., datingData)
# summary(datingTree)
# 
# plot(datingTree)
# text(datingTree, pretty = 0)

# treePred2 <- predict(tree, testSet, type = "class")
# table(treePred2,testSet$match)
# mean(treePred2==testSet$match)
# 
# 
# datingTree2 <- tree(match ~., datingData,subset = train)
# treePred2 <- predict(datingTree2, testSet, type = "class")
# table(treePred2,testSet$match)
# mean(treePred2==testSet$match)
# 
# cv_dating <- cv.tree(datingTree2, FUN = prune.misclass)
# par(mfrow = c(1, 2))
# plot(cv_dating$size, cv_dating$dev, type = "b")
# plot(cv_dating$k, cv_dating$dev, type = "b")
# 
# prune_datingData <- prune.misclass(datingTree2, best = 5)
# plot(prune_datingData)
# text(prune_datingData, pretty = 0)
# 
# tree_prune_pred <- predict(prune_datingData, testSet, type = "class")
# table(tree_prune_pred, testSet$match) 
# mean(tree_prune_pred==testSet$match)

plotcp(tree)
printcp(tree)

tree <- rpart(match ~., data=trainingSet, control=rpart.control(cp=0.0095969), minsplit = 10, minbucket=3)
best <- tree$cptable[which.min(tree$cptable[,"xerror"]),"CP"]
pruned_tree <- prune(tree, cp=best)
prp(pruned_tree)

treePred <- predict(pruned_tree, testSet, type = "class")
table(treePred,testSet$match)
mean(treePred==testSet$match)

bagDating <- randomForest(match~., data = datingData, subset = sample, mtry = 35, importance = TRUE)
bagDating

bagPred <- predict(bagDating, testSet)
bagPred
print(mean((bagPred==testSet$match)^2))
mean(bagPred==testSet$match)

vip::vip(bagDating, num_features = 35, bar = FALSE)

rfDating <- randomForest(match~., data = datingData,subset = sample, mtry = 12, importance = TRUE)
rfPRed <- predict(rfDating, newdata = testSet)
mean(rfPRed==testSet$match)
vip::vip(rfDating, num_features = 35, bar = FALSE)

varImpPlot(rfDating)

boostDating <- gbm(match~.,data = trainingSet,distribution = "gaussian", n.trees = 5000, interaction.depth = 4)
summary(boostDating)
boostPred <- predict(boostDating, newdata = testSet, n.trees = 5000)
predict_glm_values = ifelse(boostPred > 1.85,1, 0)
mean(predict_glm_values==testSet$match)

vip::vip(boostDating, num_features = 35, bar = FALSE)

df <- data.frame(summary(boostDating))
write_xlsx(df,"/Users/oliverreidmiller/Desktop/DATA 300/FINAL PROJECT/boosted.xlsx")


# numList <- seq(1.8, 1.95, by=.01)
# for(i in numList){
#   predict_glm_values = ifelse(boostPred > i, 1, 0)
#   accuracy<-mean(predict_glm_values==testSet$match)
#   print(paste(i,': ',accuracy))
# }
# for (i in 3:6){
#   boostDating <- gbm(match~.,data = trainingSet,distribution = "gaussian", n.trees = 5000, interaction.depth = i)
#   boostPred <- predict(boostDating, newdata = testSet, n.trees = 5000)
#   predict_glm_values = ifelse(boostPred > 1.85,1, 0)
#   accuracy <- mean(predict_glm_values==testSet$match)
#   print(paste(i,': ', accuracy))
# }

# Classfication -----------------------------------------------------------
glm_fits <- glm(match ~ samerace + field_cd_F + imprace_F + attr_F + attr_M + 
                  sinc_F + sinc_M + intel_M + fun_F + fun_M + like_F + like_M + 
                  prob_F + prob_M + imprace_M + imprelig_M + date_M+intel_F, data = trainingSet, family = binomial)
summary(glm_fits)

#backwardSel <- stepAIC(glm_fits, direction = 'backward', trace = 0)

predict_glm = predict(glm_fits, newdata = testSet,type = "response")
predict_glm_values = ifelse(predict_glm > 0.63, 1, 0)
mean(predict_glm_values==testSet$match)

ldaModel <- lda(match~., trainingSet)
ldaPred <- predict(ldaModel, testSet)$class
tab <- table(Predicted = ldaPred, Actual = testSet$match)
sum(diag(tab))/sum(tab)

qda_fit <- qda(match~imprace_F+attr_F+attr_M+sinc_M+fun_F+fun_M+like_F+like_M+prob_F+prob_M+imprelig_M,data = trainingSet)
predict_qda = predict(qda_fit, newdata = testSet,type = "response")
predict_qda_values = ifelse(predict_qda$posterior[,2] > 0.65, 1, 0)
mean(predict_qda_values==testSet$match)

roc_logreg = roc(response = testSet$match,
                 predictor = predict_glm) 
roc_logreg$auc
roc_lda = roc(response = testSet$match,
              predictor = predict_lda$posterior[,2])
roc_lda$auc
roc_qda = roc(response = testSet$match,
              predictor = predict_qda$posterior[,2])
roc_qda$auc
ggroc(list(logreg = roc_logreg,
           lda = roc_lda,
           qda = roc_qda))

#ADD DISTRIBUTIONS datingData
#ADD DISTRIBUTIONS OF MATCHES IN TRAINING AND TEST SET 



# SVM ---------------------------------------------------------------------
train_control = trainControl(method = "cv", number = 5)
model <- train(match~., data = trainingSet, method = "svmLinear", trControl = train_control)

pred_y = predict(model, testSet)
confusionMatrix(data = pred_y, testSet$match)
model$bestTune

model <- train(match~., data = trainingSet, method = "svmRadial", trControl = train_control)
pred_y = predict(model, testSet)
confusionMatrix(data = pred_y, testSet$match)
model$bestTune

model <- train(match~., data = trainingSet, method = "svmPoly", trControl = train_control)
pred_y = predict(model, testSet)
confusionMatrix(data = pred_y, testSet$match)
model$bestTune


# tune_out <- tune(svm,match~., data = trainingSet, kernel = "linear",
#                  ranges = list(cost = c(1,2,3,4,5)))
# summary(tune_out)

# svm_linear <- svm(match~., data = trainingSet, kernel = "linear",gamma = 1, cost = 2)
# summary(svmfit_radial)
# svm_linear_pred = predict(svm_linear, testSet)
# 
# mean(svm_linear_pred==testSet$match)
# 
# pROC_obj <- roc(testSet$match,as.numeric(svm_linear_pred),
#                 smoothed = TRUE,
#                 # arguments for ci
#                 ci=TRUE, ci.alpha=0.9, stratified=FALSE,
#                 # arguments for plot
#                 plot=TRUE, auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE,
#                 print.auc=TRUE, show.thres=TRUE)
# 
# 
# tune_out <- tune(svm, match~., data = trainingSet,
#                  kernel = "radial", 
#                  ranges = list(
#                    cost = c(0.1, 1, 10),
#                    gamma = c(0.5, 1, 2, 3, 4) )
#                  )
# pred_radial = predict(tune_out$best.model, testSet)
# 
# mean(pred_radial==testSet$match)
# 
# 
# ctrl <- trainControl(method="cv",
#                      number = 2,
#                      summaryFunction=twoClassSummary,
#                      classProbs=TRUE)
# 
# grid <- expand.grid(sigma = c(.01, .015, 0.2),
#                     C = c(0.75, 0.9, 1, 1.1, 1.25))
# 
# svm.tune <- train(x=trainX,
#                   y= svm.train$Class,
#                   method = "svmRadial",
#                   metric="ROC",
#                   tuneGrid = grid,
#                   trControl=ctrl)


modelDf <- read.csv('/Users/oliverreidmiller/Desktop/DATA 300/FINAL PROJECT/ModelAccuracy.csv')
modelDf<-modelDf[-c(10:21),-c(3:7)]
  ggplot(data=modelDf, aes(x=reorder(Model, -Test.Set.Accuracy), y=Test.Set.Accuracy)) +
    geom_bar(stat="identity", width=0.5)+ geom_text(aes(label = Test.Set.Accuracy), vjust = -0.2)+
    theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))+ggtitle('Accuracy of Classification Models')+
    labs(subtitle = 'Figure 14')+ coord_cartesian(ylim=c(75,100))+ xlab('Model')+
    ylab('Test-Set Accuracy (%)')+labs(caption = 'Figure 14')
  