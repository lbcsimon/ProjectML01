# Title     : main.R.R
# Author    : Menamot
# Created on: 2021/5/23

library('rpart')
library('randomForest')
library('MASS')
library('FNN')
library('naivebayes')
library('nnet')
library('smotefamily')
library('costsensitive')
library("pROC")

# 读取数据
# origin_data <- read.csv('./mitbih_train.csv',header = FALSE)
# test_data <- read.csv('./mitbih_test.csv',header = FALSE)

# 随随便便看看数据


# 决策树
tree_model <- rpart(origin_data$V188~.,data = origin_data,method = "class",control=rpart.control(xval=10,minbucket=10,cp=0))
# 看一下决策树长什么样
# plot(tree_model,margin = 0.1)
# 看一下cp值
# plotcp(tree_model)

pred_origin_tree <- predict(tree_model,newdata=test_data,type="class")
table(test_data$V188,pred_origin_tree)
err_origin_tree<- 1-mean(test_data$V188==pred_origin_tree)

# 修剪一下树
pruned_tree <- prune(tree_model,cp=9e-05)
pred_pruned_tree <- predict(pruned_tree,newdata=test_data,type="class")
table(test_data$V188,pred_pruned_tree)
err_pruned_tree<- 1-mean(test_data$V188==pred_pruned_tree)

# 随机森林模型(训练时间长)
forest_model <- randomForest(as.factor(origin_data$V188)~.,data=origin_data,mtry=3,importance=TRUE)
pred_forest <- predict(forest_model,newdata=test_data,type="class")
table(test_data$V188,pred_forest)
err_forest <- 1-mean(test_data$V188==pred_forest)
# 明显随机森林的错误率的表现要好很多，分类三的错误率少很多

# KNN模型
pred_knn <- knn(origin_data,test_data,origin_data$V188,k=5)
table(test_data$V188,pred_knn)
err_knn <- 1-mean(test_data$V188==pred_knn)
# 如何解释KNN模型准确律如此高的原因？
# KNN的原理是计算测试点与训练集中的各个点之间的高斯距离，取k个最近的点进行分类投票
# 那么对于本数据集来说，这个过程相当于找到与测试数据波形最为相似的几个数据，进行投票
# 在医学上，判断一种心跳是否异常也是看其与典型患病者的心跳是否相似，所以KNN更加符合这个想法

# LDA模型
lda_model <- lda(origin_data$V188~.,data=origin_data)
pred_lda <- predict(lda_model,newdata=test_data,type="class")
table(test_data$V188,pred_lda$class)
err_lda <- 1-mean(test_data$V188==pred_lda$class)

# 朴素贝叶斯模型
bayes_model <- naive_bayes(as.factor(origin_data$V188)~.,data=origin_data)
pred_bayes <- predict(bayes_model,newdata=test_data,type="class")
table(test_data$V188,pred_bayes)
err_bayes <- 1-mean(test_data$V188==pred_bayes)
# 如何解释贝叶斯模型错误率如此之高的原因？？？
# 朴素贝叶斯通过假设数据中p维度之间是无关的，这也是朴素两个字的含义，但是在本训练集中，数据的各个输入量
# 是明显暗含一定关系的，比如相邻两个点的差值一定在一个小范围内，我们不能认为项于项之间独立，所以贝叶斯模型不适用

# logistic模型
logistic_model <- multinom(origin_data$V188~.,data=origin_data)
pred_logistic <- predict(logistic_model,newdata=test_data,type="class")
table(test_data$V188,pred_logistic)
err_logistic <- 1-mean(test_data$V188==pred_logistic)

# 神经网络模型
bp_model <- nnet(as.factor(origin_data$V188)~.,data=origin_data,size = 5,decay=5e-4,linout = TRUE,maxit=400,MaxNWts=2000)
pred_bp <- predict(bp_model,newdata=test_data,type="class")
table(test_data$V188,pred_bp)
err_bp <- 1-mean(test_data$V188==pred_bp)

# 经过上述训练，我们可以观察到，训练总体性能不错，但是在少数类上性能很差，例如在分类三种，准确率只有一半不到
# 原因：类0占据了绝大数样本，对于机器学习而言，其训练效果是与训练集的总体代价成直接关系的，如果一个类的样本
# 数量过多，那么训练的总体代价也明显取决于这个类，从而我们会得到一个在大类样本上表现良好，但是在小类样本上表
# 线很差的模型

# 解决方案：1.减少大类数据量。但是这样做会明显丢失大类的样本信息，我们应该要尽量减少这种情况
#         2.增加少数类的数据量，这样可以在不丢失大类数据的信息时，提高少数类对总体代价的影响，从而改善
#           训练模型在少数类上的训练表现
#         3.增加少数类的训练代价，在传统模型中，我们认为每个类的训练代价权值是一样的，但是我们可以通过
#           增加少数类的代价权重，改善模型,但是实际上，如果想将少数类的权值提高三倍，只需要增加三倍的数据量就行


# 利用SMOTE增加数据集中最少样本类的样本数量，在这里我们先增加了类3的数量，后增加了类1的数量
smote_data <- SMOTE(origin_data[,1:187],as.numeric(origin_data[,c(188)]),dup_size = 6)
smote_data1 <- SMOTE(origin_data[,1:187],as.numeric(origin_data[,c(188)]),dup_size = 9)
smote_data <- SMOTE(smote_data$data[,1:187],as.numeric(smote_data$data[,c(188)]),dup_size = 4)
smote_data1 <- SMOTE(smote_data$data[,1:187],as.numeric(smote_data$data[,c(188)]),dup_size = 2)

# 由于SMOTE的方法产生的class为字符类型，需要转化为numeric类型才能被knn方法使用
smote_data$data[,c(188)] <- as.numeric(unlist(smote_data$data[,c(188)]))
smote_data1$data[,c(188)] <- as.numeric(unlist(smote_data1$data[,c(188)]))

# 进行knn预测

pred_smote_knn <- knn(smote_data$data,test_data,smote_data$data$class,k=5)
pred_smote_knn1 <- knn(smote_data1$data,test_data,smote_data1$data$class,k=5)
table(test_data$V188,pred_smote_knn)
table(test_data$V188,pred_smote_knn1)
err_smote_knn <- 1-mean(test_data$V188==pred_smote_knn)
err_smote_knn1 <- 1-mean(test_data$V188==pred_smote_knn1)

# 决策树
tree_model_smote <- rpart(smote_data$data$class~.,data = smote_data$data,method = "class",control=rpart.control(xval=10,minbucket=10,cp=0))
pred_smote_origin_tree <- predict(tree_model_smote,newdata=test_data,type="class")
table(test_data$V188,pred_smote_origin_tree)
err_smote_origin_tree <- 1-mean(test_data$V188==pred_smote_origin_tree)

# 随机森林
forest_model_smote <- randomForest(as.factor(smote_data$data$class)~.,data=smote_data$data,mtry=3,importance=TRUE)
pred__smote_forest <- predict(forest_model_smote,newdata=test_data,type="class")
table(test_data$V188,pred__smote_forest)
err_smote_forest <- 1-mean(test_data$V188==pred__smote_forest)

# LDA
lda_smote_model <- lda(smote_data$data$class~.,data=smote_data$data)
pred_smote_lda <- predict(lda_smote_model,newdata=test_data,type="class")
table(test_data$V188,pred_smote_lda$class)
err_smote_lda <- 1-mean(test_data$V188==pred_smote_lda$class)

# logistic
logistic_model_smote <- multinom(smote_data$data$class~.,data=smote_data$data)
pred_smote_logistic <- predict(logistic_model_smote,newdata=test_data,type="class")
table(test_data$V188,pred_smote_logistic)
err_smote_logistic <- 1-mean(test_data$V188==pred_smote_logistic)

# 神经网络
bp_smote_model <- nnet(as.factor(smote_data$data$class)~.,data=smote_data$data,size = 5,decay=5e-4,linout = TRUE,maxit=400,MaxNWts=2000)
pred_smote_bp <- predict(bp_smote_model,newdata=test_data,type="class")
table(test_data$V188,pred_smote_bp)
err_smote_bp <- 1-mean(test_data$V188==pred_smote_bp)

# 该函数用于计算分类效果，即每个类的准确率的平均值作为该预测的效果
cal_per <- function (pred){
  avr_0 <- mean(pred[test_data$V188==0] == test_data$V188[test_data$V188==0])
  avr_1 <- mean(pred[test_data$V188==1] == test_data$V188[test_data$V188==1])
  avr_2 <- mean(pred[test_data$V188==2] == test_data$V188[test_data$V188==2])
  avr_3 <- mean(pred[test_data$V188==3] == test_data$V188[test_data$V188==3])
  avr_4 <- mean(pred[test_data$V188==4] == test_data$V188[test_data$V188==4])
  avr <- ((avr_0+avr_1+avr_2+avr_3+avr_4)/5)
  return(avr)
}

perfomance_bp <- cal_per(pred_bp)
perfomance_lda <- cal_per(pred_lda$class)
perfomance_origin_tree <- cal_per(pred_origin_tree)
perfomance_forest <- cal_per(pred_forest)
perfomance_logisitic <- cal_per(pred_logistic)
perfomance_knn <- cal_per(pred_knn)

perfomance_smote_bp <- cal_per(pred_smote_bp)
perfomance_smote_lda <- cal_per(pred_smote_lda$class)
perfomance_smote_origin_tree <- cal_per(pred_smote_origin_tree)
perfomance_smote_forest <- cal_per(pred__smote_forest)
perofomace_smote_logisitic <- cal_per(pred_smote_logistic)
perfomance_smote_knn <- cal_per(pred_smote_knn)
# 经过过采样后，性能有所上升