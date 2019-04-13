library(precrec)
library(ggplot2)
#use the 5e04 and 251-300 result

#ResNet50
y.FI.Noutlier=as.matrix(read.table(paste('./outlier_first50/MNIST_ResNet50_128_best_setup0_FI_n_outlier.txt',sep='') ) )
y.FI.outlier=as.matrix(read.table(paste('./outlier_first50/MNIST_ResNet50_128_best_setup0_FI_outlier.txt',sep='') ))

y.Jacob.Noutlier=as.matrix(read.table(paste('./outlier_first50/MNIST_ResNet50_128_best_setup0_Jacob_n_outlier.txt',sep='') ))
y.Jacob.outlier=as.matrix(read.table(paste('./outlier_first50/MNIST_ResNet50_128_best_setup0_Jacob_outlier.txt',sep='') ))

y.FI=rbind(y.FI.Noutlier,y.FI.outlier)
y.Jacob=rbind(y.Jacob.Noutlier,y.Jacob.outlier)

y=c(rep(0,dim(y.FI.Noutlier)[1]),rep(1,dim(y.FI.outlier)[1]))


scores1=join_scores(y.FI[,1],y.Jacob[,1])
labels1=join_labels(y,y)
msmdat <- mmdata(scores1, labels1, modnames = c("FI", "JN"))
mscurves <- evalmod(msmdat)
autoplot(mscurves)


#DenseNet121
y.FI.Noutlier=as.matrix(read.table(paste('./outlier_first50/MNIST_DenseNet121_128_best_setup0_FI_n_outlier.txt',sep='') ) )
y.FI.outlier=as.matrix(read.table(paste('./outlier_first50/MNIST_DenseNet121_128_best_setup0_FI_outlier.txt',sep='') ))

y.Jacob.Noutlier=as.matrix(read.table(paste('./outlier_first50/MNIST_DenseNet121_128_best_setup0_Jacob_n_outlier.txt',sep='') ))
y.Jacob.outlier=as.matrix(read.table(paste('./outlier_first50/MNIST_DenseNet121_128_best_setup0_Jacob_outlier.txt',sep='') ))

y.FI=rbind(y.FI.Noutlier,y.FI.outlier)
y.Jacob=rbind(y.Jacob.Noutlier,y.Jacob.outlier)

y=c(rep(0,dim(y.FI.Noutlier)[1]),rep(1,dim(y.FI.outlier)[1]))


scores1=join_scores(y.FI[,1],y.Jacob[,1])
labels1=join_labels(y,y)
msmdat <- mmdata(scores1, labels1, modnames = c("FI", "JN"))
mscurves <- evalmod(msmdat)
autoplot(mscurves)


