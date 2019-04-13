#setwd('E:/Hai_Windows/work/DLSensitivity/code/result')

setup3.train=numeric(45000)
for(j in 1:1000)
{
  setup3.train[((j-1)*45+1):(j*45)]=as.matrix(read.table(paste('./result/CIFAR10_ResNet50_128_best_setup3_part',j,'_train.txt',sep='') ))
}

setup3.test=numeric(10000)
for(j in 1:1000)
{
  setup3.test[((j-1)*10+1):(j*10)]=as.matrix(read.table(paste('./result/CIFAR10_ResNet50_128_best_setup3_part',j,'_test.txt',sep='') ))
}

label.train=scan("CIFAR10_train_label.txt")
label.train.conti=numeric(45000)

label.test=scan("CIFAR10_test_label.txt")
label.test.conti=numeric(10000)

for(i in 0:9){
  n_i_train = sum(label.train==i)
  label.train.conti[label.train==i]=head(seq(i,i+1,by=1/n_i_train),-1)
  n_i_test = sum(label.test==i)
  label.test.conti[label.test==i]=head(seq(i,i+1,by=1/n_i_test),-1)
}

save.train=cbind(setup3.train,label.train.conti)
write.table(save.train,'setup3_CIFAR10_ResNet50_train_manhattan.txt',sep=' ',row.names = F,
            col.names = F)

save.test=cbind(setup3.test,label.test.conti)
write.table(save.test,'setup3_CIFAR10_ResNet50_test_manhattan.txt',sep=' ',row.names = F,
            col.names = F)


setup3_CIFAR10_ResNet50_train_manhattan=read.table('setup3_CIFAR10_ResNet50_train_manhattan.txt')
setup3_CIFAR10_ResNet50_test_manhattan=read.table('setup3_CIFAR10_ResNet50_test_manhattan.txt')

quantile(setup3_CIFAR10_ResNet50_train_manhattan[,1],c(seq(0.75,1,0.05),0.98,0.99))
quantile(setup3_CIFAR10_ResNet50_test_manhattan[,1],c(seq(0.75,1,0.05),0.98,0.99))


setup3_CIFAR10_DenseNet121_train_manhattan=read.table('setup3_CIFAR10_DenseNet121_train_manhattan.txt')
setup3_CIFAR10_DenseNet121_test_manhattan=read.table('setup3_CIFAR10_DenseNet121_test_manhattan.txt')

quantile(setup3_CIFAR10_DenseNet121_train_manhattan[,1],c(seq(0.75,1,0.05),0.98,0.99))
quantile(setup3_CIFAR10_DenseNet121_test_manhattan[,1],c(seq(0.75,1,0.05),0.98,0.99))

##############################################the below is nothing
log.train=log10(1+setup3.train)
log.test=log10(1+setup3.test)

par(mfrow=c(1,2),mar=c(5, 5, 2, 1))
plot(log.train)
plot(log.test)

hist(log.train,breaks=100)
hist(log.test,breaks=100)

library(sm)
library(vioplot)

plot(1, 1, xlim = c(0, 4), ylim = range(c(log.train, log.test)), type = 'n', xlab = '', ylab = '', xaxt = 'n')
vioplot(log.train, at = 1, add = T, col = 'green')
vioplot(log.test, at = 3, add = T, col = 'magenta')



# set horizontal axis
axis(1, at = c(1,3), labels = c('New York', 'Ozonopolis'))

# set vertical axis
# at = 150 sets height of label at y = 150
# pos = -0.45 ensures that the label is at x -0.45 and not overlapping the numbers along the tick labels
# tck = 0 ensures that the label does not get its own tick mark
# try not including the at, pos, and tck options and see what you get
axis(2, at = 150, pos = -0.45, tck = 0, labels = 'Ozone Concentration (ppb)')

# add title
title(main = 'Violin Plots of Ozone Concentration\nNew York and Ozonopolis')

# add legend
# the option bty = 'n' ensures that there is no border around the legend
# the option lty must be specified, because I'm drawing lines as the markers for each city in the legend
# the option lwd specifies the width of the lines in the legend
legend(0.4, 250, legend = c('New York', 'Ozonopolis'), bty = 'n', lty = 1, lwd = 7, col = c('green', 'magenta'))
