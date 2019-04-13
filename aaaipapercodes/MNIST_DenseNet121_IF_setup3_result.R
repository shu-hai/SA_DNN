#setwd('E:/Hai_Windows/work/DLSensitivity/code/result')

setup3.train=numeric(54000)
for(j in 1:1000)
{
  setup3.train[((j-1)*54+1):(j*54)]=as.matrix(read.table(paste('./result/MNIST_DenseNet121_128_best_setup3_part',j,'_train.txt',sep='') ))
}

setup3.test=numeric(10000)
for(j in 1:1000)
{
  setup3.test[((j-1)*10+1):(j*10)]=as.matrix(read.table(paste('./result/MNIST_DenseNet121_128_best_setup3_part',j,'_test.txt',sep='') ))
}

label.train=scan("MNIST_train_label.txt")
label.train.conti=numeric(54000)

label.test=scan("MNIST_test_label.txt")
label.test.conti=numeric(10000)

for(i in 0:9){
  n_i_train = sum(label.train==i)
  label.train.conti[label.train==i]=head(seq(i,i+1,by=1/n_i_train),-1)
  n_i_test = sum(label.test==i)
  label.test.conti[label.test==i]=head(seq(i,i+1,by=1/n_i_test),-1)
}

save.train=cbind(setup3.train,label.train.conti)
write.table(save.train,'setup3_MNIST_DenseNet121_train_manhattan.txt',sep=' ',row.names = F,
            col.names = F)

save.test=cbind(setup3.test,label.test.conti)
write.table(save.test,'setup3_MNIST_DenseNet121_test_manhattan.txt',sep=' ',row.names = F,
            col.names = F)

which(setup3.test%in%tail(sort(setup3.test),5))
#917 2370 3074 4741 9665

#which(setup3.train==max(setup3.train))
#23874
