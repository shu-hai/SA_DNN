setwd('E:/Hai_Windows/work/DLSensitivity/code/')
####################all layers' parameters

all=numeric(45000)
for(j in 1:1000)
{
  all[((j-1)*45+1):(j*45)]=as.matrix(read.table(paste('./result/CIFAR10_DenseNet121_128_best_setup2_part',j,'_all.txt',sep='') ))
}

label=scan("CIFAR10_train_label.txt")

label.conti=numeric(45000)

for(i in 0:9){
  n_i = sum(label==i)
  label.conti[label==i]=head(seq(i,i+1,by=1/n_i),-1)
}

save.data=cbind(all,label.conti)
write.table(save.data,'setup2_CIFAR10_DenseNet121_all_manhattan.txt',sep=' ',row.names = F,
            col.names = F)

####################Each layer's parameters

each=matrix(0,45000,545)
for(j in 1:1000)
{
  each[((j-1)*45+1):(j*45),]=as.matrix(read.table(paste('./result/CIFAR10_DenseNet121_128_best_setup2_part',j,'_layers.txt',sep='') ))
}


each.n0=each[,each[1,]!=-1]

num.n0=dim(each.n0)[2]#number of trainable layers=242

each.n0.v=as.vector(each.n0)
img.id=rep(1:45000,num.n0)
layer.id=head(seq(0,num.n0,by=1/45000),-1)#rep(1:242,45000)

save.data=cbind(each.n0.v,layer.id)
write.table(save.data,'setup2_CIFAR10_DenseNet121_layers_manhattan.txt',sep=' ',row.names = F,
            col.names = F)

