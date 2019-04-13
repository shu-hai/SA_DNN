# SA_DNN
This is the repository for the following AAAI-19 paper: 
Shu, H., and Zhu, H. (In press) Sensitivity Analysis of Deep Neural Networks. The Thirty-Third AAAI Conference on Artificial Intelligence (AAAI-19). [arXiv:1901.07152](https://arxiv.org/abs/1901.07152)

Also see the [talk slides](https://github.com/shu-hai/SA_DNN/blob/master/Slidesfor330_Shu.pdf).

Use CIFAR10_sample.py and MNIST_sample.py to obtain the CIFAR10 and MNIST datasets.

ResNet50.py and DenseNet121.py are the two networks, which are called to be trained by CIFAR10_ResNet50.py, CIFAR10_DenseNet121.py, MNIST_ResNet50.py or MNIST_DenseNet121.py.

Then for the two benchmark datasets, take CIFAR10 and DenseNet121 for example. 
Run CIFAR10_DenseNet121_IF_setupX.py for Setup X in the paper, where X=1,2,3,4. 
To summarize the results, use the R code CIFAR10_DenseNet121_IF_setupX_result.R first, then use the python code CIFAR10_DenseNet121_IF_setupX_result.py for the plots.
