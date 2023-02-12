# SA_DNN
This is the repository for the following AAAI-19 paper: 

Shu, H., and Zhu, H. (2019) Sensitivity Analysis of Deep Neural Networks. The Thirty-Third AAAI Conference on Artificial Intelligence (AAAI-19), pp. 4943-4950 [[DOI]](https://doi.org/10.1609/aaai.v33i01.33014943 )

Also see the [talk slides](https://github.com/shu-hai/SA_DNN/blob/master/Slidesfor330_Shu.pdf).

We also proposed a method, called mFI-PSO, for Adversarial Image Generation based on the Manifold-based First-order Influence (mFI) measure introduced in the paper. See https://github.com/shu-hai/mFI-PSO for details.


Instructions:

Use CIFAR10_sample.py and MNIST_sample.py to obtain the CIFAR10 and MNIST datasets.

ResNet50.py and DenseNet121.py are the two networks, which are called to be trained by CIFAR10_ResNet50.py, CIFAR10_DenseNet121.py, MNIST_ResNet50.py or MNIST_DenseNet121.py.

Then for the two benchmark datasets, take CIFAR10 and DenseNet121 for example. 
Run CIFAR10_DenseNet121_IF_setupX.py for Setup X in the paper, where X=1,2,3,4. 
To summarize the results, first use the R code CIFAR10_DenseNet121_IF_setupX_result.R, and then use the python code CIFAR10_DenseNet121_IF_setupX_result.py for the plots.
