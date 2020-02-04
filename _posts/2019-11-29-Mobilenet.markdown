---
layout:     post
title:      "MobileNets Analysis: From V1 to V3"
author:     "Jian-Hui Duan"
tags: 		网络压缩 Nets-Compression
---



## 1. **Depthwise Separable Convolution**

![a](/img/in-post/img/dsc.png)

原始图像大小为$$ D_F \times D_F $$，如果channel一共有$M$个，卷积核大小为$D_k \times D_k$。

* **Depthwise Convolution**

  Depthwise的方法是在DW层只使用$M$个卷积核，每一个图像的channel都使用一个单独的卷积核，然后产生最后的中间输出。如上图琐事，原始输入矩阵大小为$D_F \times D_F\times5$，那么在DW层使用了5个卷积核，分别对5个channel做普通卷积，得到5幅feature map，然后将5幅feature map叠加起来形成一个与原始输入大小相同的$D_F \times D_F\times5$的feature map。

* **Pointwise Convolution**

  Pointwise是在DW基础上进行的卷积操作，卷积核的大小为$1\times1\times M$，共有$N$个，通过这个卷积核同时从DW输出的feature map第一个像素开始对所有channel相同位置的像素进行卷积，输出大小为$D_F \times D_F \times N$。

#### 运算复杂度分析

原始卷积的运算复杂度，按照上述的图像大小以及卷积核的大小而言，应该为$D_F \times D_F \times M \times N \times D_k \times D_k$。其中不包括加法开销，仅计算乘法开销。

**解释**：要对原始图像每个像素都进行一次卷积，那么要做$D_F \times D_F$次；一共有$M$个channel，则每次卷积的乘法次数为$ D_k \times D_k \times M$，共有$N$个这样的卷积核，所以最后的开销为$D_F \times D_F \times M \times N \times D_k \times D_k$。

* **Depthwise Convolution**

  在DW中，卷积核已经不是一个三维的卷积核，变成了二维的卷积核。每一次的卷积乘法次数是$D_k \times D_k$次，并且一个卷积核对应一个channel，每个channel的图像大小为$D_F \times D_F$，有$M$个channel，因此最后的运算复杂度为$D_F \times D_F \times M \times D_k \times D_k$。

* **Pointwise Convolution**

  在PW中，单个卷积核的大小为$1\times1\times M$，针对每个像素进行卷积，那么最后的运算复杂度为$D_F \times D_F \times M \times N$。

采用Depthwise Separable Convolution，运算复杂度为$D_F \times D_F \times M \times D_k \times D_k + D_F \times D_F \times M \times N$。和普通的卷积层的操作相对比为：

$$ \begin{equation}

\frac{D_F \times D_F \times M \times D_k \times D_k + D_F \times D_F \times M \times N}{D_F \times D_F \times M \times N \times D_k \times D_k} = \frac{1}{N}·\frac{1}{D_k^2} 

\end{equation}$$

#### 运算优势

MobileNet的最大优势是将矩阵运算从原先的三维矩阵降低至二维矩阵，甚至在DW中降低到一维矩阵相乘。基于矩阵稀疏性进行的优化，除非矩阵非常稀疏运算复杂度才会降低非常明显，但是基于MobileNet的矩阵运算优化，使得运算复杂度大大降低。

这种优化可以被高度优化的GEMM（General Matrix Multiply）函数进行运算，接下来会有详细介绍。并且MobileNet的$1 \times 1$运算不同于普通卷积运算的一点是，普通卷积运算需要使用im2col函数把要运算的矩阵map到GEMM函数里面。

#### MobileNet的结构与参数

![](/img/in-post/img/mobilenet.png)

整个网络有28层，单独的DW层与单独的PW层都算是一层。其中的s1,s2等表示stride=1或者stride=2，用来降采样。当然**每个卷积层之后都要紧跟着BN层，然后通过ReLU层。**整体网络的运算复杂度95%都在$1 \times 1$ 卷积的地方，并且这个那个网络的75%的参数也都在$1\times 1$卷积。

<center class="half"> 
  <img src="/img/in-post/img/conv.png" width = "300" height = "200" alt="/img/in-post/img/conv.png" />
  <img src="/img/in-post/img/para.png" width = "400" height = "200" alt="/img/in-post/img/conv.png" />
</center>  

#### 两个参数

- Width Multiplier：Thinner Models

  这个参数用来控制网络的宽度，其实就是同时控制输入channel以及输出channel到原始大小的$\alpha$倍。所以最后的计算复杂度就变成了$D_F \times D_F \times \alpha M \times D_k \times D_k + D_F \times D_F \times \alpha M \times \alpha N$，其中$\alpha \in (0, 1]$，并且在论文中将其设置为了1，0.75，0.5，0.25。根据上面的算式，应该不难发现，对于计算复杂度的降低应该是$O(\alpha^2)$。

- Resolution Multiplier: Reduced Representation

  这个参数用来控制网络中每一层输入图像的大小，控制输入图像的size变为原先的$\rho$倍。此时的计算复杂度应该为$\rho D_F \times \rho D_F \times \alpha M \times D_k \times D_k + \rho D_F \times \rho D_F \times \alpha M \times \alpha N$。其中$\rho \in (0, 1]$，这时候根据图像的size需要决定$\rho$的大小，图像size在论文中设置为224，192，160，128。根据上述的算式，最后的计算复杂度相对于这个参数而言，减少量为$O(\rho^2)$。

<center class="half"> 
  <img src="/img/in-post/img/45.png" width = "350" height = "200" />
  <img src="/img/in-post/img/67.png" width = "350" height = "200" />
</center> 

对于两个参数如何应用解释如下：

针对$\alpha$参数，直接在网络结构中的卷积核心数目上乘该参数即可。如下代码所示：

```python
  # 第一层全卷积有alpha参数是因为alpha控制所有层的输入、输出channel  
  x = Convolution2D(int(32 * alpha), (3, 3), strides=(2, 2), padding='same', use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = DepthwiseConvolution2D(int(32 * alpha), (3, 3), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Convolution2D(int(64 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    ...
```

针对$\rho$参数，论文中是简单地改变了输入图像的分辨率，采用金字塔采样的方式得到192 / 168 / 128大小的输入图像。

#### MobileNet的性能

参考论文。



#### MobileNet的问题

1. 结构非常简单，并且整体网络是堆叠形成的，性价比不高。类似AlexNet、VGG一样，所以希望能够达到ResNet、DenseNet的性能，需要进行feature map的复用，复用的功能也就是Residual block的功能。
2. 非常重要的一点是由于$1\times1$的kernel维度太小了，在训练过程中非常容易被ReLU非线形映射到0，而一旦这个值达到0，并且只有一个维度，那么急于ReLU的话就无法再恢复到0以上。



### MobileNet-V2

针对V1中出现的问题，Google在V2中进行了解释以及调整。

* **Linear Bottlenecks**

  假设某层的输出的feature map大小为HxWxD，经过ReLU激活层后称之为`manifold of interest`，可以理解为兴趣流形或有用的信息的分布呈现为流形态，大小仍为HxWxD，经验证明`manifold of interest`完全可以压缩到低维子空间（**目前我认为这种说法只是从神经网络的角度说的，因为从拟合的角度而言，一个三维的甜甜圈从正向看过去变成一个平面同样可以进行分类与拟合**），因此可以利用这种性质进行矩阵压缩。在V1版本中便可以通过`width multiplier parameter`来降低激活空间的维数使得`manifold of interest`充恰好满整个空间。问题就来了，在使用ReLU函数进行激活时，负数直接变为0，这样就会导致失去较多有用的信息（这在`manifold of interest`占激活空间较小时不是一个问题）。
   总结一下，有以下两点：

  - 如果`manifold of interest`经过ReLU后存在非零部分，这一部分意味着只经过了一个线性变换，因为ReLU的正向激活是线性激活。
  - 除非`input manifold`位于输入空间的低维子空间，经过ReLU后才能保持完整的信息。

  因此，论文中使用了`linear bottleneck`来解决由于非线性激活函数造成的信息损失问题。`linear bottleneck`本质上是不带ReLU的1x1的卷积层。

  **(图看论文)**

  用线性变换层替换channel数较少的层中的ReLU，这样做的理由是ReLU会对channel数低的张量造成较大的信息损耗。ReLU会使负值置零，channel数较低时会有相对高的概率使某一维度的张量值全为0，即张量的维度减小了，而且这一过程无法恢复。张量维度的减小即意味着特征描述容量的下降。因而，在需要使用ReLU的卷积层中，将channel数扩张到足够大，再进行激活，被认为可以降低激活层的信息损失。

* **Inverted residuals**

  颠倒的正是block 内数据维度与bottleneck数据维度的大小，这从上图的中数据块的深度情况可以看出。 
  这种颠倒基于作者的直觉：bottleneck层包含了所有的必要信息，扩展的层做的仅仅是非线性变换的细节实现。 




### MobileNet-V3

整体来说MobileNetV3有两大创新点

（1）互补搜索技术组合：由资源受限的NAS执行模块级搜索，NetAdapt执行局部搜索。

（2）网络结构改进：将最后一步的平均池化层前移并移除最后一个卷积层，引入h-swish激活函数。











