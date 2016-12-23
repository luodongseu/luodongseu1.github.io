---
layout: post
title: TensorFlow入门与实例
category: ml
keywords: ml,dl,data
---





# TensorFlow入门与实例

------

## 1. 前言

虽然标题写的是TensorFlow入门与实践，但是完成了这个实例之后感觉一般化的deeplearning问题基本都是可以解决的。TensorFlow在神经网络方面，提供了非常完整的工具化支持，其中包括：

> * 全连接层、半连接层、CNN层、RNN层不同神经元的定义
> * BP自实现，损失函数，下降梯度等自计算
> * 完整的WebUI统计工具（包括变量可视化，过程可视化，计算模型可视化）
> * 导入导出模型


相比起来，h2o的操作就太傻瓜式了，我想定义几层神经网络，直接整改个参数，reformat一个数据类型丢进去就好了。

在此之前我有写了一些非DL(deep learning，下面都用DL代替)的印象笔记大约有17篇，其中主要讲了一些分类聚类等算法，可以给大家参考下。下载见附件

废话不多说，观看此片文章需要读者具有如下的知识积累：

> * 基本的非DL的算法基础，看的懂Loss算法，晓得什么叫下降梯度
> * 基础的DL基础，晓得什么叫CNN，有哪些激活函数
> * 有Python基础，最好会一些Numpy
> * 了解ML的一般化过程，训练集，测试集等


这些不了解都没关系，我将在第二节都介绍下，但是其中公式我只列出结果，具体推到过程我就不细说了，如果了解的话可以直接看第三章，我们将实现一个**门牌号识别**的实例


**注意:tensorflow对python2支持很差，所以以下python都是py3**


## 2. 预备知识


### 2.1. one-hot encoding 

One-Hot编码，又称为一位有效编码，是一种对数据预处理的方式，主要用来特征化数字，他的具体实现方式是：**采用N位状态寄存器来对个N状态进行编码**

例子：假如我有0-9个特征值，那么进行了One-Hot编码后得到的结果是:

[1,0,0,...,0]
[0,1,0,...,0]
[0,0,1,...,0]
...
[0,0,0,...,1]

简单来说就是大小为10的数组，为0-9下标的数字分别赋值1，得到特征矩阵


### 2.2. 单层感知器

单个神经元就可以构成一个最简单的神经网络——感知机。在单层神经元感知机中，网络接收若干过输入，并通过输入函数、传输函数给出一个网络的输出。

感知器（Perceptron），是神经网络中的一个概念，在1950s由Frank Rosenblatt第一次引入。

单层感知器（Single Layer Perceptron）是最简单的神经网络。它包含输入层和输出层，而输入层和输出层是直接相连的。


![感知机](http://neuralnetworksanddeeplearning.com/images/tikz0.png)


单层感知机的公式为：

![gz-gs](http://7xkw0v.com1.z0.glb.clouddn.com/Image.png)

其实多层的单个神经节点公式也是如此


利用公式1计算输出层，这个公式也是很好理解。首先计算输入层中，每一个输入端和其上的权值相乘，然后将这些乘机相加得到乘机和。对于这个乘机和做如下处理，如果乘机和大于临界值（一般是0），输入端就取1；如果小于临界值，就取-1。

感知器的学习规则为 ： 

![xxgz](http://7xkw0v.com1.z0.glb.clouddn.com/Capture.PNG)

e为误差，e=t-a ， t为期望输出，a为实际输出

此规则含义是，如果感知机的输出有误，则首先计算误差e，e为期望输出和实际输出差值。新的权值等于旧值加上误差和输入p的乘积。同理，偏置可以看做是输入p恒为1的输入信号，故新的偏置等于旧偏置加入误差。当计算机出新的权重和偏置后，使用测试数据再次测试感知机，直到没有误差或误差在可接受范围内为止。




### 2.3. 全连接神经网络


全连接网络：每一层的节点互相不相连。 每一层的节点都和上一层与下一层的所有节点相连。

好处：

> * 每一层都可以个用一个矩阵单独表示。
> * 每一层到下一层的运算都可以用矩阵操作来并行。

实际上全连接神经网络是多层感知机的一种，因为多层感知机可以是非全连接的.

![dcgzj](http://7xkw0v.com1.z0.glb.clouddn.com/Image2.png)


我的理解是，它是多个单层感知器来逐渐形成的，每一层都有多个神经元，所有有多个输入与多个输出，并且每一层都有多个权重W和偏置B(之后这个我们将会以矩阵来表示他们)



### 2.4. BP回溯算法

我们有个全连接神经网络之后，那么我们的权重和偏置更新过程就不像单层感知机那样更新学习了。在06年的时候有两个人提出BP算法来回溯更新每一层的每一个权重和偏置


公式为：

![bp](http://7xkw0v.com1.z0.glb.clouddn.com/Image3.png)

在简单的感知机学习规则中，由于只有一层网络，因此通过误差可以很容易的进行权值调整，在多层感知机网络中，这种方法同样适用，唯一不同的是，误差要通过最后一层网络，逐层向前传播，用于调衡各层的权值连接。


其还有一种变形，叫做Nguyen-Widrow随机算法，它的过程更快速。就不详细介绍了




### 2.5. CNN


卷积神经网络(Convolutional Neural Network, CNN)是深度学习技术中极具代表的网络结构之一，在图像处理领域取得了很大的成功，在国际标准的ImageNet数据集上，许多成功的模型都是基于CNN的。CNN相较于传统的图像处理算法的优点之一在于，避免了对图像复杂的前期预处理过程（提取人工特征等），可以直接输入原始图像。


它主要包括了两点：**局部连接与权值共享**

下图是一个很经典的图示，左边是全连接，右边是局部连接。

![jblj](http://7xkw0v.com1.z0.glb.clouddn.com/srjacnnv.bmp)

对于一个1000 × 1000的输入图像而言，如果下一个隐藏层的神经元数目为10^6个，采用全连接则有1000 × 1000 × 10^6 = 10^12个权值参数，如此数目巨大的参数几乎难以训练；而采用局部连接，隐藏层的每个神经元仅与图像中10 × 10的局部图像相连接，那么此时的权值参数数量为10 × 10 × 10^6 = 10^8，将直接减少4个数量级。

尽管减少了几个数量级，但参数数量依然较多。能不能再进一步减少呢？能！方法就是权值共享。具体做法是，在局部连接中隐藏层的每一个神经元连接的是一个10 × 10的局部图像，因此有10 × 10个权值参数，将这10 × 10个权值参数共享给剩下的神经元，也就是说隐藏层中10^6个神经元的权值参数相同，那么此时不管隐藏层神经元的数目是多少，需要训练的参数就是这 10 × 10个权值参数（也就是卷积核(也称滤波器)的大小），如下图。

![jblj2](http://7xkw0v.com1.z0.glb.clouddn.com/e4n8kuhr.bmp)


它的具体实现过程主要是通过卷积和池化来实现的

卷积过程：

> * 每张图片都可以被表示为一个三维矩阵：宽x高x颜色通道数
> * 我们需要一个过滤器（Filter/Feature Map）, 过滤器的大小一般就是**N x N x 颜色通道数**的矩阵（颜色通道个NxN的Filter Map）
> * 输出矩阵的Depth == Filter （过滤器数量等于输出矩阵数量）
>  * Stride : 每一步的大小即每一次滑多少步 # 
>  * padding: 填充。这个是向外填充的过程，不浪费特征
> * 完整过程flash : http://cs231n.github.io/convolutional-networks


![](http://7xkw0v.com1.z0.glb.clouddn.com/Image10.png)


这张图（网页中的flash截图）展示了滑窗过程，我们有两个过滤器（W0和W1） ，过滤器 大小是3x3x3，过滤器 每次的滑动间隔为2，则可以产生9个数字（计算过程为滑窗乘过滤器 ，非矩阵相乘）即输出的大小。因为过滤器数目是2，所以输出结果的数量是2

滑窗相乘：矩阵元素一一相乘求和

[4,2] [1,8]   *   [9,4] [5,2]    =   [4,2,1,8] * [9,4,5,2] = 4 x 9 + 2 x 4 + 1 x 5 + 8 x 2 = 65 


池化过程：卷积结束后，我们将对图片进行缩小即损失精度的压缩

![pool](http://7xkw0v.com1.z0.glb.clouddn.com/Image11.png)

如图过程，将滑窗经过区域的最大值提取出来成为新的输出。2x2的滑窗，每次滑两步
我觉得这也是一种特殊卷积过程。



## 3. 数据准备与预处理


### 3.1 数据准备

数据来源于： [http://ufldl.stanford.edu/housenumbers/](http://ufldl.stanford.edu/housenumbers/)

只要下载train_32x32.mat 和 test_32x32.mat就可以，这个是matlabl导出的文件，python是可以用scipy读取的

### 3.2 数据预处理

先对train数据loadmat

```python
from scipy.io import loadmat as load
train = load('../data/train_32x32.mat')
```

可以看下train数据的key和数据shape

```python
print(train.keys)
# > ['y', 'X', '__version__', '__header__', '__globals__']
print('Train Samples Shape:', train['X'].shape)
# > (32, 32, 3, 73257)
print('Train Samples Shape:', train['X'].shape)
# > (73257, 1)
```

可以观察出来这是个图片集，train['X']是含有73257张32x32且3色道的图片，而train['y']是这些图片的label即特征。

再打印下数据可以看到我们需要处理以下问题：

> * x中需要把 （图片高，图片宽，通道数，图片数） -> （图片数 ，图片高，图片宽，通道数 ） ，前者第四维度存放着的是每一张图在该高宽坐标上且该通道上的灰度， 而后者表达的是每一张图片是一个三维数组.
> * y中需要把 labels 进行one-hot encode 
> * x中需要把三色通道线性映射到-1.0 ~ 1.0 ，并且将三色通道转化成单色通道 ， 因为多色道对数字识别无任何用处


```python

def reformat(samples,labels):

        dataset = np.transpose(samples,(3,0,1,2)) # reshape the x

        one_hots_label = []
        for label in labels:
                one_hot = [0 for i in range(10)]
                if label == 10:
                        one_hot[0] = 1.0
                else :
                        one_hot[label] = 1.0

                one_hots_label.append(one_hot)
        return dataset,np.array(one_hots_label).astype(np.float32)

# np.add.reduce 实际上是降维操作，通过add的方式进行降维,是第四个元素进行add合并，而keepdims是不让他降维，只是合并到了一维上
def normalize(samples):
        a = np.add.reduce(samples,keepdims=True,axis=3)
        a = a / 3.0
        samples = samples / 128 - 1


```


## 4.计算



### 4.1 定义多层神经网络

在tensorflow中定义与计算神经网络的方法是：


卷积：

```python

#　define

conv1_weights = tf.Variable(tf.truncated_normal([self.patch_size, self.patch_size,num_channels, self.conv1_depth], stddev=0.1))
conv1_biases = tf.Variable(tf.zeros([self.conv1_depth]))

# calc   
conv1 = tf.nn.conv2d(data, filter=conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
addition = conv1 + conv1_biases
hidden = tf.nn.relu(addition)

```

这样的一层，定义了神经网络的weights和biases，并且进行运算 

池化：

```python

hidden = tf.nn.max_pool(
			hidden,
			ksize=[1,self.pooling_scale,self.pooling_scale,1],
			strides=[1,self.pooling_stride,self.pooling_stride,1],
			padding='SAME')

```




全连接层定义:

```python

# define
fc2_weights = tf.Variable(tf.truncated_normal([self.num_hidden, num_labels], stddev=0.1), name='fc2_weights')
fc2_biases = tf.Variable(tf.constant(0.1, shape=[num_labels]), name='fc2_biases')
self.train_summaries.append(tf.histogram_summary('fc2_weights', fc2_weights))
self.train_summaries.append(tf.histogram_summary('fc2_biases', fc2_biases))

#calc 
fc1_model = tf.matmul(reshape, fc1_weights) + fc1_biases
hidden = tf.nn.relu(fc1_model)

``` 

我们的架构是： **四层CNN + 双层神经网络输出**

    即 Conv3x3_1 Relu MaxPool1 Conv3x3_2 Relu MaxPool2  WB relu WB -> softmax


为什么这么做呢？ 可以看看这么一个例子

![41d](http://7xkw0v.com1.z0.glb.clouddn.com/Image12.png)

    
relu定义小于0的数做0处理，大于0的数保持不变

可以看出我们先对图片进行conv处理，这样会将图片进行虚化，在经过RELU特征会变得更加明显（汽车的模型变得明显，RELU可以强化CONV特征提取）

而POOLING过程是将图片维度降解，即我们提取了特征，再放大了特征之后我们再将特征提取到低维度


















