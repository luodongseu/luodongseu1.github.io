---
layout: post
title: TensorFlow入门与实例
category: ml
keywords: ml,dl,data
---


## 1. 前言

虽然标题写的是TensorFlow入门与实践，但是完成了这个实例之后感觉一般化的deeplearning问题基本都是可以解决的。TensorFlow在神经网络方面，提供了非常完整的工具化支持，其中包括：

> * 全连接层、半连接层、CNN层、RNN层不同神经元的定义
> * BP自实现，损失函数，下降梯度等自计算
> * 完整的WebUI统计工具（包括变量可视化，过程可视化，计算模型可视化）
> * 导入导出模型


相比起来，h2o的操作就太傻瓜式了，我想定义几层神经网络，直接整改个参数，reformat一个数据类型丢进去就好了。

在此之前我有写了一些非DL(deep learning，下面都用DL代替)的印象笔记大约有17篇，其中主要讲了一些分类聚类等算法，可以给大家参考下。下载见最后

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

### 2.6 损失函数与正规化函数


对于一个监督学习模型来说，过小的特征集合使得模型过于简单，过大的特征集合使得模型过于复杂。

> * 对于特征集过小的情况，称之为欠拟合（underfitting）
> *　对于特征集过大的情况，称之为过拟合（overfitting）

而对于过拟合问题，是可以用正规化或者损失函数来解决的

为了防止overfitting，可以用的方法有很多。有一个概念需要先说明，在机器学习算法中，我们常常将原始数据集分为三部分：training data、validation data、testing data。

这个validation data是什么？它其实就是用来避免过拟合的，在训练过程中，我们通常用它来确定一些超参数（比如根据validation data上的accuracy来确定early stopping的epoch大小、根据validation data确定learning rate等等）。

那为啥不直接在testing data上做这些呢？因为如果在testing data做这些，那么随着训练的进行，我们的网络实际上就是在一点一点地overfitting我们的testing data，导致最后得到的testing accuracy没有任何参考意义。

因此，training data的作用是计算梯度更新权重，validation data如上所述，testing data则给出一个accuracy以判断网络的好坏


损失函数用来计算当前模型的损失值，损失值越大，表示模型越不稳定

其公式为：

![0999](http://7xkw0v.com1.z0.glb.clouddn.com/i24.PNG)


而正规化函数

![21321343](http://7xkw0v.com1.z0.glb.clouddn.com/i25.PNG)


实际上简单理解可以将正规化比作一个BP过程，用L1或者L2函数来更新权重达到一个损失最小值。

### 2.7 Dropout

同样的，我们能用随机丢失一些神经元达到更好地效果值，丢失了一些神经元同样可以达到

理由如下

> * 由于每次用输入网络的样本进行权值更新时，隐含节点都是以一定概率随机出现，因此不能保证每2个隐含节点每次都同时出现，这样权值的更新不再依赖于有固定关系隐含节点的共同作用，阻止了某些特征仅仅在其它特定特征下才有效果的情况。
> * 可以将dropout看作是模型平均的一种。对于每次输入到网络中的样本（可能是一个样本，也可能是一个batch的样本），其对应的网络结构都是不同的，但所有的这些不同的网络结构又同时share隐含节点的权值。这样不同的样本就对应不同的模型，是bagging的一种极端情况。个人感觉这个解释稍微靠谱些，和bagging，boosting理论有点像，但又不完全相同。
> * native bayes是dropout的一个特例。Native bayes有个错误的前提，即假设各个特征之间相互独立，这样在训练样本比较少的情况下，单独对每个特征进行学习，测试时将所有的特征都相乘，且在实际应用时效果还不错。而Droput每次不是训练一个特征，而是一部分隐含层特征。
> * 还有一个比较有意思的解释是，Dropout类似于性别在生物进化中的角色，物种为了使适应不断变化的环境，性别的出现有效的阻止了过拟合，即避免环境改变时物种可能面临的灭亡。


### 2.8 Update Function 

这个没什么好说的就是BP过程


### 2.9 学习效率衰减

这个实际上是一个很土的方法，说实话我不是很理解。我的理解是在工程的角度来减少一些loss。

举个例子：假如我们有个模型，模型数据中有N个训练集合，我们定义了M个Step去学习（M << N） , 那么我们做学习效率衰减就是，在第J步的时候我们随机的丢掉f(J,M）个集合，而且随着loss率越小，我门Dropout的越多。

这样实际到了接近预期的值时，每一step就越准确，而且偏差就越来越小。就不会出现从90%突然调到50%的情况了。



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

### 4.1 TensorFlow简单运算与概念



> * 标量表示值
> * 矢量表示位置 ， 可以用一维数组表示
> * 张量表示整个空间 ， 可以用多维数组，矩阵



数据类型及过程


> * 计算图谱 ： 过程+数据
> * @Varibale 变量维护图执行过程中的状态信息.
> * @Tensor TensorFlow 程序使用 tensor 数据结构来代表所有的数据, 计算图中, 操作间传递的数据都是 tensor. 你可以把 TensorFlow tensor 看作是一个 n 维的数组或列表. 一个 tensor 包含一个静态类型 rank, 和 一个 shape.
> * @Graph 一个计算图谱
> * @Session 用来计算一个计算图谱,简单来说是个runtime



在TensorFlow中，任何计算都是会产生Tensor 对象


简单例子：


```python 

import tensorflow as tf

v1 = tf.Variable(10) # 变量
v2 = tf.Variable(5)
v3 = tf.constant(5) # 返回一个tensor

addv = v1 + v2 # 实际上Variable（或constant）在运算时，会产生Tensor.而Tensor计算是需要session runtime支持的
v1,v2,addv

```

结果 ：

```python

(<tensorflow.python.ops.variables.Variable at 0x7f35280daa50>,
 <tensorflow.python.ops.variables.Variable at 0x7f35280da910>,
 <tf.Tensor 'add_4:0' shape=() dtype=int32>)
```


例子2 ：


```python
# 创建一个session，并且初始化再进行运算
session = tf.Session()
tf.initialize_all_variables().run(session=session)
 

# 等同
print addv.eval(session=session)
print session.run(addv) 

```

结果
```python
15 15
```


所以tensorFlow是分两部分：1. 定义计算 2. 用Session计算



### 4.2 定义多层神经网络

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

网络定义完了，可以直接运算了

```python 


def get_chunk(samples, labels, chunkSize):
	'''
	Iterator/Generator: get a batch of data
	这个函数是一个迭代器/生成器，用于每一次只得到 chunkSize 这么多的数据
	用于 for loop， just like range() function
	'''
	
	if len(samples) != len(labels):
		raise Exception('Length of samples and labels must equal')
	stepStart = 0	# initial step
	i = 0
	while stepStart < len(samples):
		stepEnd = stepStart + chunkSize
		if stepEnd < len(samples):
			yield i, samples[stepStart:stepEnd], labels[stepStart:stepEnd]
			i += 1
		stepStart = stepEnd


def run(self):
	'''
	用到Session
	'''
	# private function
	def print_confusion_matrix(confusionMatrix):
		print('Confusion    Matrix:')
		for i, line in enumerate(confusionMatrix):
			print(line, line[i]/np.sum(line))
		a = 0
		for i, column in enumerate(np.transpose(confusionMatrix, (1, 0))):
			a += (column[i]/np.sum(column))*(np.sum(column)/26000)
			print(column[i]/np.sum(column),)
		print('\n',np.sum(confusionMatrix), a)


	with self.session as session:
		tf.initialize_all_variables().run()

		### 训练
		print('Start Training')
		# batch 1000
		for i, samples, labels in get_chunk(train_samples, train_labels,    chunkSize=self.batch_size):
			_, l, predictions, summary = session.run(
				[self.optimizer, self.loss, self.train_prediction, self.merged_train_summary],
				feed_dict={self.tf_train_samples: samples, self.tf_train_labels: labels}
			)
			self.writer.add_summary(summary, i)
			# labels is True Labels
			accuracy, _ = self.accuracy(predictions, labels)
			if i % 50 == 0:
				print('Minibatch loss at step %d: %f' % (i, l))
				print('Minibatch accuracy: %.1f%%' % accuracy)
			
			### 测试
		accuracies = []
		confusionMatrices = []
		for i, samples, labels in get_chunk(test_samples, test_labels, chunkSize=self.test_batch_size):
			result, summary = session.run(
				[self.test_prediction, self.merged_test_summary],
				feed_dict={self.tf_test_samples: samples}
			)
			# result = self.test_prediction.eval()
			self.writer.add_summary(summary, i)
			accuracy, cm = self.accuracy(result, labels, need_confusion_matrix=True)
			accuracies.append(accuracy)
			confusionMatrices.append(cm)
			print('Test Accuracy: %.1f%%' % accuracy)
		print(' Average  Accuracy:', np.average(accuracies))
		print('Standard Deviation:', np.std(accuracies))
		print_confusion_matrix(np.add.reduce(confusionMatrices))
			###

def accuracy(self, predictions, labels, need_confusion_matrix=False):
		'''
		计算预测的正确率与召回率
		@return: accuracy and confusionMatrix as a tuple
		'''
		_predictions = np.argmax(predictions, 1)
		_labels = np.argmax(labels, 1)
		cm = confusion_matrix(_labels, _predictions) if need_confusion_matrix else None
		# == is overloaded for numpy array
		accuracy = (100.0 * np.sum(_predictions == _labels) / predictions.shape[0])
		return accuracy, cm


```


但是这样直接定义的网络基本是没有什么准确率的。

执行后发现准确率只有20%左右


### 4.3 可视化TensorBoard


它是一个web可视化的UI工具，主要用来分析在运算过程中定义的Graph文件。

TensorBoard 涉及到的运算，通常是在训练庞大的深度神经网络中出现的复杂而又难以理解的运算。

为了更方便 TensorFlow 程序的理解、调试与优化，我们发布了一套叫做 TensorBoard 的可视化工具。你可以用 TensorBoard 来展现你的 TensorFlow 图像，绘制图像生成的定量指标图以及附加数据。


使用

#### 4.3.1 运行


在我们定义完Graph，并且Session让变量初始化后：

```python
self.session =  tf.Session(graph=self.graph)
# tensorboard 可视化
writer = tf.train.SummaryWriter('./board',self.graph)
with self.session as session:
     tf.initialize_all_variables().run()
```

运行后会发现在.board中多出一个event.out.xxx文件

执行 tensorboard -logdir ./board

这样Tensorboard的webui就运行起来了 127.0.0.1:6006


#### 4.3.2 过程打包

当我们进入web ui 会发现图形复杂且不能操作

![432](http://7xkw0v.com1.z0.glb.clouddn.com/Image13.png)

这时我们使用模组化编程，将我们的输入层、隐藏层和输出层全部打包起来

```python
def define_graph(self):
    with self.graph.as_default():
        # define some placeholder
        with tf.name_scope('input'): # 这个是模组化打包的过程，为tensorboard可视化更方便
            self.tf_train_samples = ...
            self.tf_train_labels = ...
            self.tf_test_samples = ...

            # input层到hidden层的权重和偏置
            with tf.name_scope('fc1'):
                fc1_weights = ...
                fc1_biases = ...

            # 隐藏层到输出层的权重和偏置
            with tf.name_scope('fc2'):
                ...
            
            def model(data):
                shape = data.get_shape().as_list()
                reshape = tf.reshape(data, [shape[0], shape[1] * shape[2] * shape[3]])
                with tf.name_scope('fc1_model'):
                    hidden = ...
                with tf.name_scope('fc2_model'):
                    return tf.matmul(hidden, fc2_weights) + fc2_biases

            logits = model(self.tf_train_samples)
            with tf.name_scope('loss_model'):
            self.loss = ...
            with tf.name_scope('optimizer'):
                self.optimizer = ...
           
            ....
```


可以看到加入了一些 with tf.name_scope('') , 并且一些变量附上了name的参数

![213213](http://7xkw0v.com1.z0.glb.clouddn.com/Image21.png)
![34](http://7xkw0v.com1.z0.glb.clouddn.com/Image22.png)


其他的还有一些比如

    writer.add_summary(summary,i)


可以将变量变化统计

![3213213](http://7xkw0v.com1.z0.glb.clouddn.com/Image23.png)

### 4.4 调优

前面提到，我们定义了多层神经网络，但是准确率依然很低，只有20%那么我们如何提升到90%呢？

那么我们就得通过引入一些调优函数来做这件事了


具体的调优函数已经在第二章介绍过了，下面只写出代码

#### 4.4.1 正规化Regularization

```python 

def apply_regularization(self, _lambda):
        # L2 regularization for the fully connected parameters
        regularization = 0.0
        for weights, biases in zip(self.fc_weights, self.fc_biases):
            regularization += tf.nn.l2_loss(weights) + tf.nn.l2_loss(biases)
        # 1e5
        return _lambda * regularization

     self.loss += self.apply_regularization(_lambda=5e-4)


```

在计算完loss插入

#### 4.4.2 Dropout

```python
### Dropout
    if train and i == len(self.fc_weights) - 1:
        data_flow =  tf.nn.dropout(data_flow, 0.5, seed=4926)
###

```

#### 4.4.3 Update Function 


```python

logits = model(self.tf_train_samples)
with tf.name_scope('loss'):
	self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, self.tf_train_labels))
	self.train_summaries.append(tf.scalar_summary('Loss', self.loss))

with tf.name_scope('optimizer'):
	self.optimizer = tf.train.GradientDescentOptimizer(0.0001).minimize(self.loss)
```

#### 4.4.4 学习效率衰减

```python
global_step = tf.Variable(0)
lr = 0.001
dr = 0.99
learning_rate = tf.train.exponential_decay(
    learning_rate=lr,
    global_step=global_step*self.train_batch_size,
    decay_steps=100,
    decay_rate=1,
    staircase=True
)

```


## 5. 模型参数


实际上有了上面4个优化函数就可以

如果用默认值就可以达到70%左右的正确率

最后我是设置了

> * decay_rate = 0.9
> * 迭代了3000次，每次100个
> * lambda=5e-4


可以达到90%左右的accuacy

    Average  Accuracy: 89.35
    Standard Deviation: 1.46491007863



**全文完**

附件 ： [https://pan.baidu.com/s/1jIjYAsI](https://pan.baidu.com/s/1jIjYAsI)


