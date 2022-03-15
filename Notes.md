### Lecture 3
* One-layer Neural Network = 1线性层LL + 1非线形层SoftMax Layer
* Page 13, 14: SoftMax Layer计算
* Page 27, 31, 32, 38, 39, 44: 权重W = 权重W + lr(误差向量Error Vector ⊗ 输入图像Input Picture)

### Lecture 4
* Page 8: Qmean计算
    * Qmean = Neural Network Quality，即神经网络的质量，0 <= Qmean <= 1，质量越高越好 => Qmean越接近1越好
* Page 29, 32: -logQmean = 交叉熵损失函数Cross-entropy Loss，0 <= -logQmean <= ∞，损失越小越好，=> -logQmean越接近0越好
* Page 37, 44, 45: 使用交叉熵损失函数时权重更新变为减号

### Lecture 5
* Page 8: 损失值是Batch中每个图片的平均损失
* Page 21: 反向传播Backpropagation
    * [神经网络多分类中softmax+cross-entropy的前向传播和反向传播过程](https://zhuanlan.zhihu.com/p/86184547)

### Lecture 6
* Page 6: 权重符号表示方式，权重矩阵中每一行代表输出层的一个神经元上的所有权重，每一行的权重个数(权重矩阵的列的数量)即为输入层的神经元的个数
* Page 15: 梯度消失Vanishing Gradient

### Lecture 7
* [如何理解卷积神经网络（CNN）中的卷积和池化？](https://www.zhihu.com/question/49376084?sort=created)
* [卷积神经网络中感受野的详细介绍](https://blog.csdn.net/program_developer/article/details/80958716)
* 一个Kernel是2D的
* 一个过滤器Filter可能是是高维的(>= 2D)
* 一个过滤器Filter = n个卷积核Kernel堆叠（多通道图片，一个Kernel代表一个通道）
* 一个过滤器Filter卷积扫描一个输入图像 = 一个Activation Map
* 一个Feature Map = N个Activation Map的值相加
* 进一步解释
    * Feature Map其实是一个理论意义上的图片，可能是512 * 512 * 3，可能是256 * 256 * 10，就是长，宽，通道数的意思，就是每次经过卷积后的一个结果
    * 每次卷积前会把这个Feature Map(这个图)输入，然后用过滤器Filter扫描，每个过滤器Filter扫描这个整个个Feature Map(这个图)就会得到一个2D的结果，因此这个过滤器Filter扫描的过程是扫描全部的通道然后再矩阵相加，所以最后成了2D的256 * 256 * 1的东西（即每次扫描后都会降到2D）
    * 有n个过滤器Filter，例如10个，每个过滤器Filter扫描后的结果堆叠一起，就成了256 * 256 * 10
* Page 15: 感受野Receptive Field
    * [深度神经网络中的感受野(Receptive Field)](https://zhuanlan.zhihu.com/p/28492837)
    * [卷积神经网络中感受野的详细介绍](https://blog.csdn.net/program_developer/article/details/80958716)
* Page 19: 一层卷积网络One-layer ConvNet = 1卷积层CL + 1池化层MP + 损失函数

### Lecture 8
* Page 10: 输入尺寸Input Size(n)，过滤器尺寸Filter Size(f)，填充Padding(p)，步长Stride(s)
    * 卷积：输出尺寸 = ⌊(输入尺寸n + 2 * 填充p - 过滤器尺寸f) / 步长s + 1⌋
    * 池化：输入尺寸 = ⌈(输入尺寸n - 过滤器尺寸f) / 步长s + 1⌉
    * [卷积:kernel size/padding/stride](https://blog.csdn.net/weixin_42490152/article/details/100160864)
* Page 38, 46: 神经网络参数数量计算
    * [Convolutional Neural Networks](https://www.cs.toronto.edu/~lczhang/360/lec/w04/convnet.html)

### Lecture 11
* [NMS——非极大值抑制](https://blog.csdn.net/shuzfan/article/details/52711706)

### Lecture 12
* [图像处理之双线性插值法](https://blog.csdn.net/qq_37577735/article/details/80041586)
