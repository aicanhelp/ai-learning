# 大模型加速



一、背景

随着chatGPT的火爆，如何训练大模型成为一个重要问题，训练大模型需要考虑

1、计算耗时：多个GPU如何进行模型计算加速训练：

2、通信耗时：模型并行多GPU通信耗时： 通信量 通信宽带 通信延时

3、memory问题：大数据大模型参数量多

![](https://pic1.zhimg.com/80/v2-5e326dd4e6e7bdc420ce39645d1648ac_720w.webp)

模型参数量&显存发展

![](https://pic4.zhimg.com/80/v2-36464a2b76e2805e454e50ecd794499f_720w.webp)

H100 对比A100比较 GPU CPU对比图

如图GPU和CPU架构，GPU适合高并行、计算密度大的 其将更多的晶体管用于计算，而不是缓存和流控

## **二、解决方案**

如上所说，模型的加速要从计算量、通信、内存三个角度综合分析考虑。下面从所需资源，不同角度分别如何进行加速分析。

### 1、计算加速

### 2.1.1 计算所需资源分析

模型的算子类型主要分为两类：

a) 计算密集型：GEMM batchGemm

估计方法：tensor core峰值使用效率。计算密度类【Airthmetic Intensity(AR)】直接影响GEMM计算效率，计算密度越高越容易发挥峰值计算能力。可以通过构建GEMM效率表，对于给定的问题规模使用插值计算的方式计算tensor core的峰值计算效率。

![](https://pic4.zhimg.com/80/v2-c18f386f454056059f544987f427e697_720w.webp)

AR估计方法

对于 A、B、C、D四个矩阵做如下运算，其中下标为矩阵维度：

![](https://pic4.zhimg.com/80/v2-3246523e40b95f5013d747f5a46b6fab_720w.webp)

完成上述运算的所需的总加乘次数

![](https://pic3.zhimg.com/80/v2-07ad54dda71995072fbbcf32381f5f96_720w.webp)

b) 访存密集型：softmax、layernomal、GELU、Transpose等

估计方法：显存峰值宽带使用效率

AR计算效率说明：

![](https://pic1.zhimg.com/80/v2-ac5e068cae44832a7a41ea546f0c6fb8_720w.webp)

AR计算效率

MN/(M+N)和K越大越能实现峰值的计算效率

1)构建GEMM峰值效率表，使用插值的方法获得对应GEMM的峰值效率

2)进一步推测模型计算的时间大小

3)通过不同硬件配置、不同模型、不同数据规模大小分析训练所需计算资源；为模型设计,硬件选型提供指导；未来模型在未来硬件上的表现

### 2.1.2 3D并行

### 1）什么是3D并行：

数据并行实现了良好的计算效率，但它复制了模型状态，无法利用聚合分布式内存。

![](https://pic4.zhimg.com/80/v2-c10d30a6654f75c6d5313119275e1673_720w.webp)

3D并行示例图

数据并行流程：

数据并行对于数据并行的分布式深度学习训练来说,其具体训练流程为:

1）每个节点分别从硬盘或网络上读取总共miniＧbatch 大小的数据并拷贝至内存; 【IO需要通信】

2）从CPU 内存拷贝数据至GPU 内存;【需要通信：将数据通过PCIＧe传输到GPU 中】

3）加载GPUkernel并由前向后逐层计算(前向传播过程);

4）计算损失函数(loss)并进行反向传播,逐层计算梯度值;

5）同步各节点梯度值(发送自身各层梯度,并接收其他节点的各层梯度);【需要通信：取决于数据和模型的大小】

6）根据同步后的梯度值更新神经网络参数

### 2）梯度同步的方式：

分布式系统的可以支持多种同步模式，支持异步的ASP、同步的BSP、半同步的SSP方法

![](https://pic3.zhimg.com/80/v2-e9bcbf05728abdcbdf4094adb07a650a_720w.webp)

异步的ASP

每一个worker的iteration计算完成之后就updata，进行下一轮的计算，完全不顾worker之间的顺序 缺点：大部分模型woker无法收敛

缺点：大部分模型woker无法收敛

![](https://pic3.zhimg.com/80/v2-9d7ae25abdc1fe38445df12c44c4b782_720w.webp)

同步的BSP

BSP的同步模式（spark采用这种模式）：

如图所示，只有所有的worker完成iteration的计算之后，才会进行一次woker和server的同步更新操作缺点：速度取决于最慢的那个worker

![](https://pic4.zhimg.com/80/v2-94de4b0dc727ee3425002cafdd10f413_720w.webp)

半同步的SSP方法

SSP：将上述两种方法进行折中，ASP不同worker之间iteration任意大，BSP不同worker之间iteration隔数任意大，如图所示允许最快和最慢之间的iteration设置为3【S不是无穷大，经过一定迭代之后一定可以收敛】

### 3）pipline 并行算法

![](https://pic2.zhimg.com/80/v2-bdc5df142148bdf0e5547f8de1a96df9_720w.webp)

pipline并行方法

如何让pipeline上的时间尽可能的少：PipeDream核心在于解决两个问题：

(1) 对于一个给定的模型与分布式系统，如何划分任务（即哪个节点负责哪些layer，某些layer是数据并行还是模型并行）

(2) 对于流水线模型，如何避免流水线本身带来的训练的问题。使用动态规划的思想

参考论文：

PipeDream: Fast and Efficient Pipeline Parallel DNN Training

pipDream-2BW:Memort-Efficent Pipeline-Parallel DNN Traning

![](https://pic2.zhimg.com/80/v2-48e362e597dee0260c554018cadcc119_720w.webp)

pipline并行流程

Pipeline的训练模式会引入两种类型的参数不一致：

（1）同一个minibatch的前向和后向计算使用的参数不一致。

（2）同一个minibatch在不同worker上使用的参数版本不一致。

为解决这两个问题，PipeDream 提出如下两种技术：

（1）weight stashing: 为每个active minibatch都保存一份参数。前向计算时，每个stage 都是用最新的参数处理输入的minibatch，然后将这份参数保存下来用于同一个minibatch的后向计算。

（2）Vertical Sync: 每个minibatch进入pipeline时都使用输入stage最新版本的参数，并且参数的版本号会伴随该minibatch数据整个生命周期，在各个阶段都是用同一个版本的参数（而不是前文所说的都使用最新版本的参数），从而实现了stage间的参数一致性。

### 4）3D并行优缺点分析:

![](https://pic1.zhimg.com/80/v2-9a1106a61a54e69deb0811dc00230ea0_720w.webp)

3D并行优缺点分析

### 2.1.3 混合精度训练

![](https://pic1.zhimg.com/80/v2-5da64e8b0bf0f42464111a3cd63a61fc_720w.webp)

混合精度训练

**为什么要进行混合精度：**

1)减少显存计算

2)减少推断和训练时间

3) Nvidi的Tensor Corr的普及，随着其不断普及

**混合精度带来的问题：**

1)溢出问题 上溢： 下溢问题：激活函数比较小，更容易出现下溢

2)舍入错误，其固定间隔是2的（-13）次方，所以精度小于这个区间的会存在舍入误差

**解决以上问题的方法（如何进行混合精度训练）**:采用混合的精度方法和动态损失方法（loss Scaling）

1)混合精度训练的精髓在于“在内存中用FP16做储存和乘法从而加速计算，用FP32做累加避免舍入误差”。混合精度训练的策略有效地缓解了舍入误差的问题。

2)动态损失方法：对于激活函数值太小存在的下溢问题，我们使用动态损失的方法，反向传播前,将损失变化（dLoss）手动增大2的k次方倍，因此反向传播时得到的中间变量（激活函数梯度）则不会溢出；反向传播后，将权重梯度缩小2的k（默认32）次方倍，恢复正常值。

**如何使用混合精度**:

![](https://pic4.zhimg.com/80/v2-b4335809aecde248cd891a6f3050e757_720w.webp)

使用apex使用混合精度

### 2.1.4梯度累积

**为什么要进行梯度累积：**

目的是为了解决由于内存不足导致某些大型网络无法训练大Batch_size的问题。

**如何进行梯度累积：**

首先对每个mini-batch进行梯度和loss的计算，但是不进行梯度的更新，累加N个mini-bacth之后,用累积的梯度之后进行梯度的更新，最终达到和N*mini-batch一样的效果

### 2、通信加速

### 2.2.1通信架构

![](https://pic1.zhimg.com/80/v2-421c429ce447d38baceb0e902ff2173c_720w.webp)

多卡训练常用算子

![](https://pic4.zhimg.com/80/v2-f052e91fc60c2242a06dc2154e85a013_720w.webp)

ring-all reduce架构示意图

### 2.2.2 模型通信需求量分析

从数据并行、模型并行、管道并行使用不同的方式进行通信量的统计

### 2.2.3 模型通信优化方向

![](https://pic1.zhimg.com/80/v2-3f2a0dac582caf53e98390fd79e9c3ac_720w.webp)

模型通信优化方向

张量融合技术：提出解决同步通信时传输量不均等问题：将通信的包拆分、合并，从而形成相同大小的数据包Horovod中实现张量融合的步骤为:

１)判定哪些张量准备开始做Reduce操作,选择最初几个可以适配配置标准尺寸的且同数据类型的张量;

２)为标准尺寸通信单元分配内存;

３)将选定的张量拷贝至标准尺寸通信单元；【性能损耗 拷贝工作】

４)对标准尺寸通信单元执行All-Reduce;

５)将同步后的数据从标准尺寸单元拷贝至输出张量; 【性能损耗，拷贝工作】６)重复以上过程直到本轮没有多余的张量需要传输．

### 2.2.4 量化压缩方案（adam的压缩）：

![](https://pic1.zhimg.com/80/v2-a53d0f907f04052f565a7fe975aec52c_720w.webp)

1-bit adam压缩流程

![](https://pic1.zhimg.com/80/v2-57fe645e835a162b4013fe094326c654_720w.webp)

1-bit adam压缩两阶段流程

![](https://pic1.zhimg.com/80/v2-ecc04e5ba3939532fe0b4a5d7ac841c0_720w.webp)

adam计算过程

使用常数取代vt，vt变化趋势稳定之后切换到第二阶段，并且取代后是否可以达到和原Adam算法一样的收敛速度呢？

只需要10%-15%的训练步数作为Warmup，然后使用1-bit压缩，压缩Momentum，此时达到了32倍的压缩率。总的来说，我们获得了5倍的压缩率。尽管在这么高的压缩率的情况下，无论在哪种测试方法下，我们都可以确保1-bit Adam和原来的Adam取得了一样的收敛效率。

参考文章：1-bit Adam: Communication Efficient Large-Scale Training with Adam’s Convergence Speed

### 3、内存优化

### 2.3.1模型内存分析

我们以GPT-3（模型参数量175B）模型为例进行分析，使用gpt模型，其含有

transform layer96层

语句长度2048 隐层维度：12288

词表规模：51200

使用fp32存储，其需求内存是2.8TB，我们使用数据并行和张量并行进行模型训练，假设模型对应参数配置如下图所示：

![](https://pic4.zhimg.com/80/v2-09a81b022fdb1194d013dee94e789377_720w.webp)

模型对应参数

使用如上数据并行、tensor并行参数，分析每个GPU上模型参数量的大小

![](https://pic2.zhimg.com/80/v2-c187d76b2d9808725ecee1b562fc82dd_720w.webp)

每张卡上的参数量

所以每张卡模型的存储开销大小为

![](https://pic4.zhimg.com/80/v2-2e93ed9db1812e86263fd18c581790bb_720w.webp)

模型所需内存开销

其中优化器的存储状态主要是：fp32类型的参数拷贝、fp32类型的梯度拷贝、fp32类型的动量、fp32类型的方差

其他开销包括激活开销、中间变量存储开销、存储需求取决于三个极大值点的最大值：

![](https://pic4.zhimg.com/80/v2-b531fd3ed2b9fd204326adf8f9a1b447_720w.webp)

模型开销的三个极大值点

模型开销的三个极大值点：

第一次反向传播开始之前（前向完成）

反向传播完成之后（反向完成）

完成第一次完整的iteration（all_reduce完成）

### 2.3.2 内存优化 传输换空间

![](https://pic2.zhimg.com/80/v2-98ae06daa8c5394863d62d7084d0e1c5_720w.webp)

cpu offload

为了解决数据并行存在的内存问题，ZeRO 提供了三阶段的优化方法：

优化器分割状态：ZeRO 降低了 3/4 的内存，通信量和数据并行相同；

加入梯度分割：降低了 7/8 的内存，通信量和数据并行相同；

加入参数分割：内存减少与数据并行度呈线性关系。例如，在 64 个 GPU 上进行分割的时候，可以将内存降至 1/64。

![](https://pic2.zhimg.com/80/v2-2abf958e17e463a1187143f443134d41_720w.webp)

CPU offload ZeRO-Offload: Democratizing Billion-Scale Model Training

上图使用zero的不同阶段理论上可以使用的模型参数量。

### 2.3.3 内存优化 计算换空间

![](https://pic1.zhimg.com/80/v2-3cc15042cd491c10e68446d0b4b3d290_720w.webp)

计算换空间

将一个层网络平均分为d分区。只保存分区边界处的激活, 并在 workers之间进行通信, 计算梯度仍然需要分区内层的中间激活, 因此在向后传递过程中重新计算梯度，使用激活重新计算。训练的内存成本为

![](https://pic4.zhimg.com/80/v2-1b5aad18af4a2556f3aed7db7d82bfeb_720w.webp)
