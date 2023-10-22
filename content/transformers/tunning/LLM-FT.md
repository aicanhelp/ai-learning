# 大模型微调（finetune）方法总结-LoRA,Adapter,Prefix-tuning，P-tuning，Prompt-tuning

1. **LoRA**  
paper：LoRA: Low-Rank Adaptation of Large Language Models（<u><a rel="nofollow noreferrer" class="external" href="https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2106.09685.pdf"><span class="invisible">https://</span><span class="visible">arxiv.org/pdf/2106.0968</span><span class="invisible">5.pdf</span><span class="ellipsis"></span></a></u>）  
code:<u><a rel="nofollow noreferrer" class="wrap external" href="https://link.zhihu.com/?target=https%3A//github.com/microsoft/LoRA">GitHub - microsoft/LoRA: Code for loralib, an implementation of &amp;quot;LoRA: Low-Rank Adaptation of Large Language Models&amp;quot;</a></u>  
**简介**  
自然语言处理目前存在一个重要范式：一般领域数据的大规模预训练，对特定任务或领域的适应（finetune）。  
但是随着预训练语言模型越来越大，这个范式存在以下问题：  
● 当我们finetune大模型时，由于训练成本太高，不太可能重新训练所有模型参数  
● 以前的方法（论文发表于2021年）都或多或少有其它性能问题，如adapter增加了模型层数，引入了额外的推理延迟；prefix-tuning比较难训练，效果不如直接finetune。  
基于上述背景，论文作者得益于前人的一些关于内在维度（intrinsic dimension）的发现：模型是过参数化的，它们有更小的内在维度，模型主要依赖于这个低的内在维度（low intrinsic dimension）去做任务适配。假设模型在任务适配过程中权重的改变量是低秩（low rank）的，由此提出低秩自适应（LoRA）方法，LoRA允许我们通过优化适应过程中密集层变化的秩分解矩阵来间接训练神经网络中的一些密集层，同时保持预先训练的权重不变。  
**方法**  
LoRA的实现思想很简单，如下图所示，就是冻结一个预训练模型的矩阵参数，并选择用A和B矩阵来替代，在下游任务时只更新A和B。  

![](https://pic1.zhimg.com/80/v2-27acf53fcfe3c3c594a4e5cbf4f8959c_720w.webp)

结合图片来看，LoRA的实现流程如下：  
● 在原始预训练语言模型（PLM）旁边增加一个旁路，做一个降维再升维的操作，来模拟所谓的内在秩。  
● 训练的时候固定PLM的参数，只训练降维矩阵A与升维矩阵B。  
● 模型的输入输出维度不变，输出时将BA与PLM的参数叠加。  
● 用随机高斯分布初始化A，用0矩阵初始化B，保证训练的开始此旁路矩阵依然是0矩阵。  
**实现**  
接下来我们从公式上解释LoRA的实现。  
假设要在下游任务微调一个预训练语言模型（如GPT3），则需要更新预训练模型参数，公式表示如下：  
W0是预训练模型初始化的参数，ΔW就是需要更新的参数。如果是全参数微调，则它的参数量=W0参数量（如果是GPT3，则ΔW≈175B）。从这可以看出要全参数微调大语言模型，小家小户是不可能的。  
由于前人的工作发现预训练的语言模型具有较低的“内部维度（intrinsic dimension）”，在任务适配过程中，即使随机投影到较小的子空间，仍然可以有效地学习。因此，LoRA做的就是增加小参数模块去学习改变量ΔW。  

![](https://pic1.zhimg.com/80/v2-d2cf686559cc3719f02d78004a2891a8_720w.webp)

在训练过程中，W0是固定不变的，只有A和B包含训练参数，是变化的。  
而在推理的过程中，只需要把改变量放回原模型，就不会有任何延迟。  
如果想切换任务，只需要切换任务的过程中，减去BA，然后换上用其它任务训练好的BʹAʹ就可以了。  
**总结**  
总的来说，**基于大模型的内在低秩特性，增加旁路矩阵来模拟full finetuning，LoRA是一个能达成lightweight finetuning的简单有效的方案**。目前该技术已经广泛应用于大模型的微调，如Alpaca，stable diffusion+LoRA，而且能和其它参数高效微调方法有效结合，例如 State-of-the-art Parameter-Efficient Fine-Tuning (PEFT)  

2. **Adapter**  
**paper：**Parameter-Efficient Transfer Learning for NLP （<u><a rel="nofollow noreferrer" class="external" href="https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1902.00751.pdf"><span class="invisible">https://</span><span class="visible">arxiv.org/pdf/1902.0075</span><span class="invisible">1.pdf</span><span class="ellipsis"></span></a></u>）  
MAD-X: An Adapter-Based Framework for Multi-Task Cross-Lingual Transfer（<u><a rel="nofollow noreferrer" class="external" href="https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2005.00052.pdf"><span class="invisible">https://</span><span class="visible">arxiv.org/pdf/2005.0005</span><span class="invisible">2.pdf</span><span class="ellipsis"></span></a></u>）  
**简介**  
2019年，Houlsby N等人将Adapter引入NLP领域，作为全模型微调的一种替代方案。Adapter主体架构下图所示。  

![](https://pic2.zhimg.com/80/v2-653ed26ec39836bfaaae12ea00c619ed_720w.webp)

在预训练模型每一层(或某些层)中添加Adapter模块(如上图左侧结构所示)，微调时冻结预训练模型主体，由Adapter模块学习特定下游任务的知识。每个Adapter模块由两个前馈子层组成，第一个前馈子层将Transformer块的输出作为输入，将原始输入维度d投影到m，通过控制m的大小来限制Adapter模块的参数量，通常情况下m<<d。在输出阶段，通过第二个前馈子层还原输入维度，将m重新投影到d，作为Adapter模块的输出(如上图右侧结构)。通过添加Adapter模块来产生一个易于扩展的下游模型，每当出现新的下游任务，通过添加Adapter模块来避免全模型微调与灾难性遗忘的问题。Adapter方法不需要微调预训练模型的全部参数，通过引入少量针对特定任务的参数，来存储有关该任务的知识，降低对模型微调的算力要求。  

**Adapter算法改进**  
2020年，Pfeiffer J等人对Adapter进行改进，**「提出AdapterFusion算法，用以实现多个Adapter模块间的最大化任务迁移」**(其模型结构如下图所示)。  

![](https://pic1.zhimg.com/80/v2-b03724439d5b674f0f11203be7c74800_720w.webp)

AdapterFusion将学习过程分为两个阶段：  
● 1.**「知识提取阶段」**：训练Adapter模块学习下游任务的特定知识，将知识封装在Adapter模块参数中。  
● 2.**「知识组合阶段」**：将预训练模型参数与特定于任务的Adapter参数固定，引入新参数学习组合多个Adapter中的知识，提高模型在目标任务中的表现。  
其中首先，对于N的不同的下游任务训练N个Adapter模块。然后使用AdapterFusion组合N个适配器中的知识，将预训练参数Θ和全部的Adapter参数Φ固定，引入新的参数Ψ，使用N个下游任务的数据集训练，让AdapterFusion学习如何组合N个适配器解决特定任务。参数Ψ在每一层中包含Key、Value和Query（上图右侧架构所示）。在Transformer每一层中将前馈网络子层的输出作为Query，Value和Key的输入是各自适配器的输出，将Query和Key做点积传入SoftMax函数中，根据上下文学习对适配器进行加权。在给定的上下文中，AdapterFusion学习经过训练的适配器的参数混合，根据给定的输入识别和激活最有用的适配器。**「作者通过将适配器的训练分为知识提取和知识组合两部分，解决了灾难性遗忘、任务间干扰和训练不稳定的问题。Adapter模块的添加也导致模型整体参数量的增加，降低了模型推理时的性能」**。  
Adapter Fusion 在 Adapter 的基础上进行优化，通过将学习过程分为两阶段来提升下游任务表现。作者对全模型微调(Full)、Adapter、AdapterFusion三种方法在各个数据集上进行和对比试验。AdapterFusion在大多数情况下性能优于全模型微调和Adapter，特别在MRPC(相似性和释义任务数据集)与RTE(识别文本蕴含数据集)中性能显著优于另外两种方法。  

3. **Prefix-tuning**  
paper：Prefix-Tuning: Optimizing Continuous Prompts for Generation（<u><a rel="nofollow noreferrer" class="external" href="https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2101.00190.pdf"><span class="invisible">https://</span><span class="visible">arxiv.org/pdf/2101.0019</span><span class="invisible">0.pdf</span><span class="ellipsis"></span></a></u>）  
code：<u><a rel="nofollow noreferrer" class="wrap external" href="https://link.zhihu.com/?target=https%3A//github.com/XiangLi1999/PrefixTuning">GitHub - XiangLi1999/PrefixTuning: Prefix-Tuning: Optimizing Continuous Prompts for Generation</a></u>  
**简介**  
前缀微调（prefix-tunning），用于生成任务的轻量微调。前缀微调将一个连续的特定于任务的向量序列添加到输入，称之为前缀，如下图中的红色块所示。与提示（prompt）不同的是，前缀完全由自由参数组成，与真正的token不对应。相比于传统的微调，前缀微调只优化了前缀。因此，我们只需要存储一个大型Transformer和已知任务特定前缀的副本，对每个额外任务产生非常小的开销。  

![](https://pic2.zhimg.com/80/v2-b59ea75db687e0cb63c8d8c4aab2889d_720w.webp)

**方法**  
本文考虑两个生成任务：table-to-text 和摘要任务。  

![](https://pic4.zhimg.com/80/v2-3e030d719f46be1acf0f65fcd9ee6227_720w.webp)

对于table-to-text任务，本文使用自回归语言模型GPT-2，输入为source（ x ）和target（ y ）的拼接，模型自回归地生成  ：  

![](https://pic2.zhimg.com/80/v2-3e86db41dae39307b52ce741d4add32d_720w.webp)

对于摘要任务，本文使用BART模型，编码器输入source文本 x ，解码器输入target黄金摘要（ y ），模型预测摘要文本  。  
**实现**  
在传统微调方法中，模型使用预训练参数进行初始化，然后用对数似然函数进行参数更新。  

![](https://pic3.zhimg.com/80/v2-2d23923223955c6a995ff00b7e4fb11e_720w.webp)

关于前缀/提示的设计，我们可以给模型若干的字词作为提示，比如我们想让模型生成“Obama”，那我们可以在其常见的搭配前加上上下文(例如，Barack)，那么LM就会把更高的可能性分配给想要的单词。但是对于很多生成任务来说，找到合适的离散的前缀进行优化是非常困难的，尽管它的效果是不错的。因此本文将指令优化为连续的单词嵌入，而不是通过离散的token进行优化，其效果将向上传播到所有Transformer激活层，并向右传播到后续的token。严格来说，这比离散提示符更具表达性，后者需要匹配嵌入的真实单词。对于自回归模型，加入前缀后的模型输入表示：  

![](https://pic2.zhimg.com/80/v2-242d792d6208c118fe74b0e56110f1ed_720w.webp)

对于编解码器结构的模型，加入前缀后的模型输入表示：  

![](https://pic3.zhimg.com/80/v2-51a07ba64c0af8e8fd8f58125813777a_720w.webp)

本文构造一个矩阵  

![](https://pic1.zhimg.com/80/v2-00a39aeb6d83d7ffe9cbcd260050c910_720w.webp)

去存储前缀参数，该前缀是自由参数。  

![](https://pic4.zhimg.com/80/v2-d7d07536d8407a368a9ee0fdb641e337_720w.webp)

目标函数依旧是公式（2），但是语言模型的参数是固定的，只更新前缀参数。  

除此之外，作者发现直接更新前缀参数会出现不稳定的情况，甚至模型表现还有轻微的下降，因此作者对前缀参数矩阵进行重参数化：  

![](https://pic3.zhimg.com/80/v2-169e4987dbc9d7d0a6e4d9130a3c308a_720w.webp)

其中  在第二维的维数要比  小，然后经过一个扩大维数的MLP，一旦训练完成，这些重参数化的参数就可以丢弃，只保留  。  

4. **P-tuning**  
paper:<u><a rel="nofollow noreferrer" class="wrap external" href="https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2103.10385">[2103.10385] GPT Understands, Too</a></u>  
code:<u><a rel="nofollow noreferrer" class="wrap external" href="https://link.zhihu.com/?target=https%3A//github.com/THUDM/P-tuning">GitHub - THUDM/P-tuning: A novel method to tune language models. Codes and datasets for paper ``GPT understands, too&amp;#39;&amp;#39;.</a></u>  
P-tuning是稍晚些的工作，主要针对NLU任务。对于BERT类双向语言模型采用模版(P1, x, P2, [MASK], P3)，对于单向语言模型采用(P1, x, P2, [MASK])：  

![](https://pic3.zhimg.com/80/v2-655bc13bfcf70f87d75a87d4c7b697d2_720w.webp)

同时加了两个改动：  

1. 考虑到预训练模型本身的embedding就比较离散了（随机初始化+梯度传回来小，最后只是小范围优化），同时prompt本身也是互相关联的，所以作者先用LSTM对prompt进行编码  
2. 在输入上加入了anchor，比如对于RTE任务，加上一个问号变成[PRE][prompt tokens][HYP]?[prompt tokens][MASK]后效果会更好  
   p-tuning的效果很好，之前的Prompt模型都是主打小样本效果，而P-tuning终于在整个数据集上超越了精调的效果：  

![](https://pic2.zhimg.com/80/v2-fc74af8a1980f233d165487d13ed49c5_720w.webp)

5. **prompt-tuning**  
Prompt-tuning给每个任务定义了自己的Prompt，拼接到数据上作为输入，同时freeze预训练模型进行训练，**在没有加额外层的情况下**，可以看到随着模型体积增大效果越来越好，最终追上了精调的效果：  

![](https://pic4.zhimg.com/80/v2-31cc195aab084411bdbc33ab1a8d8d43_720w.webp)

同时，Prompt-tuning还提出了Prompt-ensembling，也就是在一个batch里同时训练同一个任务的不同prompt，这样相当于训练了不同「模型」，比模型集成的成本小多了。
