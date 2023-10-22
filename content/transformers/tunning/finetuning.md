# 大模型参数高效微调技术原理综述（一）-背景、参数高效微调简介

## **背景**

目前，基于 Transformers 架构的大型语言模型 (LLM)，如 GPT、T5 和 BERT，已经在各种自然语言处理 (NLP) 任务中取得了 SOTA 结果。此外，还开始涉足其他领域，例如：计算机视觉 (VIT、Stable Diffusion、LayoutLM) 和音频 (Whisper、XLS-R)。

将预训练好的语言模型（LM）在下游任务上进行微调已成为处理 NLP 任务的一种范式。与使用开箱即用的预训练 LLM (例如：零样本推理) 相比，在下游数据集上微调这些预训练 LLM 会带来巨大的性能提升。

但是，随着模型变得越来越大，在消费级硬件上对模型进行全部参数的微调（full fine-tuning）变得不可行。

此外，为每个下游任务独立存储和部署微调模型变得非常昂贵，因为微调模型（调整模型的所有参数）与原始预训练模型的大小相同。

因此，近年来研究者们提出了各种各样的参数高效迁移学习方法（Parameter-efficient Transfer Learning），即固定住Pretrain Language model（PLM）的大部分参数，仅调整模型的一小部分参数来达到与全部参数的微调接近的效果（调整的可以是模型自有的参数，也可以是额外加入的一些参数）。

## **Tansformer**

上面谈到当前主流大语言模型都是基于 Transformers 架构，下面我们来详细看看 Transformers 架构内部构造。

Transformer（论文：**Attention is All You Need**）是谷歌在 2017年的提出的，它针对RNN的弱点进行重新设计，解决了RNN效率问题和传递中的缺陷等，在很多问题上都超过了RNN的表现。

Transformer 的整体结构如下图所示，图中 Transformer 用于中英文翻译。可以看到，Transformer 由 Encoder 和 Decoder 两个部分组成，Encoder 和 Decoder 都包含 6 个 block。

![](https://pic1.zhimg.com/80/v2-7e9756e7afaec9e27a09388bcc0b1be4_720w.webp)

image.png

而 Transformer 的内部结构如下图，左侧为 Encoder block，右侧为 Decoder block。 可以看到 Encoder block 包含一个 Multi-Head Attention（Multi-Head Attention是由多个 Self-Attention组成的），而 Decoder block 包含两个 Multi-Head Attention (其中，有一个用到 Masked)。Multi-Head Attention 上方还包括一个 Add & Norm 层，Add 表示残差连接 (Residual Connection) 用于防止网络退化，Norm 表示 Layer Normalization，用于对每一层的激活值进行归一化。

![](https://pic1.zhimg.com/80/v2-77411a69d32cbedf2f2587daa669a9ac_720w.webp)

image.png

而 Transformer 的核心就是自注意力（Self-Attention）。引入Self Attention后会更容易捕获句子中长距离的相互依赖的特征，因为如果是RNN或者LSTM，需要依次序序列计算，对于远距离的相互依赖的特征，要经过若干时间步步骤的信息累积才能将两者联系起来，而距离越远，有效捕获的可能性越小。但是Self Attention在计算过程中会直接将句子中任意两个单词的联系通过一个计算步骤直接联系起来，所以，远距离依赖特征之间的距离被极大缩短，有利于有效地利用这些特征。除此之外，Self Attention 对于增加计算的并行性也有直接帮助作用。

而**多头注意力的机制**则进一步细化了注意力层，多头注意力是由多个自注意力组成。它通过以下两种方式提高了注意力层的性能：

- 扩展了模型专注于不同位置的能力。当多头注意力模型和自注意力机制集合的时候，比如：我们翻译“动物没有过马路，因为它太累了”这样的句子的时候，我们想知道“它”指的是哪个词，如果能分析出来代表动物，就很有用。
- 为注意力层提供了多个“表示子空间”。对于多头注意力，我们有多组Query/Key/Value权重矩阵，这些权重矩阵集合中的每一个都是随机初始化的。然后，在训练之后，每组用于将输入Embedding投影到不同的表示子空间中。多个head学习到的Attention侧重点可能略有不同，这样给了模型更大的容量。

Self-Attention和 Multi-Head Attention 的内部结构如下所示：

![](https://pic3.zhimg.com/80/v2-4c13fc9c45f4443f5328f80aa34e6902_720w.webp)

image.png

总之，Transformer 架构的提出，奠定了其作为当前大模型领域主流的算法架构的基础。它不仅让模型能够支持更大的容量，使得模型参数能够轻松突破达到上亿规模。同时，还能够使模型较好地并行训练（Token并行、张量并行、流水线并行）。

## **Bert**

随着Transformer在2017年发布后，2018年谷歌又发布了BERT（论文：**BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**），一经面世便一举击败 11 个 NLP 任务的 State-of-the-art (Sota)结果，成为了 NLP 界新的里程碑；

Bert的结构如下图所示，左边是Bert模型预训练过程，右边是对于具体任务的微调过程。 其中，微调阶段是后续用于一些下游任务的时候进行微调，例如：文本分类，词性标注，问答系统等，BERT 无需调整结构就可以在不同的任务上进行微调。通过”预训练语言模型 + 下游任务微调”的任务设计，带来了强大的模型效果。从此，“预训练语言模型 + 下游任务微调”便成为了 NLP 领域主流训练范式。

![](https://pic4.zhimg.com/80/v2-c48c48643163dd05207ae2a0e1b82093_720w.webp)

image.png

## **全量参数微调与参数高效微调对比**

上面提到以BERT模型为代表的“预训练语言模型 + 下游任务微调”训练模式成为了自然语言处理研究和应用的新范式。此处的下游任务微调是基于模型全量参数进行微调。

但是，以 GPT3 为代表的预训练语言模型（PLM）参数规模变得越来越大，这使得在消费级硬件上进行全量微调变得不可行。

下表展示了在一张A100 GPU（80G 显存）以及 CPU 内存 64GB以上的硬件上进行模型全量微调以及参数高效微调对于 CPU/GPU 内存的消耗情况。

| 模型名                           | 全量参数微调                   | PEFT-LoRA （PyTorch）     | PEFT-LoRA（DeepSpeed+CPU Offloading技术） |
| ----------------------------- | ------------------------ | ----------------------- | ------------------------------------- |
| bigscience/T0_3B (3B 参数)      | 47.14GB GPU / 2.96GB CPU | 14.4GB GPU / 2.96GB CPU | 9.8GB GPU / 17.8GB CPU                |
| bigscience/bloomz-7b1 (7B 参数) | OOM GPU                  | 32GB GPU / 3.8GB CPU    | 18.1GB GPU / 35GB CPU                 |
| bigscience/mt0-xxl (12B 参数)   | OOM GPU                  | 56GB GPU / 3GB CPU      | 22GB GPU / 52GB CPU                   |

除此之外，模型全量微调还会损失多样性，存在灾难性遗忘的问题。

因此，如何高效的进行模型微调就成了业界研究的重点，这也为参数高效微调技术的快速发展带来了研究空间。

## **高效参数微调**

参数高效微调是指微调少量或额外的模型参数，固定大部分预训练模型（LLM）参数，从而大大降低了计算和存储成本，同时，也能实现与全量参数微调相当的性能。参数高效微调方法甚至在某些情况下比全量微调效果更好，可以更好地泛化到域外场景。

高效微调技术可以粗略分为以下三大类：增加额外参数（A）、选取一部分参数更新（S）、引入重参数化（R）。而在增加额外参数这类方法中，又主要分为类适配器（Adapter-like）方法和软提示（Soft prompts）两个小类。

![](https://pic4.zhimg.com/80/v2-eaaf1c00d0c4ea350cd3a79b47de26d3_720w.webp)

image.png

常见的参数高效微调技术有BitFit、Prefix Tuning、Prompt Tuning、P-Tuning、Adapter Tuning、LoRA等，后续文章将对一些主流的参数高效微调方法进行讲解。

# 大模型参数高效微调技术原理综述（二）-BitFit、Prefix Tuning、Prompt Tuning

## **BitFit**

### **背景**

虽然对每个任务进行全量微调非常有效，但它也会为每个预训练任务生成一个独特的大型模型，这使得很难推断微调过程中发生了什么变化，也很难部署， 特别是随着任务数量的增加，很难维护。

理想状况下，我们希望有一种满足以下条件的高效微调方法：

- 到达能够匹配全量微调的效果。
- 仅更改一小部分模型参数。
- 使数据可以通过流的方式到达，而不是同时到达，便于高效的硬件部署。
- 改变的参数在不同下游任务中是一致的。

上述的问题取决于微调过程能多大程度引导新能力的学习以及暴露在预训练LM中学到的能力。

虽然，之前的高效微调方法Adapter-Tuning、Diff-Pruning也能够部分满足上述的需求。但是，作者提出了一种参数量更小的稀疏的微调方法BitFit，来满足上述的需求。

### **技术原理**

BitFit（论文：**BitFit: Simple Parameter-efficient Fine-tuning or Transformer-based Masked Language-models**）是一种稀疏的微调方法，它训练时只更新bias的参数或者部分bias参数。

对于Transformer模型而言，冻结大部分 transformer-encoder 参数，只更新bias参数跟特定任务的分类层参数。涉及到的bias参数有attention模块中计算query,key,value跟合并多个attention结果时涉及到的bias，MLP层中的bias，Layernormalization层的bias参数。

在Bert-Base/Bert-Large这种模型里，bias参数仅占模型全部参数量的0.08%～0.09%。但是通过在Bert-Large模型上基于GLUE数据集进行了 BitFit、Adapter和Diff-Pruning的效果对比发现，BitFit在参数量远小于Adapter、Diff-Pruning的情况下，效果与Adapter、Diff-Pruning想当，甚至在某些任务上略优于Adapter、Diff-Pruning。

![](https://pic4.zhimg.com/80/v2-a314d803e2e2115491310aa0a95971af_720w.webp)

image.png

同时，通过实验结果还可以看出，BitFit微调结果相对全量参数微调而言, 只更新极少量参数的情况下，在多个数据集上都达到了不错的效果，虽不及全量参数微调，但是远超固定全部模型参数的Frozen方式。

![](https://pic4.zhimg.com/80/v2-c2911e7fd4030ee2f522f1efdbc15843_720w.webp)

image.png

同时，通过对比BitFit训练前后的参数，发现很多bias参数并没有太多变化（例如：跟计算key所涉及到的bias参数）。发现计算query和将特征维度从N放大到4N的FFN层（intermediate）的bias参数变化最为明显，只更新这两类bias参数也能达到不错的效果，反之，固定其中任何一者，模型的效果都有较大损失。

![](https://pic4.zhimg.com/80/v2-9932333ebf50803dcf8e647bbcd7e407_720w.webp)

image.png

## **Prefix Tuning**

### **背景**

在Prefix Tuning之前的工作主要是人工设计离散的模版或者自动化搜索离散的模版。对于人工设计的模版，模版的变化对模型最终的性能特别敏感，加一个词、少一个词或者变动位置都会造成比较大的变化。而对于自动化搜索模版，成本也比较高；同时，以前这种离散化的token搜索出来的结果可能并不是最优的。

除此之外，传统的微调范式利用预训练模型去对不同的下游任务进行微调，对每个任务都要保存一份微调后的模型权重，一方面微调整个模型耗时长；另一方面也会占很多存储空间。

基于上述两点，Prefix Tuning提出固定预训练LM，为LM添加可训练，任务特定的前缀，这样就可以为不同任务保存不同的前缀，微调成本也小；同时，这种Prefix实际就是连续可微的Virtual Token（Soft Prompt/Continuous Prompt），相比离散的Token，更好优化，效果更好。

![](https://pic4.zhimg.com/80/v2-f13d18b75046452ba0cd4986d7605177_720w.webp)

image.png

### **技术原理**

Prefix Tuning（论文：**Prefix-Tuning: Optimizing Continuous Prompts for Generation**），在输入token之前构造一段任务相关的virtual tokens作为Prefix，然后训练的时候只更新Prefix部分的参数，而PLM中的其他部分参数固定。

针对不同的模型结构，需要构造不同的Prefix。

- 针对自回归架构模型：在句子前面添加前缀，得到 `z = [PREFIX; x; y]`，合适的上文能够在固定 LM 的情况下去引导生成下文（比如：GPT3的上下文学习）。
- 针对编码器-解码器架构模型：Encoder和Decoder都增加了前缀，得到 `z = [PREFIX; x; PREFIX0; y]`。Encoder端增加前缀是为了引导输入部分的编码，Decoder 端增加前缀是为了引导后续token的生成。

![](https://pic2.zhimg.com/80/v2-1dda287347e7eeed655598f2df63d295_720w.webp)

image.png

该方法其实和构造Prompt类似，只是Prompt是人为构造的“显式”的提示，并且无法更新参数，而Prefix则是可以学习的“隐式”的提示。

![](https://pic3.zhimg.com/80/v2-0a7d452ba87d1d0ed666d877906119ae_720w.webp)

同时，为了防止直接更新Prefix的参数导致训练不稳定和性能下降的情况，在Prefix层前面加了MLP结构，训练完成后，只保留Prefix的参数。

![](https://pic1.zhimg.com/80/v2-0bf4ac54160cb44f5cbdfa7cb38a4c1c_720w.webp)

image.png

除此之外，通过消融实验证实，只调整embedding层的表现力不够，将导致性能显著下降，因此，在每层都加了prompt的参数，改动较大。

![](https://pic1.zhimg.com/80/v2-c24840aa6ebac63b5fcc450cd8354aa0_720w.webp)

image.png

另外，实验还对比了位置对于生成效果的影响，Prefix-tuning也是要略优于Infix-tuning的。其中，Prefix-tuning形式为 `[PREFIX; x; y]`，Infix-tuning形式为 `[x; INFIX; y]`。

![](https://pic1.zhimg.com/80/v2-f7be2bbc41070aeb0f3ed6edf727a28c_720w.webp)

image.png

## **Prompt Tuning**

### **背景**

大模型全量微调对每个任务训练一个模型，开销和部署成本都比较高。同时，离散的prompts（指人工设计prompts提示语加入到模型）方法，成本比较高，并且效果不太好。

基于此，作者提出了Prompt Tuning，通过反向传播更新参数来学习prompts，而不是人工设计prompts；同时冻结模型原始权重，只训练prompts参数，训练完以后，用同一个模型可以做多任务推理。

### **技术原理**

Prompt Tuning（论文：**The Power of Scale for Parameter-Efficient Prompt Tuning**），该方法可以看作是Prefix Tuning的简化版本，它给每个任务定义了自己的Prompt，然后拼接到数据上作为输入，但**只在输入层加入prompt tokens**，并且不需要加入 MLP 进行调整来解决难训练的问题。

![](https://pic1.zhimg.com/80/v2-d2eaf41d3da078a87ebe9e63b4c199d8_720w.webp)

image.png

通过实验发现，随着预训练模型参数量的增加，Prompt Tuning的方法会逼近全参数微调的结果。

![](https://pic2.zhimg.com/80/v2-06a7fd88bd29877341a3b6fc0bbcbb69_720w.webp)

image.png

同时，Prompt Tuning 还提出了 Prompt Ensembling，也就是在一个批次（Batch）里同时训练同一个任务的不同 prompt（即采用多种不同方式询问同一个问题），这样相当于训练了不同模型，比模型集成的成本小多了。

![](https://pic4.zhimg.com/80/v2-57f517168ec95f694ec9f5020b95b4cf_720w.webp)

image.png

除此之外，Prompt Tuning 论文中还探讨了 Prompt token 的初始化方法和长度对于模型性能的影响。通过消融实验结果发现，与随机初始化和使用样本词汇表初始化相比，Prompt Tuning采用类标签初始化模型的效果更好。不过随着模型参数规模的提升，这种gap最终会消失。

Prompt token 的长度在20左右时的表现已经不错（超过20之后，提升Prompt token长度，对模型的性能提升不明显了），同样的，这个gap也会随着模型参数规模的提升而减小（即对于超大规模模型而言，即使 Prompt token 长度很短，对性能也不会有太大的影响）。

![](https://pic3.zhimg.com/80/v2-d0e8a236f95fc534595511377775d352_720w.webp)

image.png

## **结语**

本文针对讲述了仅更新一部分参数高效微调方法BitFit以及通过增加额外参数的软提示高效微调方法Prefix Tuning、Prompt Tuning，下文将对高效微调方法P-Tuning、P-Tuning v2进行讲解。

# 大模型参数高效微调技术原理综述（三）-P-Tuning、P-Tuning v2

## **P-Tuning**

### **背景**

该方法的提出主要是为了解决这样一个问题：大模型的Prompt构造方式严重影响下游任务的效果。比如：GPT-3采用人工构造的模版来做上下文学习（in context learning），但人工设计的模版的变化特别敏感，加一个词或者少一个词，或者变动位置都会造成比较大的变化。

![](https://pic1.zhimg.com/80/v2-ee64b07b92401a452dd7f277cbddb4bc_720w.webp)

image.png

同时，近来的自动化搜索模版工作成本也比较高，以前这种离散化的token的搜索出来的结果可能并不是最优的，导致性能不稳定。

基于此，作者提出了P-Tuning，设计了一种连续可微的virtual token（同Prefix-Tuning类似）。

![](https://pic1.zhimg.com/80/v2-9edbb528db0177166667c53a5cae6970_720w.webp)

image.png

### **技术原理**

P-Tuning（论文：**GPT Understands, Too**），该方法将Prompt转换为可以学习的Embedding层，并用MLP+LSTM的方式来对Prompt Embedding进行一层处理。

![](https://pic3.zhimg.com/80/v2-4e810f340db4f48d186b5f1622dcd78e_720w.webp)

image.png

相比Prefix Tuning，P-Tuning加入的可微的virtual token，但仅限于输入层，没有在每一层都加；另外，virtual token的位置也不一定是前缀，插入的位置是可选的。这里的出发点实际是把传统人工设计模版中的真实token替换成可微的virtual token。

![](https://pic2.zhimg.com/80/v2-889f62f436d180f85f59f90e1330988d_720w.webp)

image.png

经过预训练的LM的词嵌入已经变得高度离散，如果随机初始化virtual token，容易优化到局部最优值，而这些virtual token理论是应该有相关关联的。因此，作者通过实验发现用一个prompt encoder来编码会收敛更快，效果更好。即用一个LSTM+MLP去编码这些virtual token以后，再输入到模型。

从对比实验证实看出，P-Tuning获得了与全参数一致的效果。甚至在某些任务上优于全参数微调。

![](https://pic1.zhimg.com/80/v2-295063e68ab2c96f72a61cf5d7dd600c_720w.webp)

image.png

![](https://pic3.zhimg.com/80/v2-682cf38dfd9430e584b8777533f03086_720w.webp)

image.png

并且在实验中还发现，相同参数规模，如果进行全参数微调，Bert的在NLU任务上的效果，超过GPT很多；但是在P-Tuning下，GPT可以取得超越Bert的效果。

![](https://pic2.zhimg.com/80/v2-5f7a1004db23d83c2b98a6d9a62e7f71_720w.webp)

image.png

## **P-Tuning v2**

### **背景**

之前的Prompt Tuning和P-Tuning等方法存在两个主要的问题：

第一，缺乏模型参数规模和任务通用性。

- 缺乏规模通用性：Prompt Tuning论文中表明当模型规模超过100亿个参数时，提示优化可以与全量微调相媲美。但是对于那些较小的模型（从100M到1B），提示优化和全量微调的表现有很大差异，这大大限制了提示优化的适用性。
- 缺乏任务普遍性：尽管Prompt Tuning和P-tuning在一些 NLU 基准测试中表现出优势，但提示调优对硬序列标记任务（即序列标注）的有效性尚未得到验证。

第二，缺少深度提示优化，在Prompt Tuning和P-tuning中，连续提示只被插入transformer第一层的输入embedding序列中，在接下来的transformer层中，插入连续提示的位置的embedding是由之前的transformer层计算出来的，这可能导致两个可能的优化挑战。

- 由于序列长度的限制，可调参数的数量是有限的。
- 输入embedding对模型预测只有相对间接的影响。

考虑到这些问题，作者提出了Ptuning v2，它利用深度提示优化（如：Prefix Tuning），对Prompt Tuning和P-Tuning进行改进，作为一个跨规模和NLU任务的通用解决方案。

### **技术原理**

P-Tuning v2（论文： **P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks**），该方法在每一层都加入了Prompts tokens作为输入，而不是仅仅加在输入层，这带来两个方面的好处：

- 更多可学习的参数（从P-tuning和Prompt Tuning的0.01%增加到0.1%-3%），同时也足够参数高效。
- 加入到更深层结构中的Prompt能给模型预测带来更直接的影响。

![](https://pic1.zhimg.com/80/v2-f29fb24c605951634320a0090742efa4_720w.webp)

image.png

具体做法基本同Prefix Tuning，可以看作是将文本生成的Prefix Tuning技术适配到NLU任务中，然后做了一些改进：

- **移除重参数化的编码器**。以前的方法利用重参数化功能来提高训练速度和鲁棒性（如：Prefix Tuning中的MLP、P-Tuning中的LSTM））。在 P-tuning v2 中，作者发现重参数化的改进很小，尤其是对于较小的模型，同时还会影响模型的表现。
- **针对不同任务采用不同的提示长度**。提示长度在提示优化方法的超参数搜索中起着核心作用。在实验中，我们发现不同的理解任务通常用不同的提示长度来实现其最佳性能，这与Prefix-Tuning中的发现一致，不同的文本生成任务可能有不同的最佳提示长度。
- **引入多任务学习**。先在多任务的Prompt上进行预训练，然后再适配下游任务。多任务学习对我们的方法来说是可选的，但可能是相当有帮助的。一方面，连续提示的随机惯性给优化带来了困难，这可以通过更多的训练数据或与任务相关的无监督预训练来缓解；另一方面，连续提示是跨任务和数据集的特定任务知识的完美载体。我们的实验表明，在一些困难的序列任务中，多任务学习可以作为P-tuning v2的有益补充。
- **回归传统的分类标签范式，而不是映射器**。标签词映射器（Label Word Verbalizer）一直是提示优化的核心组成部分，它将one-hot类标签变成有意义的词，以利用预训练语言模型头。尽管它在few-shot设置中具有潜在的必要性，但在全数据监督设置中，Verbalizer并不是必须的。它阻碍了提示调优在我们需要无实际意义的标签和句子嵌入的场景中的应用。因此，P-Tuning v2回归传统的CLS标签分类范式，采用随机初始化的分类头（Classification Head）应用于tokens之上，以增强通用性，可以适配到序列标注任务。

![](https://pic1.zhimg.com/80/v2-3c00778fc4a8525a39175fd61f94bae0_720w.webp)

image.png

论文中展示了P-tuning v2在不同模型规模下的表现。对于简单的NLU任务，如SST-2（单句分类），Prompt Tuning和P-Tuning在较小的规模下没有显示出明显的劣势。但是当涉及到复杂的挑战时，如：自然语言推理（RTE）和多选题回答（BoolQ），它们的性能会非常差。相反，P-Tuning v2在较小规模的所有任务中都与微调的性能相匹配。并且，P-tuning v2在RTE中的表现明显优于微调，特别是在BERT中。

![](https://pic2.zhimg.com/80/v2-8c4e0f059fc6ae825d93000f0078bf35_720w.webp)

image.png

上面讨论了P-Tuning v2无论何种规模都可以与微调相媲美。然而，GLUE和SuperGLUE的大多数任务都是相对简单的NLU问题。

为了评估P-Tuning v2在一些困难的NLU挑战中的能力，作者选择了三个典型的序列标注任务（名称实体识别、抽取式问答（QA）和语义角色标签（SRL）），共八个数据集。我们观察到P-Tuning v2在所有任务上都能与全量微调相媲美。

![](https://pic4.zhimg.com/80/v2-9f8e42d45369910f39442761e9856d87_720w.webp)

image.png

论文还通过消融实验研究了不同任务上Prompt Length的影响：

- 针对简单任务：如情感分析，较短的Prompt（~20）即可取得不错的效果。
- 针对复杂任务：如阅读理解，需要更长的Prompt（~100）。

![](https://pic1.zhimg.com/80/v2-3e2d795dd3018cab162679978d267c98_720w.webp)

image.png

总之，P-Tuning v2是一种在不同规模和任务中都可与微调相媲美的提示方法。P-Tuning v2对从330M到10B的模型显示出一致的改进，并在序列标注等困难的序列任务上以很大的幅度超过了Prompt Tuning和P-Tuning。P-Tuning v2可以成为微调的综合替代方案和未来工作的基线（Baseline）。

## **结语**

本文针对讲述了来自清华大学的团队发布的两种参数高效Prompt微调方法P-Tuning、P-Tuning v2，可以简单的将P-Tuning认为是针对Prompt Tuning的改进，P-Tuning v2认为是针对Prefix Tuning的改进。

![](https://pic1.zhimg.com/80/v2-6445ffb71fc5677af63d6c73473c5e7c_720w.webp)

image.png

# 大模型参数高效微调技术原理综述（四）-Adapter Tuning及其变体

## **Adapter Tuning**

### **背景**

随着计算机硬件性能的提高，预训练模型参数量越来越多，在训练下游任务时进行全量微调变得昂贵且耗时。

基于此，作者提出了Adapter Tuning，Adapter 的出现缓解了上述问题 Adapter 在预训练模型每层中插入用于下游任务的参数（针对每个下游任务，仅增加3.6%的参数），在微调时将模型主体冻结，仅训练特定于任务的参数，从而减少了训练时的算力开销。

### **技术原理**

Adapter Tuning（论文：**Parameter-Efficient Transfer Learning for NLP**），该方法设计了Adapter结构，并将其嵌入Transformer的结构里面，针对每一个Transformer层，增加了两个Adapter结构(分别是多头注意力的投影之后和第二个feed-forward层之后)，在训练时，固定住原来预训练模型的参数不变，只对新增的 Adapter 结构和 Layer Norm 层进行微调，从而保证了训练的高效性。

每当出现新的下游任务，通过添加Adapter模块来产生一个易于扩展的下游模型，从而避免全量微调与灾难性遗忘的问题。

![](https://pic3.zhimg.com/80/v2-c7a60e5065a325b848f48cb8031eb26e_720w.webp)

image.png

**Adapter结构具体细节**：

每个 Adapter 模块主要由两个前馈（Feedforward）子层组成，第一个前馈子层（down-project）将Transformer块的输出作为输入，将原始输入维度d（高维特征）投影到m（低维特征），通过控制m的大小来限制Adapter模块的参数量，通常情况下，m<<d。然后，中间通过一个非线形层。在输出阶段，通过第二个前馈子层（up-project）还原输入维度，将m（低维特征）重新映射回d（原来的高维特征），作为Adapter模块的输出。同时，通过一个skip connection来将Adapter的输入重新加到最终的输出中去，这样可以保证，即便 Adapter 一开始的参数初始化接近0，Adapter也由于skip connection的设置而接近于一个恒等映射，从而确保训练的有效性。

![](https://pic2.zhimg.com/80/v2-aee879a7574d6f24d528b7cd27de694d_720w.webp)

image.png

通过实验发现，只训练少量参数的Adapter方法的效果可以媲美全量微调，这也验证了Adapter是一种高效的参数训练方法，可以快速将语言模型的能力迁移到下游任务中去。同时，可以看到，Adapter 最佳的中间层特征维度m视数据集的大小而异，如：MINI数据集为256，最小的RTE数据集为8。如果始终将维度限制在64，将导致平均准确率略微下降。

![](https://pic2.zhimg.com/80/v2-9e0d951f3ef22fc92488d3423e808781_720w.webp)

image.png

总之，Adapter通过引入0.5%～5%的模型参数可以达到不落后全量微调模型1%的性能。

## **AdapterFusion**

### **背景**

为了整合来自多个任务的知识，传统的两个方法是按一定顺序微调（Sequential fine-tuning）或者多任务学习（multi-task learning）。前者的一大问题是需要先验知识来确定顺序，且模型容易遗忘之前任务学到的知识，后者的问题是不同的任务会互相影响，也难以平衡数据集大小差距很大的任务。

而之前的工作，Adapter Tuning的一个优势就是不用更新预训练模型的参数，而是插入比较少的新的参数就可以很好地学会一个任务。此时，Adapter 的参数某种程度上就表达了解决这个任务需要的知识。

作者受此启发，如果想要把来自多个任务的知识结合起来，是否可以考虑把多个任务的Adapter的参数结合起来？基于此，作者提出了 AdapterFusion，这是一种新的两阶段学习算法，可以利用来自多个任务的知识。

### **技术原理**

Adapter Fusion（论文：**AdapterFusion:Non-Destructive Task Composition for Transfer Learning**），一种融合多任务信息的Adapter的变体，在 Adapter 的基础上进行优化，通过将学习过程分为两阶段来提升下游任务表现。

- 知识提取阶段：在不同任务下引入各自的Adapter模块，用于学习特定任务的信息。
- 知识组合阶段：将预训练模型参数与特定于任务的Adapter参数固定，引入新参数（AdapterFusion）来学习组合多个Adapter中的知识，以提高模型在目标任务中的表现。

![](https://pic1.zhimg.com/80/v2-9104b71432f7243fdd2e15677306535c_720w.webp)

image.png

对于第一阶段，有两种训练方式，分别如下：

- Single-Task Adapters(ST-A)：对于N个任务，模型都分别独立进行优化，各个任务之间互不干扰，互不影响。
- Multi-Task Adapters(MT-A)：N个任务通过多任务学习的方式，进行联合优化。

对于第二阶段，为了避免通过引入特定任务参数而带来的灾难性遗忘问题，AdapterFusion提出了一个共享多任务信息的结构。针对特定任务m，AdapterFusion联合了第一阶段训练得到的N个Adapter信息。固定语言模型的参数跟N个Adapter的参数，新引入AdapterFusion的参数，目标函数也是学习针对特定任务m的AdapterFusion的参数。

**AdapterFusion结构**：

AdapterFusion具体结构就是一个Attention，它的参数包括query，key, value的矩阵参数，在transformer的每一层都存在，它的query是transformer每个子模块的输出结果，它的key跟value则是N个任务的adapter的输出。通过AdapterFusion，模型可以为不同的任务对应的adapter分配不同的权重，聚合N个任务的信息，从而为特定任务输出更合适的结果。

![](https://pic4.zhimg.com/80/v2-c2a2314600b1d805391395f4bdb335f7_720w.webp)

image.png

通过对全量微调、Adapter Tuning、AdapterFusion这三种方法在各个数据集上进行对比实验可以看出，AdapterFusion在大多数情况下性能优于全模型微调和Adapter Tuning，特别在MRPC与RTE数据集中，性能显著优于另外两种方法。

同时，还可以看到第一阶段采用ST-A+第二阶段AdapterFusion是最有效的方法，在多个数据集上的平均效果达到了最佳。而第一阶段采用MT-A+第二阶段AdapterFusion没有取得最佳的效果，在于第一阶段其实已经联合了多个任务的信息了，所以AdapterFusion的作用没有那么明显，同时MT-A这种多任务联合训练的方式需要投入较多的成本，并不算一种高效的参数更新方式。另外，ST-A的方法在多个任务上都有提升，但是MT-A的方法则不然，这也表明了MT-A虽然可以学习到一个通用的表征，但是由于不同任务的差异性，很难保证在所有任务上都取得最优的效果。

![](https://pic4.zhimg.com/80/v2-cadecd9c428752b45480ea7de79fe7c3_720w.webp)

image.png

总之，通过将适配器的训练分为知识提取和知识组合两部分，解决了灾难性遗忘、任务间干扰和训练不稳定的问题。但是，Adapter模块的添加也导致模型整体参数量的增加，降低了模型推理时的性能。

## **AdapterDrop**

### **背景**

近年来Adapter已被证明可以很好地用于机器翻译、跨语言迁移、社区问答和迁移学习的任务组合。尽管它们最近很受欢迎，但Adapter的计算效率尚未在参数效率之外得到探索。

作者通过对Adapter的计算效率进行分析，发现与全量微调相比，Adapter在训练时快60%，但是在推理时慢4%-6%。

基于此，作者提出了AdapterDrop方法缓解该问题。

### **技术原理**

AdapterDrop（论文：AdapterDrop: On the Efficiency of Adapters in Transformers），在不影响任务性能的情况下，对Adapter动态高效的移除，尽可能的减少模型的参数量，提高模型在反向传播（训练）和正向传播（推理）时的效率。

![](https://pic2.zhimg.com/80/v2-314db36574cdc556165340b905cad935_720w.webp)

image.png

实验表明，从较低的 Transformer 层中删除Adapter可以显着提高多任务设置中的推理速度。 例如，将前五个Transformer层中的Adapter丢弃，在对 8 个任务进行推理时，速度提高了 39%。并且即使有多个丢弃层，AdapterDrop 也能保持良好的结果。

![](https://pic3.zhimg.com/80/v2-29d2ca5a17f4f2701fbe9fb074e78d5e_720w.webp)

image.png

除此之外，作者还研究了对 AdapterFusion中的Adapter进行剪枝后的效果。

![](https://pic2.zhimg.com/80/v2-d6431b06b2a4be614e0155f2aad438ad_720w.webp)

通过实验表明可以移除 AdapterFusion 中的大多数Adapter而不影响任务性能。使用剩余的两个Adapter，实现了与具有八个Adapter的完整 AdapterFusion 模型相当的结果，并将推理速度提高了 68%。

![](https://pic2.zhimg.com/80/v2-bf86f888ceb53604d0d0efddb8435429_720w.webp)

因此，作者建议在实际部署这些模型之前执行 AdaperFusion 剪枝。 这是一种简单而有效的技术，即使在完全保持性能的情况下也能实现效率提升。

总之，AdapterDrop 通过从较低的 Transformer 层删除可变数量的Adaper来提升推理速度。 当对多个任务执行推理时，动态地减少了运行时的计算开销，并在很大程度上保持了任务性能。

## **结语**

本文讲述了参数高效微调方法Adapter Tuning及其变体（AdapterFusion、AdapterDrop），下文将对高效微调方法LoRA、AdaLoRA、QLoRA进行讲解。

# 大模型参数高效微调技术原理综述（五）-LoRA、AdaLoRA、QLoRA

## **LoRA**

### **背景**

神经网络包含很多全连接层，其借助于矩阵乘法得以实现，然而，很多全连接层的权重矩阵都是满秩的。当针对特定任务进行微调后，模型中权重矩阵其实具有很低的本征秩（intrinsic rank），因此，论文的作者认为权重更新的那部分参数矩阵尽管随机投影到较小的子空间，仍然可以有效的学习，可以理解为针对特定的下游任务这些权重矩阵就不要求满秩。

### **技术原理**

LoRA（论文：**LoRA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS**），该方法的核心思想就是通过低秩分解来模拟参数的改变量，从而以极小的参数量来实现大模型的间接训练。

在涉及到矩阵相乘的模块，在原始的PLM旁边增加一个新的通路，通过前后两个矩阵A,B相乘，第一个矩阵A负责降维，第二个矩阵B负责升维，中间层维度为r，从而来模拟所谓的本征秩（intrinsic rank）。

![](https://pic3.zhimg.com/80/v2-1ee7ae98a860e5f9aff51d5b1c833296_720w.webp)

image.png

可训练层维度和预训练模型层维度一致为d，先将维度d通过全连接层降维至r，再从r通过全连接层映射回d维度，其中，r<<d，r是矩阵的秩，这样矩阵计算就从d x d变为d x r + r x d，参数量减少很多。

![](https://pic3.zhimg.com/80/v2-4361dd8f8e301f76d0d5f3d41b30faea_720w.webp)

image.png

在下游任务训练时，固定模型的其他参数，只优化新增的两个矩阵的权重参数，将PLM跟新增的通路两部分的结果加起来作为最终的结果（两边通路的输入跟输出维度是一致的），即h=Wx+BAx。第一个矩阵的A的权重参数会通过高斯函数初始化，而第二个矩阵的B的权重参数则会初始化为零矩阵，这样能保证训练开始时新增的通路BA=0从而对模型结果没有影响。

![](https://pic2.zhimg.com/80/v2-43e234f226c6a965cfab4c2a8173cc75_720w.webp)

image.png

在推理时，将左右两部分的结果加到一起即可，h=Wx+BAx=(W+BA)x，所以只要将训练完成的矩阵乘积BA跟原本的权重矩阵W加到一起作为新权重参数替换原本PLM的W即可，对于推理来说，不会增加额外的计算资源。

此外，Transformer的权重矩阵包括Attention模块里用于计算query, key, value的Wq，Wk，Wv以及多头attention的Wo,以及MLP层的权重矩阵，LoRA只应用于Attention模块中的4种权重矩阵，而且通过消融实验发现同时调整 Wq 和 Wv 会产生最佳结果。

实验还发现，保证权重矩阵的种类的数量比起增加隐藏层维度r更为重要，增加r并不一定能覆盖更加有意义的子空间。

![](https://pic4.zhimg.com/80/v2-51fadc5415c03f6f4ca1e7ab908f8817_720w.webp)

image.png

![](https://pic4.zhimg.com/80/v2-adfea491ace2bbc57f07bb4d387dc647_720w.webp)

image.png

那么关于秩的选择，通常情况下，rank为4，8，16即可。

![](https://pic4.zhimg.com/80/v2-d5b64ac7875887d3143939169db7e467_720w.webp)

image.png

通过实验也发现，在众多数据集上LoRA在只训练极少量参数的前提下，最终在性能上能和全量微调匹配，甚至在某些任务上优于全量微调。

![](https://pic3.zhimg.com/80/v2-9d63eb91cc84736b230d04f673f982ea_720w.webp)

image.png

## **AdaLoRA**

### **背景**

在NLP领域，对于下游任务进行大型预训练语言模型的微调已经成为一种重要的做法。一般而言，我们会采用对原有的预训练模型进行全量微调的方法来适配下游任务，但这种方法存在两个问题。

- 训练阶段。对于预训练模型进行微调的时候，为了更新权重参数，需要大量的显存来存储参数的梯度和优化器信息，在当今预训练模型的参数变得越来越大的情况下，针对下游任务微调门槛变得越来越高。
- 推理阶段。由于我们训练的时候是对于模型参数进行全量的更新，所以多个下游任务需要为每个任务维护一个大型模型的独立副本，这样就导致我们在实际应用的时候浪费了不必要的存储。

为了解决这些问题，研究者提出了两个主要研究方向，以减少微调参数的数量，同时保持甚至提高预训练语言模型的性能。

- **方向一：添加小型网络模块**：将小型网络模块添加到PLMs中，保持基础模型保持不变的情况下仅针对每个任务微调这些模块，可以用于所有任务。这样，只需引入和更新少量任务特定的参数，就可以适配下游的任务，大大提高了预训练模型的实用性。如：Adapter tuning、Prefix tuning、Prompt Tuning等，这类方法虽然大大减少了内存消耗。但是这些方法存在一些问题，比如：Adapter tuning引入了推理延时；Prefix tuning或Prompt tuning直接优化Prefix和Prompt是非单调的，比较难收敛，并且消耗了输入的token。
- **方向二：下游任务增量更新**：对预训练权重的增量更新进行建模，而无需修改模型架构，即W=W0+△W。比如：Diff pruning、LoRA等， 此类方法可以达到与完全微调几乎相当的性能，但是也存在一些问题，比如：Diff pruning需要底层实现来加速非结构化稀疏矩阵的计算，不能直接使用现有的框架，训练过程中需要存储完整的∆W矩阵，相比于全量微调并没有降低计算成本。 LoRA则需要预先指定每个增量矩阵的本征秩 r 相同，忽略了在微调预训练模型时，权重矩阵的重要性在不同模块和层之间存在显著差异，并且只训练了Attention，没有训练FFN，事实上FFN更重要。

基于以上问题进行总结：

- 第一，我们不能预先指定矩阵的秩，需要动态更新增量矩阵的R，因为权重矩阵的重要性在不同模块和层之间存在显著差异。
- 第二，需要找到更加重要的矩阵，分配更多的参数，裁剪不重要的矩阵。找到重要的矩阵，可以提升模型效果；而裁剪不重要的矩阵，可以降低参数计算量，降低模型效果差的风险。

为了弥补这一差距，作者提出了AdaLoRA，它根据权重矩阵的重要性得分，在权重矩阵之间自适应地分配参数预算。

### **技术原理**

AdaLoRA（论文：**ADAPTIVE BUDGET ALLOCATION FOR PARAMETEREFFICIENT FINE-TUNING**），是对LoRA的一种改进，它根据重要性评分动态分配参数预算给权重矩阵。具体做法如下：

- **调整增量矩分配**。AdaLoRA将关键的增量矩阵分配高秩以捕捉更精细和任务特定的信息，而将较不重要的矩阵的秩降低，以防止过拟合并节省计算预算。
- **以奇异值分解的形式对增量更新进行参数化，并根据重要性指标裁剪掉不重要的奇异值，同时保留奇异向量**。由于对一个大矩阵进行精确SVD分解的计算消耗非常大，这种方法通过减少它们的参数预算来加速计算，同时，保留未来恢复的可能性并稳定训练。

![](https://pic2.zhimg.com/80/v2-d04c618ee963a38a249389793d1aa029_720w.webp)

- **在训练损失中添加了额外的惩罚项**，以规范奇异矩阵P和Q的正交性，从而避免SVD的大量计算并稳定训练。

通过实验证明，AdaLoRA 实现了在所有预算、所有数据集上与现有方法相比，性能更好或相当的水平。 例如，当参数预算为 0.3M 时，AdaLoRA 在RTE数据集上，比表现最佳的基线（Baseline）高 1.8%。

![](https://pic2.zhimg.com/80/v2-b15d686c210f00891a57c0dd7faa5525_720w.webp)

image.png

## **QLoRA**

### **背景**

微调大型语言模型 (LLM) 是提高其性能以及添加所需或删除不需要的行为的一种非常有效的方法。然而，微调非常大的模型非常昂贵；以 LLaMA 65B 参数模型为例，常规的 16 bit微调需要超过 780 GB 的 GPU 内存。

虽然最近的量化方法可以减少 LLM 的内存占用，但此类技术仅适用于推理场景。

基于此，作者提出了QLoRA，并首次证明了可以在不降低任何性能的情况下微调量化为 4 bit的模型。

### **技术原理**

QLoRA（论文： **QLORA: Efficient Finetuning of Quantized LLMs**），使用一种新颖的高精度技术将预训练模型量化为 4 bit，然后添加一小组可学习的低秩适配器权重，这些权重通过量化权重的反向传播梯度进行微调。QLORA 有一种低精度存储数据类型（4 bit），还有一种计算数据类型（BFloat16）。实际上，这意味着无论何时使用 QLoRA 权重张量，我们都会将张量反量化为 BFloat16，然后执行 16 位矩阵乘法。QLoRA提出了两种技术实现高保真 4 bit微调——4 bit NormalFloat(NF4) 量化和双量化。此外，还引入了分页优化器，以防止梯度检查点期间的内存峰值，从而导致内存不足的错误，这些错误在过去使得大型模型难以在单台机器上进行微调。具体说明如下：

- **4bit NormalFloat**（NF4）：对于正态分布权重而言，一种信息理论上最优的新数据类型，该数据类型对正态分布数据产生比 4 bit整数和 4bit 浮点数更好的实证结果。
- **双量化**：对第一次量化后的那些常量再进行一次量化，减少存储空间。
- **分页优化器**：使用NVIDIA统一内存特性，该特性可以在在GPU偶尔OOM的情况下，进行CPU和GPU之间自动分页到分页的传输，以实现无错误的 GPU 处理。该功能的工作方式类似于 CPU 内存和磁盘之间的常规内存分页。使用此功能为优化器状态（Optimizer）分配分页内存，然后在 GPU 内存不足时将其自动卸载到 CPU 内存，并在优化器更新步骤需要时将其加载回 GPU 内存。

![](https://pic2.zhimg.com/80/v2-8f5ee665d6de71414097733f80ec7c11_720w.webp)

image.png

实验证明，无论是使用16bit、8bit还是4bit的适配器方法，都能够复制16bit全参数微调的基准性能。这说明，尽管量化过程中会存在性能损失，但通过适配器微调，完全可以恢复这些性能。

![](https://pic3.zhimg.com/80/v2-88f8d638bfb392dc4e10c9a2341d17ca_720w.webp)

image.png

实验还比较了不同的4bit数据类型对效果（zero-shot均值）的影响，其中，NFloat 显著优于Float，而NFloat + DQ略微优于NFloat，虽然DQ对精度提升不大，但是对于内存控制效果更好。

![](https://pic2.zhimg.com/80/v2-54882cd01a46d53fb9e26869091beda5_720w.webp)

image.png

除此之外，论文中还对不同大小模型、不同数据类型、在 MMLU数据集上的微调效果进行了对比。使用QLoRA（NFloat4 + DQ）可以和Lora(BFloat16)持平，同时， 使用QLORA（ FP4）的模型效果落后于前两者一个百分点。

![](https://pic4.zhimg.com/80/v2-00fa28cf93bc27181fd919cb1c464a5f_720w.webp)

image.png

作者在实验中也发现了一些有趣的点，比如：指令调优虽然效果比较好，但只适用于指令相关的任务，在聊天机器人上效果并不佳，而聊天机器人更适合用Open Assistant数据集去进行微调。通过指令类数据集的调优更像是提升大模型的推理能力，并不是为聊天而生的。

总之，QLoRA的出现给大家带来一些新的思考，不管是微调还是部署大模型，之后都会变得更加容易。每个人都可以快速利用自己的私有数据进行微调；同时，又能轻松的部署大模型进行推理。

## **结语**

本文针对讲述了参数高效微调方法LoRA、AdaLoRA、QLoRA，下文将对混合高效微调方法MAM Adapter、UniPELT进行讲解。

# 大模型参数高效微调技术原理综述（六）-MAM Adapter、UniPELT

## **MAM Adapter**

### **背景**

近年来提出了多种参数高效的迁移学习方法，这些方法仅微调少量（额外）参数即可获得强大的性能。虽然有效，但人们对为什么有效的关键要素以及各种高效微调方法之间的联系知之甚少。

下图展示了不同的微调方法，在Xsum数据集上做英文文本摘要任务的效果（ROUGE-2是该任务的评价指标（越大越好））以及其他高效微调方法参数量相对于全参数微调参数量的百分比。图中的左上角的位置是理想化的方法。从图中发现，Adapter，Prefix Tuning和LoRA都是性能比较好的方法。

![](https://pic1.zhimg.com/80/v2-e5b9d25e820588d154e086eb10b52c18_720w.webp)

image.png

为什么看起来Adapter、Prefix Tuning、LoRA（在结构上和公式上）都不太一样，尤其是Prefix Tuning，但是这三种方法有近似的效果？

基于此，作者分解了当下最先进的参数高效迁移学习方法（Adapter、Prefix Tuning和LoRA）的设计，并提出了一种新方法MAM Adapter，一个在它们之间建立联系的统一框架。具体来说，将它们重新构建为对预训练模型中特定隐藏状态的修改，并定义一组设计维度，不同的方法沿着这些维度变化。

![](https://pic3.zhimg.com/80/v2-8602d7e8119a507082ecbe8024492f6e_720w.webp)

image.png

首先，作者通过对Prefix Tuning变换，发现Prefix Tuning和Adapters的公式高度相似。

然后，分析不同微调方法的内部结构和结构插入形式的相似之处。下图展示了高效微调方法Adapter、Prefix Tuning、LoRA以及新变体（通过更换一些元素，设计了前人的工作里没有的变体） Parallel Adapter、 Scaled PA的结构。

![](https://pic1.zhimg.com/80/v2-e73a7ad34cfbd3015218ccd820f81fa4_720w.webp)

下表展示了高效微调方法Adapter、Prefix Tuning、LoRA以及新变体在新增可训练参数结构形式（functional form）、结构插入形式（Insertion form）、新增结构在PLM修改的具体位置（modified representation）、新增结构与PLM的组合函数（composition function）。其中，新增可训练参数结构形式为需要学习的部分（注：Prefix Tuning为经过转换后的格式）；插入形式有串联或并联；模型修改的具体位置有Attention、FFN层。

![](https://pic3.zhimg.com/80/v2-87e112265dbf8d31d5517f7e9bf5a23e_720w.webp)

image.png

### **技术原理**

MAM Adapter（论文：TOWARDS A UNIFIED VIEW OF PARAMETER-EFFICIENT TRANSFER LEARNING），一个在Adapter、Prefix Tuning和LoRA之间建立联系的统一方法。

**具体实现**：

作者对Adapter的放置和软提示（soft prompt）进行了详细的调查。得出如下结论：

- 并行放置的Adapter优于顺序放置的Adapter，并且与 FFN 并行放置的Adapter优于多头注意力（MHA）并行放置的Adapter（模型修改的位置如下图中所示，蓝色表示修改Attention、红色表示修改FFN）。
- 软提示可以通过仅更改 0.1% 的参数来有效地修改注意力。

![](https://pic1.zhimg.com/80/v2-9d8b3e57eab6e4ae171cb764d18cd00c_720w.webp)

image.png

然后，提出了“mix-and-match”（MAM）。 因此，最终模型 MAM Adapter 是用 FFN 层的并行Adapter和软提示的组合。

通过最终的实验结果，可以看到 MAM Adapter 在仅用了6.7%参数量（相比全量微调）的情况下，在Xsum和MT这两个任务上达到了和全量微调相近的效果，并且该方法大大优于 BitFit 和 Prompt Tuning，并始终优于 LoRA、Adapter 和 Prefix Tuning。

![](https://pic3.zhimg.com/80/v2-14950c0ec3865af70c16db7fabdb43d2_720w.webp)

## **UniPELT**

### **背景**

近年来，涌现出了许多针对语言模型的参数高效微调（PELT）方法，在模型训练参数极大的减少的情况下，模型效果与全量微调相当。但是不同的PELT方法在同一个任务上表现差异可能都非常大，这让针对特定任务选择合适的方法非常繁琐。

基于此，作者提出了UniPELT方法，将不同的PELT方法作为子模块，并通过门控机制学习激活最适合当前数据或任务的方法。

### **技术原理**

UniPELT（论文： UNIPELT: A Unified Framework for Parameter-Efficient Language Model Tuning）是 LoRA、Prefix Tuning和Adapter的门控组合。

更具体地说，LoRA 重新参数化用于 WQ 和 WV 注意力矩阵，Prefix Tuning应用于每一Transformer层的key和value，并在Transformer块的feed-forward子层之后添加Adapter。 对于每个模块，门控被实现为线性层，通过GP参数控制Prefix-tuning方法的开关，GL控制LoRA方法的开关，GA控制Adapter方法的开关。可训练参数包括 LoRA 矩阵 WA（Down）和WB（Up），提示调优参数Pk和Pv、Adapter参数和门函数权重。即图中蓝颜色的参数为可学习的参数。

![](https://pic3.zhimg.com/80/v2-19837ee8e46ff80cedb640462effa0c6_720w.webp)

image.png

UniPELT 仅用 100 个示例就在低数据场景中展示了相对于单个 LoRA、Adapter 和 Prefix Tuning 方法的显著改进。在更高数据的场景中，UniPELT 的性能与这些方法相当或更好。

![](https://pic3.zhimg.com/80/v2-d1f7204d89425f93e88bf5ce903502fe_720w.webp)

image.png

实验还对不同 PELT 方法训练时间和推理时间进行了分析。

- 从训练速度来看，UniPELT比之前微调的方法多一些，但是还在能接受的范围，
- 从推理时间来看，BitFit方法增加的最少，UniPELT方法时间增加了27%。
- 从训练参数量来看，LoRA，BitFit，Prefix-tuning都比较小，UniPELT参数量相对会多一些。

![](https://pic4.zhimg.com/80/v2-3aec866c6dcb4bb8c48174c117358213_720w.webp)

image.png

总之，本方法始终优于常规的全量微调以及它在不同设置下包含的子模块，通常超过在每个任务中单独使用每个子模块的最佳性能的上限；并且，通过研究结果表明，多种 PELT 方法的混合涉及到PLM 的不同部分可能对模型有效性和鲁棒性都有好处。

## **结语**

本文讲述了多维混合参数高效微调方法MAM Adapter、UniPELT，接下来将对大模型参数高效微调技术原理综述的微调方法进行简单的总结。


