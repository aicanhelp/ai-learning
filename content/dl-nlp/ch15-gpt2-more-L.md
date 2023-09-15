> 这是一篇以「介绍GPT-2」之名，行「推坑金庸」之实的自然语言处理文章。

嗨，会点击进来，我想你应该至少看过武侠小说泰斗[金庸](https://zh.wikipedia.org/wiki/金庸)的一部著作吧！

这篇文章将简单介绍[OpenAI](https://openai.com/)在今年提出的知名**语言模型** [GPT-2](https://openai.com/blog/better-language-models)，并展示一个能够用来生成金庸风格文本的小型GPT-2。在读完本文之后，你也能使用我的[Colab笔记本](https://colab.research.google.com/drive/1MaT8-HUHfZkdCra0OqZEIr0IFCq0MJBx)来生成属于你自己的金庸小说。文中也将透过视觉化工具[BertViz](https://github.com/jessevig/bertviz)让你能够直观地感受GPT-2等[基于Transformer架构的NLP模型](https://leemeng.tw/neural-machine-translation-with-transformer-and-tensorflow2.html)如何利用[注意力机制（Attention Mechanism）](https://www.youtube.com/watch?v=jd9DtlR90ak&feature=youtu.be)来生成文本。

<video autoplay="" loop="" muted="" playsinline="" poster="https://leemeng.tw/images/gpt2/gpt2-colab-demo.jpg" style="box-sizing: inherit; display: block; max-width: 100%; height: auto; margin: auto; width: 880px;"></video>
本文的Colab 笔记本让你可以自己生成金庸桥段并可视化结果



如果你想要直观地了解[自然语言处理（**N** atural **L** anguage **P** rocessing, NLP）](http://research.sinica.edu.tw/nlp-natural-language-processing-chinese-knowledge-information/)以及[深度学习](https://leemeng.tw/deep-learning-resources.html)可以如何被用来生成金庸小说，这篇应该很适合你。

## 前置知识

如果你已读过我写的几篇NLP 文章，我相信你可以非常轻松地理解本文提及的GPT-2 概念：

- [进入NLP 世界的最佳桥梁：写给所有人的自然语言处理与深度学习入门指南](https://leemeng.tw/shortest-path-to-the-nlp-world-a-gentle-guide-of-natural-language-processing-and-deep-learning-for-everyone.html)
- [让AI 写点金庸：如何用TensorFlow 2.0 及TensorFlow.js 写天龙八部](https://leemeng.tw/how-to-generate-interesting-text-with-tensorflow2-and-tensorflow-js.html)
- [浅谈神经机器翻译& 用Transformer 与TensorFlow 2 英翻中](https://leemeng.tw/neural-machine-translation-with-transformer-and-tensorflow2.html)
- [进击的BERT：NLP 界的巨人之力与迁移学习](https://leemeng.tw/attack_on_bert_transfer_learning_in_nlp.html)

基本上排越后面越进阶。别担心，我还是会用最平易近人的方式介绍GPT-2！但你之后可能会想要回来参考这些文章。不管如何，现在先让我们开始这趟GPT-2 之旅吧！

## 先睹为快：看看GPT-2生成的金庸桥段

首先，你可以把本文想成是[如何用TensorFlow 2及TensorFlow.js写天龙八部](https://leemeng.tw/how-to-generate-interesting-text-with-tensorflow2-and-tensorflow-js.html)的升级版本（是的，我太爱金庸所以要写第两篇文章）。本文跟该篇文章的最大差异在于：

- **模型升级**：当初我们使用轻量的[长短期记忆LSTM](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)作为语言模型，这篇则使用更新、更强大的GPT-2
- **数据增加**：当初我们只让LSTM「阅读」一本[《天龙八部》](https://bit.ly/2luD3JM)，这篇则用了整整14部金庸武侠小说来训练GPT-2
- **强调概念**：当初我们用[TensorFlow](https://www.tensorflow.org/)一步步实作LSTM，这篇则会专注在GPT-2的运作原理并额外提供[Colab笔记本](https://colab.research.google.com/drive/1MaT8-HUHfZkdCra0OqZEIr0IFCq0MJBx)供你之后自己生成文本以及视觉化结果

![img](imgs/金庸全作品-20200116133636663.jpg)用来训练GPT-2 的金庸武侠小说：第一排由左到右：飞雪连天射白鹿；第二排：笑书神侠倚碧鸳



> 飞雪连天射白鹿，笑书神侠倚碧鸳。
> ─金庸作品首字诗

从《飞狐外传》、《倚天屠龙记》、《笑傲江湖》到《鸳鸯刀》，这14 部经典的金庸著作你读过几本呢？哪一本是你的最爱呢？还记得多少桥段呢？

在实际介绍GPT-2 之前，让我们先看看将这些作品读过上百遍的GPT-2 会生成出怎么样的桥段。你可以从底下的这些生成例子感受一下GPT-2 的语言能力以及脑补技巧：

![img](https://leemeng.tw/images/gpt2/4_%E5%A4%A9%E9%BE%8D%E5%85%AB%E9%83%A8.jpg)❮❯

点击左右箭头可查看GPT-2 生成不同金庸著作的桥段



没错，很ㄎ ㄧ ㄤ ！但这些文本不是我自己吸麻瞎掰出来的。（事实上，GPT-2比我厉害多了）在用14部金庸武侠小说训练完GPT-2之后，我从这些小说中随意抽取一段文字作为**前文脉络**，接着就让它自己脑补后续桥段。你可以从左上角得知模型是在生成哪部武侠小说。

这些文本当然不完美，但跟[我们当初用LSTM生成《天龙八部》](https://leemeng.tw/how-to-generate-interesting-text-with-tensorflow2-and-tensorflow-js.html)的结果相比，已有不少进步：

- 生成的文本更加通顺、语法也显得更为自然
- 记忆能力好，能够持续生成跟前文相关的文章而不乱跳人物

值得一提的是，如果你读过《天龙八部》，就会知道第一个例子的前文脉络（context）是`段譽`与`王語嫣`坠入井内最终两情相悦的桥段。尽管每次生成的结果会因为随机抽样而有所不同，GPT-2在看了这段前文后为我们生成了一本超级放闪的言情小说，尽管放闪的两人貌似跟我们预期地不太一样。且在该平行时空底下，貌似`慕容復`也到了井里（笑

你可以跟当初LSTM 的生成结果比较，感受GPT-2 进步了多少：

![img](https://leemeng.tw/images/gpt2/gpt-vs-lstm.jpg)❮❯

点击左右箭头切换LSTM 与GPT-2 的生成桥段



虽然很有金庸架势，细看会发现LSTM的用词不太自然（比如说`四具屍體匆匆忙逼`、`伸掌在膝頭褲子`）。且明明前文脉络提到的是`段譽`与`王語嫣`，LSTM开头马上出现不相干的`南海鱷神`、接着跳到`虛竹`、然后又扯到`蕭峰`...

无庸置疑，[LSTM](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)是你在NLP路上必学的重要神经网路。不过很明显地，跟当初用词不顺且不断跳tune的LSTM相比，新的GPT-2的生成结果稳定且流畅许多（当然，训练文本及参数量的差异不小）。在看过生成结果以后，让我们看看GPT-2实际上是个怎么样的语言模型。

## GPT-2：基于Transformer的巨大语言模型

GPT-2的前身是[GPT](https://blog.openai.com/language-unsupervised/)，其全名为[**G** enerative **P** re- **T** raining](https://openai.com/blog/better-language-models/)。在[GPT-2的论文](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)里头，作者们首先从网路上爬了将近40 GB，名为[WebText（开源版）](https://skylion007.github.io/OpenWebTextCorpus/)的文本数据，并用此庞大文本训练了数个以[Transformer](https://arxiv.org/abs/1706.03762)架构为基底的[语言模型（language model ）](https://youtu.be/iWea12EAu6U)，让这些模型在读进一段文本之后，能够预测下一个字（word）。

![img](imgs/lm-equation-20200116133636682.jpg)给定前t 个在字典里的词汇，语言模型要去估计第t + 1 个词汇的机率分布 P



如今无人不知、无人不晓的神经网路架构Transformer在2017年由至今已超过3,000次引用的论文[Attention Is All You Need](https://arxiv.org/abs/1706.03762)提出，是一个不使用循环神经网路、[卷积神经网路](https://demo.leemeng.tw/)并完全仰赖[注意力机制](https://www.youtube.com/watch?v=jd9DtlR90ak&feature=youtu.be)的[Encoder-Decoder模型](https://leemeng.tw/neural-machine-translation-with-transformer-and-tensorflow2.html#Encoder-Decoder-模型-+-注意力機制)。在[前置知识一节](https://leemeng.tw/gpt2-language-model-generate-chinese-jing-yong-novels.html#前置知識)提过的[神经机器翻译& Transformer](https://leemeng.tw/neural-machine-translation-with-transformer-and-tensorflow2.html)一文里已经用了大量动画带你理解并实作自注意力机制及Transformer，这边就不再赘述了。

基本上只要了解Transformer架构，你马上就懂GPT-2了。因为该语言模型的本体事实上就是Transformer里的**Decoder**：

![img](https://leemeng.tw/images/gpt2/elmo-bert-gpt2.jpg)GPT-2与两知名模型ELMo与BERT使用的参数量比较（[图片来源](https://youtu.be/UYPa347-DdE?list=PLJV_el3uVTsOK_ZK5L0Iv_EQoL1JefRL4&t=2967)）



更精准地说，GPT-2使用的Transformer Decoder是[原始Transformer论文](https://arxiv.org/abs/1706.03762)的[小变形](https://arxiv.org/abs/1801.10198)（比方说没有了关注Encoder的Decoder-Encoder Attention Layer），但[序列生成（Sequence Generation）](https://youtu.be/f1KUUz7v8g4?list=PLJV_el3uVTsPMxPbjeX7PicgWbY7F8wW9)的概念是完全相同的。

架构本身没什么特别。但GPT-2之所以出名，是因为它训练模型时所使用的数据以及参数量都是前所未有地**庞大**：

- **训练数据**：使用从800万个网页爬来的40 GB高品质文本。把金庸14部著作全部串起来也不过50 MB。WebText的数据量是金庸著作的800倍。想像一下光是要看完这14部著作**一遍**所需花费的时间就好。
- **模型参数**：15亿参数，是已经相当巨大、拥有3.4亿参数的[BERT-Large](https://github.com/google-research/bert)语言代表模型的4.5倍之多。BERT-Large使用了24层Transformer blocks，GPT-2则使用了48层。

这可是有史以来最多参数的语言模型。而GPT-2独角兽（unicorn）的形象则是因为当初作者在[论文](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)里生成文本时，给了GPT-2一段关于「住在安地斯山脉，且会说英文的一群独角兽」作为前文脉络，而GPT-2接着生成的结果有模有样、头头是道，让许多人都惊呆了：

![img](https://leemeng.tw/images/gpt2/gpt2-unicorns.jpg)GPT-2作者用跟本文生成金庸桥段一样的方式让模型生成独角兽文章（[图片来源](https://openai.com/blog/better-language-models/)）



你可以前往在[由浅入深的深度学习资源整理](https://leemeng.tw/deep-learning-resources.html)就已经介绍过的[Talk to Transformer](https://talktotransformer.com/)生成上例的独角兽文章、复仇者联盟剧本或是任何其他类型的**英文**文章。

我懂，对非英文母语的我们来说，其实很难深切地感受GPT-2生成的文章到底有多厉害。这也是为何我决定要使用金庸小说来训练一个中文模型并介绍GPT-2，因为这样你比较能够实际感受并了解模型生成的文本。[官方释出的GPT-2能够输出中文字](https://github.com/openai/gpt-2/issues/31)，但因为大部分文本都是透过[Reddit](https://www.reddit.com/)爬下来的英文文章，因此是没有办法做到如同本文的中文生成的。

让GPT-2在社群上被热烈讨论的另个原因是作者等人[当初在部落格上展示惊人的生成结果后表示](https://openai.com/blog/better-language-models/)：

> 因为顾虑到这项技术可能会遭到恶意运用，我们目前并不打算释出已训练好的模型。但为了促进研究我们将释出规模小很多的模型供研究者参考。
> ─ OpenAI, 2019/02

此言一出，一片哗然。看看隔壁棚几乎可以说是以开源为志向的[Google BERT](https://github.com/google-research/bert)！网路上有人甚至嘲讽OpenAI一点都不open，而是CloseAI；也有人说OpenAI只是为了炒话题，GPT-2并没有那么厉害；当然也有人认为作者们的论文已经有足够的学术贡献，并非一定得释出模型。

不过至少目前看来OpenAI只是采取相对谨慎的态度在释出模型。该团队在今年2月将最小的124M GPT-2 Small（1.2亿参数）与[论文](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)一起释出，并在5月释出355M的GPT-2 Medium。而就在[不久前的8月释出了有7.74亿参数的GPT-2 Large](https://openai.com/blog/gpt-2-6-month-follow-up/)，其模型大小是1558M GPT-2完全体的一半。

![img](https://leemeng.tw/images/gpt2/gpt2-model-sizes.jpg)作者们实验的4种不同大小的GPT-2模型，774M版本在上个月被释出（[图片来源](https://jalammar.github.io/illustrated-gpt2/)）



一群人欢欣鼓舞，迫不及待地把玩最新玩具GPT-2 Large。刚刚提到的[Talk to Transformer](https://talktotransformer.com/)事实上就已经是在使用最新的GPT-2 Large了，手脚很快。

其他相关应用多如牛毛。比方说之前介绍过的[This Waifu Does Not Exist](https://www.thiswaifudoesnotexist.net/)在使用GAN生成动漫头像的同时也利用GPT-2随机生成一段动漫剧情；而[TabNine](https://tabnine.com/)则是一个加拿大团队利用GPT-2做智慧auto-complete的开发工具，志在让工程师们减少不必要的打字，甚至推荐更好的写法：

<video autoplay="" loop="" muted="" playsinline="" poster="https://leemeng.tw/images/gpt2/tabnine_demo_java_3.jpg" style="box-sizing: inherit; display: block; max-width: 100%; height: auto; margin: auto; width: 880px;"></video>
TabNine 透过GPT-2 让工程师更有效率地开发程式（以Java 为例）



由强大的深度学习模型驱动，可以想像未来（现在已经是了！）会有更多如TabNine 的应用影响我们的工作与生活。而这也是为何你最好花点时间follow 深度学习以及AI 发展趋势。当然，你也可以选择在文末订阅此部落格，只是我不敢保证文章的更新速度（笑

GPT-2公布时在多个language modeling任务取得SOTA结果，因此所有人都在引颈期盼着OpenAI将最大、拥有15亿参数的GPT-2模型释出。而该团队也表示[他们会先观察人们怎么使用774M GPT-2，并持续考虑开源的可能性](https://openai.com/blog/gpt-2-6-month-follow-up/)。

![img](https://leemeng.tw/images/gpt2/bert-gpt2-researcher.jpg)在有了BERT 之后，不少研究者开始垂涎着后来发表的GPT-2



不过别走开！GPT-2 的故事还没有结束。可别以为OpenAI 会仅仅满足于能够生成独角兽文章的一个语言模型。

重头戏现在才要开始。

## 论文作者：GPT-2能做的可不只是生成文本

要了解GPT-2，先看其前身GPT。

我们前面就已经提过，[GPT](https://blog.openai.com/language-unsupervised/)的全名是**G** enerative **P** re- **T** raining。**G** enerative（生成）指的是上节看到的language modeling，将爬来的文本喂给GPT，并要求它预测（生成）下一个字；**P** re- **T** raining则是最近NLP界非常流行的**两阶段**迁移学习的第一阶段：[无监督式学习（Unsupervised Learning）](https://www.youtube.com/watch?v=iwh5o_M4BNU)。相关概念我们在[进击的BERT：NLP界的巨人之力与迁移学习](https://leemeng.tw/attack_on_bert_transfer_learning_in_nlp.html)就已经非常详细地探讨过了，但为了帮助你理解，让我很快地再次简单说明。

近年NLP界十分流行的两阶段迁移学习会先搜集大量文本（无需任何标注数据），并以无监督的方式训练一个**通用** NLP模型，接着再微调（Fine-tune）该模型以符合**特定**任务的需求。常见的NLP任务有文章分类、自然语言推论、问答以及阅读理解等等。

![img](https://leemeng.tw/images/bert/bert-intro.jpg)Google的语言代表模型BERT则是Transformer中的Encoder （[图片来源](https://leemeng.tw/attack_on_bert_transfer_learning_in_nlp.html)）



值得一提的是，[OpenAI提出的GPT](https://openai.com/blog/language-unsupervised/)跟[Google的语言代表模型BERT](https://leemeng.tw/attack_on_bert_transfer_learning_in_nlp.html)都信奉着两阶段迁移学习：利用大量文本训练出一个**通用**、具有高度自然语言理解能力的NLP模型。有了一个这样的通用模型之后，之后就能透过简单地微调**同一个**模型来解决各式各样的NLP任务，而无需每次都为不同任务设计特定的神经网路架构，省时省力有效率。

两者的差别则在于进行无监督式训练时选用的训练目标以及使用的模型有所不同：

- GPT选择Transformer里的**Decoder**，训练目标为一般的语言模型，预测下个字
- BERT选择Transformer里的**Encoder**，训练目标则为克漏字填空以及下句预测

我们这边不会细谈，但基本上不同模型架构适合的训练目标就会有所不同。不管如何，两者都使用了[Transformer](https://leemeng.tw/neural-machine-translation-with-transformer-and-tensorflow2.html)架构的一部份。而这主要是因为Transformer里头的[自注意力机制（Self-Attention Mechanism）](https://leemeng.tw/neural-machine-translation-with-transformer-and-tensorflow2.html#Transformer：Seq2Seq-模型-+-自注意力機制)十分有效且相当适合平行运算。GPT(-2)的Transformer Decoder里头叠了很多层Decoder blocks，以下则是某一层Decoder block透过自注意力机制处理一段文字的示意图：

![img](https://leemeng.tw/images/gpt2/decoder-block-attention.jpg)训练好的Transformer Decoder在处理某词汇时能关注前方相关的其他词汇，进而为该词汇的representation融入语境资讯（[图片来源](https://jalammar.github.io/illustrated-gpt2/)）



给定一段文本：

```
<s> a robot must obey the orders given it ...
```

你可以很轻易地看出`it` [指代](http://ckip.iis.sinica.edu.tw/project/coreference/)前面出现过的`robot`。而这是因为你懂得去**关注**（pay **attention** to）前文并修正当前词汇`it`的语意。在给定相同句子时，[传统词嵌入（Word Embeddings）](https://youtu.be/kEMJRjEdNzM)方法是很难做到这件事情的。所幸，透过强大的自注意力机制，我们可以让模型学会**关注**上文以决定每个词汇所代表的语意。

以上例而言，训练好的GPT可以在看到`it`时知道该去关注前面的`a`及`robot`，并进而调整`it`在当下代表的意思（即修正该词汇的vector representation）。而被融入前文语意的新representation就是所谓的[Contextual Word Representation](https://youtu.be/S-CspeZ8FHc)。

我们可以再看一个[BERT文章](https://leemeng.tw/attack_on_bert_transfer_learning_in_nlp.html)里出现过的自注意力例子：

![img](https://leemeng.tw/images/bert/bert-coreference.jpg)BERT用自注意力机制理解句中的「他」究竟代表什么意思（[图片来源](https://leemeng.tw/attack_on_bert_transfer_learning_in_nlp.html)）



关于自注意力机制（Self-Attention），有个值得记住的重要概念：在GPT之后问世的BERT是同时关注**整个**序列来修正一个特定词汇的representation，让该词汇的repr.同时隐含**上下文**资讯；而GPT是一个由左到右的常见语言模型，会额外透过[遮罩技巧（Masking）](https://leemeng.tw/neural-machine-translation-with-transformer-and-tensorflow2.html#遮罩：Transformer-的祕密配方)来确保模型只会关注到某词汇**以左**的**上文**资讯。

![img](https://leemeng.tw/images/gpt2/self-attention-vs-masked-version.jpg)在原始的Transformer架构里头就包含了Encoder与Decoder，分别使用左侧与右侧的自注意力机制。BERT跟GPT其实只是各选一边来用（[图片来源](https://jalammar.github.io/illustrated-gpt2/)）



再次提醒，如果你想要深入了解如何实际用TensorFlow来实作遮罩，并将左侧的自注意力机制变成右侧的遮罩版本，可以参考[之前的Transformer文章](https://leemeng.tw/neural-machine-translation-with-transformer-and-tensorflow2.html)。假设你仍无法理解我在胡扯些什么的话，只要记得：

> Transformer 的自注意力机制让我们可以用更有意义、具备当下语境的方式来表达一段文字里头的每个词汇，进而提升模型的自然语言理解能力。

我在[下一节](https://leemeng.tw/gpt2-language-model-generate-chinese-jing-yong-novels.html#用-BertViz-觀察-GPT-2-生成文本)还会用些额外的例子让你能更直观地理解这个概念。

了解GPT后GPT- **2**就容易了解了，因为GPT- **2**基本上就是GPT第二代：一样是Transformer Decoder架构，但使用的数据及模型大小都直接霸气地乘上**10倍**。有了如此庞大的数据与模型，在做完第一阶段的无监督式训练以后，GPT-2的作者们决定做些疯狂的事情：不再遵循两阶段迁移学习，直接做zero-shot learning！这也就意味着直接把只看过WebText的GPT-2带上考场，「裸测」它在多个跟WebText无关的NLP任务上的表现。而实验结果如下：

![img](https://leemeng.tw/images/gpt2/gpt2-zero-shot-result.jpg)由左至右分别为阅读理解、翻译、摘要以及问答任务



乍看之下你可能会觉得这张图没什么。毕竟就算是最大、最右侧的GPT-2 模型（1542M）在多数特定的NLP 任务上还是比不过专门为那些任务设计的神经网路架构（比方说阅读理解的DrQA + PGNet）。但GPT-2 的作者们认为他们最大的贡献在于展现了用大量无标注数据训练巨大语言模型的潜力：数大就是美！除了摘要（Summarization）任务之外，基本上只要模型参数越大，zero-shot 的结果就越好。

且因为在模型参数越来越大时，训练/ 测试集的结果仍然都持续进步且表现水准相仿，作者们认为就算是最大的GPT-2 模型也还underfit 他们爬的WebText 数据集，还有进步空间。

![img](https://leemeng.tw/images/gpt2/gpt-underfit.jpg)不同大小的GPT-2在WebText上表现（[图片来源](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)）



令某些研究者兴奋的是，这实验结果隐含的一个讯息是「或许只要用够多的文本训练够大的语言模型，就能让该模型在没有监督的情况下完成更多NLP 任务」 。

总而言之，GPT-2 整篇论文的核心思想可以被这样总结：

> 给定越多参数以及越多样、越大量的文本，无监督训练一个语言模型或许就可让该模型具备更强的自然语言理解能力，并在没有任何监督的情况下开始学会解决不同类型的NLP任务。

这个概念用一个简单但合适的比喻就是「触类旁通」：GPT-2在被要求预测WebText里头各式各样文章的下一个字时，逐渐掌握理解自然语言的能力，最后就算不经过特别的训练也能做些简单的问答、翻译以及阅读理解任务。现在回头看看，你从论文标题：[Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)就能理解当初GPT-2作者们想要传达的想法了。他们也希望这些实验结果能吸引更多研究者往这个方向发展。当然，不是每个人都有能力与资源做这种庞大模型的研究，且我个人事实上比较喜欢如[DistilBERT](https://medium.com/huggingface/distilbert-8cf3380435b5)等轻量级模型的研究与应用，之后有时间再撰文分享。

好啦！基本上你现在应该已经掌握GPT-2的核心概念与思想了。如果你欲罢不能、想要了解更多，我在[最后一节](https://leemeng.tw/gpt2-language-model-generate-chinese-jing-yong-novels.html#延伸閱讀：課外參考資源)会附上更多相关连结供你参考。在下一节，让我们再回去看看金庸GPT-2。

## 用BertViz观察GPT-2生成文本

取决于超参数设定，GPT-2是一个有上千万、甚至上亿参数的语言模型，因此有时你很难理解它里头究竟在做些什么。这节我想花点时间说明如何透过[BertViz](https://github.com/jessevig/bertviz)工具来视觉化（visualize）本文的金庸GPT-2里头的自注意力机制，以加深你对GPT-2的理解。

让我们先比较一下金庸GPT-2 跟论文里4 个模型的差距：

![img](https://leemeng.tw/images/gpt2/gpt2-model-comparison.jpg)本文的金庸GPT-2与其他GPT-2论文模型的规模比较（[图片来源](https://jalammar.github.io/illustrated-gpt2/)）



是的，金庸GPT-2不管是在训练数据还是模型尺寸都比论文里最小的GPT-2 Small还来得小，因此你不该期待它表现地跟任何使用大型GPT-2模型的线上demo一样好。但因为它是读**中文**文章，非常适合我们理解，是作为教学用途的好伙伴。另外注意金庸GPT-2使用了10层Decoder blocks，而我们可以用BertViz轻易地视觉化每一层的自注意力机制。

比方说给定一个在金庸原著里不存在，但我超级想要实现的《天龙八部》剧情：

> 乔峰带阿朱回到北方，乔峰对她说：「我们两人永远留在这里！」

透过我为你准备好的[Colab笔记本及预先训练好的金庸GPT-2](https://colab.research.google.com/drive/1MaT8-HUHfZkdCra0OqZEIr0IFCq0MJBx)，你只需几行Python程式码就能视觉化每一层Decoder block处理这段文本的结果：

```python
from bertviz.pytorch_transformers_attn import GPT2Model
gpt2_model = GPT2Model.from_pretrained('.')

text = '喬峯帶阿朱回到北方，喬峯對她說：「我們兩人永遠留在這裡！」'
view = 'model'
show(gpt2_model, tokenizer, text, view)
```

<video autoplay="" loop="" muted="" playsinline="" poster="https://leemeng.tw/images/gpt2/gpt-bertviz-model-view.jpg" style="box-sizing: inherit; display: block; max-width: 100%; height: auto; margin: auto; width: 880px;"></video>
BertViz 的model view 让你轻松「鸟瞰」整个模型。这里只显示第6 - 9 层blocks（zero-index）



我们在这边不会细谈，但你会发现**上下**每一层Decoder block **左右**各有12个heads。这就是[之前介绍过的Multi-head Attention机制](https://leemeng.tw/neural-machine-translation-with-transformer-and-tensorflow2.html#Multi-head-attention：你看你的，我看我的)，目的是让模型能够给予每个词汇多个不同的representations并在不同representation spaces里关注不同位置，增加表达能力。

这些图的解读方式是当GPT-2看**左侧**特定词汇时，关注**右侧**同序列中出现在该词汇**之前**（包含自己）的其他词汇。关注的程度则透过线条**粗**细来表示。

而如果我们将GPT-2 生成这段文本的自注意力机制的变化依照词汇的生成顺序显示出来的话，会看起来像这样：

```python
# BertViz 的 neuron view 可以看到 key, value 的匹配
view = 'neuron'
show(gpt2_model, tokenizer, text, view)
```

<video autoplay="" loop="" muted="" playsinline="" poster="https://leemeng.tw/images/gpt2/gpt-seq-generation.jpg" style="box-sizing: inherit; display: block; max-width: 100%; height: auto; margin: auto; width: 880px; mix-blend-mode: initial;"></video>
GPT-2 在生成新词汇时会持续透过自注意力机制关注前文



这边的重点是前面讲过的Masked Self-Attention：一个传统、**单向**的语言模型在处理新词汇时只会、也只能**关注**前面已经被生成的其他词汇，而不该去看「未来」的词汇。

不知道你有没有注意到，在上面的例子中，GPT-2 在处理词汇「她」时会去关注前面靠近人名的「阿」，这是一件很了不起的事情：

![img](https://leemeng.tw/images/gpt2/gpt2-bertviz-attention-example.jpg)GPT-2 透过自注意力机制建立具有语境的word representations



如同我们在[前面章节](https://leemeng.tw/gpt2-language-model-generate-chinese-jing-yong-novels.html#論文作者：GPT-2-能做的可不只是生成文本)提过的，中文字「她」与「这」本身并没有太多含义，只有在了解情境之后才能判别它们所代表的意义。而这是具有自注意力机制的NLP模型可以想办法学会的。

我们也可以看看不同层的Decoder blocks 在关注同样的文本时有什么变化：

```python
# 實際上只會產生一個可以選擇不同層的 UI，我隨意選了 3 層的截圖結果
text = '喬峯帶阿朱回到北方，喬峯對她說：「我們兩人永遠留在這裡！」'
view = 'head'
show(gpt2_model, tokenizer, text, view)
```

![img](https://leemeng.tw/images/gpt2/layerwise-self-attention.jpg)不同层的Decoder blocks 关注相同文本的结果



你可以明显地观察到底层（第一层）的Decoder block 在处理词汇「她」时将注意力放到多个词汇上以撷取整个前文资讯。而越上层的block 的注意力越显集中，到了最后一层的Decoder block 时就相当专注在较远的人名附近。

透过这些视觉化结果，你现在应该对GPT-2 的运作模式有着更直观的理解了。

## 故事尾声

呼！我希望你享受这趟跟我一起探索GPT-2 以及金庸著作的旅程。

就我所知，这应该是网路上第一篇以中文GPT-2 为背景，用最白话的方式讲解相关概念的中文文章了。我在撰写本文时尝试用初学者最容易理解的章节编排方式、省略不必要的艰涩词汇并避免丢上一长串程式码。如果你读完觉得本文内容很简单，GPT-2 也不过就这样，或者迫不及待想要知道更多细节，那就达成我撰写本文的目标了。

[![img](https://leemeng.tw/images/gpt2/previous-nlp-posts.jpg)](https://leemeng.tw/gpt2-language-model-generate-chinese-jing-yong-novels.html#前置知識)跟本篇相关的NLP 文章



本文为了顾及更多刚刚入门NLP的读者省略了不少技术细节，如长距离依赖（Long-range Dependencies）的探讨、自注意力机制的实作细节及研究GPT-2是真的学习还是只是记忆等课题。想要深入研究相关概念的读者，[下节的延伸阅读](https://leemeng.tw/gpt2-language-model-generate-chinese-jing-yong-novels.html#延伸閱讀：課外參考資源)可供你做参考。而如果你想要打好NLP基础以及本文提到的相关知识，我会推荐你回到前面的[前置知识](https://leemeng.tw/gpt2-language-model-generate-chinese-jing-yong-novels.html#前置知識)一节，选择最合你胃口的NLP文章开始阅读。

正如[统计学家乔治·E·P·博克斯](https://en.wikipedia.org/wiki/George_E._P._Box)所说的：

> 所有模型都是错的；但有些是有用的。
> ─ George EP Box

等到哪天有更好的语言模型可以拿来生成金庸武侠小说，你就会再次看到我撰写相关文章了。

但今天就到这里啦！我现在还得烦恼该从哪部金庸小说开始复习呢...

## 延伸阅读：课外参考资源

我在文中讲述GPT-2 时已经附上相当多连结，如果你很好学，事实上应该已经花了不少时间才读到这里。这节系统性地列出相关连结供你做延伸阅读。

- 教学文章、课程影片
  - [李宏毅教授2019 机器学习课程的ELMO, BERT, GPT 影片](https://youtu.be/UYPa347-DdE?list=PLJV_el3uVTsOK_ZK5L0Iv_EQoL1JefRL4)
  - [The Illustrated GPT-2 (Visualizing Transformer Language Models)](https://jalammar.github.io/illustrated-gpt2/)
  - [Better Language Models and Their Implications](https://openai.com/blog/better-language-models/)
  - [CS224n: Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/)
- 论文
  - [Generating Wikipedia by Summarizing Long Sequences](https://arxiv.org/abs/1801.10198)
  - [Improving language understanding by generative pre-training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
  - [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
  - [Visualizing Attention in Transformer-Based Language Representation Models](https://arxiv.org/abs/1904.02679)
- 实作
  - [pytorch-transformers](https://github.com/huggingface/pytorch-transformers)
  - [GPT2-Chinese](https://github.com/Morizeyao/GPT2-Chinese)