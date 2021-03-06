# Hedge_Ch_BLSTM_Adversarial
Keras 构建基于BLSTM+对抗学习的跨领域中文模糊限制语识别

## 摘要
模糊限制语常用来表示不确定性和可能性的含义，由模糊限制语所引导的信息为模糊限制信息，开展中文模糊限制信息检测研究，对中文事实信息抽取意义重大。中文模糊限制语识别研究尚处于起步阶段，且都是基于某个特定领域的语料学习该领域的模糊限制语识别模型。然而，模糊限制语在不同领域分布不同，所以特定领域的模糊限制语识别模型很难推广应用于其他领域。为充分利用源领域的标注数据，减少目标领域的标注代价，本文提出一种基于共享表示的跨领域模糊限制语识别方法。该方法利用双向长短期记忆网络，通过参数共享机制交替地学习源领域和目标领域的训练数据，同时引入对抗学习把各领域私有特征从共享特征中剥离，从而获得不同领域间的共享语义表示。基于共享表示对目标领域的未标注数据进行分类，实现跨领域的中文模糊限制语识别。在中文生物医学和维基百科两个领域上的实验表明，本文方法的跨领域中文模糊限制语识别性能明显优于传统基于实例和基于特征的迁移学习方法。
### 关键词
中文模糊限制语识别；跨领域；共享表示；对抗学习

## 数据处理及特征抽取
数据预处理采用Zhou等[14]的方法，即先利用词典匹配获得模糊限制语的候选词，然后抽取候选词的相关特征作为模型输入。
相关特征包括窗口大小为2的上下文特征、词性特征、以及对应的共现特征

## 实验数据及设置
  实验采用周等[15]构建的中文模糊限制语及其限制范围语料库，包含生物医学和维基百科两个领域共计24414 句已标注句子，约 100 万词。
本文仅使用其中的维基百科与生物医学领域的实验结果、摘要和讨论四部分语料（Wiki.xml、Result.xml、Abstract.xml、Discuss.xml），包含模糊限制语的个数分别是1958，1622，2759和4674。维基百科领域中，33.78%的句子包含模糊限制信息；生物医学领域中，实验结果27.8%的句子，摘要25.28%的句子和讨论47.69%的句子包含模糊限制信息。为检测维基百科和生物医学两个领域间的跨领域中文模糊限制语识别性能，共设置了六组实验，如表1所示。
  我们从万方数据库下载了6.19M的生物医学文献摘要和106M的中文维基百科语料库，加上实验所用的4.16M语料，共计117M的语料用于训练词向量。本文采用Mikolov等[16]提出的Word2vec工具训练词向量。词性向量和共现特征向量均为随机初始化，通过模型训练进行调整。词向量、词性向量和共现特征向量分别是100维、50维和10维。模型采用随机梯度下降策略进行参数更新，对抗学习的权重系数。
  
## 结论与展望
本文提出了一种基于共享表示的跨领域中文模糊限制语识别方法。通过大量的源领域训练数据和少量的目标领域训练数据（200个），利用对抗学习策略学习源领域和目标领域间的共享语义表示。在生物医学和维基百科领域的实验中，共享表示方法取得了较好的跨领域中文模糊限制语识别性能。本文仅研究了两个领域间的跨领域中文模糊限制识别，如何利用多个源领域的数据，辅助目标领域的模糊限制语识别，是本文下一步主要研究工作。
