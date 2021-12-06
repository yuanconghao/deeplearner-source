---
title: ImageNet Classification with Deep Convolutional Neural Networks[未完]
date: 2021-12-01 21:17:49
toc: true
tags:
    - AlexNet
    - ImageNet
    - LSVCR-2010
    - CNN
categories:
    - 论文复现
---

&emsp;&emsp;AlexNet，2012年ImageNet竞赛冠军模型；模型论文一作Alex，所以网络结构称之为AlexNet。
&emsp;&emsp;文章中模型为ImageNet LSVRC-2010，ImageNet数据集1.2 million幅高分辨率图像，共有1000个类别。测试集为Top1和Top5，错误率为37.5%和17%。
&emsp;&emsp;AlexNet网络结构先卷积，然后全连接。有60 million个参数，65 thousand个神经元，五层卷积，三层全连接网络，输出层为1000通道的softmax。利用了GPU进行计算，大大提高了运算效率。

<!--more-->




<!-- <object data="/pdf/AlexNet.pdf" type="application/pdf" width="800px" height="1000px"> -->



### Reference
[1.ImageNet Classification With Deep Convolutional Neural Networks][1]





[1]: https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf

