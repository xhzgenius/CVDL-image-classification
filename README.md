# CVDL image classification

 CVDL课程平时作业1：自然景观图像分类

## 一、作业任务

课程作业采用在线竞赛平台Kaggle，请通过以下链接访问自然景观图像分类的竞赛：
https://www.kaggle.com/datasets/puneet6060/intel-image-classification

本任务需要大家预测6种自然景观的种类。数据集共有25k张图像。要求同学们建立模型，在给定的测试集上进行预测。

## 二、实现要求

针对给定的分类数据集，每位同学需要分别实现基于传统算法和深度学习方法的分类模型：

传统算法：对比不同的特征提取 + 分类器组合
特征提取：SIFT、HOG等
分类器：SVM、kernel SVM、k-means clustering等

结果：

* Accuracy of SIFT & SVM: 0.5571285571285571
* Accuracy of SIFT & Kernel-SVM: 0.7277992277992278
* Accuracy of SIFT & k-means: 0.33697983697983697
* Accuracy of HOG & SVM: 0.7166452166452166
* Accuracy of HOG & Kernel-SVM: 0.8787358787358788
* Accuracy of HOG & k-means: 0.389031889031889

深度学习方法：对比网络架构、优化器、数据增强和预处理、正则化等方面不同的设定
网络架构：MLP、VGG、ResNet等
优化器：SGD、AdaGrad、Adam等，可以尝试不同的学习率
数据增强和预处理：随机翻转、随机裁剪、标准化等
正则化：weight decay，dropout等

红色部分是必须完成的对比项目，有兴趣的同学也可以探索其他设定（如损失函数、BN）对模型性能的影响。以上内容可以调取
已有的算法包，不必从头实现。

## 三、提交要求

实现分类模型并在线评测结果

提交一份不超过3页的报告(pdf格式)，说明自己所使用的方法以及主要工作，特别要把以上要求对比的部分写清楚，同时提交
代码

请大家将最终材料打包发送到cvdl23@163.com，命名格式为 学号_姓名_第一次作业.zip，提交的截止日期为5月14日晚23点
59分(北京时间)

注意：测试结果仅供参考，对于最终的成绩评定我们主要衡量报告和代码的质量，请不要将过多时间花在刷点上
