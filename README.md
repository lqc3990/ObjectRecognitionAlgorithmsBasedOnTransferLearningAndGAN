# ObjectRecognitionAlgorithmsBasedOnTransferLearningAndGAN\<br> 
=======
简介：\<br> 
----
基于迁移学习和对抗生成网络的分类器算法，\<br> 目标是实现源域和目标域的半监督域适应，将源域训练的分类器模型应用于目标域数据分类任务。\<br> 
本文采用CycleGAN来实现域适应。 CycleGAN采用循环对称结构，加入了循环损失函数和重建损失函数，可以使源域和目标域双向生成，从而实现从源域到目标域的域适应。\<br> 
使得源域训练的分类器可以用于目标域。实验结果表明，用域适应后的源域数据训练SVM和KNN分类器，对目标域样本进行分类，性能远远好于源域直接训练的分类器。\<br> 
实验环境：\<br> 
-----
MATLAB \<br> 
python2.7以上\<br> 
运行：
----
下载数据“MSRC_vs_VOC” 运行oppo_disco_GAT_MSRC_alltrain.m\<br> 
下载数据文件夹“Handwritten_digits“下所有数据，运行 oppo_disco_GAT_SEMEION_alltrain.m\<br> 
