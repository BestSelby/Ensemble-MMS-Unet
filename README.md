#### 多模态MRI乳腺癌分割

----

##### 项目介绍：

​	本项目用于对多个功能模态的乳腺MRI进行病灶分割，主要用到多路径的神经网络`MMS-Unet`将不同MRI模态融合，然后使用Bagging算法思想，将多个训练好的`MMS-Unet`集成，对分割结果进行像素级的多数投票，进一步利用模态，提高分割精度。

----

##### 代码的目录结构如下：

* `Breast_Datasets`：包含9种MRI功能模态，分别为PDMapping、PDW、STIR、T1Mapping、T1W、T1WFLAIR、T2Mapping、T2W、T2WFLAIR，训练数据与标签都采用`npy`格式文件，每个模态文件夹下包含如下内容：
  * `Images`:以npy格式保存的原图像；
  * `Labels`:标签数据，同一样本对应的数据与标签文件名相同；
  * `ImageSets`：以txt文件记录样本文件名，模型读取数据时，只需记录该文件的所有文件名内容，然后找到对应数据与标签；

- `logs`：保存模型在训练过程中，较优的模型权重；
- `metrics_logs`：通过`TensorBoard`可视化模型训练过程，保存loss，miou，dice，ppv等；
- `model_data`：存放`vgg16`权重文件，本项目部分模型主干网络使用预训练模型`vgg16`；
- `nets`：基本的分割模型，包括：`deeplabv3+`，`FCN`，`Segnet`，`Unet`，`vgg`；
- `utils`：包括图像预处理，数据集读取，训练每个epoch过程，分割度量指标的计算。

------

- 
- `Result`：以`npy`格式保存模型`loss`结果；
- `main.py`：训练模型，代码运行入口处使用`argparse`模块调节训练超参数，如`batch_size`、`num_classes`、`epoches`等；
- `metrices.py`：用于度量模型精度，包括`混淆矩阵`、`f1`、`acc`、`pre`、`recall`等；
- `myNetwork.py`：网络模型；
- `vgg.py`：分类模型`vgg16`；
- `predict.py`：分割测试，使用`lugSeg_test`文件夹中的数据；
- 
