# 《神经网络与深度学习》的练习代码

中文版 | [English](./README_eng.md)

基本上按照[原书代码](https://github.com/mnielsen/neural-networks-and-deep-learning)写的，针对python3的修改，参考了[DeepLearningPython35](https://github.com/MichalDanielDobrzanski/DeepLearningPython35)，没什么差别🌚（代码肯定比原书丑🙈）

不想直接用别人的代码，就自己比葫芦画瓢写(chao)一(yi)写(chao)，加深理解🤣

## 代码说明

network.py依照原文编写，并根据作者提出的问题进行改进，每个mini_batch一起使用矩阵的形式进行运算，而不是一个一个数据进行训练，加快训练的速度

network2.py依照原文编写，直接使用network.py里面的矩阵形式的运算（由于我长时间没有用命令行操作git，不小心`git checkout -- network2.py`。我也不知道自己撤销了什么修改，所以……放弃了，不尝试恢复了。😭）

network3.py……emmm不写了，不想安装Theano。