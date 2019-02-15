"""
不知由于何种原因，运行时使用默认初始化方法训练正常，之后再使用原本的初始化时，在SGD_matrix中            evaluation_data = list(evaluation_data)会得到一个空列表。若直接使用原本的初始化运行，则可以正常运行，且结果符合作者得到的结果
"""

#%% [markdown]
# # 对比两种初始化方法

#%% [markdown]
# ## 导入库

#%%
import json
import random
import sys

import mnist_loader
import network2

import matplotlib.pyplot as plt 
import numpy as np 

#%% [markdown]
# ## 主函数

#%%
def main(filename, n, eta):
    run_network(filename, n, eta)
    make_plot(filename)

#%% [markdown]
# ## 运行网络

#%%
def run_network(filename, n, eta):
    """Train the network using both the default and the large starting
    weights.  Store the results in the file with name ``filename``,
    where they can later be used by ``make_plots``.

    """
    # Make results more easily reproducible
    random.seed(12345678)
    np.random.seed(12345678)
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = network2.Network([784, n, 10], cost=network2.CrossEntropyCost)
    print("Train the network using the default starting weights.")
    default_vc, default_va, default_tc, default_ta \
        = net.SGD_matrix(training_data, 30, 10, eta, lmbda=5.0,
                  evaluation_data=validation_data, 
                  monitor_evaluation_accuracy=True)
    print("Train the network using the large starting weights.")
    net.large_weight_initializer()
    large_vc, large_va, large_tc, large_ta \
        = net.SGD_matrix(training_data, 30, 10, eta, lmbda=5.0,
                  evaluation_data=validation_data, 
                  monitor_evaluation_accuracy=True)
    f = open(filename, "w")
    json.dump({"default_weight_initialization":
               [default_vc, default_va, default_tc, default_ta],
               "large_weight_initialization":
               [large_vc, large_va, large_tc, large_ta]}, 
              f)
    f.close()

#%% [markdown]
# ## 绘图

#%%
def make_plot(filename):
    """Load the results from the file ``filename``, and generate the
    corresponding plot.

    """
    f = open(filename, "r")
    results = json.load(f)
    f.close()
    default_vc, default_va, default_tc, default_ta = results[
        "default_weight_initialization"]
    large_vc, large_va, large_tc, large_ta = results[
        "large_weight_initialization"]
    # Convert raw classification numbers to percentages, for plotting
    default_va = [x/100.0 for x in default_va]
    large_va = [x/100.0 for x in large_va]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(0, 30, 1), large_va, color='#2A6EA6',
            label="Old approach to weight initialization")
    ax.plot(np.arange(0, 30, 1), default_va, color='#FFA933', 
            label="New approach to weight initialization")
    ax.set_xlim([0, 30])
    ax.set_xlabel('Epoch')
    ax.set_ylim([85, 100])
    ax.set_title('Classification accuracy')
    plt.legend(loc="lower right")
    plt.show()

#%% [markdown]
# ## 运行主体

#%%
main("outputs/weight_initialization1.json", 30, 0.1)