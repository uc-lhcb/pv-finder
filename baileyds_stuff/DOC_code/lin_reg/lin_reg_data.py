import torch
from torch import nn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

def map_to(vec):
    #maps 0,1 to -1,1
    #print(vec)
    out = 2*vec - np.ones_like(vec)
    #print(out)
    return out

def getLinRegDatasets(num_train, num_test, x_dim=1, y_dim=1, m_var=0.25, b_var=0.25, preset=(False, 0, 0)):

    train = []
    test = []

    if preset[0]==False:
        m = np.random.rand(y_dim, x_dim)
    if preset[0]==True:
        m = preset[1]
    train_m_adj = m_var*map_to(np.random.rand(num_train, y_dim, x_dim))
    test_m_adj = m_var*map_to(np.random.rand(num_test, y_dim, x_dim))


    if preset[0]==False:
        b = np.random.rand(y_dim)
    if preset[0]==True:
        b = preset[2]
    train_b_adj = b_var*map_to(np.random.rand(num_train, y_dim))
    test_b_adj = b_var*map_to(np.random.rand(num_test, y_dim))

    #x_train = map_to(np.random.rand(num_train, x_dim))
    x_train = np.random.rand(num_train, x_dim)
    y_train = np.zeros((num_train, y_dim))
    for i in range(num_train):
        y_train[i] = np.matmul((train_m_adj[i] + m), x_train[i]) + (train_b_adj[i] + b)

    #x_test = map_to(np.random.rand(num_test, x_dim))
    x_test = np.random.rand(num_test, x_dim)
    y_test = np.zeros((num_test, y_dim))
    for i in range(num_test):
        y_test[i] = np.matmul((test_m_adj[i] + m), x_test[i]) + (test_b_adj[i] + b)

    tensor_x_train = torch.Tensor(x_train)
    tensor_y_train = torch.Tensor(y_train)
    tensor_x_test = torch.Tensor(x_test)
    tensor_y_test = torch.Tensor(y_test)

    #print(tensor_x_test.shape)

    return TensorDataset(tensor_x_train, tensor_y_train), TensorDataset(tensor_x_test, tensor_y_test), m, b

#returns data in form approx y = ax^2 + mx + b
def getQuadRegDatasets(num_train, num_test, x_dim=1, y_dim=1, m_var=0.25, b_var=0.25, preset=(False, 0, 0)):

    train = []
    test = []

    if preset[0]==False:
        m = np.random.rand(y_dim, x_dim)
    if preset[0]==True:
        m = preset[1]
    train_m_adj = m_var*map_to(np.random.rand(num_train, y_dim, x_dim))
    test_m_adj = m_var*map_to(np.random.rand(num_test, y_dim, x_dim))


    if preset[0]==False:
        b = np.random.rand(y_dim)
    if preset[0]==True:
        b = preset[2]
    train_b_adj = b_var*map_to(np.random.rand(num_train, y_dim))
    test_b_adj = b_var*map_to(np.random.rand(num_test, y_dim))

    #x_train = map_to(np.random.rand(num_train, x_dim))
    x_train = np.random.rand(num_train, x_dim)
    y_train = np.zeros((num_train, y_dim))
    for i in range(num_train):
        y_train[i] = 10*np.matmul(x_train[i].T, x_train[i]) + np.matmul((train_m_adj[i] + m), x_train[i]) + (train_b_adj[i] + b)

    #x_test = map_to(np.random.rand(num_test, x_dim))
    x_test = np.random.rand(num_test, x_dim)
    y_test = np.zeros((num_test, y_dim))
    for i in range(num_test):
        y_test[i] = 10*np.matmul(x_test[i].T, x_test[i]) + np.matmul((test_m_adj[i] + m), x_test[i]) + (test_b_adj[i] + b)

    tensor_x_train = torch.Tensor(x_train)
    tensor_y_train = torch.Tensor(y_train)
    tensor_x_test = torch.Tensor(x_test)
    tensor_y_test = torch.Tensor(y_test)

    #print(tensor_x_test.shape)

    return TensorDataset(tensor_x_train, tensor_y_train), TensorDataset(tensor_x_test, tensor_y_test), m, b

