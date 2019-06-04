# coding: utf-8

from statistics import mean
from sklearn import model_selection
import numpy as np
import matplotlib.pyplot as plt
import random
np.random.seed(198)

# generating sample
# true_model : f = y = sin(pi*x)/(pi) + 0.01(x**2)

def generate_sample(xmin, xmax, sample_size):
    x = np.linspace(start=xmin, stop=xmax, num=sample_size)
    true_model = np.sin(np.pi*x)/np.pi + 0.01*x**3
    noise = 0.15 * np.random.normal(loc=0, scale=1, size=sample_size)
    y = true_model + noise
    return x, y

# dividing the data set into k subsets
# kth subset = test set
# k = 10
sample_x = generate_sample(0, 5, 10000)[0]
sample_y = generate_sample(0, 5, 10000)[1]

num_test  = 100
num_train = 100
num_all = len(sample_x)

id_all   = np.random.choice(num_all, num_all, replace=False)
id_test  = id_all[:num_test]
test_data  = [sample_x[id_test], sample_y[id_test]]
id_all = id_all[num_test:num_all]
train_data =[]
while len(id_all) >= num_train:
    id_train = id_all[:num_train]
    train_data.append([sample_x[id_train], sample_y[id_train]])
    id_all = id_all[num_train:num_all]

# いくつか描写してみる
"""
true_x = np.linspace(start=0, stop=5, num=100)
true_y = np.sin(np.pi*true_x)/np.pi + 0.01*true_x**3
fig = plt.figure(figsize=(6,12))
plt.subplot(5, 2, 1); x = test_data[0]; y = test_data[1];plt.scatter(x, y, c="green", s = 1);plt.plot(true_x, true_y)
plt.subplot(5, 2, 2); x = train_data[0][0]; y = train_data[0][1];plt.scatter(x, y, c="blue", s = 1);plt.plot(true_x, true_y)
plt.subplot(5, 2, 3); x = train_data[1][0]; y = train_data[1][1];plt.scatter(x, y, c="red", s = 1);plt.plot(true_x, true_y)
plt.subplot(5, 2, 4); x = train_data[2][0]; y = train_data[2][1];plt.scatter(x, y, c="yellow", s = 1);plt.plot(true_x, true_y)
plt.show()
plt.close()
"""

# kernel matrix
def calc_kernel_matrix(x, c, h):
    kernel_matrix = np.exp(-(x[None]-c[:, None])**2 / (2 * (h **2)))
    return kernel_matrix

# calculate design matrix
# solve the least square with l2 and #make prediction
def solve_and_prediction(x, y, X, h, l):
    k = calc_kernel_matrix(x=x, c=x, h=h)
    theta = np.linalg.solve(
        k.T.dot(k) + l * np.identity(len(y)), 
        k.T.dot(y[:, None]))
    K = calc_kernel_matrix(x=x, c=X, h=h)
    prediction = K.dot(theta)
    return prediction, theta

# meausure the errorness
def get_error(pred):
    error = 0.01 * np.sum((prediction-test_data[1])**2)
    return(error)

#get general error of each model and compare them
error_list = []
for i in range(10):
    prediction = solve_and_prediction(x=train_data[i][0], 
        y=train_data[i][1],X=test_data[0], h=0.3, l=0.1)[0]
    error = get_error(prediction)
    error_list.append(error)
print("k=0.3, lambda=0.1 のときの汎化誤差 : {}".format(mean(error_list)))

#visualizationのための下準備
h_ = [0.03, 0.3, 3] 
l_ = [0.00001, 0.01, 100]
candidates =[[h_[0],l_[0]],
    [h_[1],l_[0]],
    [h_[2],l_[0]],
    [h_[0],l_[1]],
    [h_[1],l_[1]],
    [h_[2],l_[1]],
    [h_[0],l_[2]],
    [h_[1],l_[2]],
    [h_[2],l_[2]]]
x_tempo = np.linspace(0,5,100)
y_predict = []
for i in candidates:
    y_tempo = solve_and_prediction(x=train_data[2][0], 
        y=train_data[2][1], X=x_tempo, h=i[0], l=i[1])[0]
    y_predict.append(y_tempo)

x = train_data[0][0]
y = train_data[0][1]

fig = plt.figure(figsize=(12,12))
for i in range(9):
    plt.subplot(3,3,i+1);
    plt.title("(h, lambda) = ({}, {})".format(candidates[i][0], candidates[i][1]) )
    plt.ylim(-1,2.2);plt.scatter(x, y, c="orange", s=3);
    plt.plot(x_tempo, y_predict[i], c="blue", linewidth=1)#;plt.plot(true_x, true_y, c="red")
plt.savefig("result.png")
plt.show()