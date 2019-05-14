# coding: utf-8
# 先端データ解析論第3回宿題２
# スパース回帰

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(198)

# generating sample
# true_model : f = y = sin(pi*x)/(pi) + 0.01(x**2)

def generate_sample(xmin, xmax, sample_size):
    x = np.linspace(start=xmin, stop=xmax, num=sample_size)
    true_model = np.sin(np.pi*x)/np.pi + 0.01*x**3
    noise = 0.15 * np.random.normal(loc=0, scale=1, size=sample_size)
    y = true_model + noise
    return x, y

#とりあえず可視化

sample_x = generate_sample(0, 5, 100)[0]
sample_y = generate_sample(0, 5, 100)[1]
"""
true_x = np.linspace(0, 5, 1000)
true_y = np.sin(np.pi*true_x)/np.pi + 0.01*true_x**3
fig = plt.figure(figsize=(9,9))
plt.plot(true_x, true_y, c="red", linewidth=2)
plt.scatter(sample_x, sample_y, s=2)
plt.show()
"""

# kernel matrix
def calc_desgin_matrix(x, c, h=0.3): #ガウス幅はl2と比較するために同じにした。
    karnel_matrix = np.exp(-(x[None]-c[:, None])**2 / (2 * (h **2)))
    return karnel_matrix

# initialization
def initailization():
    theta, z, u = np.random.randn(3)
    return theta, z, u

# augmented Lagrangian
def augmented_Lagrangian(k, theta, z, u, lambda_=0.1): #ラムダも0.1にしておく
    _ = np.linalg.inv(k.T.dot(k) + np.identity(len(k))).dot(k.T.dot(sample_y)+z-u)
    update_quantity = np.linalg.norm(theta - _)
    theta = _
    z = np.maximum(0, theta + u - lambda_) + np.minimum(0, theta + u + lambda_)
    u = u + theta - z
    return update_quantity, theta, z, u

# update parameters
k = calc_desgin_matrix(x=sample_x, c=sample_x)
theta, z,u = initailization()
epsilon = 0.0001
update_quantity = 1
num = 0 
while update_quantity > epsilon:
    update_quantity,theta,z,u = augmented_Lagrangian(k, theta, z, u)
    num += 1

print("{}回更新しました。".format(num))

# create data to visualize the prediction
pred_x = np.linspace(0, 5, 100)
K = calc_desgin_matrix(sample_x, pred_x)
pred_y = K.dot(theta)

# reguralization with l2
theta_l2 = np.linalg.solve(
    k.T.dot(k) + 0.1 * np.identity(len(k)),
    k.T.dot(sample_y[:, None]))
pred_y_l2 = K.dot(theta_l2)

# visualization
true_x = np.linspace(0, 5, 1000)
true_y = np.sin(np.pi*true_x)/np.pi + 0.01*true_x**3
fig = plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
#plt.plot(true_x, true_y, c="red", linewidth=2)
plt.title("L1-reguralization") ;plt.scatter(sample_x, sample_y,c="orange", s=2); plt.plot(pred_x, pred_y, c="blue", linewidth=2)
plt.subplot(1,2,2)
plt.title("L2-reguralization"); plt.scatter(sample_x, sample_y,c="orange", s=2); plt.plot(pred_x, pred_y_l2, c="green", linewidth=2)
plt.savefig("result_0507.png")
plt.show()
plt.close()

# evaluate the sparseness
theta_digit = np.log10(abs(theta))
theta_l2_digit = np.log10(abs(theta_l2))
print("（ほとんど）0となるパラメータの数 : {}個".format(np.sum(theta_digit < -5)))
plt.clf()
#plt.figure(figsize=(6,6))
plt.title("# of Parameters\nL1: blue  L2: orange")
plt.xlabel("log10|x|")
plt.ylabel("#")
plt.ylim(0,30)
plt.hist(theta_digit, bins=np.linspace(-8, 0, 41), alpha=0.6)
plt.hist(theta_l2_digit, bins=np.linspace(-8, 0, 41), alpha=0.6)
plt.savefig("result_0507_2.png")
plt.show()
