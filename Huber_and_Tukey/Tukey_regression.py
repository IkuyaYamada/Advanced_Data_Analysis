# coding: utf-8
# 先端データ解析論第4回宿題3
# テューキー回帰

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(198)

# generating sample
# true_model : y = x

# generate samples
def generate_sample(xmin, xmax, sample_size, outlier):
    x = np.linspace(start=xmin, stop=xmax, num=sample_size)
    noise = 0.3 * np.random.normal(loc=0, scale=1, size=sample_size)
    y = x + noise
    y = [y[i] if i < sample_size/2 or i%outlier!=0  else -3 for i in range(len(y))]
    return x, y

# initialization
def initailization():
    theta = np.random.randn(2,1)
    return theta

# updating
def huber_loss(x, y,theta, eta=1): 
	num = 0
	while True:
		phi = np.array([[1]*len(sample_x),theta[1]*x]).squeeze().T
		r =  (theta[0] + theta[1] * x) - y
		W = np.diag([1 if abs(i) <= eta else eta/abs(i) for i in r])
		_ = np.linalg.inv(phi.T.dot(W).dot(phi)).dot(phi.T).dot(W).dot(y)
		if np.linalg.norm(theta-_) > 0.001 and num < 1000:
			theta = _
			num += 1
		else:
			theta =_
			return(num, theta)

def tukey_loss(x, y,theta, eta=1):
	num = 0
	while True:
		phi = np.array([[1]*len(sample_x),theta[1]*x]).squeeze().T
		r =  (theta[0] + theta[1] * x) - y
		W = np.diag([((1 - i**2) / (eta**2))**2 if abs(i) <= eta else 0 for i in r])
		_ = np.linalg.inv(phi.T.dot(W).dot(phi)).dot(phi.T).dot(W).dot(y)
		if np.linalg.norm(theta-_) > 0.001 and num < 1000:
			theta = _
			num += 1
		else:
			theta =_
			return(num, theta)

def tukey_loss(x, y,theta, eta=1):
	num = 0
	while True:
		phi = np.array([[1]*len(sample_x),theta[1]*x]).squeeze().T
		r =  (theta[0] + theta[1] * x) - y
		W = np.diag([((1 - i**2) / (eta**2))**2 if abs(i) <= eta else 0 for i in r])
		_ = np.linalg.inv(phi.T.dot(W).dot(phi)).dot(phi.T).dot(W).dot(y)
		if np.linalg.norm(theta-_) > 0.001 and num < 1000:
			theta = _
			num += 1
		else:
			theta =_
			return(num, theta)

# いろいろな条件のもとでグラフ描写
fig = plt.figure(figsize=(12,12))
outlier_num = [0, 10, 20, 50]
for i, o in enumerate([100, 10, 5, 2]):
	sample_x, sample_y = generate_sample(-3, 3, 200, o)
	theta = initailization()
	huber = huber_loss(x=sample_x, y=sample_y,theta=theta, eta=1)
	tukey = tukey_loss(x=sample_x, y=sample_y,theta=theta, eta=1)
	print("huberでは{}回反復を行いました。".format(huber[0]))
	print("tukeyでは{}回反復を行いました。".format(tukey[0]))

	# visualizaion
	huber_x = np.linspace(-3, 3, 100)
	huber_y = huber[1][0] + huber[1][1]*huber_x
	tukey_x = huber_x
	tukey_y = tukey[1][0] + tukey[1][1]*huber_x
	plt.subplot(2,2,i+1)
	plt.title("# of outliers: {}".format(outlier_num[i]))
	plt.xlim(-3.5,3.5)
	plt.ylim(-3.5,3.5)
	plt.plot(huber_x, huber_y, c="red", linewidth=3)
	plt.plot(tukey_x, tukey_y, c="blue", linewidth=3)
	plt.scatter(sample_x, sample_y,c="orange", s=5)
fig.suptitle("blue: tukey regression  red: huber regression",fontsize=20)
plt.savefig("result_0514.png")
plt.show()