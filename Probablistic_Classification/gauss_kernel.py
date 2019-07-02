import numpy as np
import matplotlib.pyplot as plt

np.random.seed(10)

def generate_data(sample_size=120, n_class=4):
    x = (np.random.normal(size=(sample_size // n_class, n_class))
         + np.linspace(-4., 4., n_class)).flatten()
    y = np.broadcast_to(np.arange(n_class),
                        (sample_size // n_class, n_class)).flatten()
    return x, y

def least_square_l2(x, y, h, lamda):
	K = np.exp(-(x - x[:, None]) ** 2 / (2 * h ** 2))
	pi_y = []
	for i in range(4):
		pi_y.append(np.zeros(120))
		pi_y[i][y==i] = 1
	theta = []
	for i in range(4):
		a = np.linalg.solve(
			K.T.dot(K) + lamda * np.identity(len(y)),
			K.T.dot(pi_y[i])
			)
		theta.append(a)
	theta = np.array(theta)
	return K, theta

def calc_prob(theta, K):
	numerator = []
	for i in range(4):
		numerator.append(theta[i].T.dot(K))
	numerator = np.array(numerator)
	numerator = np.where(numerator > 0, numerator, 0)
	denominator = np.sum(numerator, axis = 0)
	prob = []
	for i in range(4):
		prob.append(numerator[i] / denominator)
	return np.array(prob)


def visualize(x, theta, h):
	X = np.linspace(-5., 5., num=100)
	K = np.exp(-(x - X[:, None]) ** 2 / (2 * h ** 2))
	
	a, b, c, d = [K.dot(theta[i]) for i in range(4)]
	a, b, c, d = [i / (a + b + c + d) for i in [a, b, c, d]]

	plt.clf()
	plt.xlim(-5, 5)
	plt.ylim(-1.8, 2.3)

	plt.plot(X, a, c='blue', linestyle=':', label='p(y|x=1)')
	plt.plot(X, b, c='red', linestyle='-', label='p(y|x=2)')
	plt.plot(X, c, c='green', linestyle='--', label='p(y|x=3)')
	plt.plot(X, d, c="yellow", linestyle='-.', label='p(y|x=4)')

	plt.scatter(x[y == 0], -1 * np.ones(len(x) // 4), c='blue', marker='o')
	plt.scatter(x[y == 1], -1.5 * np.ones(len(x) // 4), c='red', marker='x')
	plt.scatter(x[y == 2], -1 * np.ones(len(x) // 4), c='green', marker='v')
	plt.scatter(x[y == 3], -1.5 * np.ones(len(x) // 4), c='yellow', marker='o')

	plt.legend()

	plt.savefig("result.pdf")
	plt.show()
	
# 実行
h = 0.4
lamda = 1
x, y = generate_data(120, 4)
K, theta = least_square_l2(x, y, h, lamda)
prob = calc_prob(theta, K)
visualize(x, prob, h)
