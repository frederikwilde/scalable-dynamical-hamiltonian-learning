import numpy as np


class GradientDescentOptimizer:
    def init(self, parameters, **hyper_parameters):
        self.parameters = np.copy(parameters)
        self.step_size = hyper_parameters.get('step_size', .01)
        self.iter = 0

class Adam(GradientDescentOptimizer):
    def __init__(self, parameters, **hyper_parameters):
        GradientDescentOptimizer.init(self, parameters, **hyper_parameters)
        self.beta1 = hyper_parameters.get('beta1', .9)
        self.beta2 = hyper_parameters.get('beta2', .999)
        self.eps = hyper_parameters.get('eps', 1e-8)
        self.m = np.zeros(parameters.shape, dtype='double')
        self.v = np.zeros(parameters.shape, dtype='double')
        self.m_hat = np.zeros(parameters.shape, dtype='double')
        self.v_hat = np.zeros(parameters.shape, dtype='double')
    def step(self, gradient):
        self.iter += 1
        self.m = self.beta1 * self.m + (1-self.beta1) * gradient
        self.v = self.beta2 * self.v + (1-self.beta2) * gradient**2
        self.m_hat = self.m / (1-self.beta1**self.iter)
        self.v_hat = self.v / (1-self.beta2**self.iter)
        self.parameters -= self.step_size * self.m_hat / (np.sqrt(self.v_hat) + self.eps)