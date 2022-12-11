class Sgd:
    def __init__(self, learningrate):
        self.learning_rate = learningrate

    def calculate_update(self, weight_tensor, gradient_tensor):
        updated_wts = weight_tensor - self.learning_rate * gradient_tensor
        return updated_wts

class SgdWithMomentum:
    def __init__(self, learningrate, m):
        self.learning_rate = learningrate
        self.m = m
        self.v0 = 0
        
    def calculate_update(self, weight_tensor, gradient_tensor):
        v1 = self.m * self.v0 - self.learning_rate* gradient_tensor
        updated_wts = weight_tensor + v1
        self.v0 = v1
        return updated_wts


class Adam:
    def __init__(self, learningrate, b1, b2):
        self.learning_rate = learningrate
        self.k = 1
        self.mu = b1
        self.rho = b2
        self.v0 = 0
        self.p0 = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        v1 = self.mu * (self.v0) + (1 - self.mu) * gradient_tensor
        p1 = self.rho * (self.p0) + (1 - self.rho) * gradient_tensor * gradient_tensor

        vel_bias = v1 / (1 - self.mu ** self.k)
        pos_bias = p1 / (1 - self.rho ** self.k)

        updated_wts = weight_tensor - self.learning_rate * (vel_bias / (pos_bias ** 0.5 + 0.000000000000001))
        self.v0 = v1
        self.p0 = p1
        self.k += 1
        return updated_wts

