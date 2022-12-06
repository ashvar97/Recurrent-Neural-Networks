from Layers import Base

class Flatten(Base.BaseLayer):
    def __init__(self):
        super().__init__()
    
    def forward(self, input_tensor):
        self.shape =  input_tensor.shape
        print(self.shape)
        bsize,l,w,h = self.shape
        return input_tensor.reshape(bsize, -1)
    
    def backward(self, error_tensor):
        return error_tensor.reshape(self.shape)
    