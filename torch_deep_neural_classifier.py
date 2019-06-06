import torch.nn as nn

class TorchDeepNeuralClassifier(TorchShallowNeuralClassifier):
    def __init__(self, dropout_prob=0.7, **kwargs):
        self.dropout_prob = dropout_prob
        super().__init__(**kwargs)
    
    def define_graph(self):