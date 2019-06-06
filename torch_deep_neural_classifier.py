from torch_shallow_neural_classifier import TorchShallowNeuralClassifier
import torch.nn as nn

class TorchDeepNeuralClassifier(TorchShallowNeuralClassifier):
    def __init__(self, dropout_prob=0.0, **kwargs):
        self.dropout_prob = dropout_prob
        super().__init__(**kwargs)
    
    def define_graph(self):
    	return nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.Dropout(p = self.dropout_prob),
            self.hidden_activation,
            nn.Linear(self.hidden_dim, self.n_classes_))