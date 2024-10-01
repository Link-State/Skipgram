import torch
from torch import nn

class dotModel(nn.Module) :
    def __init__(self, voc_sz, d_emb) :
        super(dotModel, self).__init__()
        self.embedding1 = nn.Embedding(voc_sz, d_emb)
        self.embedding2 = nn.Embedding(voc_sz, d_emb)
        self.logistic = nn.Sigmoid()

    def forward(self, X) :
        wc = X[:,0]
        wo = X[:,1]
        center_word_vector = self.embedding1(wc)
        context_word_vector = self.embedding2(wo)

        s = torch.einsum('ij,ij->i', center_word_vector, context_word_vector)
        y = self.logistic(s)
        return y
