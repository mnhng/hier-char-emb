#!/usr/bin/env python
import torch
from nnblk import HierarchicalEmbedding

char2index = {'白': 0, '山': 1, '名': 2, '風': 3}
emb = HierarchicalEmbedding(num_embeddings=len(char2index), embedding_dim=4, char2index=char2index)
input_ = torch.LongTensor(list(char2index.values()))
print(emb(input_))
