import faiss
import numpy as np

index = faiss.IndexFlatL2(4)
vecs = np.random.rand(3, 4).astype('float32')
index.add(vecs)
v1 = index.reconstruct(0)
print("Reconstructed:", v1)
print("Original:", vecs[0])
