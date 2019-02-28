import numpy as np
import numpy.linalg as LA

# X = np.array([[72.,101.,94.],[50.,96.,70.],[14.,79.,10.],[8.,70.,1.]], np.float64);
X = np.array([[0.,101.,94.],[0.,96.,70.],[0.,79.,10.],[0.,70.,1.]], np.float64);
print(X)

μ=np.mean(X,axis=0)
print("μ ->{}".format(μ))
Z=X-μ
print("Z ->{}".format(Z))
print("Z mean ->{}".format(np.mean(Z,axis=0)))

C=np.cov(Z,rowvar=False)
print("C ->{}".format(C))

[λ,V]=LA.eigh(C)
print("λ,V->{}\n\n{}".format(λ,V))

