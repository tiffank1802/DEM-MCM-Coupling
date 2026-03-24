import numpy as np
import scipy as sp

""""Ce code implemente la méthode des Elements finis """

# Definition des variables

L=1
R=.01
E=2e11
rho=7800
N=10
delta_L=L/N
S=np.pi*R**2

# Construction de la matrice de raideur elementaire
K_e=(E*S/delta_L)*np.array([[1,-1],[-1,1]])

# Construction de la matrice de Masse elementaire

M_e=(rho*S*delta_L)*np.array([
    [1/3,1/6],
    [1/6,1/3]
    ]
                             )

# Assemblage
K=np.zeros((N+1,N+1))
M=np.zeros((N+1,N+1))

for i in range(N):
    n0=i
    n1=i+1
    K[np.ix_([n0,n1],[n0,n1])]+=K_e
    M[np.ix_([n0,n1],[n0,n1])]+=M_e

# Application des conditions aux limites
K=np.delete(K,0,axis=0)
K=np.delete(K,0,axis=1)
M=np.delete(M,0,axis=0)
M=np.delete(M,0,axis=1)

V,D=sp.linalg.eigh(K,M)


# print(K.shape)
print(f"Valeurs propres: {np.sqrt(V)}")



def puissances_iteres(K=K,M=M,iter=10):
    x=np.random.randn(K.shape[0],1)
    lmd=np.zeros(iter)
    for i in range(iter):
        alpha=np.linalg.norm(x)
        v=K@x
        x=np.linalg.inv(M)@v
        R=(x.T@K@x)/(x.T@M@x)
        x/=alpha
        lmd[i]=np.sqrt(R)
    return lmd

def algo_inverse(K=K,M=M,iter=10):
    x=np.random.randn(K.shape[0],1)
    lmd=np.zeros(iter)
    for i in range(iter):
        alpha=np.linalg.norm(x)
        v=M@x
        x=np.linalg.inv(K)@v
        R=(x.T@K@x)/(x.T@M@x)
        x/=alpha
        lmd[i]=np.sqrt(R)
    return lmd
    
    
lmd1=puissances_iteres()
lmd2=algo_inverse()
print("\n"*3)
print(f"lambda pour application des puissances itérées:\n\n{lmd1}")

print("\n"*3)
print(f"lambda pour application de l'algo inverse:\n\n{lmd2}")

