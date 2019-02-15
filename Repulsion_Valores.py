import numpy as np
import matplotlib.pylab as plt
n=100 # dimension del espacio a utilizar
m= 200 #dimensión del espacio sobre el que se traza (ambiente)
eigen_states=np.array([])
entropias=np.array([])
for i in range(10000):
    Psi=np.random.normal(0,1,(m,n))+1j*np.random.normal(0,1,(m,n))
    Psi=np.matmul(Psi.conjugate().T,Psi)
    Psi=Psi/Psi.trace()
    w,v = np.linalg.eig(Psi)
    # U,Sigma,V=np.linalg.svd(Psi)
    w=np.real(w)
    # w=np.absolute(w)**2
    eigen_states=np.append(eigen_states,w)
    e=-np.sum(w*np.log(w))
    entropias=np.append(entropias,e)
    if i%1000==0:
        print(i/100 + 10,"%")
# print(eigen_states)
# print(entropias)

plt.hist(eigen_states,density=True,bins=50)
plt.savefig("eigen.png")
plt.show()

plt.hist(entropias,density=True,bins=50)
#plt.axvline(np.log(n))
plt.title("Pico de entropía {} vs maximo ({})".format(round(np.mean(entropias),3),round(np.log(n),3)))
plt.savefig("entropy.png")
# plt.xlim(2*np.min(entropias),np.log(n)+0.01)
plt.show()
