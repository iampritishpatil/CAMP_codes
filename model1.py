from functions import *

connectivity_matrix = create_martix(p=1,N=100)
plot_matrix(connectivity_matrix)
plot_eigenvalues(connectivity_matrix)
print np.mean(connectivity_matrix),np.var(connectivity_matrix)
