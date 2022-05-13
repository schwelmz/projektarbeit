import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt

def make_laplace(N, hx = 1, bounds=None):
    if hasattr(hx, "__len__"):
        """
        0    1    2
        |----|----|----|--
        h0   h1   h2
        """
        assert(len(hx) == N - 1), f"len(hx) = {len(hx)}, N = {N}"
        h = lambda i: hx[i]
    else:
        h = lambda i: hx


    rows = []
    cols = []
    vals = []
    for i in range(N):
        if bounds=='skip' and i in [0, N-1]:
            continue
        if bounds=='dirichlet' and i in [0, N-1]:
            rows.append(i)
            cols.append(i)
            vals.append(1)
            continue
        if bounds=='neumann' and i in [0, N-1]:
            if i == 0:
                rows.append(i)
                cols.append(i)
                vals.append(-1/h(0))
                rows.append(i)
                cols.append(i+1)
                vals.append(1/h(0))
            else:
                rows.append(i)
                cols.append(i-1)
                vals.append(1/h(N-2))
                rows.append(i)
                cols.append(i)
                vals.append(-1/h(N-2))
            continue

        if i != 0:
            rows.append(i)
            cols.append(i-1)
            vals.append(1/h(i-1)) # ∇φ_i ∇φ_i-1
        rows.append(i)
        cols.append(i)
        vals.append(-1/h(i-1) - 1/h(i)) # ∇φ_i ∇φ_i
        if i != N-1:
            rows.append(i)
            cols.append(i+1)
            vals.append(1/h(i)) # ∇φ_i ∇φ_i+1
    # negate as Δ = - <∇φ, ∇φ>
    return -sparse.csr_matrix((vals, (rows, cols)), shape=(N,N))

def arr2str(arr, **kwargs):
    return np.array2string(arr, formatter={'float_kind': lambda x: '{:+.2e}'.format(x)}, **kwargs).replace('+0.00e+00', '    -    ')

def second_order_coefficients(j, u, x):
    xi = [x[j-1], x[j], x[j+1]]
    ui = [u[j-1], u[j], u[j+1]]
    a = ui[0]*xi[1]*xi[2]/((xi[0]-xi[1])*(xi[0]-xi[2]))+ui[1]*xi[0]*xi[2]/((xi[1]-xi[0])*(xi[1]-xi[2]))+ui[2]*xi[0]*xi[1]/((xi[2]-xi[0])*(xi[2]-xi[1]))
    b = -(ui[0]*(xi[1]+xi[2])/((xi[0]-xi[1])*(xi[0]-xi[2]))+ui[1]*(xi[0]+xi[2])/((xi[1]-xi[0])*(xi[1]-xi[2]))+ui[2]*(xi[0]+xi[1])/((xi[2]-xi[0])*(xi[2]-xi[1])))
    c = ui[0]/((xi[0]-xi[1])*(xi[0]-xi[2]))+ui[1]/((xi[1]-xi[0])*(xi[1]-xi[2]))+ui[2]/((xi[2]-xi[0])*(xi[2]-xi[1]))
    return [a,b,c]

def first_order_coefficients(j, u, x):
    x = [x[j], x[j+1]]
    u = [u[j], u[j+1]]
    c = (u[1]*x[0]-u[0]*x[1])/(x[0]-x[1])
    m = (u[0]-u[1])/(x[0]-x[1])
    return [m, c]

Nx = 200; Ne = Nx - 1
#Ne = 20; Nx = Ne + 1
x = np.linspace(0,1, Nx)
h = 1 / Ne
print(Nx, Ne, h)

laplace = make_laplace(Nx, h, bounds='dirichlet')
print('  '+arr2str(laplace.todense(), prefix='  '))

rhs = np.ones(Nx) * h # ∫f φᵢdx
# boundary
rhs[0] = 0
rhs[-1] = 0
print('f', rhs)

J = rhs
u = sparse.linalg.gmres(laplace, rhs)
z = sparse.linalg.gmres(laplace, J)
assert u[1] == 0; u = u[0]
assert z[1] == 0; z = z[0]
print('u', u)

u_prime = (u[1:] - u[:-1]) / h # on each element
z_prime = (z[1:] - z[:-1]) / h # on each element
print('p', u_prime)

jumps_u = u_prime[1:] - u_prime[:-1] # on the inner nodes
jumps_z = z_prime[1:] - z_prime[:-1] # on the inner nodes
print('j', jumps_u)

#Iterationsfehler:
    def calc_disc_error():
    t1 = 0
    t2 = 0
    t3 = 0
    t4 = 0
    for ie in range(Ne):
        t1 += h * (z[ie] + z[ie+1])/2
        [m, c] = first_order_coefficients(ie, z, x)
        #m = (z[ie+1]-z[ie])/(x[ie+1]-x[ie])
        #c = z[ie+1]-(z[ie]-z[ie+1])/(x[ie]-x[ie+1])*x[ie+1]
        t4 += c*(x[ie+1]-x[ie]) + 1/2*m*(x[ie+1]**2-x[ie]**2)
        if ie != 0:
            t2 += .5 * jumps_u[ie - 1] * z[ie]
        if ie != Ne - 1:
            t3 += .5 * jumps_u[ie] * z[ie+1]
    #print(t1, t4, t2, t3)
    #print(t1 + t2 + t3)
    return (t1+t2+t3)

#Diskretisierungsfehler:
def discretization_error(x, z_h, u1, u0, Nx):
    for ie in range(1, Ne):
        
        