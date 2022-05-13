import sys
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg
from functools import lru_cache
import matplotlib.pyplot as plt

########################
def extract_channels(file):
    import numpy as np
    if file.endswith('.py'):
        channel_names = ['membrane/V', 'sodium_channel_m_gate/m', 'sodium_channel_h_gate/h', 'potassium_channel_n_gate/n']
        return extractX(file, 'solution', channel_names)
    raise "FileType not understood: "+file

def extract_xyz(file):
    import numpy as np
    if file.endswith('.py'):
        channel_names = ['membrane/V', 'sodium_channel_m_gate/m', 'sodium_channel_h_gate/h', 'potassium_channel_n_gate/n']
        return extractX(file, 'solution', channel_names)
    raise "FileType not understood: "+file

def extractX(file, name, channel_names):
    # name = solution or geometry
    import numpy as np
    if file.endswith('.py'):
        # data = py_reader.load_data([file])
        data = np.load(file, allow_pickle=True)
        data = [data]
        if data == []:
            print("  \033[1;31mCould not parse file\033[0m '{}'".format(file))
        # data[0]['timeStepNo'] is current time step number IN THE SPLITTING (i.e. not global)
        # data[0]['currentTime'] is current SIMULATION time
        tcurr = data[0]['currentTime']
        print('  Loaded data at simulation time:', '\033[1;92m'+str(tcurr)+'\033[0m') # bold green
        data = data[0]['data']
        solution  = next(filter(lambda d: d['name'] == name, data))
        componentX = lambda x: next(filter(lambda d: d['name'] == str(x), solution['components']))
        return {'val':np.vstack([componentX(i)['values'] for i in channel_names]).T, 'tcurr':tcurr}
    raise "FileType not understood: "+file
########################

"""
Right hand side for Hodgkin-Huxley.
"""
def rhs_hh(Vmhn, style='numpy'):
    # numpy takes [[v,m,h,n]]*N: shape=(n,4)
    # jax takes [[v,m,h,n]]*N: shape=(n,4)
    # jax1 takes [v,m,h,n]: shape=(4)
    # sympy only variables [[v,m,h,n]]
    if style=='numpy':
        exp = np.exp
        pow = np.power
        STATES = Vmhn.T
        array = np.array
    elif style=='sympy':
        exp = sp.exp
        pow = lambda x,y: x**y
        STATES = Vmhn[0]
        array = lambda a: np.empty(a,dtype=object)
    elif style=='jax':
        exp = jnp.exp
        pow = jnp.power
        STATES = Vmhn.T
        array = jnp.array
    elif style=='jax1':
        exp = jnp.exp
        pow = jnp.power
        STATES = Vmhn
        array = jnp.array
    else:
        raise RuntimeError(f"Unknown array style '{style}'")

    # copied from inputs/hodgkin_huxley_1952.c

    # init constants
    # STATES[0] = -75;
    CONSTANTS_0 = -75;
    CONSTANTS_1 = 1;
    CONSTANTS_2 = 0;
    CONSTANTS_3 = 120;
    # STATES[1] = 0.05;
    # STATES[2] = 0.6;
    CONSTANTS_4 = 36;
    # STATES[3] = 0.325;
    CONSTANTS_5 = 0.3;
    CONSTANTS_6 = CONSTANTS_0+115.000;
    CONSTANTS_7 = CONSTANTS_0 - 12.0000;
    CONSTANTS_8 = CONSTANTS_0+10.6130;

    # compute rates
    ALGEBRAIC_1 = ( - 0.100000*(STATES[0]+50.0000))/(exp(- (STATES[0]+50.0000)/10.0000) - 1.00000);
    ALGEBRAIC_5 =  4.00000*exp(- (STATES[0]+75.0000)/18.0000);
    RATES_1 =  ALGEBRAIC_1*(1.00000 - STATES[1]) -  ALGEBRAIC_5*STATES[1];
    ALGEBRAIC_2 =  0.0700000*exp(- (STATES[0]+75.0000)/20.0000);
    ALGEBRAIC_6 = 1.00000/(exp(- (STATES[0]+45.0000)/10.0000)+1.00000);
    RATES_2 =  ALGEBRAIC_2*(1.00000 - STATES[2]) -  ALGEBRAIC_6*STATES[2];
    ALGEBRAIC_3 = ( - 0.0100000*(STATES[0]+65.0000))/(exp(- (STATES[0]+65.0000)/10.0000) - 1.00000);
    ALGEBRAIC_7 =  0.125000*exp((STATES[0]+75.0000)/80.0000);
    RATES_3 =  ALGEBRAIC_3*(1.00000 - STATES[3]) -  ALGEBRAIC_7*STATES[3];
    ALGEBRAIC_0 =  CONSTANTS_3*pow(STATES[1], 3.00000)*STATES[2]*(STATES[0] - CONSTANTS_6);
    ALGEBRAIC_4 =  CONSTANTS_4*pow(STATES[3], 4.00000)*(STATES[0] - CONSTANTS_7);
    ALGEBRAIC_8 =  CONSTANTS_5*(STATES[0] - CONSTANTS_8);
    RATES_0 = - (- CONSTANTS_2+ALGEBRAIC_0+ALGEBRAIC_4+ALGEBRAIC_8)/CONSTANTS_1;

    RATES = array([RATES_0, RATES_1, RATES_2, RATES_3])
    return RATES.T

"""
build system matrix for diffusion term


make_laplace(4).todense()
matrix([[-2.,  1.,  0.,  0.],
        [ 1., -2.,  1.,  0.],
        [ 0.,  1., -2.,  1.],
        [ 0.,  0.,  1., -2.]])

make_laplace(4,bounds='skip').todense()
matrix([[ 0.,  0.,  0.,  0.],
        [ 1., -2.,  1.,  0.],
        [ 0.,  1., -2.,  1.],
        [ 0.,  0.,  0.,  0.]])

make_laplace(4,bounds='dirichlet').todense()
matrix([[ 1.,  0.,  0.,  0.],
        [ 1., -2.,  1.,  0.],
        [ 0.,  1., -2.,  1.],
        [ 0.,  0.,  0.,  1.]])

make_laplace(4,bounds='neumann').todense()
matrix([[-1.,  1.,  0.,  0.],
        [ 1., -2.,  1.,  0.],
        [ 0.,  1., -2.,  1.],
        [ 0.,  0.,  1., -1.]])

returns sparse matrix
"""
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

def make_inv_lumped_mass_matrix(N, hx):
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
        lumped_val = 0
        if i != 0:
            lumped_val += h(i-1) / 3 # φ_i|_left φ_i|_left
            lumped_val += h(i-1) / 6 # φ_i|_left φ_i-1|_right
        if i != N-1:
            lumped_val += h(i) / 3 # φ_i|_right φ_i|_right
            lumped_val += h(i) / 6 # φ_i|_right φ_i+1|_left
        rows.append(i)
        cols.append(i)
        vals.append(1/lumped_val)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(N,N))

def make_lumped_mass_matrix(N, hx):
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
        lumped_val = 0
        if i != 0:
            lumped_val += h(i-1) / 3 # φ_i|_left φ_i|_left
            lumped_val += h(i-1) / 6 # φ_i|_left φ_i-1|_right
        if i != N-1:
            lumped_val += h(i) / 3 # φ_i|_right φ_i|_right
            lumped_val += h(i) / 6 # φ_i|_right φ_i+1|_left
        rows.append(i)
        cols.append(i)
        vals.append(lumped_val)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(N,N))

def make_support_matrix(N, hx):
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
        val = 0
        if i != 0:
            val += h(i-1)
        if i != N-1:
            val += h(i)
        rows.append(i)
        cols.append(i)
        vals.append(val)
    # This is equivalent to the lumped mass matrix!
    return sparse.csr_matrix((vals, (rows, cols)), shape=(N,N))



"""
model parameters for diffusion term
"""
Conductivity = 3.828    # sigma, conductivity [mS/cm]
Am = 500.0              # surface area to volume ratio [cm^-1]
Cm = 0.58               # membrane capacitance [uF/cm^2]

#######################
# error analysis
#######################

def second_order_coefficients(j, u, x):
    xi = [xs[j-1], xs[j], xs[j+1]]
    ui = [u[j-1], u[j], u[j+1]]
    a = ui[0]*xi[1]*xi[2]/((xi[0]-xi[1])*(xi[0]-xi[2]))+ui[1]*xi[0]*xi[2]/((xi[1]-xi[0])*(xi[1]-xi[2]))+ui[2]*xi[0]*xi[1]/((xi[2]-xi[0])*(xi[2]-xi[1]))
    b = -(ui[0]*(xi[1]+xi[2])/((xi[0]-xi[1])*(xi[0]-xi[2]))+ui[1]*(xi[0]+xi[2])/((xi[1]-xi[0])*(xi[1]-xi[2]))+ui[2]*(xi[0]+xi[1])/((xi[2]-xi[0])*(xi[2]-xi[1])))
    c = ui[0]/((xi[0]-xi[1])*(xi[0]-xi[2]))+ui[1]/((xi[1]-xi[0])*(xi[1]-xi[2]))+ui[2]/((xi[2]-xi[0])*(xi[2]-xi[1]))
    return [a,b,c]

def first_order_coefficients(j, u, x):
    x = [x[j-1], x[j]]
    u = [u[j-1], u[j]]
    a = (u[1]*x[0]-u[0]*x[1])/(x[0]-x[1])
    b = (u[0]-u[1])/(x[0]-x[1])
    return [a, b]

#Sprungterme berechnen
def abl(x1, x2, v1, v2):
    return (v2-v1)/(x2-x1)

def calc_sprungterme(j, v, x, Nx):
    if(j == 1):
        sprung_links = 0
        sprung_rechts = abl(x[j-1],x[j],v[j-1],v[j]) - abl(x[j],x[j+1],v[j],v[j+1])
    if(j == Nx-1):
        sprung_rechts = 0
        sprung_links = abl(x[j-2],x[j-1],v[j-2],v[j-1]) - abl(x[j-1],x[j],v[j-1],v[j])
    if(j>1 and j< Nx-1):
        sprung_links = abl(x[j-2],x[j-1],v[j-2],v[j-1]) - abl(x[j-1],x[j],v[j-1],v[j])
        sprung_rechts = abl(x[j-1],x[j],v[j-1],v[j]) - abl(x[j],x[j+1],v[j],v[j+1])
    return [-sprung_links, -sprung_rechts]

#Fehlerabschätzung
def discretization_error(x, z_h, u1, u0, Nx):
    error = 0
    j = 1
    while j < Nx-1:
        #linkes Intervall 1
        [a1,b1] = first_order_coefficients(j, u1, x)
        [c1,d1] = first_order_coefficients(j, u0, x)
        [d1_,e1_] = first_order_coefficients(j, z_h, x)
        #rechtes Intervall 2
        [a2,b2] = first_order_coefficients(j+1, u1, x)
        [c2,d2] = first_order_coefficients(j+1, u0, x)
        [d2_,e2_] = first_order_coefficients(j+1, z_h, x)
        #2nd order polynom bilden
        [a_,b_,c_] = second_order_coefficients(j, z_h, x)
        #linke Seite
        k1_l = a1*a_ - a1*d1_ + a_*c1 - c1*d1_
        k2_l = a1*b_ - a1*e1_ + a_*b1 - b1*d1_ + b_*c1- c1*e1_ + a_*d1 - d1*d1_
        k3_l = a1*c_ + b1*b_ - b1*e1_ + c1*c_ + b_*d1 - d1*e1_
        k4_l = b1*c_ + c_*d1
        error_l = -k1_l*(x[j]-x[j-1]) - 1/2*k2_l*(x[j]**2-x[j-1]**2) - 1/3*k3_l*(x[j]**3-x[j-1]**3) - 1/4*k4_l*(x[j]**4-x[j-1]**4)
        #rechte Seite
        k1_r = a2*a_ - a2*d1_ + a_*c1 - c2*d2_
        k2_r = a2*b_ - a2*e2_ + a_*b2 - b2*d2_ + b_*c2- c2*e2_ + a_*d2 - d2*d2_
        k3_r = a2*c_ + b2*b_ - b2*e2_ + c2*c_ + b_*d2 - d2*e2_
        k4_r = b2*c_ + c_*d2
        error_r = -k1_r*(x[j+1]-x[j]) - 1/2*k2_r*(x[j+1]**2-x[j]**2) - 1/3*k3_r*(x[j+1]**3-x[j]**3) - 1/4*k4_r*(x[j+1]**4-x[j]**4)
        #2 Schritte zusammenfügen
        error += abs(error_l) + abs(error_r)
        j += 2
    return error

def iteration_error(x, z_h, u1, u0, Nx):
    error = 0
    for j in range(1, Nx):
        #erster Term
        [a,b] = first_order_coefficients(j,u1,x)
        [c,d] = first_order_coefficients(j,u0,x)
        [f,g] = first_order_coefficients(j,z_h,x)
        term1 = -(a*f+c*f)*(x[j]-x[j-1]) - 1/2*(b*f+d*f+a*g+c*g)*(x[j]**2-x[j-1]**2) - 1/3*(b*g+d*g)*(x[j]**3-x[j-1]**3)
        #zweiter term
        [sprung_l, sprung_r] = calc_sprungterme(j,u1,x,Nx)
        term2 = 1/2 * (sprung_l * z_h[j-1] + sprung_r * z_h[j])
        error += abs(term1+term2)
    return error

def error_analysis(x, V0, V1, ht, i):
    #duales Problem lösen: A^T z = (1,1,1,1,1)^T  --CG-->  z_h
    M = dual_problem_matrix(ht)
    J = np.ones(M.shape[1]) * hxs[2]
    J[0] = 0
    J[-1] = 0
    z_h = sparse.linalg.cg(M , J)[0]             #A^T z = (1,1,1,1,1)^T  --CG-->  z_h
    if i == 10:
        print('u1:     ', V1)
        print('sys matrix duales problem:')
        print('  '+arr2str(M.todense(), prefix='  '))
        print('z_h:     ', z_h)
        plt.plot(x, z_h)
        plt.show()
    #Diskretisierungsfehler berechnen
    eta_h = discretization_error(x, z_h, V1, V0, Nx)
    #Iterationsfehler berechnen
    eta_m = iteration_error(x, z_h, V1, V0, Nx)
    #print error
    print('Diskretisierungsfehler: ', eta_h, 'Iterationerror: ', eta_m)

#######################
# time stepping methods
#######################
def heun_step(Vmhn, rhs, t, ht):
    Vmhn0 = Vmhn + ht * rhs(Vmhn, t)
    Vmhn1 = Vmhn + ht * rhs(Vmhn0, t + ht)
    return (Vmhn0 + Vmhn1) / 2

"""
Crank-Nicolson step for Finite-Differences formulation
dt_linear: function(state, t) for rhs of the system
"""
def crank_nicolson_step(Vmhn0, dt_linear, t, ht, maxit=1000, eps=1e-10):
    Vmhn = Vmhn0.reshape(-1)
    linop_shape = (Vmhn.shape[0], Vmhn.shape[0])
    linop = sparse.linalg.LinearOperator(linop_shape, matvec=lambda x: dt_linear(x.reshape(Vmhn0.shape), t).reshape(-1))
    linop_cn = sparse.linalg.LinearOperator(linop_shape, matvec=lambda x: x - ht * linop * x)
    V0 = Vmhn + ht * linop * Vmhn
    # solve the linear system
    V1 = sparse.linalg.cg(linop_cn, Vmhn, maxiter=maxit, tol=eps)[0]
    return (V0 + V1).reshape(Vmhn0.shape)/2

"""
Crank-Nicolson step for Finite-Element formulation
dt_linear: tuple of 2 functions
             the first returns the system matrix for the explicit term,
             the second function returns the system matrix for the implicit term,
           system matrices will act only on the V-channel
"""
def crank_nicolson_FE_step(Vmhn0, sys_expl_impl, t, ht, i, maxit=1000, eps=1e-10):
    # get explicit and implicit system matrix
    cn_sys_expl, cn_sys_impl = sys_expl_impl    #cn_sys_impl = (eye - ht * mass_inv_stiffness)  ; cn_sys_expl = eye     (for implicit euler) 

    Vmhn0 = np.array(Vmhn0)

    # only act on V channel
    V0 = Vmhn0[:,0]             #V0
    A = cn_sys_impl(ht)
    rhs = cn_sys_expl(ht)*V0
    Vmhn0[:,0] = sparse.linalg.cg(A, rhs, maxiter=maxit, tol=eps)[0]        #V1
    error_analysis(xs, V0, Vmhn0[:,0], ht, i)
    return Vmhn0

"""
dt_linear: linear rhs function. Is applied to all channels
"""
def implicit_euler_step(Vmhn0, dt_linear, t, ht, maxit=1000, eps=1e-10):
    Vmhn = Vmhn0.reshape(-1)
    linop_shape = (Vmhn.shape[0], Vmhn.shape[0])
    # rhs lin op: flattened Vmhn -> Vmhn -> dt_linear -> dt Vmhn -> flatten dt Vmhn
    linop = sparse.linalg.LinearOperator(linop_shape, matvec=lambda x: dt_linear(x.reshape(Vmhn0.shape), t).reshape(-1))
    # system matrix
    linop_ie = sparse.linalg.LinearOperator(linop_shape, matvec=lambda x: x - ht * linop * x)
    # solve the linear system
    V1 = sparse.linalg.cg(linop_ie, Vmhn, maxiter=maxit, tol=eps)[0]
    return V1.reshape(Vmhn0.shape)

"""
repeats many steps of a time stepping method
integator: time step method
Vmhn0: initial values
dt: right hand side of system
t0: initial time
t1: end time
ht: time step width
traj=False: return only the final solution at t1
traj=True:  return intermediate solution for each time step t0+k*ht for k = 0,1,...
"""
def stepper(integator, Vmhn0, rhs, t0, t1, ht, traj=False, **kwargs):
    Vmhn = Vmhn0

    if not traj:
        result = Vmhn
    else:
        result = [Vmhn]

    n_steps = max(1, int((t1-t0)/ht + 0.5)) # round to nearest integer
    ht_ = (t1-t0) / n_steps

    for i in range(n_steps):
        print(i, '/', n_steps, 'timesteps')
        Vmhn = integator(Vmhn, rhs, t0+i*ht_, ht_, i, **kwargs)
        if not traj:
            result = Vmhn
        else:
            result.append(Vmhn)

    return np.asarray(result) # cast list to array if we store the trajectory

def heun(Vmhn0, rhs, t0, t1, ht, maxit=None, eps=None, traj=False):
    return stepper(heun_step, Vmhn0, rhs, t0, t1, ht, traj=traj)
def crank_nicolson(Vmhn0, rhs, t0, t1, ht, maxit=1000, eps=1e-10, traj=False):
    return stepper(crank_nicolson_step, Vmhn0, rhs, t0, t1, ht, maxit=maxit, eps=eps, traj=traj)
def crank_nicolson_FE(Vmhn0, rhs, t0, t1, ht, maxit=1000, eps=1e-10, traj=False):
    return stepper(crank_nicolson_FE_step, Vmhn0, rhs, t0, t1, ht, maxit=maxit, eps=eps, traj=traj)
def implicit_euler(Vmhn0, dt, t0, t1, ht, maxit=1000, eps=1e-10, traj=False):
    return stepper(implicit_euler_step, Vmhn0, dt, t0, t1, ht, maxit=maxit, eps=eps, traj=traj)

def make_strang_step(int0, ht0, int1, ht1, maxit=1000, eps=1e-10):
    def strang_step(Vmhn, rhs, t, ht):
        # solve the first equation for the first half time interval
        Vmhn = int0(Vmhn, rhs[0], t     , t+ht/2, ht0)
        # solve the second equation for one time interval
        Vmhn = int1(Vmhn, rhs[1], t     , t+ht  , ht1, maxit, eps)
        # solve the first equation for the second half time interval
        Vmhn = int0(Vmhn, rhs[0], t+ht/2, t+ht  , ht0)
        return Vmhn
    return strang_step

def godunov_step_1H_1CN(Vmhn0, rhs_reaction, system_matrices_expl_impl, t, ht, **kwargs):
    # 1 interval for reaction term with Heun
    Vmhn = heun_step(Vmhn0, rhs_reaction, t, ht)
    # 1 interval for diffusion with Crank-Nicolson
    Vmhn = crank_nicolson_FE_step(Vmhn, system_matrices_expl_impl, t, ht, **kwargs)

    return Vmhn

def strang_step_1H_1CN_FE(Vmhn0, rhs, t, ht, i, **kwargs):
    # unpack rhs for each component
    rhs_reaction, system_matrices_expl_impl = rhs

    # 1/2 interval for reaction term with Heun
    Vmhn = heun_step(Vmhn0, rhs_reaction, t, ht/2)
    # 1 interval for diffusion with Crank-Nicolson
    Vmhn = crank_nicolson_FE_step(Vmhn, system_matrices_expl_impl, t, ht, i, **kwargs)
    # 1/2 interval for reaction term with Heun
    Vmhn = heun_step(Vmhn, rhs_reaction, t+ht/2, ht/2)

    return Vmhn

def strang_1H_1CN_FE(Vmhn, rhs0, system_matrices_expl_impl, t0, t1, hts, maxit=1000, eps=1e-10, traj=False):
    return stepper(strang_step_1H_1CN_FE, Vmhn, (rhs0, system_matrices_expl_impl), t0, t1, hts, maxit=maxit, eps=eps, traj=traj)

def strang_H_CN(Vmhn, rhs0, rhs1, t0, t1, ht0, ht1, hts, maxit=1000, eps=1e-10, traj=False):
    int0 = heun
    int1 = crank_nicolson
    return stepper(make_strang_step(int0, ht0, int1, ht1, maxit, eps), Vmhn, [rhs0,rhs1], t0, t1, hts, traj=traj)

def strang_H_CN_FE(Vmhn, rhs0, rhs1, t0, t1, ht0, ht1, hts, maxit=1000, eps=1e-10, traj=False):
    int0 = heun
    int1 = crank_nicolson_FE
    return stepper(make_strang_step(int0, ht0, int1, ht1, maxit, eps), Vmhn, [rhs0,rhs1], t0, t1, hts, traj=traj)

def strang_H_IE(Vmhn, rhs0, rhs1, t0, t1, ht0, ht1, hts, maxit=1000, eps=1e-10, traj=False):
    int0 = heun
    int1 = implicit_euler
    return stepper(make_strang_step(int0, ht0, int1, ht1, maxit, eps), Vmhn, [rhs0,rhs1], t0, t1, hts, traj=traj)

def coupled_EE_FE(Vmhn0, Minv, Minv_A, hh, t0, t1, ht, traj=False):
    """
    Minv should be a lumped mass matrix Mbarinv
    """
    Minv_Mbar = lambda x: x

    Vmhn = Vmhn0

    if not traj:
        result = Vmhn
    else:
        result = [Vmhn]

    n_steps = max(1, int((t1-t0)/ht + 0.5)) # round to nearest integer
    ht_ = (t1-t0) / n_steps

    for i in range(n_steps):
        Vmhn_ = np.array(Vmhn)
        hh_Vmhn = hh(Vmhn, i*ht_)

        """
        ⟨u⁺, φⱼ⟩ = ⟨u, φⱼ⟩ + h a ⟨∇u, ∇φⱼ⟩ + h ⟨f(u), φⱼ⟩ ∀ⱼ

        u =  ∑ ūᵢ φᵢ

        M ū⁺ = M ū + h A ū + h F

        Fⱼ =  ⟨f(∑ ūᵢ φᵢ), φⱼ⟩    // now: trapezoidal rule
           = ½ h₋ fⱼ₋₁ φⱼ(xⱼ₋₁) + ½ (h₋ + h₊) fⱼ φⱼ(xⱼ) + ½ h₊ fⱼ₊₁ φⱼ(xⱼ₊₁)
           =           0        + ½ (h₋ + h₊) fⱼ 1      +           0
           = ½ (h₋ + h₊) fⱼ 1
           = M̄ⱼ fⱼ    where M̄ is equivalent to the lumped mass matrix

        ⇒ M ū⁺ = M ū + h A ū + h M̄ fⱼ
        ⇔ ū⁺ = ū + h M⁻¹ A ū + h M⁻¹ M̄ fⱼ
             = ū + h M⁻¹ A ū + h fⱼ    if we use mass lumping for the inverse


        M V+ = V + h A V + h M̄ HH0(Vmnh)
        M m+ = m         + h M̄ HH1(Vmnh)
        M n+ = n         + h M̄ HH2(Vmnh)
        M h+ = h         + h M̄ HH3(Vmnh)
        """
        Vmhn_[:,0] += ht_ * Minv_A @ Vmhn[:,0]
        Vmhn_[:,0] += ht_ * Minv_Mbar(hh_Vmhn[:,0])
        Vmhn_[:,1] += ht_ * Minv_Mbar(hh_Vmhn[:,1])
        Vmhn_[:,2] += ht_ * Minv_Mbar(hh_Vmhn[:,2])
        Vmhn_[:,3] += ht_ * Minv_Mbar(hh_Vmhn[:,3])

        Vmhn = Vmhn_

        if not traj:
            result = Vmhn
        else:
            result.append(Vmhn)

    return np.asarray(result) # cast list to array if we store the trajectory

def error_est_test():
    Nx = 200; Ne = Nx - 1
    Ne = 20; Nx = Ne + 1
    h = 1 / Ne
    print(Nx, Ne, h)
    laplace = make_laplace(Nx, h, bounds='dirichlet')
    print(laplace)
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

    t1 = 0
    t2 = 0
    t3 = 0
    for ie in range(Ne):
        t1 += h * (z[ie] + z[ie+1])/2
        if ie != 0:
            t2 += .5 * jumps_u[ie - 1] * z[ie]
        if ie != Ne - 1:
            t3 += .5 * jumps_u[ie] * z[ie+1]

    print(t1, t2, t3)
    print(t1 + t2 + t3)
# error_est_test()


# if False and __name__ == '__main__':
if __name__ == '__main__':
    initial_value_file = sys.argv[1]

    tend = float(sys.argv[2])
    hts = float(sys.argv[3])
    if len(sys.argv) > 4:
        ht0 = hts / int(sys.argv[4])
        ht1 = hts / int(sys.argv[5])
    else:
        # strang splitting with one step for each component
        ht0 = hts / 2 # first and last step in strang splitting
        ht1 = hts / 1 # central step in strang splitting


    if initial_value_file != '':
        print('initial values:', initial_value_file)
        xyz = extractX(initial_value_file, 'geometry', 'xyz')['val'] # fiber location in space
        Vmhn0 = extract_channels(initial_value_file)['val'] # initial values
        hxs = np.sum((xyz[1:,:] - xyz[:-1,:])**2, axis=1)**.5
        xs = np.zeros(hxs.shape[0] + 1) # 1D location
        xs[1:] = np.cumsum(hxs)
        print("Loaded fiber")
    else:
        xs = np.linspace(0,11.9, 1191)
        hxs = xs[1:] - xs[:-1]
        Vmhn0 = np.zeros((1191, 4))
        Vmhn0[:,0] = -75.0,
        Vmhn0[:,1] =   0.05,
        Vmhn0[:,2] =   0.6,
        Vmhn0[:,3] =   0.325,
        # initial acivation
        Vmhn0[1191//2 - 3 : 1191//2 + 3, 0] = 50
        print("Created fiber")
    Nx = xs.shape[0]
    print(f"  length: {xs[-1]:>5.2f}cm")
    print(f"  nodes:  {Nx:>4}")
    print(f"Initial values:")
    print(f"  V: {np.min(Vmhn0[:,0]):>+.2e} -- {np.max(Vmhn0[:,0]):>+.2e}")
    print(f"  m: {np.min(Vmhn0[:,1]):>+.2e} -- {np.max(Vmhn0[:,1]):>+.2e}")
    print(f"  h: {np.min(Vmhn0[:,2]):>+.2e} -- {np.max(Vmhn0[:,2]):>+.2e}")
    print(f"  n: {np.min(Vmhn0[:,3]):>+.2e} -- {np.max(Vmhn0[:,3]):>+.2e}")
    print(f"Model parameters:")
    print(f"  sigma: {Conductivity:>7.3f}")
    print(f"  Am:    {Am:>7.3f}")
    print(f"  Cm:    {Cm:>7.3f}")

    def arr2str(arr, **kwargs):
        return np.array2string(arr, formatter={'float_kind': lambda x: '{:+.2e}'.format(x)}, **kwargs).replace('+0.00e+00', '    -    ')

    # cite opendihu in crank_nicolson.tpp: "compute the system matrix (I - dt*M^{-1}K) where M^{-1} is the lumped mass matrix"
    # matrices marked with `[opendihu: ...]` have been checked to match the opendihu matrices on 18.06.2021 with
    #    env PETSC_OPTIONS="-mat_view ascii" ./multiple_fibers_CN ../settings_multiple_fibers.py --nt_0D 1 --nt_1D 1 --dt_splitting 1e-3 | less
    # and the simulation outputs match in picture-norm.
    # The effect, that the V-channel increases just before the end of the fibers is also present in the opendihu results.
    # opendihu commit: 44cadd4060552f6d1ad4e89153f37d1b843800da
    laplace = make_laplace(Nx, hxs, bounds='neumann')
    print("Laplace")
    print('  '+arr2str(laplace.todense(), prefix='  '))
    prefactor = Conductivity / Am / Cm # mS / uF = [mS/cm] / ([cm^-1] * [uF/cm^2])
    # we have _negative_ laplace on rhs
    stiffness = -prefactor * laplace
    print("Stiffness \033[90m[opendihu: `Mat Object: stiffnessMatrix`]\033[m")
    print('  '+arr2str(stiffness.todense(), prefix='  '))
    mass_inv = make_inv_lumped_mass_matrix(Nx, hxs)
    print("Mass^-1 (lumped) \033[90m[opendihu: `Mat Object: inverseLumpedMassMatrix`]\033[m")
    print('  '+arr2str(mass_inv.todense(), prefix='  '))
    mass_inv_stiffness = mass_inv @ stiffness
    print("Mass^-1 Stiffness")
    print('  '+arr2str(mass_inv_stiffness.todense(), prefix='  '))
    mass = make_lumped_mass_matrix(Nx, hxs)
    mass[0,0] = 0
    mass[-1,-1] = 0
    print("mass matrix")
    print('  '+arr2str(mass.todense(), prefix='  '))


    # system matrices for crank_nicolson
    @lru_cache(maxsize=8)
    def cn_sys_expl(ht):
        print(f"> build expl. CN matrix for ht={ht} \033[90m[opendihu: crank_nicolson.tpp:setIntegrationMatrixRightHandSide]\033[m")
        return sparse.eye(Nx) + 0.5 * ht * mass_inv_stiffness
    @lru_cache(maxsize=8)
    def cn_sys_impl(ht):
        print(f"> build impl. CN matrix for ht={ht} \033[90m[opendihu: crank_nicolson.tpp:setSystemMatrix]\033[m")
        return sparse.eye(Nx) - 0.5 * ht * mass_inv_stiffness
    # system matrices for implicit euler
    @lru_cache(maxsize=8)
    def ie_sys_expl(ht):
        print(f"> build expl. IE matrix for ht={ht} \033[90m\033[m")
        return sparse.eye(Nx)
    @lru_cache(maxsize=8)
    def ie_sys_impl(ht):
        print(f"> build impl. IE matrix for ht={ht} \033[90m\033[m")
        return sparse.eye(Nx) - ht * mass_inv_stiffness

    def rhs_hodgkin_huxley(Vmhn, t):
        return rhs_hh(Vmhn)
    
    def dual_problem_matrix(ht):
        M = mass + ht * stiffness
        return M.T

    # Solve the equation

    time_discretization=dict(
        t0=0, t1=tend, # start and end time of the simulation
        hts=hts, # time step width of the splitting method
        ht0=ht0, # time step width for the reaction term (Hodgkin-Huxley)
        ht1=ht1, # time step width for the diffusion term
        eps=1e-12, # stopping criterion for the linear solver in the diffusion step
        maxit=1000, # maximum number of iterations for the implicit solver
    )

    #method = 'opendihu'
    #method = 'opendihu1'
    #method = 'FE-EE'
    method = 'H-IE'

    if method == 'opendihu':
        # solve reaction (Hodgkin-Huxley) with Heun's method (explicit, second order)
        # solve diffusion with Crank-Nicolson (implicit, second order)
        # combine both with strang splitting
        trajectory2 = strang_H_CN_FE(
            Vmhn0, # initial values
            rhs0=rhs_hodgkin_huxley,
            rhs1=(cn_sys_expl, cn_sys_impl),
            traj=True, # return full trajectory
            **time_discretization
        )
    elif method == 'opendihu1':
        # equivalent, but with easier implementation
        trajectory = strang_1H_1CN_FE(
            Vmhn0,
            rhs_hodgkin_huxley,
            (cn_sys_expl, cn_sys_impl),
            time_discretization['t0'],
            time_discretization['t1'],
            time_discretization['hts'],
            traj=True,
            eps=time_discretization['eps'],
            maxit=time_discretization['maxit'],
        )
    elif method == 'FE-EE':
        # stable dt=3.6e-3, unstable dt=3.7e-3
        trajectory = coupled_EE_FE(Vmhn0, mass_inv, mass_inv_stiffness, rhs_hodgkin_huxley,
            time_discretization['t0'],
            time_discretization['t1'],
            time_discretization['hts'],
            traj=True
        )
    elif method == 'H-IE':
        # reuse CN code by replacing the implicit and explicit system matrices
        trajectory = strang_1H_1CN_FE(
            Vmhn0,
            rhs_hodgkin_huxley,
            (ie_sys_expl, ie_sys_impl),
            time_discretization['t0'],
            time_discretization['t1'],
            time_discretization['hts'],
            traj=True,
            eps=time_discretization['eps'],
            maxit=time_discretization['maxit'],
        )

    # indices: trajectory[time_index, point alog x-axis, variable]
    out_stride = 1
    np.save("out.npy",trajectory[::out_stride, :, :])

    ######## plot results
    time_steps = trajectory.shape[0]
    step_stride = 400
    cs = np.linspace(0,1, time_steps // step_stride + 1)
    import matplotlib.pyplot as plt
    fig = plt.figure()

    ax = fig.add_subplot(411)
    # plot the transmembrane voltage
    ax.plot(xs, trajectory[::-step_stride, :, 0].T)
    for idx,line in enumerate(ax.lines): line.set_color((cs[idx], 0.5, 0.5))
    ax.plot(xs, trajectory[0, :, 0], '--', color='black')
    ax.plot(xs, trajectory[-1, :, 0], color='black')

    ax = fig.add_subplot(412)
    ax.plot(xs, trajectory[::-step_stride, :, 1].T)
    for idx,line in enumerate(ax.lines): line.set_color((cs[idx], 0.5, 0.5))
    ax.plot(xs, trajectory[0, :, 1], '--', color='black')
    ax.plot(xs, trajectory[-1, :, 1], color='black')

    ax = fig.add_subplot(413)
    ax.plot(xs, trajectory[::-step_stride, :, 2].T)
    for idx,line in enumerate(ax.lines): line.set_color((cs[idx], 0.5, 0.5))
    ax.plot(xs, trajectory[0, :, 2], '--', color='black')
    ax.plot(xs, trajectory[-1, :, 2], color='black')

    ax = fig.add_subplot(414)
    ax.plot(xs, trajectory[::-step_stride, :, 3].T)
    for idx,line in enumerate(ax.lines): line.set_color((cs[idx], 0.5, 0.5))
    ax.plot(xs, trajectory[0, :, 3], '--', color='black')
    ax.plot(xs, trajectory[-1, :, 3], color='black')

    plt.show()
