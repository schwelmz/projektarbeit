import numpy as np
import sys
import scipy.sparse as sparse
from functools import lru_cache
import scipy.sparse.linalg
import matplotlib.pyplot as plt

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

def stepper(integator, Vmhn0, rhs, t0, t1, ht, traj=False, **kwargs):
    Vmhn = Vmhn0

    if not traj:
        result = Vmhn
    else:
        result = [Vmhn]

    n_steps = max(1, int((t1-t0)/ht + 0.5)) # round to nearest integer
    ht_ = (t1-t0) / n_steps

    for i in range(n_steps):
        #print('timestep: ', i, '/', n_steps)
        Vmhn = integator(Vmhn, rhs, t0+i*ht_, ht_, i, **kwargs)
        if not traj:
            result = Vmhn
        else:
            result.append(Vmhn)
        if i > 2:
            break

    return np.asarray(result) # cast list to array if we store the trajectory

def crank_nicolson_FE_step(Vmhn0, sys_expl_impl, t, ht, ti, maxit=1000, eps=1e-10):
    # get explicit and implicit system matrix
    cn_sys_expl, cn_sys_impl = sys_expl_impl    #cn_sys_impl = (eye - ht * mass_inv_stiffness)  ; cn_sys_expl = eye     (for implicit euler) 
    Vmhn0 = np.array(Vmhn0)

    #store the old iterate
    V_alt = np.array(Vmhn0)[:,0]

    # only act on V channel
    V0 = Vmhn0[:,0]             #V0
        
    Vmhn0[:,0] = sparse.linalg.cg(cn_sys_impl(ht), cn_sys_expl(ht)*V0, maxiter=maxit, tol=eps)[0]        #V1

    return Vmhn0

def heun_step(Vmhn, rhs, t, ht):
    Vmhn0 = Vmhn + ht * rhs(Vmhn, t)
    Vmhn1 = Vmhn + ht * rhs(Vmhn0, t + ht)
    return (Vmhn0 + Vmhn1) / 2

def strang_step_1H_1CN_FE(Vmhn0, rhs, t, ht, ti, **kwargs):
    # unpack rhs for each component
    rhs_reaction, system_matrices_expl_impl = rhs

    # 1/2 interval for reaction term with Heun
    Vmhn = heun_step(Vmhn0, rhs_reaction, t, ht/2)
    # 1 interval for diffusion with Crank-Nicolson
    Vmhn = crank_nicolson_FE_step(Vmhn, system_matrices_expl_impl, t, ht, ti, **kwargs)
    # 1/2 interval for reaction term with Heun
    Vmhn = heun_step(Vmhn, rhs_reaction, t+ht/2, ht/2)

    return Vmhn

def strang_1H_1CN_FE(Vmhn, rhs0, system_matrices_expl_impl, t0, t1, hts, maxit=1000, eps=1e-10, traj=False):
    return stepper(strang_step_1H_1CN_FE, Vmhn, (rhs0, system_matrices_expl_impl), t0, t1, hts, maxit=maxit, eps=eps, traj=traj)

def second_order_coefficients(j, u, x):
    xi = [x[j-1], x[j], x[j+1]]
    ui = [u[j-1], u[j], u[j+1]]
    a = ui[0]*xi[1]*xi[2]/((xi[0]-xi[1])*(xi[0]-xi[2]))+ui[1]*xi[0]*xi[2]/((xi[1]-xi[0])*(xi[1]-xi[2]))+ui[2]*xi[0]*xi[1]/((xi[2]-xi[0])*(xi[2]-xi[1]))
    b = -(ui[0]*(xi[1]+xi[2])/((xi[0]-xi[1])*(xi[0]-xi[2]))+ui[1]*(xi[0]+xi[2])/((xi[1]-xi[0])*(xi[1]-xi[2]))+ui[2]*(xi[0]+xi[1])/((xi[2]-xi[0])*(xi[2]-xi[1])))
    c = ui[0]/((xi[0]-xi[1])*(xi[0]-xi[2]))+ui[1]/((xi[1]-xi[0])*(xi[1]-xi[2]))+ui[2]/((xi[2]-xi[0])*(xi[2]-xi[1]))
    return [a,b,c]  #a+b*x+c*x²

def first_order_coefficients(j, u, x):
    x = [x[j], x[j+1]]
    u = [u[j], u[j+1]]
    a = (u[1]*x[0]-u[0]*x[1])/(x[0]-x[1])
    b = (u[0]-u[1])/(x[0]-x[1])
    return [a, b]   #a+b*x

def calc_disc_error_exact(V_approx, V_exact, h, h_exact, mult):
    N_exact = 6337
    N_approx=101
    N_interval=2
    for i in range(1,mult):
        N_approx = N_approx*2-1
        N_interval = N_interval*2-1
    
    V_approx_filled = np.zeros(N_exact)
    for j in range(0, N_approx-1):
        [a,b] = first_order_coefficients(j, V_approx, xs)
        for k in range(0, N_interval-1):
            V_approx_filled[j+k] = a + b*xs[]

    error = 0
    for i in range(0, N_exact-1):
        error1 += (V_approx[i+1]+V_approx[i]-V_exact[i+1]-V_exact[i])/2*h

    error1 = 0
    error2 = 0
    for i in range(0, Nx-1):
        error1 += (V_approx[i+1]+V_approx[i])/2*h
    for j in range (0,3999):
        error2 += (V_exact[i+1]+V_exact[i])/2*h_exact
    return abs(error1-error2)

def calc_disc_error(u0,u1,z,x):
    error = 0
    ix = 1
    while (ix <= Nx-1):
        #Interpolation 2.Ordnung
        [a2,b2,c2] = second_order_coefficients(ix, z, x)
        #left side
        [a0,b0] = first_order_coefficients(ix-1, u0, x)
        [a1,b1] = first_order_coefficients(ix-1, u1, x)
        [a3,b3] = first_order_coefficients(ix-1, z, x)
        t1 = (a0-a1)*(a2-a3)*(x[ix]-x[ix-1]) + 1/2*((a0-a1)*(b2-b3)+(b0-b1)*(a2-a3))*(x[ix]**2-x[ix-1]**2) + 1/3*((a0-a1)*c2+(b0-b1)*(b2-b3))*(x[ix]**3-x[ix-1]**3) + 1/4*(b0-b1)*c2*(x[ix]**4-x[ix-1]**4)
        #right side
        [a0,b0] = first_order_coefficients(ix, u0, x)
        [a1,b1] = first_order_coefficients(ix, u1, x)
        [a3,b3] = first_order_coefficients(ix, z, x)
        t2 = (a0-a1)*(a2-a3)*(x[ix+1]-x[ix]) + 1/2*((a0-a1)*(b2-b3)+(b0-b1)*(a2-a3))*(x[ix+1]**2-x[ix]**2) + 1/3*((a0-a1)*c2+(b0-b1)*(b2-b3))*(x[ix+1]**3-x[ix]**3) + 1/4*(b0-b1)*c2*(x[ix+1]**4-x[ix]**4)
        #update
        error += t1+t2
        ix += 2
    return abs(error)

disc_error_exact_list = []
disc_error_est_list = []
nodes_list = []
#discretization error test
done = False
for mult in range(0,6):
    if mult == 0:
        N = 6337
        print('Calculating "exact" solution...')
    else:
        N=101
        for i in range(1,mult):
            N = N*2-1
        print('Calculating "exact" discretization error for ',N,' nodes')
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

    Conductivity = 3.828    # sigma, conductivity [mS/cm]
    Am = 500.0              # surface area to volume ratio [cm^-1]
    Cm = 0.58               # membrane capacitance [uF/cm^2]

    xs = np.linspace(0,11.9, N)
    hxs = xs[1:] - xs[:-1]
    Vmhn0 = np.zeros((N, 4))
    Vmhn0[:,0] = -75.0,
    Vmhn0[:,1] =   0.05,
    Vmhn0[:,2] =   0.6,
    Vmhn0[:,3] =   0.325,
    # initial acivation
    Vmhn0[(N)//2 - 3 : (N)//2 + 3, 0] = 50
    print("Created fiber")
    Nx = xs.shape[0]

    laplace = make_laplace(Nx, hxs, bounds='neumann')
    prefactor = Conductivity / Am / Cm
    stiffness = -prefactor * laplace
    mass_inv = make_inv_lumped_mass_matrix(Nx, hxs)
    mass_inv_stiffness = mass_inv @ stiffness

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

    #Solve the dual problem
    Ne = Nx-1
    h = (xs[-1] - xs[0])/Ne
    ht = hts
    mass = sparse.linalg.inv(make_inv_lumped_mass_matrix(Nx-2, hxs[1:-1]))
    mass[0,0] = 1.5 * h
    mass[-1,-1] = 1.5 * h
    lap = -make_laplace(Nx, hxs, bounds = 'neumann')[1:-1,1:-1]
    lhs = mass + prefactor * ht * lap
    rhs = np.ones(Nx-2)*h
    z = sparse.linalg.cg(lhs.T, rhs)
    assert z[1] == 0; z = z[0]
    z = np.insert(z, 0, z[0])
    z = np.insert(z, -1, z[-1])

    time_discretization=dict(
        t0=0, t1=tend, # start and end time of the simulation
        hts=hts, # time step width of the splitting method
        ht0=ht0, # time step width for the reaction term (Hodgkin-Huxley)
        ht1=ht1, # time step width for the diffusion term
        eps=1e-12, # stopping criterion for the linear solver in the diffusion step
        maxit=1000, # maximum number of iterations for the implicit solver
    )
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

    if N == 6337:
        V_exact = trajectory[1][:,0]
        h_exact = hxs[1]
        N = 100
    else:
        V0 = trajectory[0][:,0]
        V = trajectory[1][:,0]
        disc_error_exact = calc_disc_error_exact(V,V_exact,hxs[1], h_exact, mult)

        #estimate the discretization error
        disc_error_est = calc_disc_error(V0, V, z, xs)

        disc_error_exact_list.append(disc_error_exact)
        disc_error_est_list.append(disc_error_est)
        nodes_list.append(N)

print("Diskretisierungsfehler (exact) : ",disc_error_exact_list)
print("Diskretiserungsfehler (estimation): ", disc_error_est_list)
plt.plot(np.array(nodes_list), np.array(disc_error_exact_list), label="exact")
plt.plot(np.array(nodes_list), np.array(disc_error_est_list), label="estimation")
plt.yscale('log', base=2)
#plt.xscale('log', base=2)
plt.show()