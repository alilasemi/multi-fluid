import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import scipy.optimize

def exact_solution(
        r4, p4, u4, v4, r1, p1, u1, v1, g, t_list):
    make_plots = True
    # Domain
    n_points = 1601
    xL = -10
    x = np.linspace(xL, -xL, n_points)

    for t in t_list:

        # Compute speed of sound
        def compute_c(g, p, r): return np.sqrt(g * p / r)
        c1 = compute_c(g, p1, r1)
        c4 = compute_c(g, p4, r4)

        # Compute the pressure ratio
        def p_rhs(p2p1, u1, u4, c1, c4, p1, p4, g):
            return p2p1 * (
                1 + (g - 1) / (2 * c4) * (
                    u4 - u1 - (c1/g) * (
                        (p2p1 - 1) /
                        np.sqrt(((g+1) / (2 * g)) * (p2p1 - 1) + 1)
                    )
                )
            )**(-(2 * g) / (g - 1)) - p4/p1

        if (p1 > p4):
            guess = p4/p1
        else:
            guess = p1/p4
        guesses = np.linspace(0, np.max([p1/p4, p4/p1]), 100)
        success = False
        for guess in guesses:
            p2p1, infofict, ier, msg = scipy.optimize.fsolve(p_rhs,
                    guess, args=(u1, u4, c1, c4, p1, p4, g),
                    full_output=True)
            if ier == 1:
                success = True
                break
        p2 = p2p1 * p1

        # Compute u2
        u2 = u1 + (c1 / g) * (p2p1-1) / (np.sqrt( ((g+1)/(2*g)) * (p2p1-1) + 1))
        # Compute V
        V = u1 + c1 * np.sqrt( ((g+1)/(2*g)) * (p2p1-1) + 1)
        # Compute c2
        c2 = c1 * np.sqrt(
                p2p1 * (
                    (((g+1)/(g-1)) + p2p1
                    ) / (
                    1 + ((g+1)/(g-1)) * p2p1)
                    )
        )
        # Compute r2
        def compute_r(g, p, c): return g * p / (c**2)
        r2 = compute_r(g, p2, c2)

        # p and u same across contact
        u3 = u2
        p3 = p2
        # Compute c3
        c3 = .5 * (g - 1) * (u4 + ((2*c4)/(g-1)) - u3)
        # Compute r3
        r3 = compute_r(g, p3, c3)

        # Flow inside expansion
        u_exp = (2/(g+1)) * (x/t + ((g-1)/2) * u4 + c4)
        c_exp = (2/(g+1)) * (x/t + ((g-1)/2) * u4 + c4) - x/t
        # Clip the speed of sound to be positive. This is not entirely necessary
        # (the spurious negative speed of sound is only outside the expansion,
        # so in the expansion everything is okay) but not doing this makes Numpy
        # give warnings when computing pressure.
        c_exp[c_exp < 0] = 1e-16
        p_exp = p4 * (c_exp/c4)**(2*g/(g-1))
        r_exp = compute_r(g, p_exp, c_exp)

        # Figure out which flow region each point is in
        r = np.empty_like(x)
        u = np.empty_like(x)
        p = np.empty_like(x)
        for i in range(n_points):
            xt = x[i] / t
            # Left of expansion
            if xt < (u4 - c4):
                r[i] = r4
                u[i] = u4
                p[i] = p4
            # Inside expansion
            elif xt < (u3 - c3):
                r[i] = r_exp[i]
                u[i] = u_exp[i]
                p[i] = p_exp[i]
            # Right of expansion
            elif xt < u3:
                r[i] = r3
                u[i] = u3
                p[i] = p3
            # Left of shock
            elif xt < V:
                r[i] = r2
                u[i] = u2
                p[i] = p2
            # Right of shock
            elif xt > V:
                r[i] = r1
                u[i] = u1
                p[i] = p1

        # Write to file
        with open(f'data/r_exact_t_{t}.npy', 'wb') as f:
            np.save(f, r)
        with open(f'data/u_exact_t_{t}.npy', 'wb') as f:
            np.save(f, u)
        with open(f'data/p_exact_t_{t}.npy', 'wb') as f:
            np.save(f, p)
        with open(f'data/x_exact_t_{t}.npy', 'wb') as f:
            np.save(f, x)

    if make_plots:
        # Plot
        rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
        rc('text', usetex=True)

        # rho
        fig = plt.figure(figsize=(4,4))
        plt.plot(x, r, 'k', linewidth=3, label='Exact')
        plt.plot(x, r2*np.ones_like(x), 'r', linewidth=1.5, label='Exact')
        plt.plot(x, r3*np.ones_like(x), 'r', linewidth=1.5, label='Exact')
        plt.xlabel('$x$ (m)', fontsize=20)
        plt.ylabel('$\\rho$ (kg/m$^3$)', fontsize=20)
        plt.tick_params(labelsize=12)
        plt.grid(linestyle='--')
        plt.savefig('r_exact.pdf', bbox_inches='tight')
        print(f'rho_2 = {r2}')
        print(f'rho_3 = {r3}')
        # u
        fig = plt.figure(figsize=(4,4))
        plt.plot(x, u, 'k', linewidth=3, label='Exact')
        plt.xlabel('$x$ (m)', fontsize=20)
        plt.ylabel('$u$ (m/s)', fontsize=20)
        plt.tick_params(labelsize=12)
        plt.grid(linestyle='--')
        plt.savefig('u_exact.pdf', bbox_inches='tight')
        print(f'u_2 = {u2}')
        print(f'u_3 = {u3}')
        # p
        fig = plt.figure(figsize=(4,4))
        plt.plot(x, p, 'k', linewidth=3, label='Exact')
        plt.xlabel('$x$ (m)', fontsize=20)
        plt.ylabel('$p$ (N/m$^2$)', fontsize=20)
        plt.tick_params(labelsize=12)
        plt.grid(linestyle='--')
        plt.savefig('p_exact.pdf', bbox_inches='tight')
        print(f'p_2 = {p2}')
        print(f'p_3 = {p3}')
        plt.show()

if __name__ == "__main__":
    #r4, u4, v4, p4 = 1000, 0, 0, 1e5
    #r1, u1, v1, p1 = 1, 0, 0, .02e5
    r1, u1, v1, p1 = .125, 50, 0, 1e4
    r4, u4, v4, p4 = 1, 100, 0, 1e5
    g = 1.4
    t_list = [.01]
    exact_solution(r4, p4, u4, v4, r1, p1, u1, v1, g, t_list)
