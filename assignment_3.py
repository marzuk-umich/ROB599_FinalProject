import numpy as np
import matplotlib.pyplot as plt
from assignment_3_helper import LCPSolve, assignment_3_render

# DEFINE GLOBAL PARAMETERS
L = 0.4
MU = 0.3
EP = 0.5
dt = 0.01
m = 0.3
g = np.array([0., -9.81, 0.])
rg = 1./12. * (2 * L * L) #TODO: Rename this to rg_squared since it is $$r_g^2$$ - Do it also in the master
M = np.array([[m, 0, 0], [0, m, 0], [0, 0, m * rg]])
Mi = np.array([[1./m, 0, 0], [0, 1./m, 0], [0, 0, 1./(m * rg)]])
DELTA = 0.001
T = 150

def get_contacts(q):
    """
        Return jacobian of the lowest corner of the square and distance to contact
        :param q: <np.array> current configuration of the object
        :return: <np.array>, <float> jacobian and distance
    """
    # ------------------------------------------------
    # FILL WITH YOUR CODE
    half = L / 2 
    corner = np.array([[half, half], [-half, half], [-half, -half], [half, -half]])
    rot_mat = np.array([[np.cos(q[-1]), -np.sin(q[-1])], [np.sin(q[-1]), np.cos(q[-1])]])
    rot_corner = q[0:2] + np.dot(corner, rot_mat.T)
    idx = np.argmin(rot_corner[:, 1])
    lower = rot_corner[idx]
    phi = lower[1]
    r = lower - q[0:2]
    J_t = np.array([1, 0, -r[1]])
    J_n = np.array([0, 1, r[0]])  
    jac = np.column_stack((J_t, J_n)) # TODO: Replace None with your result
    # ------------------------------------------------
    return jac, phi

def form_lcp(jac, v):
    """
        Return LCP matrix and vector for the contact
        :param jac: <np.array> jacobian of the contact point
        :param v: <np.array> velocity of the center of mass
        :return: <np.array>, <np.array> V and p
    """
    # ------------------------------------------------
    # FILL WITH YOUR CODE
    Jt = jac[:,0]
    Jn = jac[:,1]
    fe = m * g
    V = np.zeros((4, 4))  # TODO: Replace None with your result
    V[0] = [Jn.T @ np.linalg.inv(M) @ Jn * dt, -Jn.T @ np.linalg.inv(M) @ Jt * dt, Jn.T @ np.linalg.inv(M) @ Jt * dt, 0]
    V[1] = [-Jt.T @ np.linalg.inv(M) @ Jn * dt, Jt.T @ np.linalg.inv(M) @ Jt * dt, -Jt.T @ np.linalg.inv(M) @ Jt * dt, 1]
    V[2] = [Jt.T @ np.linalg.inv(M) @ Jn * dt, -Jt.T @ np.linalg.inv(M) @ Jt * dt, Jt.T @ np.linalg.inv(M) @ Jt * dt, 1]
    V[3] = [MU, -1, -1, 0]
    p = np.zeros((4,))
    p[0] = Jn.T @ ((1 + EP) * v  + dt * np.linalg.inv(M) @ fe)
    p[1] = -Jt.T @ (v  + dt * np.linalg.inv(M) @ fe)
    p[2] = Jt.T @ (v  + dt * np.linalg.inv(M) @ fe)
    # ------------------------------------------------
    return V, p

def step(q, v):
    """
        predict next config and velocity given the current values
        :param q: <np.array> current configuration of the object
        :param v: <np.array> current velocity of the object
        :return: <np.array>, <np.array> q_next and v_next
    """
    # ------------------------------------------------
    # FILL WITH YOUR CODE
    jac, phi = get_contacts(q)
    Jt = jac[:,0]
    Jn = jac[:,1]
    fe = m * g
    qp = np.array([0, DELTA, 0])
    v_next = None
    q_next = None
    if phi < DELTA:
        V, p = form_lcp(jac, v)
        fc = lcp_solve(V, p)
        v_next = v + dt * np.linalg.inv(M) @ (fe + Jn * fc[0] - Jt * fc[1] + Jt * fc[2])
        q_next = q + dt * v_next + qp
    else:
        v_next = v + dt * np.linalg.inv(M) @ fe
        q_next = q + dt * v_next  # TODO: Replace None with your result
    # ------------------------------------------------
    return q_next, v_next

def simulate(q0, v0):
    """
        predict next config and velocity given the current values
        :param q0: <np.array> initial configuration of the object
        :param v0: <np.array> initial velocity of the object
        :return: <np.array>, <np.array> q and v trajectory of the object
    """
    # ------------------------------------------------
    # FILL WITH YOUR CODE
    q = np.zeros((3, T))  # TODO: Replace with your result
    v = np.zeros((3, T))
    q[:, 0] = q0
    v[:, 0] = v0
    for t in range(T - 1):
        q[:, t+1], v[:, t+1] = step(q[:, t], v[:, t])
    # ------------------------------------------------
    return q, v

def lcp_solve(V, p):
    """
        DO NOT CHANGE -- solves the LCP
        :param V: <np.array> matrix of the LCP
        :param p: <np.array> vector of the LCP
        :return: renders the trajectory
    """
    sol = LCPSolve(V, p)
    f_r = sol[1][:3]
    return f_r

def render(q):
    """
        DO NOT CHANGE -- renders the trajectory
        :param q: <np.array> configuration trajectory
        :return: renders the trajectory
    """
    assignment_3_render(q)

if __name__ == "__main__":
    # to test your final code, use the following initial configs
    q0 = np.array([0.0, 1.5, np.pi / 180. * 30.])
    v0 = np.array([0., -0.2, 0.])
    q, v = simulate(q0, v0)

    plt.plot(q[1, :])
    plt.show()

    render(q)