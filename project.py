import numpy as np
import matplotlib.pyplot as plt
from assignment_3_helper import LCPSolve, assignment_3_render
from matplotlib.patches import Polygon

# DEFINE GLOBAL PARAMETERS
L = 0.4
MU = 0.5
EP = 0.6    # Restitution Coefft 0-> inelastic 1-> elastic
dt = 0.01
m = 0.3
g = np.array([0., -9.81, 0.])
rg = 1./12. * (2 * L * L)
M = np.array([[m, 0, 0], [0, m, 0], [0, 0, m * rg]])
Mi = np.array([[1./m, 0, 0], [0, 1./m, 0], [0, 0, 1./(m * rg)]])
DELTA = 0.001
T = 500

def generate_polygon_vertices(sides, radius=0.2):
    """
    Generate vertices of a regular polygon with a given number of sides and radius.
    :param sides: <int> number of sides of the polygon
    :param radius: <float> radius of the polygon
    :return: <np.array> array of vertices
    """
    # Angle between each vertex in radians
    angle_step = 2 * np.pi / sides
    
    # Generate vertices using polar to Cartesian conversion
    vertices = np.array([
        [radius * np.cos(i * angle_step), radius * np.sin(i * angle_step)]
        for i in range(sides)
    ])
    
    return vertices


def check_collision(poly1, poly2):
    min1, max1 = np.min(poly1, axis=0), np.max(poly1, axis=0)
    min2, max2 = np.min(poly2, axis=0), np.max(poly2, axis=0)
    return not (max1[0] < min2[0] or max2[0] < min1[0] or
                max1[1] < min2[1] or max2[1] < min1[1])

def get_collision_normal(poly1, poly2):
    center1 = np.mean(poly1, axis=0)
    center2 = np.mean(poly2, axis=0)
    normal = center2 - center1
    return normal / np.linalg.norm(normal)

def get_rotation_matrix(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

def get_contacts(q, geometry_params):
    pos, theta = q[0:2], q[2]
    rotation = get_rotation_matrix(theta)
    vertices = geometry_params
    transformed_vertices = pos + np.dot(vertices, rotation.T)
    idx = np.argmin(transformed_vertices[:, 1])
    contact_point = transformed_vertices[idx]
    r = contact_point - pos
    phi = contact_point[1]
    J_t = np.array([1, 0, -r[1]])
    J_n = np.array([0, 1, r[0]])
    jac = np.column_stack((J_t, J_n))
    return jac, phi

def form_lcp(jac, v):
    Jt = jac[:, 0]
    Jn = jac[:, 1]
    fe = m * g
    V = np.zeros((4, 4))
    V[0] = [Jn.T @ np.linalg.inv(M) @ Jn * dt, -Jn.T @ np.linalg.inv(M) @ Jt * dt, Jn.T @ np.linalg.inv(M) @ Jt * dt, 0]
    V[1] = [-Jt.T @ np.linalg.inv(M) @ Jn * dt, Jt.T @ np.linalg.inv(M) @ Jt * dt, -Jt.T @ np.linalg.inv(M) @ Jt * dt, 1]
    V[2] = [Jt.T @ np.linalg.inv(M) @ Jn * dt, -Jt.T @ np.linalg.inv(M) @ Jt * dt, Jt.T @ np.linalg.inv(M) @ Jt * dt, 1]
    V[3] = [MU, -1, -1, 0]
    p = np.zeros((4,))
    p[0] = Jn.T @ ((1 + EP) * v + dt * np.linalg.inv(M) @ fe)
    p[1] = -Jt.T @ (v + dt * np.linalg.inv(M) @ fe)
    p[2] = Jt.T @ (v + dt * np.linalg.inv(M) @ fe)
    return V, p

def step(q, v, geometry_params):
    jac, phi = get_contacts(q, geometry_params)
    Jt = jac[:, 0]
    Jn = jac[:, 1]
    fe = m * g
    qp = np.array([0, DELTA, 0])
    if phi < DELTA:
        V, p = form_lcp(jac, v)
        fc = lcp_solve(V, p)
        v_next = v + dt * np.linalg.inv(M) @ (fe + Jn * fc[0] - Jt * fc[1] + Jt * fc[2])
        q_next = q + dt * v_next + qp
    else:
        v_next = v + dt * np.linalg.inv(M) @ fe
        q_next = q + dt * v_next
    return q_next, v_next

def lcp_solve(V, p):
    sol = LCPSolve(V, p)
    f_r = sol[1][:3]
    return f_r

def resolve_collision(v1, v2, m1, m2):
    v1_new = ((m1 - m2) * v1 + 2 * m2 * v2) / (m1 + m2)
    v2_new = ((m2 - m1) * v2 + 2 * m1 * v1) / (m1 + m2)
    return v1_new, v2_new

def transform_polygon(q, vertices):
    pos, theta = q[0:2], q[2]
    rotation = get_rotation_matrix(theta)
    return pos + np.dot(vertices, rotation.T)

def simulate(q01, v01, q02, v02, geometry_params1, geometry_params2):
    q1 = np.zeros((3, T))
    v1 = np.zeros((3, T))
    q2 = np.zeros((3, T))
    v2 = np.zeros((3, T))
    q1[:, 0] = q01
    v1[:, 0] = v01
    q2[:, 0] = q02
    v2[:, 0] = v02

    for t in range(T - 1):
        q1[:, t + 1], v1[:, t + 1] = step(q1[:, t], v1[:, t], geometry_params1)
        q2[:, t + 1], v2[:, t + 1] = step(q2[:, t], v2[:, t], geometry_params2)

        poly1 = transform_polygon(q1[:, t + 1], geometry_params1)
        poly2 = transform_polygon(q2[:, t + 1], geometry_params2)

        if check_collision(poly1, poly2):
            normal = get_collision_normal(poly1, poly2)
            v1[:, t + 1], v2[:, t + 1] = resolve_collision(v1[:, t + 1], v2[:, t + 1], normal[0], normal[1])

    return q1, v1, q2, v2

def render(q1, q2, geometry_params1, geometry_params2):
    fig, ax = plt.subplots()
    ax.set_xlim(-5, 5)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')

    for t in range(q1.shape[1]):
        ax.clear()
        ax.set_xlim(-4, 4)
        ax.set_ylim(0, 2)
        ax.set_aspect('equal')

        pos1, theta1 = q1[0:2, t], q1[2, t]
        rotation1 = get_rotation_matrix(theta1)
        vertices1 = geometry_params1
        transformed_vertices1 = pos1 + np.dot(vertices1, rotation1.T)
        patch1 = Polygon(transformed_vertices1, closed=True, color='blue', alpha=0.5)
        ax.add_patch(patch1)

        pos2, theta2 = q2[0:2, t], q2[2, t]
        rotation2 = get_rotation_matrix(theta2)
        vertices2 = geometry_params2
        transformed_vertices2 = pos2 + np.dot(vertices2, rotation2.T)
        patch2 = Polygon(transformed_vertices2, closed=True, color='red', alpha=0.5)
        ax.add_patch(patch2)

        plt.pause(0.01)

    plt.show()

if __name__ == "__main__":
    square_vertices = np.array([[0.3, 0.3], [-0.3, 0.3], [-0.3, -0.3], [0.3, -0.3]])
    pentagon_vertices = np.array([    
        [0.2 * 1.5, 0 * 1.5],
        [0.0618 * 1.5, 0.1902 * 1.5],
        [-0.1618 * 1.5, 0.1176 * 1.5],
        [-0.1618 * 1.5, -0.1176 * 1.5],
        [0.0618 * 1.5, -0.1902 * 1.5]
    ])


    q01 = np.array([-0.5, 1.5, np.pi / 180. * 30.])
    v01 = np.array([0.2, -0.2, 0.])
    q02 = np.array([0.5, 1.5, np.pi / 180. * -30.])
    v02 = np.array([-0.2, -0.2, 0.])

    vertices = generate_polygon_vertices(10)

    q1, v1, q2, v2 = simulate(q01, v01, q02, v02, square_vertices, vertices)

    plt.plot(q1[1, :], label="Square")
    plt.plot(q2[1, :], label="Pentagon")
    plt.title("Height vs Time for Two Polygons")
    plt.xlabel("Time Steps")
    plt.ylabel("Height")
    plt.legend()
    plt.show()

    render(q1, q2, square_vertices, vertices)