import numpy as np
import matplotlib.pyplot as plt
from assignment_3_helper import LCPSolve, assignment_3_render
from matplotlib.patches import Ellipse  # Import Ellipse at the top of your script


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
T = 700

def get_rotation_matrix(theta):
    """
    Compute the 2D rotation matrix for a given angle.
    :param theta: <float> angle in radians
    :return: <np.array> 2x2 rotation matrix
    """
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

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


def get_contacts(q, geometry_type, geometry_params):
    """
    Return the Jacobian of the lowest contact point and the distance to contact.
    :param q: <np.array> current configuration of the object
    :param geometry_type: <str> "polygon" or "ellipse"
    :param geometry_params: parameters of the shape (vertices for polygon, radii for ellipse)
    :return: <np.array>, <float> Jacobian and distance
    """
    pos, theta = q[0:2], q[2]
    rotation = get_rotation_matrix(theta)

    if geometry_type == "polygon":
        # Transform polygon vertices
        vertices = geometry_params
        transformed_vertices = pos + np.dot(vertices, rotation.T)
        idx = np.argmin(transformed_vertices[:, 1])  # Find the lowest point
        contact_point = transformed_vertices[idx]
        r = contact_point - pos
    elif geometry_type == "ellipse":
        # Compute contact point for ellipse
        a, b = geometry_params
        theta_contact = np.arctan2(-pos[1], pos[0])
        contact_point = pos + np.array([a * np.cos(theta_contact), b * np.sin(theta_contact)])
        r = contact_point - pos
    else:
        raise ValueError("Unsupported geometry type. Use 'polygon' or 'ellipse'.")

    phi = contact_point[1]  # Distance to ground
    J_t = np.array([1, 0, -r[1]])
    J_n = np.array([0, 1, r[0]])
    jac = np.column_stack((J_t, J_n))

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

def step(q, v, geometry_type, geometry_params):
    """
    Predict the next configuration and velocity given the current values.
    :param q: <np.array> current configuration of the object
    :param v: <np.array> current velocity of the object
    :param geometry_type: <str> "polygon" or "ellipse"
    :param geometry_params: parameters of the shape (vertices for polygon, radii for ellipse)
    :return: <np.array>, <np.array> q_next and v_next
    """
    jac, phi = get_contacts(q, geometry_type, geometry_params)
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

def simulate(q0, v0, geometry_type, geometry_params):
    """
    Simulate the trajectory of the object.
    :param q0: <np.array> initial configuration of the object
    :param v0: <np.array> initial velocity of the object
    :param geometry_type: <str> "polygon" or "ellipse"
    :param geometry_params: parameters of the shape
    :return: <np.array>, <np.array> q and v trajectory
    """
    q = np.zeros((3, T))
    v = np.zeros((3, T))
    q[:, 0] = q0
    v[:, 0] = v0

    for t in range(T - 1):
        q[:, t + 1], v[:, t + 1] = step(q[:, t], v[:, t], geometry_type, geometry_params)
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


from matplotlib.patches import Polygon, Ellipse

def render(q, geometry_type, geometry_params):
    """
    Render the trajectory of the object.
    :param q: <np.array> trajectory of the configuration
    :param geometry_type: <str> "polygon" or "ellipse"
    :param geometry_params: vertices or radii
    """
    fig, ax = plt.subplots()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')

    ax.set_xlabel("POSITION")  # Updated X-axis label
    ax.set_ylabel("Y Position")  # Y-axis label remains unchanged
    ax.set_title("Object Trajectory Visualization")  # Title for the plot

    for t in range(q.shape[1]):
        ax.clear()  # Clear the axis
        ax.set_xlim(-2, 2)
        ax.set_ylim(0, 2)
        ax.set_aspect('equal')

        ax.set_xlabel("POSITION")  # Re-add X-axis label after clearing
        ax.set_ylabel("Y Position")  # Re-add Y-axis label after clearing
        ax.set_title("Object Trajectory Visualization")  # Re-add title after clearing

        pos, theta = q[0:2, t], q[2, t]
        rotation = get_rotation_matrix(theta)

        if geometry_type == "polygon":
            vertices = geometry_params
            transformed_vertices = pos + np.dot(vertices, rotation.T)
            patch = Polygon(transformed_vertices, closed=True, color='blue', alpha=0.5)
            ax.add_patch(patch)
        elif geometry_type == "ellipse":
            a, b = geometry_params
            ellipse = Ellipse(
                xy=pos, width=2 * a, height=2 * b, angle=np.degrees(theta), edgecolor="red", fill=None
            )
            ax.add_patch(ellipse)
        else:
            raise ValueError("Unsupported geometry type. Use 'polygon' or 'ellipse'.")

        plt.pause(0.01)

    plt.show()

if __name__ == "__main__":

    # # triangle    
    triangle_vertices = np.array([
        [0.2, 0.2], 
        [-0.2, 0.2], 
        [-0.2, -0.2]
    ])
     
    # pentagon
    pentagon_vertices = np.array([    
    [0.2, 0],
    [0.0618, 0.1902],
    [-0.1618, 0.1176],
    [-0.1618, -0.1176],
    [0.0618, -0.1902]
    ])

    octagon_vertices = np.array([
    [0.2, 0],
    [0.1414, 0.1414],
    [0, 0.2],
    [-0.1414, 0.1414],
    [-0.2, 0],
    [-0.1414, -0.1414],
    [0, -0.2],
    [0.1414, -0.1414]
    ])

    decagon_vertices = np.array([
        [0.2, 0],
        [0.1951, 0.0618],
        [0.1902, 0.1209],
        [0.1808, 0.1783],
        [0.1672, 0.2339],
        [0.1494, 0.2876],
        [0.1291, 0.3387],
        [0.1063, 0.3862],
        [0.0812, 0.4299],
        [0.0544, 0.4695]
    ])

    vertices = generate_polygon_vertices(10)


    q0 = np.array([0.0, 1.5, np.pi / 180. * 30.])
    v0 = np.array([0., -0.2, 0.])
    q, v = simulate(q0, v0, "polygon", vertices)

    plt.plot(q[1, :])
    plt.title("Trajectory (Height vs Time)")
    plt.xlabel("Time Steps")
    plt.ylabel("Height")
    plt.show()

    render(q, "polygon", vertices)