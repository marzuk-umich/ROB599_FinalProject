import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def cross_product(v1, v2):
    """Returns the 2D cross product of two vectors."""
    return v1[0] * v2[1] - v1[1] * v2[0]

def subtract_vectors(v1, v2):
    """Subtract vector v2 from v1."""
    return np.array([v1[0] - v2[0], v1[1] - v2[1]])

def dot_product(v1, v2):
    """Returns the dot product of two vectors."""
    return np.dot(v1, v2)

def project_polygon(polygon, axis):
    """Projects the polygon onto the given axis."""
    projections = [dot_product(vertex, axis) for vertex in polygon]
    return min(projections), max(projections)

def is_separating_axis(polygon1, polygon2, axis):
    """Checks if the given axis separates the two polygons."""
    min1, max1 = project_polygon(polygon1, axis)
    min2, max2 = project_polygon(polygon2, axis)
    return max1 < min2 or max2 < min1

def detect_collision(polygon1, polygon2):
    """Detects if there is a collision between two polygons."""
    for i in range(len(polygon1)):
        edge = subtract_vectors(polygon1[(i + 1) % len(polygon1)], polygon1[i])
        axis = np.array([-edge[1], edge[0]])  # Perpendicular axis
        if is_separating_axis(polygon1, polygon2, axis):
            return False  # Separating axis found, no collision
    for i in range(len(polygon2)):
        edge = subtract_vectors(polygon2[(i + 1) % len(polygon2)], polygon2[i])
        axis = np.array([-edge[1], edge[0]])  # Perpendicular axis
        if is_separating_axis(polygon1, polygon2, axis):
            return False  # Separating axis found, no collision
    return True  # No separating axis found, collision detected

def reflect_velocity(velocity, normal):
    """Reflects the velocity vector off the surface defined by the normal."""
    velocity_normal = dot_product(velocity, normal)
    return velocity - 2 * velocity_normal * normal

def collision_response(polygon1, polygon2, velocity1, velocity2):
    """Handles the collision response between two polygons with velocities."""
    if detect_collision(polygon1, polygon2):
        # Find the normal vector (we'll assume the first edge of polygon1 is the collision edge)
        edge = subtract_vectors(polygon1[1], polygon1[0])
        normal = np.array([-edge[1], edge[0]])  # Perpendicular to the edge
        normal = normal / np.linalg.norm(normal)  # Normalize the normal
        
        # Reflect velocities based on the normal
        velocity1_new = reflect_velocity(velocity1, normal)
        velocity2_new = reflect_velocity(velocity2, normal)
        
        return velocity1_new, velocity2_new
    return velocity1, velocity2  # No collision, velocities unchanged

def update_position(polygon, velocity, dt):
    """Updates the polygon's position based on its velocity."""
    return polygon + velocity * dt

def animate(i, polygon1, polygon2, velocity1, velocity2, dt, ax, trajectory1, trajectory2):
    """Animation function to update the positions of the polygons."""
    # Update positions based on velocity
    polygon1 = update_position(polygon1, velocity1, dt)
    polygon2 = update_position(polygon2, velocity2, dt)

    # Check for collision and update velocities
    velocity1, velocity2 = collision_response(polygon1, polygon2, velocity1, velocity2)

    # Clear previous plot
    ax.clear()
    
    # Plot polygons
    ax.fill(polygon1[:, 0], polygon1[:, 1], 'b', alpha=0.5, label="Polygon 1")
    ax.fill(polygon2[:, 0], polygon2[:, 1], 'r', alpha=0.5, label="Polygon 2")
    
    # Plot trajectories
    trajectory1.append(polygon1.mean(axis=0))  # Add the center of mass to trajectory
    trajectory2.append(polygon2.mean(axis=0))  # Add the center of mass to trajectory
    trajectory1_np = np.array(trajectory1)
    trajectory2_np = np.array(trajectory2)
    ax.plot(trajectory1_np[:, 0], trajectory1_np[:, 1], 'b--', label="Trajectory 1")
    ax.plot(trajectory2_np[:, 0], trajectory2_np[:, 1], 'r--', label="Trajectory 2")
    
    # Set plot limits
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_title(f"Step {i}")
    ax.legend()

    return polygon1, polygon2, velocity1, velocity2, trajectory1, trajectory2

# Example usage for animation
polygon1 = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])  # Square
polygon2 = np.array([[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5]])  # Overlapping square
velocity1 = np.array([0.1, 0])  # Velocity of polygon1
velocity2 = np.array([-0.1, 0])  # Velocity of polygon2
dt = 0.1  # Time step

# Set up the plot
fig, ax = plt.subplots()
trajectory1 = []
trajectory2 = []

# Run the animation
ani = animation.FuncAnimation(fig, animate, frames=500, fargs=(polygon1, polygon2, velocity1, velocity2, dt, ax, trajectory1, trajectory2), interval=100)

plt.show()

