import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Constants
g = np.array([0.0, 0.0, -9.81])  # Gravity
dt = 0.01
k = 1000.0  # Spring constant
L0 = 1.0  # Rest length of the spring
damping = 0.99  # Damping constant

# Mass Definition
class Mass:
    def __init__(self, p, v, m=1.0):
        self.m = m
        self.p = np.array(p)
        self.v = np.array(v)
        self.a = np.zeros(3,dtype=float)
        self.F = np.zeros(3,dtype=float)

# Spring Definition
class Spring:
    def __init__(self, L0, k, m1, m2):
        self.L0 = L0
        self.k = k
        self.m1 = m1
        self.m2 = m2

# Initialize 8 masses for the cube
half_L0 = L0/2
drop_height = 10.0
masses = [
    Mass([-half_L0, -half_L0, -half_L0 + drop_height], [0.0, 0.0, 0.0]),  # 0
    Mass([half_L0, -half_L0, -half_L0 + drop_height], [0.0, 0.0, 0.0]),   # 1
    Mass([-half_L0, -half_L0, half_L0 + drop_height], [0.0, 0.0, 0.0]),   # 2
    Mass([half_L0, -half_L0, half_L0 + drop_height], [0.0, 0.0, 0.0]),    # 3
    Mass([-half_L0, half_L0, -half_L0 + drop_height], [0.0, 0.0, 0.0]),   # 4
    Mass([half_L0, half_L0, -half_L0 + drop_height], [0.0, 0.0, 0.0]),    # 5
    Mass([-half_L0, half_L0, half_L0 + drop_height], [0.0, 0.0, 0.0]),    # 6
    Mass([half_L0, half_L0, half_L0 + drop_height], [0.0, 0.0, 0.0])      # 7
]

# Connect the masses with springs to form a cube
springs = [
    Spring(L0, k, masses[0], masses[1]),  # Base square
    Spring(L0, k, masses[1], masses[3]),
    Spring(L0, k, masses[3], masses[2]),
    Spring(L0, k, masses[2], masses[0]),
    Spring(L0, k, masses[4], masses[5]),  # Top square
    Spring(L0, k, masses[5], masses[7]),
    Spring(L0, k, masses[7], masses[6]),
    Spring(L0, k, masses[6], masses[4]),
    Spring(L0, k, masses[0], masses[4]),  # Vertical edges
    Spring(L0, k, masses[1], masses[5]),
    Spring(L0, k, masses[2], masses[6]),
    Spring(L0, k, masses[3], masses[7])
]

def simulation_step(masses, springs, dt):
    # Reset forces on each mass
    for mass in masses:
        mass.f = np.zeros(3, dtype=float)
        mass.f += mass.m * g  # Gravity
    
    # Calculate spring forces
    for spring in springs:
        delta_p = spring.m1.p - spring.m2.p
        delta_length = np.linalg.norm(delta_p)
        direction = delta_p / delta_length
        force_magnitude = spring.k * (delta_length - spring.L0)
        force = force_magnitude * direction

        # Apply spring force to masses
        spring.m1.f -= force
        spring.m2.f += force

    # Update positions and velocities for each mass
    for mass in masses:
        mass.a = mass.f / mass.m
        mass.v += mass.a * dt
        mass.p += mass.v * dt

        # Simple collision with the ground
        if mass.p[2] < -half_L0:
            mass.p[2] = -half_L0
            mass.v[2] = -damping * mass.v[2]  # Some damping on collision

# Visualization setup
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Initialize 8 points for the cube's vertices
points = [ax.plot([], [], [], 'ro')[0] for _ in range(8)]

# Initialize 12 lines for the springs
lines = [ax.plot([], [], [], 'b-')[0] for _ in range(12)] 

ax.set_xlim([-5, 5]) 
ax.set_ylim([-5, 5])
ax.set_zlim([0, 10])
ax.set_xlabel('X')
ax.set_ylabel('Y')  
ax.set_zlabel('Z')
ax.set_title('Dropping and Bouncing Cube in 3D')

def init():
    for point in points:
        point.set_data([], [])
        point.set_3d_properties([])
    for line in lines:
        line.set_data([], [])
        line.set_3d_properties([])
    return points + lines

def animate(i):
    if i != 0:
        simulation_step(masses, springs, dt)
    
    for mass, point in zip(masses, points):
        x, y, z = mass.p
        point.set_data([x], [y])
        point.set_3d_properties([z])  # Setting the Y value for 3D

    # Update the spring lines
    for spring, line in zip(springs, lines):
        x_data = [spring.m1.p[0], spring.m2.p[0]]
        y_data = [spring.m1.p[1], spring.m2.p[1]]
        z_data = [spring.m1.p[2], spring.m2.p[2]]
        line.set_data(x_data, y_data)
        line.set_3d_properties(z_data)
    
    return points + lines

ani = animation.FuncAnimation(fig, animate, frames=1000, init_func=init, blit=False, interval=5)

plt.show()


