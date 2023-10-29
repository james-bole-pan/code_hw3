import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Define Constants
GRAVITY = np.array([0.0, -9.81, 0.0])  # Earth gravity in y direction

# Define Data Structures

class Mass:
    def __init__(self, m, p, v, a, F):
        self.m = m
        self.p = p
        self.v = v
        self.a = a
        self.F = F

class Spring:
    def __init__(self, L0, k, m1, m2):
        self.L0 = L0
        self.k = k
        self.m1 = m1
        self.m2 = m2

# Define Cube Initialization
m = 1.0
p = np.array([0.0, 10.0, 0.0])  # starting 10 units above the ground
v = np.array([0.0, 0.0, 0.0])
a = np.array([0.0, 0.0, 0.0])
F = np.array([0.0, 0.0, 0.0])

cube = Mass(m, p, v, a, F)

# Define Simulation Functions

def apply_forces(mass):
    # Apply gravity
    F_gravity = mass.m * GRAVITY
    mass.F += F_gravity

def collision_response(mass):
    # Simple collision with ground
    if mass.p[1] < 0:
        mass.p[1] = 0
        mass.v[1] = -1 * mass.v[1]  # Simple restitution for bounce

def simulation_step(mass, dt):
    apply_forces(mass)
    mass.a = mass.F / mass.m
    mass.v = mass.v + dt * mass.a
    mass.p = mass.p + dt * mass.v
    collision_response(mass)
    mass.F = np.array([0.0, 0.0, 0.0])  # Reset forces for next step

dt = 0.01

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
point, = ax.plot([], [], [], 'ro', markersize=12)  # 'ro' means red circles
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([0, 11])
ax.set_xlabel('X')
ax.set_ylabel('Z')
ax.set_zlabel('Y')  # In this case, we're treating Y as up
ax.set_title('Dropping and Bouncing Cube in 3D')

positions = []

def init():
    point.set_data([], [])
    point.set_3d_properties([])
    return (point,)

def animate(i):
    if i != 0:
        simulation_step(cube, dt)
    positions.append(cube.p)
    x, y, z = cube.p
    point.set_data([x], [z])
    point.set_3d_properties([y])  # Setting the Y value for 3D
    return (point,)

ani = animation.FuncAnimation(fig, animate, frames=1000, init_func=init, blit=False, interval=5)

plt.show()



