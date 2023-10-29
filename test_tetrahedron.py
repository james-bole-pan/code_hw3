import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Constants
g = np.array([0.0, 0.0, -9.81])  # Gravity
dt = 0.001
k = 1000.0  # Spring constant
L0 = 1.0  # Rest length of the spring
damping = 0.999  # Damping constant


# Mass Definition
class Mass:
    def __init__(self, p, v, m=0.1):
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

# Initialize 4 masses for the tetrahedron
drop_height = 2.0
half_L0 = L0 / 2
masses = [
    Mass([0, 0, drop_height], [0.0, 0.0, 0.0]),              # Top vertex
    Mass([-half_L0, -half_L0, -half_L0 + drop_height], [0.0, 0.0, 0.0]),  # Base vertices
    Mass([half_L0, -half_L0, -half_L0 + drop_height], [0.0, 0.0, 0.0]),
    Mass([0, half_L0, -half_L0 + drop_height], [0.0, 0.0, 0.0])
]

# Connect the masses with springs to form a tetrahedron
springs = [
    Spring(L0, k, masses[0], masses[1]),
    Spring(L0, k, masses[0], masses[2]),
    Spring(L0, k, masses[0], masses[3]),
    Spring(L0, k, masses[1], masses[2]),
    Spring(L0, k, masses[1], masses[3]),
    Spring(L0, k, masses[2], masses[3])
]

KE_list = []
PE_list = []
TE_list = []

def simulation_step(masses, springs, dt):
    # Reset forces on each mass
    for mass in masses:
        mass.f = np.zeros(3, dtype=float)
        mass.f += mass.m * g  # Gravity
    
    KE = sum([0.5 * mass.m * np.linalg.norm(mass.v)**2 for mass in masses])
    PE = sum([mass.m * g[2] * mass.p[2] for mass in masses])

    # Calculate spring forces
    for spring in springs:
        delta_p = spring.m1.p - spring.m2.p
        delta_length = np.linalg.norm(delta_p)
        direction = delta_p / delta_length
        force_magnitude = spring.k * (delta_length - spring.L0)
        force = force_magnitude * direction

        PE_spring = 0.5 * spring.k * (delta_length - spring.L0)**2
        PE += PE_spring

        # Apply spring force to masses
        spring.m1.f -= force
        spring.m2.f += force

    # Update positions and velocities for each mass
    for mass in masses:
        mass.a = mass.f / mass.m
        mass.v += mass.a * dt
        mass.p += mass.v * dt

        # Simple collision with the ground
        if mass.p[2] < 0:
            mass.p[2] = 0
            mass.v[2] = -damping * mass.v[2]  # Some damping on collision

    TE = KE + PE
    KE_list.append(KE)
    PE_list.append(PE)
    TE_list.append(TE)

# Visualization setup
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Initialize 4 points for the tetrahedron's vertices
points = [ax.plot([], [], [], 'ro')[0] for _ in range(4)]

# Initialize 6 lines for the springs
lines = [ax.plot([], [], [], 'b-')[0] for _ in range(6)]
shadows = [ax.plot([], [], [], 'k-')[0] for _ in range(6)]

ax.set_xlim([-2, 2]) 
ax.set_ylim([-2, 2])
ax.set_zlim([0, 4])
ax.set_xlabel('X')
ax.set_ylabel('Y')  
ax.set_zlabel('Z')
ax.set_title('Dropping and Bouncing Tetrahedron in 3D')

def init():
    for point in points:
        point.set_data([], [])
        point.set_3d_properties([])
    for line in lines:
        line.set_data([], [])
        line.set_3d_properties([])
    for shadow in shadows:
        shadow.set_data([], [])
        shadow.set_3d_properties([])
    return points + lines + shadows

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
    
    # Update the shadow lines
    for spring, shadow in zip(springs, shadows):
        x_data = [spring.m1.p[0], spring.m2.p[0]]
        y_data = [spring.m1.p[1], spring.m2.p[1]]
        z_data = [0, 0]
        shadow.set_data(x_data, y_data)
        shadow.set_3d_properties(z_data)

    return points + lines

ani = animation.FuncAnimation(fig, animate, frames=1000, init_func=init, blit=False, interval=5)

plt.show()

N = len(KE_list)
time_list = np.arange(0, N*dt, dt)

plt.figure()
plt.plot(time_list, KE_list, label='Kinetic Energy')
plt.plot(time_list, PE_list, label='Potential Energy')
plt.plot(time_list, TE_list, label='Total Energy')
plt.xlabel('Time (s)')
plt.ylabel('Energy (J)')
plt.title('Energy vs Time for a Bouncing Tetrahedron')
plt.legend()
plt.show()

print("Length of KE_list: ", len(KE_list))
