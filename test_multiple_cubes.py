import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Constants
g = np.array([0.0, 0.0, -9.81])  # Gravity
dt = 0.001
k = 10000.0  # Spring constant
L0 = 1.0  # Rest length of the spring
damping = 0.999  # Damping constant

half_L0 = L0/2
drop_height = 1.0

def rotation_matrix(axis, theta):
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

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

class Cube:
    def __init__(self, offset, v_offset, orientation):
        self.masses = [Mass(np.dot(rotation_matrix(orientation[0], orientation[1]), np.array([x, y, z])) + offset, v_offset)
               for x in [-half_L0, half_L0]
               for y in [-half_L0, half_L0]
               for z in [-half_L0 + drop_height, half_L0 + drop_height]]
        self.springs = [
            Spring(L0, k, self.masses[0], self.masses[1]),  # Base square
            Spring(L0, k, self.masses[1], self.masses[3]),
            Spring(L0, k, self.masses[3], self.masses[2]),
            Spring(L0, k, self.masses[2], self.masses[0]),
            Spring(L0, k, self.masses[4], self.masses[5]),  # Top square
            Spring(L0, k, self.masses[5], self.masses[7]),
            Spring(L0, k, self.masses[7], self.masses[6]),
            Spring(L0, k, self.masses[6], self.masses[4]),
            Spring(L0, k, self.masses[0], self.masses[4]),  # Vertical edges
            Spring(L0, k, self.masses[1], self.masses[5]),
            Spring(L0, k, self.masses[2], self.masses[6]),
            Spring(L0, k, self.masses[3], self.masses[7])
        ]
        short_diag_length = np.sqrt(2 * L0**2)
        self.springs += [
            Spring(short_diag_length, k, self.masses[0], self.masses[3]),
            Spring(short_diag_length, k, self.masses[1], self.masses[2]),
            Spring(short_diag_length, k, self.masses[4], self.masses[7]),
            Spring(short_diag_length, k, self.masses[5], self.masses[6]),
            # Short Diagonals between opposite faces
            Spring(short_diag_length, k, self.masses[0], self.masses[5]),
            Spring(short_diag_length, k, self.masses[1], self.masses[4]),
            Spring(short_diag_length, k, self.masses[2], self.masses[7]),
            Spring(short_diag_length, k, self.masses[3], self.masses[6]),
            Spring(short_diag_length, k, self.masses[0], self.masses[7]),
            Spring(short_diag_length, k, self.masses[1], self.masses[6]),
            Spring(short_diag_length, k, self.masses[2], self.masses[5]),
            Spring(short_diag_length, k, self.masses[3], self.masses[4])
        ]

        # Long Diagonals
        long_diag_length = np.sqrt(3 * L0**2)
        self.springs += [
            Spring(long_diag_length, k, self.masses[0], self.masses[6]),
            Spring(long_diag_length, k, self.masses[1], self.masses[7]),
            Spring(long_diag_length, k, self.masses[2], self.masses[5]),
            Spring(long_diag_length, k, self.masses[3], self.masses[4])
        ]

    def simulate(self):
        for mass in self.masses:
            mass.f = np.zeros(3, dtype=float)
            mass.f += mass.m * g  # Gravity
    
        # Calculate spring forces
        for spring in self.springs:
            delta_p = spring.m1.p - spring.m2.p
            delta_length = np.linalg.norm(delta_p)
            direction = delta_p / delta_length
            force_magnitude = spring.k * (delta_length - spring.L0)
            force = force_magnitude * direction

            # Apply spring force to masses
            spring.m1.f -= force
            spring.m2.f += force

        # Update positions and velocities for each mass
        for mass in self.masses:
            mass.a = mass.f / mass.m
            mass.v += mass.a * dt
            mass.p += mass.v * dt

            # Simple collision with the ground
            if mass.p[2] < 0:
                mass.p[2] = 0
                mass.v[2] = -damping * mass.v[2]  # Some damping on collision

# Random orientations and velocities for 3 cubes
np.random.seed(42)  # for reproducibility

cubes_number = 3
offsets = [np.random.rand(3) * 3 for _ in range(cubes_number)]
v_offsets = [np.random.rand(3) for _ in range(cubes_number)]
orientations = [(np.random.rand(3), 2*np.pi*np.random.rand()) for _ in range(cubes_number)]  # (axis, angle)

cubes = [Cube(offset, v_offset, orientation) for offset, v_offset, orientation in zip(offsets, v_offsets, orientations)]

# Visualization setup
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Initialize 8 points for each cube's vertices
points_sets = [[ax.plot([], [], [], 'ro')[0] for _ in range(8)] for _ in range(cubes_number)]

# Initialize springs for each cube
lines_sets = [[ax.plot([], [], [], 'b-')[0] for _ in range(28)] for _ in range(cubes_number)]


ax.set_xlim([-3, 3]) 
ax.set_ylim([-3, 3])
ax.set_zlim([0, 6])
ax.set_xlabel('X')
ax.set_ylabel('Y')  
ax.set_zlabel('Z')
ax.set_title('Dropping and Bouncing Multiple Cube in 3D')

def init():
    for points in points_sets:
        for point in points:
            point.set_data([], [])
            point.set_3d_properties([])
    for lines in lines_sets:
        for line in lines:
            line.set_data([], [])
            line.set_3d_properties([])
    return [item for sublist in points_sets + lines_sets for item in sublist]


def animate(i):
    for cube_idx, cube in enumerate(cubes):
        if i != 0:
            cube.simulate()
        
        # Update mass positions
        for mass, point in zip(cube.masses, points_sets[cube_idx]):
            x, y, z = mass.p
            point.set_data([x], [y])
            point.set_3d_properties([z])

        # Update the spring lines
        for spring, line in zip(cube.springs, lines_sets[cube_idx]):
            x_data = [spring.m1.p[0], spring.m2.p[0]]
            y_data = [spring.m1.p[1], spring.m2.p[1]]
            z_data = [spring.m1.p[2], spring.m2.p[2]]
            line.set_data(x_data, y_data)
            line.set_3d_properties(z_data)
    
    return [item for sublist in points_sets + lines_sets for item in sublist]

ani = animation.FuncAnimation(fig, animate, frames=1000, init_func=init, blit=False, interval=5)

plt.show()




