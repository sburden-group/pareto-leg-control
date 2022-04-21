import numpy as np
from numpy.linalg import solve, norm
from .finite_difference import central_difference, hessian

"""
Definition of model constants 
"""

g = 9.81
m_body = 2.0
m_weight = 1.0
# const Jm = .5/2*(.087^2+.08^2) # rough approximation motor inertia from thick-walled cylinder model.
Jm = 2e-3

foot_offset = 0.03 # distance from kinematic chain closure to foot location in meters
spring_knee_offset = 0.015 # radial distance from "knee" joints to free end of extension springs
R = 0.186           # winding resistance
Kv = (2*np.pi*100.0/60) # (Rad/s) / Volt
Ke = 1/Kv           # Volt / (rad/s)


# indices of state variables
body_idx = 0  # body location in world frame
tht1_idx = 1    # motor 1 axis
tht2_idx = 2    # motor 2 axis
foot_idx = [3,4] # foot (x,y) location in world frame
n_coords = 5  # number of coordinate variables

# input - state map
G = np.array([
    [0., 0.],
    [1.0, 0.0],
    [0.0, 1.0],
    [0.0, 0.0],
    [0.0, 0.0],
])

""" Begin kinematics """
def hip_foot_angle(q):
    return (q[tht1_idx]-q[tht2_idx])/2.0

def interior_leg_angle(q):
    return (q[tht1_idx]+q[tht2_idx])/2.0

def leg_length(q,p):
    psi = interior_leg_angle(q)
    return p.l1*np.cos(psi)+np.sqrt(p.l2**2-(p.l1*np.sin(psi))**2)+foot_offset

"""
Computes holonomic constraints associated with five-bar mechanism
"""
def kin_constraints(q,p):
    tht = hip_foot_angle(q)
    l = leg_length(q,p)
    return q[foot_idx]-np.array([l*np.sin(tht),q[body_idx]-l*np.cos(tht)])

def kin_constraints_jac(q,p):
    f = lambda q: kin_constraints(q,p)
    return central_difference(f,q,1e-4)

def kin_constraints_hess(q,p):
    f = lambda q: kin_constraints(q,p)
    return hessian(f,q,1e-4)

""" Computes potential energy of springs. """
def spring_energy(q,p):
    l = leg_length(q,p)
    tht1 = q[tht1_idx]

    # compute energy in spring 1
    joint_location = np.array([p.l1*np.sin(tht1),-p.l1*np.cos(tht1)])
    r = (p.l1+p.s1_L+spring_knee_offset)
    fixed_end = np.array([r*np.sin(p.s1_r),-r*np.cos(p.s1_r)]) 
    free_end = joint_location + spring_knee_offset*(fixed_end-joint_location)/norm(fixed_end-joint_location)
    δx = norm(fixed_end-free_end)
    s1_energy = .5*p.s1_k*δx**2+p.s1_Fi*δx

    # compute energy in spring 3
    δx = p.l1+p.l2+foot_offset-l
    s3_energy = .5*p.s3_k*δx**2

    return s1_energy+s3_energy

def potential_energy(q,p):
    return m_body*g*q[body_idx] + spring_energy(q,p)

def kinetic_energy(q,qdot,p):
    return .5*m_body*qdot[body_idx]^2 + .5*Jm*np.sum(qdot[[tht1_idx,tht2_idx]]**2)

