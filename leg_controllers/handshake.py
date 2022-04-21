import numpy as np
from numpy.linalg import solve
from .finite_difference import central_difference
from . import model

# This is a rough approximation, a more accurate model would 
# compute the kinetic energy in all links, and then M would be given from
# the hessian of the kinetic energy metric (being configuration dependent)
load_m = 1.0
M = np.diag([model.m_body,model.Jm,model.Jm,load_m,load_m])

""" Behavior specific lagrangian quantities """
# TODO: MUST ROTATE THE BODY 90 DEGREES!!!
def potential_energy(q,p):
    return model.potential_energy(q,p) + load_m*model.g*q[model.foot_idx[0]]

def potential_gradient(q,p):
    f = lambda q: potential_energy(q,p)
    return central_difference(f,q,1e-6)

""" Computes all constraints on behavior """
def constraints(q,p):
    return np.append(model.kin_constraints(q,p),[q[model.body_idx]])


""" Computes jacobian of constraints """
def constraints_jac(q,p):
    return np.vstack([model.kin_constraints_jac(q,p),
        [1.0,0.,0.,0.,0.]])

""" Computes hessian tensor of constraints """
def constraints_hess(q,p):
    return np.vstack([
        model.kin_constraints_hess(q,p),
        np.zeros((1,model.n_coords,model.n_coords))
    ])

""" Computes the stance dynamics of system given a triple (q,qdot,u) """
def dynamics(q,qdot,u,p):
    gradV = potential_gradient(q,p)
    DA = constraints_jac(q,p)
    m = np.size(DA,0) # number of constraints
    ddtDA = constraints_hess(q,p)@qdot
    # solve system of linear equations Ax=b where x = (qddot, λ)
    A = np.vstack([
            np.hstack([M,DA.T]),
            np.hstack([DA,np.zeros((m,m))])
        ])
    
    b = np.concatenate((
            -gradV+model.G@u,
            -ddtDA@qdot
        ))
    x = solve(A,b)
    qddot = x[0:model.n_coords]
    λ = x[model.n_coords:]
    return qddot,λ

""" Computes projection from Model coordinates to Template coordinates """
# does this even match the optimization code anymore??
def anchor_projection(q,p):
    r = p.l1+p.l2
    tht = -np.pi/8
    eq = np.array([r*np.sin(tht),-r*np.cos(tht)])
    return q[model.foot_idx] - eq

""" Computes projection from Model coordinates to Template coordinates """
def anchor_pushforward():
    return np.array([
            [0.,0.,0.,1.,0.],
            [0.,0.,0.,0.,1.]
        ])

"""
Computes the template dynamics at the projection of (q,qdot)
"""
def template_dynamics(q,qdot):
    omega = 2*np.pi
    zeta = 0.5
    return -2*zeta*omega*qdot - omega**2 * q

""" Computes the Handshake controller """
def control(q,qdot,p):
    gradV = potential_gradient(q,p)
    DA = constraints_jac(q,p)
    m = np.size(DA,0) # number of constraints
    ddtDA = constraints_hess(q,p)@qdot
    dpi = anchor_pushforward()
    f = template_dynamics(anchor_projection(q,p),dpi@qdot)
    # solve system of linear equations Ax=b where x = (qddot, λ)
    A = np.vstack([
            np.hstack([M,DA.T,-model.G]),
            np.hstack([DA,np.zeros((m,m)),np.zeros((m,np.size(model.G,1)))]),
            np.hstack([dpi,np.zeros((np.size(dpi,0),m)),np.zeros((np.size(dpi,0),np.size(model.G,1)))])
        ])
    
    b = np.concatenate((
            -gradV,
            -ddtDA@qdot,
            f
        ))
    x = solve(A,b)
    u = x[model.n_coords+m:]
    return u

def computed_torque(q,qdot,a,p):
    gradV = potential_gradient(q,p)
    DA = constraints_jac(q,p)
    m = np.size(DA,0) # number of constraints
    ddtDA = constraints_hess(q,p)@qdot
    dpi = anchor_pushforward()
    # solve system of linear equations Ax=b where x = (qddot, λ)
    A = np.vstack([
            np.hstack([M,DA.T,-model.G]),
            np.hstack([DA,np.zeros((m,m)),np.zeros((m,np.size(model.G,1)))]),
            np.hstack([dpi,np.zeros((np.size(dpi,0),m)),np.zeros((np.size(dpi,0),np.size(model.G,1)))])
        ])
    
    b = np.concatenate((
            -gradV,
            -ddtDA@qdot,
            a
        ))
    x = solve(A,b)
    u = x[model.n_coords+m:]
    return u