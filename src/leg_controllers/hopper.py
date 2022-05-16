import numpy as np
from numpy.linalg import solve, lstsq

from leg_controllers.handshake import template_dynamics
from .finite_difference import central_difference
from . import model

CONTROL_WEIGHT = 1.

""" Behavior specific lagrangian quantities """
def potential_energy(q,p):
    return model.potential_energy(q,p)

def potential_gradient(q,p):
    f = lambda q: potential_energy(q,p)
    return central_difference(f,q,1e-9) 

# This is a rough approximation, a more accurate model would 
# compute the kinetic energy in all links, and then M would be given from
# the hessian of the kinetic energy metric (being configuration dependent)
M = np.diag([model.m_body,model.Jm,model.Jm,0.,0.])

""" STANCE DYNAMICS CODE """

""" Computes constraint that foot cannot move """
def contact_constraints(q,foot_loc):
    return q[model.foot_idx]-foot_loc

""" Computes all constraints on behavior """
def stance_constraints(q,p,foot_loc):
    return np.concatenate([
        model.kin_constraints(q,p),contact_constraints(q,foot_loc)
    ])

""" Computes jacobian of constraints """
def stance_constraints_jac(q,p):
    return np.vstack([model.kin_constraints_jac(q,p),
        [[0,0,0,1.0,0],
         [0,0,0,0,1.0]]
    ])

""" Computes fundamental tensor of constraints """
def stance_constraints_hess(q,p):
    return np.vstack([
        model.kin_constraints_hess(q,p),
        np.zeros((2,model.n_coords,model.n_coords))
    ])

""" Computes the stance dynamics of system given a triple (q,qdot,u) """
def stance_dynamics(q,qdot,u,p):
    gradV = potential_gradient(q,p)
    DA = stance_constraints_jac(q,p)
    m = np.size(DA,0) # number of constraints
    ddtDA = stance_constraints_hess(q,p)@qdot
    # solve system of linear equations Ax=b where x = (qddot, λ)
    A = np.vstack([
            np.hstack([M,DA.T]),
            np.hstack([DA,np.zeros((m,m))])
        ])
    
    b = np.concatenate((
            -gradV+model.G@u,
            -ddtDA@qdot
        ))
    x = lstsq(A,b)[0]
    qddot = x[:model.n_coords]
    λ = x[model.n_coords:]
    return qddot,λ


""" Computes projection from model coordinates to Template coordinates in stance mode """
def stance_anchor_projection(q,p):
    return np.array([q[0]-p.l1-p.l2-model.foot_offset])

""" Computes projection from model coordinates to Template coordinates in stance mode """
def stance_anchor_pushforward():
    return np.array([[1.,0.,0.,0.,0.]])

"""
Computes the template dynamics at the projection of (q,qdot) in stance mode
"""
omega = 5*np.pi
zeta = 0.1
def stance_template_dynamics(q,qdot):
    return -2*zeta*omega*qdot - omega**2 * q - np.array([model.g])

""" Computes the Hopper stance controller """
def stance_control(q,qdot,p):
    gradV = potential_gradient(q,p)
    DA = stance_constraints_jac(q,p)
    m = np.size(DA,0) # number of constraints
    ddtDA = stance_constraints_hess(q,p)@qdot
    dpi = stance_anchor_pushforward()
    f = stance_template_dynamics(stance_anchor_projection(q,p),dpi@qdot)
    # solve system of linear equations Ax=b where x = (qddot, λ)
    A = np.vstack([
            np.hstack([M,DA.T,-model.G/CONTROL_WEIGHT]),
            np.hstack([DA,np.zeros((m,m)),np.zeros((m,np.size(model.G,1)))]),
            np.hstack([dpi,np.zeros((np.size(dpi,0),m)),np.zeros((np.size(dpi,0),np.size(model.G,1)))])
        ])
    b = np.concatenate((
            -gradV,
            -ddtDA@qdot,
            f
        ))
    x = lstsq(A,b)[0]
    return x[-np.size(model.G,1):]/CONTROL_WEIGHT

def stance_computed_torque(q,qdot,a,p):
    gradV = potential_gradient(q,p)
    DA = stance_constraints_jac(q,p)
    m = np.size(DA,0) # number of constraints
    ddtDA = stance_constraints_hess(q,p)@qdot
    dpi = stance_anchor_pushforward()
    # solve system of linear equations Ax=b where x = (qddot, λ)
    A = np.vstack([
            np.hstack([M,DA.T,-model.G/CONTROL_WEIGHT]),
            np.hstack([DA,np.zeros((m,m)),np.zeros((m,np.size(model.G,1)))]),
            np.hstack([dpi,np.zeros((np.size(dpi,0),m)),np.zeros((np.size(dpi,0),np.size(model.G,1)))])
        ])
    b = np.concatenate((
            -gradV,
            -ddtDA@qdot,
            a
        ))
    x = lstsq(A,b)[0]
    return x[-np.size(model.G,1):]/CONTROL_WEIGHT

""" FLIGHT DYNAMICS CODE """

""" Computes constraints in flight (non-contact) mode """
def flight_constraints(q,p):
   return model.kin_constraints(q,p)

""" Computes jacobian of constraints in flight (non-contact) mode """
def flight_constraints_jac(q,p):
    return model.kin_constraints_jac(q,p)

""" Computes fundamental tensor of constraints in flight (non-contact) mode """
def flight_constraints_hess(q,p):
    return model.kin_constraints_hess(q,p)


def flight_dynamics(q,qdot,u,p):
    gradV = potential_gradient(q,p)
    DA = flight_constraints_jac(q,p)
    m = np.size(DA,0) # number of constraints
    ddtDA = flight_constraints_hess(q,p)@qdot
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

KNEE_MIN_DIST = 0.04
""" Computes projection from model coordinates to Template coordinates in flight mode """
def flight_anchor_projection(q,p):
    # TODO: this is not good enough, need to control both foot coordinates
    phi = (q[model.tht1_idx]-model.tht2_idx)/2

    return np.array([.5*q[1]-.5*q[2],.5*q[1]+.5*q[2]])

""" Computes projection from model coordinates to Template coordinates in flight mode """
def flight_anchor_pushforward(q,p):
    return np.array([
        [0.,.5,-.5,0.,0.],
        [0.,.5,.5,0.,0.]
    ])

"""
Computes the template dynamics at the projection of (q,qdot) in flight mode
"""
def flight_template_dynamics(q,qdot):
    # omega = 1.5*np.pi 
    # zeta = 5.
    # return -2*zeta*omega*qdot - omega**2 * q 
    return np.zeros(np.size(q))

""" Computes the Hopper flight controller """
def flight_control(q,qdot,p):
    gradV = potential_gradient(q,p)
    DA = flight_constraints_jac(q,p)
    m = np.size(DA,0) # number of constraints
    ddtDA = flight_constraints_hess(q,p)@qdot
    dpi = flight_anchor_pushforward(q,p)
    f = flight_template_dynamics(flight_anchor_projection(q,p),dpi@qdot)
    # solve system of linear equations Ax=b where x = (qddot, λ)
    A = np.vstack([
            np.hstack([M,DA.T,-model.G/CONTROL_WEIGHT]),
            np.hstack([DA,np.zeros((m,m)),np.zeros((m,np.size(model.G,1)))]),
            np.hstack([dpi,np.zeros((np.size(dpi,0),m)),np.zeros((np.size(dpi,0),np.size(model.G,1)))])
        ])
    
    b = np.concatenate((
            -gradV,
            -ddtDA@qdot,
            f
        ))
    x = solve(A,b)
    return x[-np.size(model.G,1):]/CONTROL_WEIGHT

def flight_computed_torque(q,qdot,a,p):
    gradV = potential_gradient(q,p)
    DA = flight_constraints_jac(q,p)
    m = np.size(DA,0) # number of constraints
    ddtDA = flight_constraints_hess(q,p)@qdot
    dpi = flight_anchor_pushforward(q,p)
    # solve system of linear equations Ax=b where x = (qddot, λ)
    A = np.vstack([
            np.hstack([M,DA.T,-model.G/CONTROL_WEIGHT]),
            np.hstack([DA,np.zeros((m,m)),np.zeros((m,np.size(model.G,1)))]),
            np.hstack([dpi,np.zeros((np.size(dpi,0),m)),np.zeros((np.size(dpi,0),np.size(model.G,1)))])
        ])
    
    b = np.concatenate((
            -gradV,
            -ddtDA@qdot,
            a
        ))
    x = solve(A,b)
    return x[-np.size(model.G,1):]/CONTROL_WEIGHT

""" MODE GUARD FUNCTIONS """
STANCE_GUARD_KNEE_DIST = KNEE_MIN_DIST+.005
FLIGHT_GUARD_KNEE_DIST = KNEE_MIN_DIST+.005

def stance_guard(q,p):
    ϕ = model.interior_leg_angle(q) 
    return STANCE_GUARD_KNEE_DIST-2*p.l1*np.sin(ϕ)


def flight_guard(q,p):
    ϕ = model.interior_leg_angle(q)
    return -FLIGHT_GUARD_KNEE_DIST+2*p.l1*np.sin(ϕ)
