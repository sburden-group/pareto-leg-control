##
from matplotlib.pyplot import thetagrids
import numpy as np
import julia
# julia.install() # don't need to execute this every time if environment is set up correctly
from julia import Pkg, Main 
Pkg.activate(".") # this string needs to point to the pareto-leg-control folder
Main.include("ParetoLegControl.jl") # this string needs to be preceeded by PATH_TO_THIS_FOLDER/

# binding included modules to variables
model = Main.Model
designs = Main.Designs
computed_torque = Main.ComputedTorque

# design parameters, see Designs.jl for description of the arguments
params = designs.Params(
    .1,
    0.,
    0.,
    3.14/4,
    .1,
    0.,
    0.,
    -3.14/4,
    .27,
    0.,
    .07,
    .2
)

# example of computing control cost

# first going to pick a configuration and velocity
# see Model.jl for definition of coordinates
q = np.array([0.,3.14/4,-3.14/4,0.,0.])
# q[0] = model.leg_length(q,params)
r = model.leg_length(q,params)
theta = .5*q[1]+.5*q[2]
q[[3,4]] = [r*np.sin(theta),-r*np.cos(theta)]
print(q)
qdot = np.zeros(len(q))

# now compute the control
u = computed_torque.control(q,qdot,params,[0.,-9.81])
print(u)

# the current is u/model.Ke
print(u/model.Ke)

print(computed_torque.dynamics(q,qdot,[0.,0.],params))

print(model.Jm)
