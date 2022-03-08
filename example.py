##
import numpy as np
import julia
# julia.install() # don't need to execute this every time if environment is set up correctly
from julia import Pkg, Main 
Pkg.activate(".") # this string needs to point to the pareto-leg-control folder
Main.include("ParetoLegControl.jl") # this string needs to be preceeded by PATH_TO_THIS_FOLDER/

# binding included modules to variables
model = Main.Model
designs = Main.Designs
hopper = Main.Hopper

# design parameters, see Designs.jl for description of the arguments
params = designs.Params(
    .1,
    100.,
    2.0,
    3.14/4,
    .1,
    100.,
    2.0,
    -3.14/4,
    .27,
    500.,
    .07,
    .2
)

# example of computing control cost

# first going to pick a configuration and velocity
# see Model.jl for definition of coordinates
q = np.array([0.,3.14/4,-3.14/4,0.,0.])
q[0] = model.leg_length(q,params)
qdot = np.zeros(len(q))

# now compute the control
u = hopper.stance_control(q,qdot,params)
print(u)

# the current is u/model.Ke
print(u/model.Ke)
