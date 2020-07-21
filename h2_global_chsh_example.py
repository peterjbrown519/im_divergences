"""
In this script we compute lower bounds on H^{uparrow}_2(AB|X=0,Y=0,E) for
devices constrained by a CHSH score.
"""

def ent(SDP):
	# Returns the entropy of the computed solution
	return -2*log2(-SDP.dual)

import numpy as np
from math import sqrt, log2
import ncpol2sdpa as ncp
from sympy.physics.quantum.dagger import Dagger
import mosek
from itertools import product

# Global level of NPA relaxation
LEVEL = 2
# Maximum CHSH score
WMAX = 0.5 + sqrt(2)/4


# Defining the measurement scheme we add additional operators to the inputs
# (X,Y) = (0,0) as the package ncpol2sdpa will automatically remove a
# projector for efficiency purposes. However, we need all projectors
# for the randomness certification inputs to ensure certain Cauchy-Schwarz
# relations are enforced.
A_config = [3,2]
B_config = [3,2]
# Measurement operators
A = [Ax for Ax in ncp.generate_measurements(A_config, 'A')]
B = [By for By in ncp.generate_measurements(B_config, 'B')]
V = ncp.generate_operators('V', 4, hermitian=False)

# Collecting all monomials of form AB for later
AB = []
for Ax, By in product(A,B):
	AB += [a*b for a, b in product(Ax,By)]

substitutions = {}
moment_ineqs = []
moment_eqs = []
operator_eqs = []
operator_ineqs = []
localizing_monos = [] # op_eqs are processed last so need to add three Nones to end

# Projectors sum to identity
# We can speed up the coputation (for potentially worse rates) by imposing these
# as moment equalities.
operator_eqs += [A[0][0] + A[0][1] - 1]
operator_eqs += [B[0][0] + B[0][1] - 1]

# Adding the constraints for the measurement operators
substitutions.update(ncp.projective_measurement_constraints(A,B))

# Defining the chsh inequality
chsh_expr = (A[0][0]*B[0][0] + A[0][1]*B[0][1] + \
			 A[0][0]*B[1][0] + A[0][1]*(1-B[1][0]) + \
			 A[1][0]*B[0][0] + (1-A[1][0])*B[0][1] + \
			 A[1][0]*(1-B[1][0]) + (1-A[1][0])*B[1][0])/4.0
# constraint on the chsh score
w_exp = 0.82
score_con = [chsh_expr - w_exp]

# CONDITIONS on V
# Commute with all Alice and Bob operators
for v in V:
	for Ax in A:
		for a in Ax:
			substitutions.update({v*a : a*v})
			substitutions.update({Dagger(v)*a : a*Dagger(v)})
	for By in B:
		for b in By:
			substitutions.update({v*b : b*v})
			substitutions.update({Dagger(v)*b : b*Dagger(v)})

# We can also impose the relations coming from the dilation theorem
for i in range(len(V)):
	substitutions.update({V[i] * Dagger(V[i]) : 1})
	for j in range(len(V)):
		if i != j:
			substitutions.update({V[i] * Dagger(V[j]) : 0})

# V* V <= 1
operator_ineqs += [1 - (Dagger(V[0])*V[0] + Dagger(V[1])*V[1] + \
				  	Dagger(V[2])*V[2] + Dagger(V[3])*V[3])]
# We build a localizing set of monomials for the constraint V* V <= 1
# This set includes the monomials AB
localizing_set = ncp.nc_utils.get_all_monomials(ncp.flatten([A,B,V]),extramonomials=None,substitutions=substitutions,degree=1)
localizing_set += AB
# We add this set to the localizing_monomials
localizing_monos += [localizing_set]
# We must also specify localizing mmonomials for the other constraints of the
# problem but by specifying None ncpol2sdpa uses a default set
localizing_monos += [None, None, None]

moment_equalities = moment_eqs[:]
moment_inequalities = moment_ineqs[:] + score_con[:]
operator_equalities = operator_eqs[:]
operator_inequalities = operator_ineqs[:]

# We now specify some extra monomials to include in the relaxation
extra_monos = []
for v in V:
	for Ax in A:
		for a in Ax:
			for By in B:
				for b in By:
					extra_monos += [a*b*v]
					extra_monos += [a*b*Dagger(v)]
					extra_monos += [a*b*Dagger(v)*v]
			extra_monos += [a*Dagger(v)*v]
	for By in B:
		for b in By:
			extra_monos += [b*Dagger(v)*v]

# Objective function
obj = A[0][0]*B[0][0]*(V[0] + Dagger(V[0]))/2.0 + \
	  A[0][0]*B[0][1]*(V[1] + Dagger(V[1]))/2.0 + \
	  A[0][1]*B[0][0]*(V[2] + Dagger(V[2]))/2.0 + \
	  A[0][1]*B[0][1]*(V[3] + Dagger(V[3]))/2.0

ops = ncp.flatten([A,B,V])
sdp = ncp.SdpRelaxation(ops, verbose = 1, normalized=True, parallel=0)
sdp.get_relaxation(level = LEVEL,
					equalities = operator_equalities,
					inequalities = operator_inequalities,
					momentequalities = moment_equalities,
					momentinequalities = moment_inequalities,
					objective = -obj,
					substitutions = substitutions,
					extramonomials = extra_monos,
					localizing_monomials = localizing_monos)

sdp.solve('mosek')
print(f"For a chsh score {w_exp} we find an entropy of {ent(sdp)}.")
