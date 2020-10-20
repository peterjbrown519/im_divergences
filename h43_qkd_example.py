"""
Example script to compute H_{4/3}^{uparrow}(A|X=0,E) - H(A|B,X=0,Y=2)

Notes:
 1. We do not need to specify any additional localizing monomials as all the
 	neccesary ones are included by default
"""

# We begin with some helper functions

def cond_ent(joint, marg):
	hab, hb = 0.0, 0.0

	for prob in joint:
		if 0.0 < prob < 1.0:
			hab += -prob*log2(prob)

	for prob in marg:
		if 0.0 < prob < 1.0:
			hb += -prob*log2(prob)

	return hab - hb

def score_constraints(sys, Aops, Bops, eta=1.0):
	"""
	Returns a list of moment equalities for a two-qubit system
					cos(theta) |00> + sin(theta) |11>
	with measurements
					A_{a|x} = (id + cos(ax) sigma_z + sin(ax) sigma_x)/2.0
					B_{b|y} = (id + cos(by) sigma_z + sin(by) sigma_x)/2.0

	sys = [theta, a0, a1, b0, b1, b2]

	This system is subject to detectors with efficiency eta and we treat no-clicks
	as the outcome 0.
	"""

	[id, sx, sy, sz] = [qtp.qeye(2), qtp.sigmax(), qtp.sigmay(), qtp.sigmaz()]
	[theta, a0, a1, b0, b1, b2] = sys[:]
	rho = (cos(theta)*qtp.ket('00') + sin(theta)*qtp.ket('11')).proj()

	# Define the first projectors for each of the measurements of Alice and Bob
	a00 = 0.5*(id + cos(a0)*sz + sin(a0)*sx)
	a01 = id - a00
	a10 = 0.5*(id + cos(a1)*sz + sin(a1)*sx)
	a11 = id - a10
	b00 = 0.5*(id + cos(b0)*sz + sin(b0)*sx)
	b01 = id - b00
	b10 = 0.5*(id + cos(b1)*sz + sin(b1)*sx)
	b11 = id - b10

	A_meas = [[a00, a01], [a10, a11]]
	B_meas = [[b00, b01], [b10, b11]]

	constraints = []

	constraints += [Aops[0][0]*Bops[0][0] - (eta**2 * (rho*qtp.tensor(A_meas[0][0], B_meas[0][0])).tr().real + \
				+ eta*(1-eta)*((rho*qtp.tensor(A_meas[0][0], id)).tr().real + (rho*qtp.tensor(id, B_meas[0][0])).tr().real) + \
				+ (1-eta)*(1-eta))]
	constraints += [Aops[0][0]*Bops[1][0] - (eta**2 * (rho*qtp.tensor(A_meas[0][0], B_meas[1][0])).tr().real + \
				+ eta*(1-eta)*((rho*qtp.tensor(A_meas[0][0], id)).tr().real + (rho*qtp.tensor(id, B_meas[1][0])).tr().real) + \
				+ (1-eta)*(1-eta))]
	constraints += [Aops[1][0]*Bops[0][0] - (eta**2 * (rho*qtp.tensor(A_meas[1][0], B_meas[0][0])).tr().real + \
				+ eta*(1-eta)*((rho*qtp.tensor(A_meas[1][0], id)).tr().real + (rho*qtp.tensor(id, B_meas[0][0])).tr().real) + \
				+ (1-eta)*(1-eta))]
	constraints += [Aops[1][0]*Bops[1][0] - (eta**2 * (rho*qtp.tensor(A_meas[1][0], B_meas[1][0])).tr().real + \
				+ eta*(1-eta)*((rho*qtp.tensor(A_meas[1][0], id)).tr().real + (rho*qtp.tensor(id, B_meas[1][0])).tr().real) + \
				+ (1-eta)*(1-eta))]

	constraints += [Aops[0][0] - eta * (rho*qtp.tensor(A_meas[0][0], id)).tr().real - (1-eta)]
	constraints += [Bops[0][0] - eta * (rho*qtp.tensor(id, B_meas[0][0])).tr().real - (1-eta)]
	constraints += [Aops[1][0] - eta * (rho*qtp.tensor(A_meas[1][0], id)).tr().real - (1-eta)]
	constraints += [Bops[1][0] - eta * (rho*qtp.tensor(id, B_meas[1][0])).tr().real - (1-eta)]

	return constraints[:]

def HAgB(sys, eta):
	"""
	Computes H(A|B,X=0,Y=2) for the specified system sys (see score_constraints()
	for explanation of sys)	and given detection	efficiency eta.
	"""
	[id, sx, sy, sz] = [qtp.qeye(2), qtp.sigmax(), qtp.sigmay(), qtp.sigmaz()]
	[theta, a0, a1, b0, b1, b2] = sys[:]
	rho = (cos(theta)*qtp.ket('00') + sin(theta)*qtp.ket('11')).proj()

	# Define the first projectors for each of the measurements of Alice and Bob
	a00 = 0.5*(id + cos(a0)*sz + sin(a0)*sx)
	a01 = id - a00
	a10 = 0.5*(id + cos(a1)*sz + sin(a1)*sx)
	a11 = id - a10
	b00 = 0.5*(id + cos(b0)*sz + sin(b0)*sx)
	b01 = id - b00
	b10 = 0.5*(id + cos(b1)*sz + sin(b1)*sx)
	b11 = id - b10
	b20 = 0.5*(id + cos(b2)*sz + sin(b2)*sx)
	b21 = id - b20

	A_meas = [[a00, a01], [a10, a11]]
	B_meas = [[b00, b01], [b10, b11], [b20,b21]]

	# Complete no bin distribution
	q00 = eta**2 * (rho*qtp.tensor(A_meas[0][0], B_meas[2][0])).tr().real
	q01 = eta**2 * (rho*qtp.tensor(A_meas[0][0], B_meas[2][1])).tr().real
	q02 = eta * (1-eta) * (rho*qtp.tensor(A_meas[0][0], id)).tr().real
	q10 = eta**2 * (rho*qtp.tensor(A_meas[0][1], B_meas[2][0])).tr().real
	q11 = eta**2 * (rho*qtp.tensor(A_meas[0][1], B_meas[2][1])).tr().real
	q12 = eta * (1-eta) * (rho*qtp.tensor(A_meas[0][1], id)).tr().real
	q20 = eta * (1-eta) * (rho*qtp.tensor(id, B_meas[2][0])).tr().real
	q21 = eta * (1-eta) * (rho*qtp.tensor(id, B_meas[2][1])).tr().real
	q22 = (1-eta)**2

	# qjoint = [q00,q01,q02,q10,q11,q12,q20,q21,q22]
	# qmarg = [q00 + q10 + q20, q01 + q11 + q21, q02, + q12 + q22]

	# Alice bins into 0
	p00 = q00 + q20
	p01 = q01 + q21
	p02 = q02 + q22
	p10 = q10
	p11 = q11
	p12 = q12
	pjoint = [p00,p01,p02,p10,p11,p12]

	pb0 = p00 + p10
	pb1 = p01 + p11
	pb2 = p02 + p12

	pmarg = [pb0, pb1, pb2]

	# Alice and bob both bin
	r00 = q00 + q20 + q02 + q22
	r01 = q01 + q21
	r10 = q10 + q12
	r11 = q11
	rjoint = [r00, r01, r10, r11]
	rb0 = r00 + r10
	rb1 = r01 + r11
	rmarg = [rb0, rb1]

	hagb_abin = cond_ent(pjoint,pmarg)
	hagb_bothbin = cond_ent(rjoint, rmarg)

	return min([hagb_bothbin, hagb_abin])

def rate(SDP,sys,eta):
	return -4*log2(-SDP.dual) - HAgB(sys,eta)

import numpy as np
from math import sqrt, log2, pi, cos, sin
import ncpol2sdpa as ncp
import qutip as qtp
from sympy.physics.quantum.dagger import Dagger
import mosek

# Global level of NPA relaxation
LEVEL = 2

# Defining the measurement scheme we add additional operators to the inputs
# X=0 as the package ncpol2sdpa will automatically remove a
# projector for efficiency purposes. However, we need all projectors
# for the randomness certification inputs to ensure certain Cauchy-Schwarz
# relations are enforced.
# We also don't need to specify Bobs third input as we can lower bound H(A|E)
# using only the measurements on inputs {0,1}
A_config = [3,2]
B_config = [2,2]

# Measurement operators
A = [Ai for Ai in ncp.generate_measurements(A_config, 'A')]
B = [Bj for Bj in ncp.generate_measurements(B_config, 'B')]
V1 = ncp.generate_operators('V1', 2, hermitian=False)
V2 = ncp.generate_operators('V2', 2, hermitian=False)

substitutions = {}
moment_ineqs = []
moment_eqs = []
operator_eqs = []
operator_ineqs = []

# Projectors sum to identity
# We can speed up the computation (for potentially worse rates) by imposing these
# as moment equalities.
moment_eqs += [A[0][0] + A[0][1] - 1]

# Adding the constraints for the measurement operators
substitutions.update(ncp.projective_measurement_constraints(A,B))

# Defining a system to generate a conditional distribution
# We pick the system that maximizes the chsh score
test_sys = [pi/4, 0.0, pi/2, pi/4, -pi/4, 0.0]
test_eta = 0.99
# Get the monomial constraints
score_cons = score_constraints(test_sys, A, B, test_eta)

# CONDITIONS on V1 and V2
# Commute with all Alice and Bob operators
for v in V1 + V2:
	for Ax in A:
		for a in Ax:
			substitutions.update({v*a : a*v})
			substitutions.update({Dagger(v)*a : a*Dagger(v)})
	for By in B:
		for b in By:
			substitutions.update({v*b : b*v})
			substitutions.update({Dagger(v)*b : b*Dagger(v)})

# V_2^* V_2 <= I
operator_ineqs += [1 - (Dagger(V2[0])*V2[0] + Dagger(V2[1])*V2[1])]
# V_2 + V_2^* >= 2 V_1^* V_1
operator_ineqs += [V2[0] + Dagger(V2[0]) - 2*Dagger(V1[0])*V1[0]]
operator_ineqs += [V2[1] + Dagger(V2[1]) - 2*Dagger(V1[1])*V1[1]]

moment_equalities = moment_eqs[:] + score_cons[:]
moment_inequalities = moment_ineqs[:]
operator_equalities = operator_eqs[:]
operator_inequalities = operator_ineqs[:]

# We include some extra monomials in the relaxation to boost rates
extra_monos = []
for v in V1 + V2:
	for Ax in A:
		for a in Ax:
			for By in B:
				for b in By:
					extra_monos += [a*b*v]
					extra_monos += [a*b*Dagger(v)]

# Objective function
obj = A[0][0]*(V1[0] + Dagger(V1[0]))/2.0 + A[0][1]*(V1[1] + Dagger(V1[1]))/2.0

ops = ncp.flatten([A,B,V1,V2])
sdp = ncp.SdpRelaxation(ops, verbose = 1, normalized=True, parallel=0)
sdp.get_relaxation(level = LEVEL,
					equalities = operator_equalities,
					inequalities = operator_inequalities,
					momentequalities = moment_equalities,
					momentinequalities = moment_inequalities,
					objective = -obj,
					substitutions = substitutions,
					extramonomials = extra_monos)

sdp.solve('mosek')
print(f"For detection efficiency {test_eta} the system {test_sys} achieves a DI-QKD rate of {rate(sdp,test_sys,test_eta)}")
