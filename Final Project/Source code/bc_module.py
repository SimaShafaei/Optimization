import numpy as np
from scipy.integrate import solve_ivp

##Summary of optimization problem:
##Maximize the value of I as a function of an array c of size 10.
##The reaction is divided into 10 stages.
##At each stage, 200 g of catalyst blend is added.
##Time is measured in terms not in seconds or minutes, but in 
##the amount of catalyst blend added.  E.g. the time difference
##between 2 stages is 200 g.  The reaction is complete when 2000 g
##have been added.
##I represents the amount of benzene that comes out of the reactor tube
##c represents the hydrogenation catalyst component of the catalyst blend

##Note:  for consistency with array indexing, there are several variables that use
##numbering starting at 0, whereas the reference papers used numbering starting at 1.
##E.g. the differential equation that is labeled z1 in [Esposito, 2000] is
##labeled z0 in the code.


##Differential equations.  There are 7 of them, one for each chemical reactant/product.
##Numbering is offset from the references.
##Inputs:  one or more chemical amounts from array A
##Inputs:  the u value at the current stage
##Output:  dz/dt where z is the amount of a chemical 
def z0(A, u):
    k0 = k(0, u)
    return -k0*A[0]

def z1(A, u):
    k0 = k(0, u)
    k1 = k(1, u)
    k2 = k(2, u)
    k3 = k(3, u)
    return k0*A[0] - (k1 + k2)*A[1] + k3*A[4]

def z2(A, u):
    k1=k(1, u)
    return k1*A[1]

def z3(A, u):
    k5 = k(5, u)
    k4 = k(4, u)
    return -k5*A[3] + k4*A[4]

def z4(A, u):
    k2 = k(2, u)
    k5 = k(5, u)
    k3 = k(3, u)
    k4 = k(4, u)
    k7 = k(7, u)
    k8 = k(8, u)
    k6 = k(6, u)
    k9 = k(9, u)
    return k2*A[1] + k5*A[3] - (k3 + k4 + k7 + k8)*A[4] + k6*A[5] + k9*A[6]

def z5(A, u):
    k7 = k(7, u)
    k6 = k(6, u)
    return k7*A[4] - k6*A[5]

def z6(A, u):
    k8 = k(8, u)
    k9 = k(9, u)
    return k8*A[4] - k9*A[6]

##Rate constant calculation function for k1...k10
##The subscripts are offset by (-1) from those in the reference papers
##Note that the rate constants are dependent on u, which can be different
##at each stage of the reaction.  Therefore k values must be recalculated at each stage.
def k(subscript, u):
    return c[subscript][0]+c[subscript][1]*u \
           + c[subscript][2]*pow(u,2) + c[subscript][3]*pow(u,3)


##A holds the starting amount of each of the 7 chemicals
A = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


##Reaction rate coefficients
##These are empirically determined values that
##are used to calculate k0 through k9
global c
c = np.array([[0.002918487, -0.008045787, 0.006749947, -0.001416647],\
              [9.509977, -35.00994, 42.83329, -17.33333],\
              [26.82093, -95.56079, 113.0398, -44.29997],\
              [208.7241, -719.8052, 827.7466, -316.6655],\
              [1.350005, -6.850027, 12.16671, -6.666689],\
              [0.01921995, -0.07945320, 0.1105666, -0.05033333],\
              [0.1323596, -0.4696255, 0.5549323, -0.2166664],\
              [7.339981, -25.27328, 29.93329, -11.99999],\
              [-0.3950534, 1.679353, -1.777829, 0.4974987],\
              [-0.00002504665, 0.01005854, -0.01986696, 0.009833470]])


##This function accepts a time from 1 to 2000
##and returns an index 1-10 corresponding to the entry in u[]
##that is in effect at that time.
def get_u(t):
    ##The expression (t-1) is used instead of t so that
    ##the first index is used through and including time 200.
    ##To handle the situation when t=0, the abs() is taken to avoid
    ##returning an index of -1.
    index = abs(int((t-1)/200))
    return u[index]


##This function combines all of the differential equations into a form usable by
##the differential equation solver
def model(t, A):
    u=get_u(t)
    v0 = z0(A,u)
    v1 = z1(A,u)
    v2 = z2(A,u)
    v3 = z3(A,u)
    v4 = z4(A,u)
    v5 = z5(A,u)
    v6 = z6(A,u)
    return [v0, v1, v2, v3, v4, v5, v6]


##This function evaluates the fitness of a given value of u[] by integrating from
##0 to 2000 and storing the results at 200,400,600,...2000
def bc(u_arg):
    global u
    u = u_arg
    solution = solve_ivp(model, [0,2000], A,\
               t_eval=[200,400,600,800,1000,1200,1400,1600,1800,2000],\
               atol=1e-7, rtol=1e-4)
    return solution.y[6][9]

##Variant version of bc function that returns the complete results of integration
##instead of just the fitness score
def bc_full_results(u_arg):
    global u
    u = u_arg
    solution = solve_ivp(model, [0,2000], A,\
               t_eval=[200,400,600,800,1000,1200,1400,1600,1800,2000],\
               atol=1e-7, rtol=1e-4)
    return solution



def model_dp(t, A):
    v0 = z0(A,u_dp)
    v1 = z1(A,u_dp)
    v2 = z2(A,u_dp)
    v3 = z3(A,u_dp)
    v4 = z4(A,u_dp)
    v5 = z5(A,u_dp)
    v6 = z6(A,u_dp)
    return [v0, v1, v2, v3, v4, v5, v6]

##This version only integrates 1 stage (200 time units)
##Stage is an integer in range [0,9]
##Returns the full solution instead of just the fitness score
##gp is an array of 7 chemical amounts existing at the start of integration
##Unlike in normal bc(), u_arg is a scalar, not an array
#will using only 1 eval point affect accuracy?
def bc_dp(u_arg, stage, gp):
    global u_dp
    u_dp = u_arg
    integration_start = (stage) * 200
    integration_end = integration_start + 200
    solution = solve_ivp(model_dp, [integration_start,integration_end], gp,\
               t_eval=[integration_end],\
               atol=1e-7, rtol=1e-4)
    return solution


