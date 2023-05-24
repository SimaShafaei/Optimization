import numpy as np
import math
import bc_module as bc
import scipy.integrate
from numpy.random import default_rng

##########Dynamic programming functions for bifunctional catalyst problem


##########
#Randomly choose allowable values of u
#Input:  r, the width of the search space in the current iteration
#Output:  An M by 10 matrix of different values of u that are within the allowable stage-specific ranges for the current iteration
#Note:  the constraints of 0.60 <= u <= 0.90 cannot be exceeded.
def get_u_values(r, u_policy, M):
    rng = default_rng()
    u_values = np.zeros(shape=(M, 10))
    for i in range(10):
        u_lowest = u_policy[i] - r/2.0
        if (u_lowest < 0.6):
            u_lowest = 0.6
        if (u_lowest + r > 0.9):
            u_lowest = 0.9 - r
        for j in range(M):
            u_value = rng.uniform(u_lowest, u_lowest + r + 0.00001) #Add 0.00001 because uniform() treats upper bound as open
            u_values[j,i]= u_value
    return u_values

##########
#Distance between 2 vectors of size 7
#Input:  2 vectors of 7 chemical quantities
#Output:  the Euclidian distance between the vectors
def A_dist(A1, A2):
    sum = 0
    for i in range(7):
        sum += pow((A1[i] - A2[i]),2)
    return math.sqrt(sum)

##########
#Determine which grid point at stage k+1 is closest to the output of integrating stage k
#Input:  stage, an index in the range [0,9]
#Input:  solution, the results of integrating one stage
#Output:  an index in the range [0, N-1]
def closest_gp(stage, solution):
    best = float('inf')
    best_index = 0
    for i in range(N):
        distance = A_dist(solution.y[:], x_grid[i,stage])
        if distance < best:
            best = distance
            best_index = i
    return best_index

##########
#Integration function
#Used to integrate the corresponding stage
#input:  s, the stage to integrate
#Input:  u_value, the value of u that is in effect for this stage
#Input:  A, a vector of 7 starting points for integration
#Output:  An object containing the full results of the scipy function solve_ivp()
def int(s, j, u_value, A):
    u_choices[j, s] = u_value
    if s < 9:
        solution = bc.bc_dp(u_value, s, A)
        #Which of the N grid vectors at the end of this stage is closest to this solution?
        closest = closest_gp(s, solution)
		  #Get the previously stored u value that worked best for integrating the next stage
        next_u = bestu[closest, s+1]
        #Integrate stage s+1 using that value of u and using
        #the result of this stage s integration as the starting point
        result = int(s+1, j, next_u, solution.y[:,0])
    #Stage 9 is the end of a recursive process
    else:
        result = bc.bc_dp(u_value, 9, A)
    return result

##########
#Identify the candidate with the best fitness
#Input:  s, the current stage
#Input:  g, the current grid, in the range [0,N-1]
#Input:  candidates, an array of M solutions for stage s
#Output:  the u value that yielded the best of the M candidates
def find_best_u(s, g, candidates):
    best = candidates[0]
    best_u = u_values[0,s]
    for j in range(1, M):
        if candidates[j][6] > best[6]:
            best = candidates[j]
            best_u = u_values[j,s]
    return best_u

##########
#The primary function that initiates a DP run
#Input:  u_initial, the initial u policy (a scalar that is in effect for all stages)
#Input:  r, the initial width of the allowable u region
#Input:  N_param, the number of rows of the x-grid
#Input:  M_param, the number of u values tested for each grid point
#Input:  gamma, the proprtion by which r shrinks with each iteration
#Input:  iters, the number of times DP restarts with an updated u policy and r
#Output:  A set of u values and the associated cost function for the best solution found in each iteration
def run(u_initial, r, N_param, M_param, gamma, iters):
    
    #Make N and M accessible to all functions
    global N
    N = N_param
    global M
    M = M_param
    #Create a matrix to hold the x-grid
    global x_grid
    x_grid = np.empty([N,10], dtype = np.ndarray)
    
    #Generate N values of u inside the region defined by u_initial and r
    #Values are evenly spaced from u_initial - r/2 to u_initial + r/2
    global grid_u_values
    grid_u_values = np.empty([N, 10], dtype = float)

    #For the first iteration, use the initial u policy which is constant across all stages
    u_policy = np.array([u_initial,u_initial,u_initial,u_initial,u_initial,\
                         u_initial,u_initial,u_initial,u_initial,u_initial])

    baseline = bc.bc(u_policy)
    print("baseline fitness:  {}".format(baseline))

    #Array used to save the u values used by the most recent 5 solutions
    #At the end of the run, one row will hold the best solution
    global u_choices
    u_choices = np.empty([M, 10], dtype = float)

    #The main loop
    for k in range(iters):
        print("r:  {}".format(r))

        #Populate the matrix the holds u values used to generate this iteration's x-grid
        grid_u_lowest = np.empty([10])
        for i in range(10):
            grid_u_lowest[i] = u_policy[i] - r/2.0
        for i in range(N):
            for j in range(10):
                grid_u_value = grid_u_lowest[j] + i*(r/(N-1))
                grid_u_values[i, j] = grid_u_value

        ##Generate x-grid points using grid_u_values
        for i in range(N):
            start = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            for j in range(10):
                x_grid[i,j] = bc.bc_dp(grid_u_values[i,j], j, start).y[:,0]
                start = x_grid[i,j]

        #The M by 10 matrix of u values used throughout this iteration of DP
        global u_values
        u_values = get_u_values(r, u_policy, M)
    
        #A matrix that holds the best u value found for each grid point in this iteration
        global bestu
        bestu = np.empty([N, 10], dtype = float)

        #An array to hold all M of the outputs from integrating with M different u values
        candidates = np.empty([M], dtype = np.ndarray)

        #Integrate stage 9, NxM times
        #A vector of M solutions that will be compared at each grid point
        for i in range(N):
            #Integrate using M different values of u
            for j in range(M):
                solution = int(9, j, u_values[j,9], x_grid[i,8])
                candidates[j] = solution.y[:,0]
            #Identify the candidate with the best fitness
            #and save the u value that generated that candidate
            bestu[i,9] = find_best_u(9, i, candidates)

        #At this point, N u values for stage 9 are saved in bestu9[]
        #Each of these is the best out of M options that were compared
                       
        #Integrate stage 8->9
		  #For each grid point in stage 8
        for i in range(N):
				#Evaluate M different u values
            for j in range(M):
                #Returns a solution integrated over stages 8 and 9, not just 8
                solution = int(8, j, u_values[j,8], x_grid[i,7])
					 #Save the 7 chemical quantities from the solution to an array
                candidates[j] = solution.y[:,0]
				#Save the u value that gave the best result
            bestu[i,8] = find_best_u(8, i, candidates)

        #Integrate stage 7->9
        for i in range(N):
            for j in range(M):
                solution = int(7, j, u_values[j,7], x_grid[i,6])
                candidates[j] = solution.y[:,0]
            bestu[i,7] = find_best_u(7, i, candidates)

        #Integrate stage 6->9
        for i in range(N):
            for j in range(M):
                solution = int(6, j, u_values[j,6], x_grid[i,5])
                candidates[j] = solution.y[:,0]
            bestu[i,6] = find_best_u(6, i, candidates)

        #Integrate stage 5->9
        for i in range(N):
            for j in range(M):
                solution = int(5, j, u_values[j,5], x_grid[i,4])
                candidates[j] = solution.y[:,0]
            bestu[i,5] = find_best_u(5, i, candidates)

        #Integrate stage 4->9
        for i in range(N):
            for j in range(M):
                solution = int(4, j, u_values[j,4], x_grid[i,3])
                candidates[j] = solution.y[:,0]
            bestu[i,4] = find_best_u(4, i, candidates)

        #Integrate stage 3->9
        for i in range(N):
            for j in range(M):
                solution = int(3, j, u_values[j,3], x_grid[i,2])
                candidates[j] = solution.y[:,0]
            bestu[i,3] = find_best_u(3, i, candidates)

        #Integrate stage 2->9
        for i in range(N):
            for j in range(M):
                solution = int(2, j, u_values[j,2], x_grid[i,2])
                candidates[j] = solution.y[:,0]
            bestu[i,2] = find_best_u(2, i, candidates)
            
        #Integrate stage 1->9
        for i in range(N):
            for j in range(M):
                solution = int(1, j, u_values[j,1], x_grid[i,1])
                candidates[j] = solution.y[:,0]
            bestu[i,1] = find_best_u(1, 1, candidates)

        #Integrate stage 0->9
        #For this stage, there is only 1 possible starting point
        #So there is no need for arrays of size N
        global bestu0
        for j in range(M):
            solution = int(0, j, u_values[j,0], bc.A)
            candidates[j] = solution
        best = candidates[0]
        best_u = u_values[0,0]
        best_choices = 0
        for j in range(1, M):
            if candidates[j].y[6] > best.y[6]:
                best = candidates[j]
                best_u = u_values[j,0]
                best_choices = j
            best0 = best
            bestu[:,0] = best_u
            
        print("best solution from iter {}: {}".format(k,best0.y[6]))
        print("This solution used u values:  {}".format(u_choices[best_choices]))
        
        #Update the u policy
        #Next iteration, center the range of possible u values
        #around the values that worked best in the past iteration
        u_policy = u_choices[best_choices,:]

        #Update r
        #Shrink it by a factor of gamma
        r = r*gamma
