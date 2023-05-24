import numpy as np
import math
import bc_module as bc

##Get the allowable values of u (M different values for each stage)
##These values are evenly spaced between (u_inital-r/2) and (u_initial+r/2)
##These values will be different in each iteration because r shrinks and
##the center of the range moves
##This code could be improved to change r if the range exceeds constraints
def get_u_values(r, u_policy, M):
    u_values = np.zeros(shape=(M, 10))
    for i in range(10):
        u_lowest = u_policy[i] - r/2.0
        if (u_lowest < 0.6):
            u_lowest = 0.6
        if (u_lowest + r > 0.9):
            u_lowest = 0.9 - r
        for j in range(M):
            u_value = u_lowest + j*(r/(M-1))
            u_values[j,i]= u_value
            
    return u_values

##Calculate the distance between 2 vectors of 7 A values
##Uses Euclidian distance
def A_dist(A1, A2):
    sum = 0
    for i in range(7):
        sum += pow((A1[i] - A2[i]),2)
    return math.sqrt(sum)


##Find the grid point that is closest to a vector of 7 A values
##Returns an index in the range 0 to N-1
def closest_gp(stage, solution):
    best = float('inf')
    best_index = 0
    for i in range(N):
        distance = A_dist(solution.y[:], x_grid[i].y[:,stage])
        if distance < best:
            best = distance
            best_index = i
    return best_index


##u_values is an array of size 10
def int9(u_value, A):
    u_choices[9] = u_value
    result = bc.bc_dp(u_value, 9, A)
    #print(A[6])
    return result

def int8(u_value, A):
    u_choices[8] = u_value
    solution8 = bc.bc_dp(u_value, 8, A)
    #Which of the 21 grid vectors at t=1800 is closest to this solution?
    closest = closest_gp(8, solution8)
    #Get the corresponding u value from best9
    u9 = bestu9[closest]
    #Integrate the next stage using that value of u and using
    #the result of this stage 8 integration as the starting point
    #of the stage 9 integration
    return int9(u9, solution8.y[:,0])
    
def int7(u_value, A):
    u_choices[7] = u_value
    solution7 = bc.bc_dp(u_value, 7, A)
    closest = closest_gp(7, solution7)
    u8 = bestu8[closest]
    return int8(u8, solution7.y[:,0])

def int6(u_value, A):
    u_choices[6] = u_value
    solution6 = bc.bc_dp(u_value, 6, A)
    closest = closest_gp(6, solution6)
    u7 = bestu7[closest]
    return int7(u7, solution6.y[:,0])

def int5(u_value, A):
    u_choices[5] = u_value
    solution5 = bc.bc_dp(u_value, 5, A)
    closest = closest_gp(5, solution5)
    u6 = bestu6[closest]
    return int6(u6, solution5.y[:,0])

def int4(u_value, A):
    u_choices[4] = u_value
    solution4 = bc.bc_dp(u_value, 4, A)
    closest = closest_gp(4, solution4)
    u5 = bestu5[closest]
    return int5(u5, solution4.y[:,0])

def int3(u_value, A):
    u_choices[3] = u_value
    solution3 = bc.bc_dp(u_value, 3, A)
    closest = closest_gp(3, solution3)
    u4 = bestu4[closest]
    return int4(u4, solution3.y[:,0])

def int2(u_value, A):
    u_choices[2] = u_value
    solution2 = bc.bc_dp(u_value, 2, A)
    closest = closest_gp(2, solution2)
    u3 = bestu4[closest]
    return int3(u3, solution2.y[:,0])

def int1(u_value, A):
    u_choices[1] = u_value
    solution1 = bc.bc_dp(u_value, 1, A)
    closest = closest_gp(1, solution1)
    u2 = bestu2[closest]
    return int2(u2, solution1.y[:,0])

def int0(u_value, A):
    u_choices[0] = u_value
    solution0 = bc.bc_dp(u_value, 0, A)
    closest = closest_gp(0, solution0)
    u1 = bestu1[closest]
    return int1(u1, solution0.y[:,0])

def run(u_initial, r, N_param, M, gamma, iters):

    ##Generate N values of u inside the region defined by u_initial and r
    ##Values are evenly spaced from u_initial - r/2 to u_initial + r/2
    global N
    N = N_param
    global grid_u_values
    grid_u_values = []
    grid_u_lowest = u_initial - r/2.0
    for i in range(N):
        grid_u_value = grid_u_lowest + i*(r/(N-1))
        grid_u_values.append(grid_u_value)

    ##Generate x-grid points
    global x_grid
    x_grid = []
    #For each point, the value of u will be constant through all stages
    for i in range(N):
        u_array = np.array([grid_u_values[i], grid_u_values[i], grid_u_values[i],\
                            grid_u_values[i], grid_u_values[i], grid_u_values[i],\
                            grid_u_values[i], grid_u_values[i], grid_u_values[i],\
                            grid_u_values[i]])
        x_entry = bc.bc_full_results(u_array)
        x_grid.append(x_entry)

    ##For the first iteration, use the initial u policy
    u_policy = np.array([u_initial,u_initial,u_initial,u_initial,u_initial,\
                         u_initial,u_initial,u_initial,u_initial,u_initial])

    baseline = bc.bc(u_policy)
    print("baseline fitness:  {}".format(baseline))

    ##Array used to save the u values used by the most recent solution
    global u_choices
    u_choices = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])

    

    for k in range(iters):
        print("r:  {}".format(r))

        ##The array of M u values used throughout this iteration of DP
        u_values = get_u_values(r, u_policy, M)

        ##Integrate stage 9, NxM times
        best9 = []
        global bestu9
        bestu9 = []
        for i in range(N):
            candidates = []
            #Integrate using M different values of u
            for j in range(M):
                solution = int9(u_values[j,9], x_grid[i].y[:,8])
                candidates.append(solution)
            #Identify the candidate with the best fitness
            best = candidates[0]
            best_u = u_values[0,9]
            for j in range(1, M):
                if candidates[j].y[6] > best.y[6]:
                    best = candidates[j]
                    best_u = u_values[j,9]
            #Save that best candidate
            best9.append(best)
            #Save the u value that led to that candidate
            bestu9.append(best_u)




        ##At this point, 21 best candidates for stage 9 are saved in best9[]

                                 
        ##Integrate stage 8
        best8 = []
        global bestu8
        bestu8 = []
        for i in range(N):
            candidates = []
            for j in range(M):
                solution = int8(u_values[j,8], x_grid[i].y[:,7])
                candidates.append(solution)
            best = candidates[0]
            best_u = u_values[0,8]
            for j in range(1, M):
                if candidates[j].y[6] > best.y[6]:
                    best = candidates[j]
                    best_u = u_values[j,8]
            #Save that best candidate
            best8.append(best)
            bestu8.append(best_u)



        ##Integrate stage 7
        best7 = []
        global bestu7
        bestu7 = []
        for i in range(N):
            candidates = []
            for j in range(M):
                solution = int7(u_values[j,7], x_grid[i].y[:,6])
                candidates.append(solution)
            best = candidates[0]
            best_u = u_values[0,7]
            for j in range(1, M):
                if candidates[j].y[6] > best.y[6]:
                    best = candidates[j]
                    best_u = u_values[j,7]
            #Save that best candidate
            best7.append(best)
            bestu7.append(best_u)


            
        ##Integrate stage 6
        best6 = []
        global bestu6
        bestu6 = []
        for i in range(N):
            candidates = []
            for j in range(M):
                solution = int6(u_values[j,6], x_grid[i].y[:,5])
                candidates.append(solution)
            best = candidates[0]
            best_u = u_values[0,6]
            for j in range(1, M):
                if candidates[j].y[6] > best.y[6]:
                    best = candidates[j]
                    best_u = u_values[j,6]
            #Save that best candidate
            best6.append(best)
            bestu6.append(best_u)



        ##Integrate stage 5
        best5 = []
        global bestu5
        bestu5 = []
        for i in range(N):
            candidates = []
            for j in range(M):
                solution = int5(u_values[j,5], x_grid[i].y[:,4])
                candidates.append(solution)
            best = candidates[0]
            best_u = u_values[0,5]
            for j in range(1, M):
                if candidates[j].y[6] > best.y[6]:
                    best = candidates[j]
                    best_u = u_values[j,5]
            #Save that best candidate
            best5.append(best)
            bestu5.append(best_u)

        ##Integrate stage 4
        best4 = []
        global bestu4
        bestu4 = []
        for i in range(N):
            candidates = []
            for j in range(M):
                solution = int4(u_values[j,4], x_grid[i].y[:,3])
                candidates.append(solution)
            best = candidates[0]
            best_u = u_values[0,4]
            for j in range(1, M):
                if candidates[j].y[6] > best.y[6]:
                    best = candidates[j]
                    best_u = u_values[j,4]
            #Save that best candidate
            best4.append(best)
            bestu4.append(best_u)

        ##Integrate stage 3
        best3 = []
        global bestu3
        bestu3 = []
        for i in range(N):
            candidates = []
            for j in range(M):
                solution = int3(u_values[j,3], x_grid[i].y[:,2])
                candidates.append(solution)
            best = candidates[0]
            best_u = u_values[0,3]
            for j in range(1, M):
                if candidates[j].y[6] > best.y[6]:
                    best = candidates[j]
                    best_u = u_values[j,3]
            #Save that best candidate
            best3.append(best)
            bestu3.append(best_u)

        ##Integrate stage 2
        best2 = []
        global bestu2
        bestu2 = []
        for i in range(N):
            candidates = []
            for j in range(M):
                solution = int2(u_values[j,2], x_grid[i].y[:,1])
                candidates.append(solution)
            best = candidates[0]
            best_u = u_values[0,2]
            for j in range(1, M):
                if candidates[j].y[6] > best.y[6]:
                    best = candidates[j]
                    best_u = u_values[j,2]
            #Save that best candidate
            best2.append(best)
            bestu2.append(best_u)
            
        ##Integrate stage 1
        best1 = []
        global bestu1
        bestu1 = []
        for i in range(N):
            candidates = []
            for j in range(M):
                solution = int1(u_values[j,1], x_grid[i].y[:,0])
                candidates.append(solution)
            best = candidates[0]
            best_u = u_values[0,1]
            for j in range(1, M):
                if candidates[j].y[6] > best.y[6]:
                    best = candidates[j]
                    best_u = u_values[j,1]
            #Save that best candidate
            best1.append(best)
            bestu1.append(best_u)

        ##Integrate stage 0
        #For this stage, there is only 1 possible starting point
        global bestu0
        candidates = []
        for j in range(M):
            solution = int0(u_values[j,0], bc.A)
            candidates.append(solution)
        best = candidates[0]
        best_u = u_values[0,0]
        for j in range(1, M):
            if candidates[j].y[6] > best.y[6]:
                best = candidates[j]
                best_u = u_values[j,0]
            best0 = best
            bestu0 = best_u
        print("best solution from iter {}: {}".format(k,best0.y[6]))
        print("This solution used u values:  {}".format(u_choices))
        ##Update the u policy
        ##Next iteration, center the range of possible u values
        ##around the values that worked best in the past iteration
        u_policy = u_choices

        ##Update r
        r = r*gamma
