import numpy as np


def page_rank(transitions):
    """ This function accepts a Numpy matrix of transitions and returns the PageRank matrix. """
    
    # convergence criteria
    EPSILON = 0.000001 
    
    # rows indicate inlinks, columns indicate outlinks
    number_columns = len(transitions.T)
  
    # need to keep track of previous iteration
    v_prev = None

    # initial guess for v
    v = np.matrix([1/number_columns]*number_columns).T
    
    while True:
        v_prev = v
        v = np.dot(transitions, v)
        
        # difference between current iteration and previous iteration
        differences = v_prev - v
        converged = True 

        for element in differences:
            # if one element in differences is greater than EPSILON,
            # the method has not converged within our desired margin 
            # of error
            if abs(element) > EPSILON:
                converged = False 
        if converged:
            break

    # make a new v that contains answer rounded to 4 decimal points
    v_truncate = []
    for x in v:
        v_truncate.append('%.4f'%(x))
    
    # turn this list into a Numpy matrix and take its transpose
    v_truncate_matrix = np.matrix(v_truncate)
    v_truncate_matrix = v_truncate_matrix.T

    return v_truncate_matrix


def example1():
    transitions = np.matrix([[0, 0, 1/3, 0, 0, 0], [1/2, 0, 0, 1/2, 1/2, 0], [0, 1/2, 0, 0, 0, 1], [1/2, 0, 1/3, 0, 0, 0], [0, 0, 0, 1/2, 0, 0], [0, 1/2, 1/3, 0, 1/2, 0]])
    v = page_rank(transitions)
    
    print('transitions:')
    print(transitions)
    print()
    print('v:')
    print(v)
    print()


if __name__ == '__main__':
    example1()
