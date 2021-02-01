import os
import argparse
import numpy as np
import pulp as p
import time

class MDP():

    def __init__(self, path):
        #init mdp from txt at path
        with open(path) as fp:
            Lines = fp.readlines()
            init_data = Lines[:4]
            init_data+=Lines[-2:]
            transition_data = Lines[4:-2]
            self.numStates = int(init_data[0].strip().split()[1])
            self.numActions = int(init_data[1].strip().split()[1])
            self.start = int(init_data[2].strip().split()[1])
            self.end = map(int, init_data[3].strip().split()[1:] )
            self.mdptype = init_data[4].strip().split()[1]
            self.discount = float(init_data[5].strip().split()[1])

            #using a 3d matrix to store T and R
            self.T = np.zeros((self.numStates, self.numActions, self.numStates))
            self.R = np.zeros((self.numStates, self.numActions, self.numStates))


            for transition in transition_data:
                _, s1, a, s2, r, p = transition.strip().split()
                s1 = int(s1)
                a = int(a)
                s2 = int(s2)
                r = float(r)
                p = float(p)
                self.T[s1, a, s2] = p
                self.R[s1, a, s2] = r
    
    #conversions between V, Q and pi
    def _q_values_from_V(self, V):
        return np.sum(self.T*(self.R + self.discount*V), axis=2)

    def _pi_from_q_values(self, q_values):
        return np.argmax(q_values, axis=1)
    
    def _pi_from_V(self, V):
        q_values = self._q_values_from_V(V)
        return self._pi_from_q_values(q_values)
    
    def _V_from_pi(self, pi):
        T = self.T[range(len(pi)), pi]
        R = self.R[range(len(pi)), pi]
        A = np.eye(self.numStates) - self.discount*T
        b = np.sum(T*R, axis=1)
        V = np.linalg.solve(A, b)
        return V


    def _bellman_optimality_operator(self, F):
        #axis 0 is s1, axis 1 is a, axis 2 is s2
        #summing over axis 2 implies summing over s2
        #taking max over axis 1 implies taking max over a
        return np.max(np.sum(self.T*(self.R + self.discount*F), axis=2), axis=1)


    def value_iteration(self, tolerance=None, init =None):
        if tolerance is None:
            tolerance = 1e-6
        error = tolerance*(1-self.discount)/self.discount
        V = init if init is not None else np.zeros(self.numStates)
        while(True):
            V_next = self._bellman_optimality_operator(V)
            if np.max(np.abs(V_next - V))<= error :
                optimal_V =  V_next
                break
            else:
                V = V_next
        return optimal_V, self._pi_from_V(optimal_V)
    
    def _howard_policy_improvement(self, pi):
        V = self._V_from_pi(pi)
        Q = self._q_values_from_V(V)
        improved_pi = pi.copy()
        #can you remove this for loop
        for i in range(self.numStates):
            improvable = np.where( (Q[i,:]-V[i])>1e-6 )[0]
            if len(improvable)>0:
                improved_pi[i] = improvable[0]
        return improved_pi
    
    def policy_iteration(self, init=None):
        pi = init if init is not None else np.zeros(self.numStates, dtype=int)
        while(True):
            improved_pi = self._howard_policy_improvement(pi)
            if (improved_pi == pi).all():
                break
            else:
                pi = improved_pi
        return self._V_from_pi(pi), pi
    
    def lpp_method(self):   
        start_t = time.time()
        # Create a LP Minimization problem 
        Lp_prob = p.LpProblem('LPP', p.LpMaximize)
        #create problem variables
        V = []
        for s in range(self.numStates):
            V.append(p.LpVariable(f"V{s}") )
        #objective function
        Lp_prob += -1*sum(V)
        #constraints
        for s in range(self.numStates):
            for a in range(self.numActions):
                rhs = []
                s_prime_arr = np.where( np.abs(self.T[s,a,:]-1.0)<1e-6 )[0]
                for s_prime in range(self.numStates):
                    rhs.append(self.T[s,a,s_prime]*
                               (self.R[s,a,s_prime] + self.discount*V[s_prime]))
                Lp_prob += V[s]>=sum(rhs)
        end_t = time.time()
        # print('execution time for creating lpp = ', end_t-start_t)
        start_t = time.time()
        #status
        # status = Lp_prob.solve()
        p.PULP_CBC_CMD(msg=False).solve(Lp_prob)
        # print(p.LpStatus[status])
        optimal_V = np.zeros(self.numStates)
        for s in range(self.numStates):
            optimal_V[s] = p.value(V[s])
        end_t = time.time()
        # print('time to solve lpp = ', end_t-start_t)
        return optimal_V, self._pi_from_V(optimal_V)

def main():
    parser = argparse.ArgumentParser(description='Planner')
    parser.add_argument('-p', '--mdp', type=str, required=True, help='Path of mdp file')
    parser.add_argument('-a', '--algorithm', type=str, required=True, help='Algorithm for planning')
    
    args = parser.parse_args()

    mdp_path = args.mdp
    mdp_algorithm = args.algorithm

    my_mdp = MDP(mdp_path)

    if mdp_algorithm == 'vi':
        V, pi = my_mdp.value_iteration()
    elif mdp_algorithm == 'hpi':
        V, pi = my_mdp.policy_iteration()
    elif mdp_algorithm == 'lp':
        V, pi = my_mdp.lpp_method()
    else:
        print('INVALID ALGORITHM')
        exit()

    for i in range(len(V)):
        print(np.round(V[i], 6), pi[i])

if __name__ == "__main__": 
    # calling the main function 
    main()

