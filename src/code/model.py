from code import tensors
from code import plot as plt

import gurobipy as gp
import numpy as np

from IPython.display import clear_output

class Model:
    '''
    A wrapper for all the functionality to perform recursive polynomial optimization
    '''
    
    def __init__(self, n, k_half, k_upper, socp=False, eigen_tol=1e-6):
        '''
        The constructor.
        
        n:           the number of variables
        k_half:      half the full required polynomial degree
        k_upper:     and upper bound on the maximum degree we will need to optimize
        socp:        should we use SOCP or LP relaxations
        eigen_tol:   the tolerance on violation of the SDP cone
        '''
        
        self.monomials = tensors.get_monomial_indices(n, k_upper)
        self.k_half = k_half
        self.k_upper = k_upper
        self.n = n
        self.socp = socp
        self.eigen_tol = eigen_tol
        self.optimized = False
        self.constraints = []
        
        # create the model
        self.model = gp.Model("polynomial_program")
        self.vars_x = self.model.addMVar(len(self.monomials), name='x', lb=-gp.GRB.INFINITY)
        
        # set the constant
        self.model.addConstr(self.vars_x[0] == 1, name='constant')
        
        # current DD cone
        self.width = len([key for key in self.monomials.keys() if sum(key) <= k_half])
        
        # cones
        self.cones = dict()
        
        # if socp we need qcp duals
        if socp:
            self.model.setParam('QCPDual', 1)
        
        # add the identity cone for the moment sequence
        self.add_cone('identity', socp=socp)
    
    def set_objective(self, polynomial):
        '''
        Set the objective function.
        
        polynomial:   a list of (coef., [monomial])
        '''
        
        self.objective = polynomial
        
        # get the coef vector
        coef = np.zeros(len(self.monomials))
        
        # set the coefs
        for p in polynomial:
            coef[self.monomials[tuple(p[1])]] = p[0]
        
        self.model.setObjective(coef @ self.vars_x, gp.GRB.MINIMIZE)
    
    def __optimize_and_feasible(self):
        '''
        We optimize, and if infeasible, we find an infeasible subsystem.
        '''
        
        # initial optimization
        self.model.optimize()
        
        # if we have an infeasible start
        if self.model.status != gp.GRB.Status.INF_OR_UNBD:
            return

        # we need to resolve to get dual rays
        self.model.setParam('DualReductions', 0)

        # Compute the IIS
        self.model.computeIIS()

        # we need to resolve to get dual rays
        self.model.setParam('DualReductions', 1)
        
    
    def optimize(self, max_steps):
        '''
        Run the optimizer.
        
        max_steps:   the maximum number of columns we will generate
        '''
        
        # turn on this flag for plotting
        self.optimized = True
        
        # initial optimize
        self.__optimize_and_feasible()
        
        # attempt the max number of steps
        for i in range(max_steps):
            
            # extend each cone that is violated
            num_violations = 0
            for key in self.cones:
                Z = self.get_violation_ray(key)
                if Z is None:
                    continue
                
                # extend the violated cone
                num_violations += 1
                self.extend_cone(key, Z)
            
            if num_violations == 0:
                break
            
            # clear the output in jupyter
            clear_output(wait=True)
            
            # reoptimize
            self.__optimize_and_feasible()
    
    def extend_cone(self, name, add_tensors=None):
        '''
        Extend an SDP cone approximation with a new extreme vector.
        '''
        
        # add to the cone generators if needed
        if add_tensors is None:
            return
        
        # cast to list
        if not isinstance(add_tensors, (list, tuple)):
            add_tensors = [add_tensors]
        
        # get the map
        values = self.cones[name]
        
        # this is not an inequality
        if values[2] is None:
            return
        
        # remove old constraint
        self.model.remove(values[1])
            
        # get cached cone props
        Ax = values[0]
        socp = values[3]
        
        # if we have LP we will build the DD cone, otherwise
        # the SDD cone is used for an SOCP
        an = len(add_tensors)
        if not socp:
            y = self.model.addMVar(an)
        
            # extend the cone
            Dy = values[2] + gp.quicksum(add_tensors[i] * y[i] for i in range(an))
        else:
            y = self.model.addMVar((an, 2, 2), lb=-gp.GRB.INFINITY)
            
            # psd constraint
            self.model.addConstrs(y[k,0,0] >= 0 for k in range(an))
            self.model.addConstrs(y[k,1,1] >= 0 for k in range(an))
            self.model.addConstrs(y[k,1,0] == y[k,0,1] for k in range(an))
            self.model.addConstrs(y[k,0,0] * y[k,1,1] >= y[k,0,1] * y[k,0,1] for k in range(an))
            
            # extend the cone
            Dy = values[2] + gp.quicksum(add_tensors[i] @ y[i,:,:] @ add_tensors[i].T for i in range(an))
        
        # set equality of Ax with z, our semidefinite vector
        Z = self.model.addConstr(Ax == Dy, name=f'{name}_Ax=Dy')
        
        # update the dictionary
        self.cones[name] = (Ax, Z, Dy, socp)
        
    def add_cone(self, name, localizer=None, inequality=True, socp=False):
        '''
        Add a new moment cone approximation.
        
        name:         the name of the constraint
        localizer:    the localizing polynomial specified as in objective
        inequality:   is this an inequality or equality
        socp:         should we use the socp or lp generation
        '''
        
        # add constraint
        if localizer is not None:
            self.constraints.append(localizer)
        
        # get the constraints
        A = tensors.get_moment_operator(
            n=self.n, 
            k_half=self.k_half, 
            monomials=self.monomials,
            localizer=localizer
        )
        
        # apply the moment operator
        Ax = gp.quicksum(A[i] * self.vars_x[i] for i in range(A.shape[0]))

        # if not an inequality
        if not inequality:
            Z = self.model.addConstr(Ax == 0, name=f'{name}_Ax=0')
            
            self.cones[name] = (Ax, Z, None, None)
            
            # we do not need to create the generators
            return
        
        # if we have LP we will build the DD cone, otherwise
        # the SDD cone is used for an SOCP
        if not socp:
            an = self.width*self.width
            y = self.model.addMVar(an, name='alpha')

            # starting cone
            D = tensors.get_cone_un2(self.width)
            Dy = gp.quicksum(D[i] * y[i] for i in range(an))

        else:
            # we need duals
            self.model.setParam('QCPDual', 1)
            
            an = self.width * (self.width - 1) // 2
            y = self.model.addMVar((an, 2, 2), name='alpha', lb=-gp.GRB.INFINITY)
            
            # psd constraint
            self.model.addConstrs(y[k,0,0] >= 0 for k in range(an))
            self.model.addConstrs(y[k,1,1] >= 0 for k in range(an))
            self.model.addConstrs(y[k,1,0] == y[k,0,1] for k in range(an))
            self.model.addConstrs(y[k,0,0] * y[k,1,1] >= y[k,0,1] * y[k,0,1] for k in range(an))
            
            # starting cone
            D = tensors.get_cone_vn2(self.width)
            Dy = gp.quicksum(D[i] @ y[i] @ D[i].T for i in range(an))
            
        # set equality of Ax with z, our semidefinite vector
        Z = self.model.addConstr(Ax == Dy, name=f'{name}_Ax=Dy')
        
        # set the properties for this cone
        self.cones[name] = (Ax, Z, Dy, socp)
    
    def get_violation_ray(self, cone):
        '''
        After solving, find a dual ray which violates the SDP requirement
        of a moment cone.
        
        cone:   the name of the cone
        '''
        
        # if we have a zero cone we dont need to extend it
        if self.cones[cone][2] is None:
            return None
        
        # get the dual variable for the sdp constr
        Z = self.cones[cone][1].Pi
        Z = (Z + Z.T) / 2
        
        # check if socp
        socp = self.cones[cone][3]
        
        # compute the violation dual eigenvectors
        # and check if the violation is severe enough
        eig = np.linalg.eigh(Z)
        ev = eig[0][0]
        
        if ev >= -self.eigen_tol:
            return None
        
        if not socp:
            # get the largest negative eigenvector
            v = eig[1][:,0]
            return np.outer(v, v)
        else:
            # get the two largest negative eigenvectors
            return eig[1][:,:2]
    
    def plot(self, bounds):
        '''
        Plot the objective function and optimizer
        
        bounds:   the bounds of the domain
        '''
        
        if not hasattr(self, 'objective'):
            return
        
        if self.n >= 3:
            print('3D plot not supported.')
            return
        
        soln = None if not self.optimized else self.get_solution()
        
        if self.n == 1:
            plt.plot_1d(self.objective, bounds, point=soln, constraints=self.constraints)
    
    def get_solution(self):
        '''
        Get the optimal solution
        '''
        
        inds = [value for key, value in self.monomials.items() if sum(key) == 1]
        
        return np.flip([self.vars_x[idx].X for idx in inds])