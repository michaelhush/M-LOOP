'''
Created on 16 Jul 2015

@author: michaelhush
'''

class NelderController(ExpController):
    
    def nelderGetCost(self,param):
        self.numExpCalls += 1
        cost, _, bad = self.interface.sendParametersGetCost(param)
        if bad:
            cost = float('inf')
        return cost
    
    def nelderInitGetCost(self,param):
        self.numExpCalls += 1
        cost, _, _ = self.interface.sendParametersGetCost(param)
        return cost
    
    def __init__(self, interface, numPlannedRuns, numParams, minBoundary, maxBoundary, initParams, initSimplexDisp, configDict):
        
        super(NelderController,self).__init__(interface, numPlannedRuns, numParams, minBoundary, maxBoundary, configDict)
        
        self.initParams = nm.asfarray(initParams).flatten()
        self.initSimplexDisp = nm.asfarray(initSimplexDisp).flatten()
        
        self.numBoundaryHits = 0
        
        N = len(self.initParams)
        self.sim = nm.zeros((N + 1, N), dtype=self.initParams.dtype)
        self.fsim = nm.zeros((N + 1,), float)
        
        self.saveEverything()
        
        #Run some extra checks
        
        if self.numPlannedRuns < self.numParams:
            sys.exit("Not enough runs to form initial simplex. Increase the number of experimental runs (much more than number of parameters).")
        
        if initParams.size!= numParams :
            sys.exit("Number of initial parameters elements does not match number of parameters.")
         
        if initSimplexDisp.size!= numParams:
            sys.exit("Number of initial simplex displacements does not match number of parameters.")
        
        if nm.any(self.initParams < self.minBoundary) :
            sys.exit("Initial parameters outside of minimum boundary.")
            
        if nm.any(self.maxBoundary < (self.initParams + self.initSimplexDisp)) :
            print("WARNING: Initial simplex outside of maximum boundary. Projecting to be within boundaries")
            self.initSimplexDisp = nm.minimum(self.initParams + self.initSimplexDisp,self.maxBoundary) - self.initParams
            
        if not nm.all(nm.isfinite(self.initParams)):
            sys.exit('Initial parameters are infinite of Nan.')
            
        if not nm.all(nm.isfinite(self.initSimplexDisp)):
            sys.exit('Initial simplex displacements are infinite of Nan.')
        
        
            
    def runOptimization(self):
        """
        Optimises attached experiment of simulation.
        
        Uses nelder-mead algorithm
        """
        
        N = len(self.initParams)
        rho = 1
        chi = 2
        psi = 0.5
        sigma = 0.5
        one2np1 = list(range(1, N + 1))
    
        self.sim[0] = self.initParams
        
        if not self.checkInBoundary(self.initParams):
            sys.exit("Initial condition outside of boundaries. Pick an initial condition inside bounds.")
        self.fsim[0] = self.nelderInitGetCost(self.initParams)
        
        #Create initial simplex
        #Use provided initial condition as a corner then stretch simplex by scale factor in all directions
        for k in range(0, N):
            y = nm.array(self.initParams, copy=True)
            y[k] = y[k] + self.initSimplexDisp[k]
            self.sim[k + 1] = y
            if not self.checkInBoundary(y):
                sys.exit("Initial simplex outside of boundaries. Pick a different initial condition and/or smaller simplex scale.")
            f = self.nelderInitGetCost(y)
            
            self.fsim[k + 1] = f
    
        ind = nm.argsort(self.fsim)
        self.fsim = nm.take(self.fsim, ind, 0)
        # sort so sim[0,:] has the lowest function value
        self.sim = nm.take(self.sim, ind, 0)
    
        while (self.numExpCalls < self.numPlannedRuns):
            
            xbar = nm.add.reduce(self.sim[:-1], 0) / N
            xr = (1 + rho) * xbar - rho * self.sim[-1]
            
            if self.checkInBoundary(xr):
                fxr = self.nelderGetCost(xr)    
            else:
                #Hit boundary so set the cost to positive infinite to ensure reflection
                fxr = float('inf')
                self.numBoundaryHits+=1
                print("Hit boundary (reflect): "+str(self.numBoundaryHits)+" times.")
            
            
            
            doshrink = 0
            
            if fxr < self.fsim[0]:
                xe = (1 + rho * chi) * xbar - rho * chi * self.sim[-1]
                
                if self.checkInBoundary(xe):
                    fxe = self.nelderGetCost(xe)
                else:
                    #Hit boundary so set the cost above maximum this ensures the algorithm does a contracting reflection
                    fxe = fxr+1.0 
                    self.numBoundaryHits+=1
                    print("Hit boundary (expand): "+str(self.numBoundaryHits)+" times.")
                
                if fxe < fxr:
                    self.sim[-1] = xe
                    self.fsim[-1] = fxe
                else:
                    self.sim[-1] = xr
                    self.fsim[-1] = fxr
            else:  # fsim[0] <= fxr
                if fxr < self.fsim[-2]:
                    self.sim[-1] = xr
                    self.fsim[-1] = fxr
                else:  # fxr >= fsim[-2]
                    # Perform contraction
                    if fxr < self.fsim[-1]:
                        xc = (1 + psi * rho) * xbar - psi * rho * self.sim[-1]
                        
                        #if check_boundary(xc):
                        fxc = self.nelderGetCost(xc)
                        #else:
                        #    print("Outside of boundary on contraction: THIS SHOULDNT HAPPEN")
                        #    numB+=1
                        #    fxc = self.interface.nelderGetCost(xc)
                            
                        if fxc <= fxr:
                            self.sim[-1] = xc
                            self.fsim[-1] = fxc
                        else:
                            doshrink = 1
                    else:
                        # Perform an inside contraction
                        xcc = (1 - psi) * xbar + psi * self.sim[-1]
                        
                        #if check_s(xcc):
                        fxcc = self.nelderGetCost(xcc)
                        #else:
                        #    print("Outside of boundary on inside contraction: THIS SHOULDNT HAPPEN")
                        #    numB+=1
                        #    fxcc = self.interface.nelderGetCost(xcc)
    
                        if fxcc < self.fsim[-1]:
                            self.sim[-1] = xcc
                            self.fsim[-1] = fxcc
                        else:
                            doshrink = 1
    
                    if doshrink:
                        for j in one2np1:
                            self.sim[j] = self.sim[0] + sigma * (self.sim[j] - self.sim[0])
                            
                            #if self.checkIn(self.sim[j]):
                            self.fsim[j] = self.nelderGetCost(self.sim[j])
                            #else:
                            #    print("Outside of boundary on shrink contraction: THIS SHOULDNT HAPPEN")
                            #    fsim[j] = self.interface.nelderGetCost(sim[j])
                                
    
            ind = nm.argsort(self.fsim)
            self.sim = nm.take(self.sim, ind, 0)
            self.fsim = nm.take(self.fsim, ind, 0)
            
            self.saveEverything()
        
        return

    def nextParams(self):
        
        N = len(self.initParams)
        rho = 1
        psi = 0.5
        
        xbar = nm.add.reduce(self.sim[:-1], 0) / N
        xr = (1 + rho) * xbar - rho * self.sim[-1]
        
        if self.checkInBoundary(xr):
            return xr    
        else:
            return (1 - psi) * xbar + psi * self.sim[-1]
                 
        
  