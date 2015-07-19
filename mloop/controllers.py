'''
Created on 16 Jul 2015

@author: michaelhush
'''

import sys
import numpy as nm
import numpy.random as nr
import numpy.linalg as nl
import scipy.io as si
import scipy.optimize as so
import dill
import math
from sklearn import gaussian_process as slgp


class ExpController():
    
    def __init__(self, interface, numPlannedRuns, numParams, minBoundary, maxBoundary, configDict, fileAppend=None):
        
        self.interface = interface
        self.numPlannedRuns = numPlannedRuns
        self.numParams = int(numParams)
        self.minBoundary = minBoundary
        self.maxBoundary = maxBoundary
        self.diffBoundary = maxBoundary - minBoundary
        
        self.numExpCalls = 0
        
        self.configDict = configDict
        
        if fileAppend==None:
            self.fileAppend = ""
        else:
            self.fileAppend = fileAppend
    
        #Do some basic checks 
        
        #Check data from file is formatted right
        if ((self.minBoundary.size!=self.numParams)|(self.minBoundary.size!=self.numParams)):
            sys.exit("Number of boundary elements does not match number of parameters.")

        if (nm.all(self.diffBoundary>0.0)==False):
            sys.exit("Maximum boundary values are not larger than minimum values.")
  
    
    def checkInBoundary(self,param):
        param = nm.array(param)
        testbool = nm.all(param >= self.minBoundary)&nm.all(self.maxBoundary >= param)
        return testbool
    
    def wrapGetCost(self,param):
        self.numExpCalls += 1
        return self.interface.sendParametersGetCost(param)
    
    def saveEverything(self):
        with open('ControllerArchive' + self.fileAppend + '.pkl','wb') as learnerDump:
            dill.dump(self,learnerDump)


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
  

class RandomController(ExpController):
    
    def __init__(self, interface, numPlannedRuns, numParams, minBoundary, maxBoundary, configDict):
        
        super(RandomController,self).__init__(interface, numPlannedRuns, numParams, minBoundary, maxBoundary, configDict)
        
        if ((nm.all(nm.isfinite(self.minBoundary))&nm.all(nm.isfinite(self.maxBoundary)))==False):
            sys.exit('Minimum and/or maximum boundaries are NaN or inf. Must both be finite for random controller.')
        
    def runOptimization(self):
        
        while (self.numExpCalls < self.numPlannedRuns):
            
            paramsSample = self.minBoundary + nr.rand(self.numParams) * self.diffBoundary
            
            self.wrapGetCost(paramsSample)
        
        self.saveEverything()
    
    def nextParams(self):
        
        return self.minBoundary + nr.rand(self.numParams) * self.diffBoundary




class LearnerController(ExpController):
    
    def __init__(self,interface, numPlannedRuns, numParams, minBoundary, maxBoundary, minCost, maxCost, trainingNum, trainingParams, trainingCosts, trainingUncers, trainingBads, initParams, configDict, thetaParticleNumber=None, thetaSearchNumber=None, paramsSearchNumber=None, corrLengths=None) :
        
        if ((nm.all(nm.isfinite(minBoundary))&nm.all(nm.isfinite(maxBoundary)))==False):
            sys.exit('Minimum and/or maximum boundaries are NaN or inf. Must both be finite for learning controller.')
        
        super(LearnerController,self).__init__(interface, numPlannedRuns, numParams, minBoundary, maxBoundary, configDict)
        
        self.minCost = minCost
        self.maxCost = maxCost
        self.diffCost = maxCost - minCost
        
        if self.diffCost<0:
            sys.exit("Minimum cost not less than maximum cost")
        
        
        self.defaultMinNugget = 1.0e-8
        self.samePointTol = 1.0e-5
        self.sameLogThetaTol = 1.0e-1
        self.relLogWeightTol = 2
        
        #Parameters are all scaled to be between 0.0 and 1.0
        self.normParamsFunc = lambda p: nm.array((p - self.minBoundary)/self.diffBoundary)
        #Cost function is normalized to be between 1.0 and 10.0 (in other words 10^0 to 10^1)
        self.normCostFunc = lambda c: float(((c - self.minCost)/self.diffCost)*9.0+1.0)
        #Uncertainty appropriately scaled given cost scaling
        self.normUncerFunc = lambda u: float(abs(u)*9.0/self.diffCost)
        #Function for determining nugget given normalized costs
        self.nuggetFunc = lambda c,u: float(max((u/c)**2,self.defaultMinNugget))
        #Normalising bads    
        self.normBadFunc = lambda b: bool(b)
        
        #Translation of scaled parameters back to real values
        self.realParamsFunc = lambda n: self.minBoundary + self.diffBoundary*n
        #Translation for jacobian and hessian
        self.realDerivFunc = lambda i: 1.0/float(self.diffBoundary[i])
        #Translate internal cost back to real value
        self.realCostFunc = lambda nc: float(self.minCost + self.diffCost*(nc - 1.0)/9.0) 
        
        #Bias between learning and searching
        self.currBias = None
        self.histBias = []
        #Variables for storing max uncertainty and costs, neccesary when biasing costs.
        self.minCUncer = None
        self.maxUUncer = None
        self.maxUCost = None
        self.minUCost = None
        
        #Storage arrays for all parameters
        self.allParams = []
        self.allCosts = []
        self.allUncers = []
        self.allNuggets = []
        self.allBads = []
        self.allEvals = []
        
        #Run some checks to make sure the training data is sane
        if any(trainingNum!= len(tlist) for tlist in [trainingParams, trainingCosts, trainingUncers, trainingBads]):
            sys.exit("One set of training data is not the expected length (trainingNum)")
        
        if trainingNum<=2:
            sys.exit("Not enough training data, take a few more poitns")
        
        #Array used to determine what the best training run was
        self.bestIndex = 0
        self.bestCost = float('inf')
        self.bestParams = None
        
        #Need to check for duplicates of points, so just add each point and check as we add.
        for p,c,u,b in zip(trainingParams,trainingCosts,trainingUncers,trainingBads):
        
            self.currParams = self.normParamsFunc(p)
            self.currCost = self.normCostFunc(c)
            self.currUncer = self.normUncerFunc(u)
            self.currBad = self.normBadFunc(b)
            
            self.saveCurrRun()
        
        if not self.checkInBoundary(initParams):
            sys.exit("Initial parameters provided to learner not within boundaries.")
            
        #Array with first set of parameters to be run
        self.currParams = self.normParamsFunc(initParams)
        
        #array used to store nextParams
        self.nextParams = self.currParams
        
        if corrLengths is None:
        
            if thetaParticleNumber is None:
                self.thetaParticleNumber = min(16,int(self.numParams)*3)
            else:
                self.thetaParticleNumber = int(thetaParticleNumber)
            if self.thetaParticleNumber<1:
                sys.exit("Number of theta particles must be greater than or equal to 1")
        
            #Fitting parameters and fits to check.
            if thetaSearchNumber is None:
                self.thetaSearchNumber = self.numParams*3
            else:
                self.thetaSearchNumber = int(thetaSearchNumber)
            #print("thetaSearchFac:"+ str(self.thetaSearchFac))
            if self.thetaSearchNumber<1:
                sys.exit("Number of theta searchs must be greater than or equal to 1")
        
            self.minLogTheta = -5;
            self.maxLogTheta = 2;
            
            self.minTheta = nm.array([10**(self.minLogTheta) for _ in range(self.numParams)])
            self.maxTheta = nm.array([10**(self.maxLogTheta) for _ in range(self.numParams)])
        
            self.logThetaParticles = []
            self.logThetaParticles.append(nm.array([-1 for _ in range(self.numParams)])) 
        
            self.corrFixed = False
            
        else:
            
            self.bestTheta = nm.array(corrLengths, dtype=float)
            
            self.numParticles = 1
            self.normWeights = []
            self.normWeights.append(1.0)
            self.weightEntropy = 1.0
            self.corrFixed = True
            
            if (self.bestTheta.size!=self.numParams):
                sys.exit("Number of correlation lengths does not match number of parameters.")
                
        self.histBestTheta = []
        self.histBestRLFV = []
        self.histNumParticles = []
        self.histWeightEntropy = []    
        
        #Leash variable if needed
        self.currLeash = None
        self.histLeash = []
        
        if paramsSearchNumber is None:
            self.paramsSearchNumber = self.numParams*3
        else:
            self.paramsSearchNumber  = int(paramsSearchNumber)
        if self.paramsSearchNumber<1:
            sys.exit("Number of parameters search base factor must be greater than or equal to 1")
        #print("maxNumParams:"+ str(self.maxNumParams))
        
        #Bounds which can change when using leash
        self.optBounds = [(0,1) for _ in range(numParams) ]
        self.currMinBoundary = self.minBoundary
        self.currMaxBoundary = self.maxBoundary
        
        #Storage of best parameters parameters etc
        self.bestParams = None
        self.histBestParams = []
        
        
        #Create a dictionary with all the variables to be output to the matlab file
        self.learnerMatDict = {}
        self.learnerMatDict['allParams'] = self.allParams
        self.learnerMatDict['allCosts'] = self.allCosts 
        self.learnerMatDict['allUncers'] = self.allUncers
        self.learnerMatDict['allNuggets'] = self.allNuggets
        self.learnerMatDict['allBads'] = self.allBads
        self.learnerMatDict['allEvals'] = self.allEvals
        self.learnerMatDict['histLeash'] = self.histLeash
        self.learnerMatDict['histBias'] = self.histBias
        self.learnerMatDict['histBestParams'] = self.histBestParams
        self.learnerMatDict['histBestTheta'] = self.histBestTheta
        self.learnerMatDict['histBestRLFV'] = self.histBestRLFV
        self.learnerMatDict['histNumParticles'] = self.histNumParticles
        self.learnerMatDict['histWeightEntropy'] = self.histWeightEntropy
        
    def findLeashedBoundsAndSearchParams(self):
        self.bestParams = self.allParams[nm.argmin(self.allCosts)]
        self.histBestParams.append(self.bestParams)
        self.currMinBoundary = nm.maximum(0, self.bestParams - self.currLeash)
        self.currMaxBoundary = nm.minimum(1, self.bestParams + self.currLeash)
        self.currDiffBoundary = self.currMaxBoundary - self.currMinBoundary
        self.optBounds = [(float(tmin),float(tmax)) for tmin,tmax in zip(self.currMinBoundary,self.currMaxBoundary) ]
        self.localParams = [t for t,b in zip(self.allParams,self.allBads) if (not b) and nm.all(t >= self.currMinBoundary) and nm.all(self.currMaxBoundary >= t)]
        self.searchParams = [self.currMinBoundary + nr.rand(self.numParams)*self.currDiffBoundary for _ in range(self.paramsSearchNumber)]
        self.searchParams.append(self.bestParams)
    
        
    def predBiasedCost(self,params):
        (predCost, predUnc) = self.predictCostAndUncer(params)
        return self.currBias*predCost - (1.0-self.currBias)*predUnc
    
    def predictCostAndUncer(self, params):
        costNUnc = nm.array([gp.predict(params.reshape(1,self.numParams), eval_MSE=True) for gp in self.gaussProcessParticles])
        costs = costNUnc[:,0,0]
        MSEs = costNUnc[:,1,0]
        predCost = float(nm.average(costs, weights=self.normWeights))
        predUnc = float(nm.sqrt(nm.average(MSEs + costs**2, weights=self.normWeights) - predCost**2))
        return (predCost, predUnc)
    
    def predictJustCost(self, params):
        costs = nm.array([gp.predict(params.reshape(1,self.numParams)) for gp in self.gaussProcessParticles])
        costs = costs[:,0]
        predCost = float(nm.average(costs, weights=self.normWeights))
        return predCost
    
    def checkNextParamsForRepeat(self):
        if any([nm.all(nm.absolute(t - self.nextParams)<=self.samePointTol) for t in self.localParams]):
            print("Repeat predicted parameter (now resampling):" + str(self.nextParams))
            self.nextParams = self.currMinBoundary + nr.rand(self.numParams)*self.currDiffBoundary
    
    def findNextParamsBiased(self):
        #Find biased minimum
        firstRunFlag = True
        tempCost = None
        for sp in self.searchParams:
            res = so.minimize(self.predBiasedCost, sp, bounds = self.optBounds, tol=1e-6)
            if firstRunFlag or res.fun < tempCost:
                self.nextParams = nm.array(res.x)
                tempCost = res.fun
                firstRunFlag = False
        
        #self.checkNextParamsForRepeat()
    
    def makeGaussianProcess(self):
        
        tempNuggs = nm.reshape(nm.array(self.allNuggets),len(self.allNuggets))
        
        if self.corrFixed:
            self.gaussProcess = slgp.GaussianProcess(nugget=tempNuggs, theta0 = self.bestTheta)
            self.gaussProcess.fit(self.allParams, self.allCosts)
            self.gaussProcessParticles = []
            self.gaussProcessParticles.append(self.gaussProcess)
          
        else:
            tempThetaParticles = self.logThetaParticles
            self.logThetaParticles = []
            self.logWeights = []
            self.gaussProcessParticles = []
            
            for ttp in tempThetaParticles:
                testTheta = nm.exp(ttp)
                #print("saved currTestTheta:" + str(testTheta))
                self.gaussProcess = slgp.GaussianProcess(nugget=tempNuggs, theta0 = testTheta, thetaL=self.minTheta, thetaU=self.maxTheta)
                self.gaussProcess.fit(self.allParams, self.allCosts)
                self.saveParticles()
            for _ in range(self.thetaSearchNumber):
                testTheta = nm.power(10,self.minLogTheta + nr.rand(self.numParams)*(self.maxLogTheta - self.minLogTheta))
                #print("random currTestTheta:" + str(testTheta))
                self.gaussProcess = slgp.GaussianProcess(nugget=tempNuggs, theta0 = testTheta, thetaL=self.minTheta, thetaU=self.maxTheta)
                self.gaussProcess.fit(self.allParams, self.allCosts)
                self.saveParticles()
            
            self.dropSmallParticles()
            
            self.normWeights = nm.exp(nm.array(self.logWeights))
            self.normWeights = self.normWeights/nm.sum(self.normWeights)
            self.numParticles = len(self.normWeights)
            
            #Diagnostics 
            maxLogWeight = max(self.logWeights)
            thetaRef = self.logWeights.index(maxLogWeight)
            self.bestTheta = 10. ** self.logThetaParticles[thetaRef]
            self.weightEntropy = -float(nm.sum(self.normWeights*nm.log(self.normWeights)))
            
            self.histBestTheta.append(nm.ravel(self.bestTheta))
            self.histBestRLFV.append(maxLogWeight)
            self.histNumParticles.append(self.numParticles)
            self.histWeightEntropy.append(self.weightEntropy)
            
    def saveParticles(self):
        saveFlag = False
        
        currLogTheta = nm.log10(self.gaussProcess.theta_)
        currRLFV = self.gaussProcess.reduced_likelihood_function_value_
        
        if len(self.logThetaParticles) ==0:
            saveFlag = True
        else:
            boolArray = [nm.all(nm.absolute(currLogTheta - tp)<=self.sameLogThetaTol) for tp in self.logThetaParticles]
            
            if any(boolArray):
                repInd = boolArray.index(True)
                if currRLFV > self.logWeights[repInd]:
                    del self.logWeights[repInd]
                    del self.logThetaParticles[repInd]
                    del self.gaussProcessParticles[repInd]
                    saveFlag=True
            else:
                if len(self.logWeights) < self.thetaParticleNumber:
                    saveFlag = True
                else:
                    lowestLogWeight = min(self.logWeights) 
                    if currRLFV > lowestLogWeight:
                        lowestRef = self.logWeights.index(lowestLogWeight)
                        del self.logWeights[lowestRef]
                        del self.logThetaParticles[lowestRef]
                        del self.gaussProcessParticles[lowestRef]
                        saveFlag = True
        
        if saveFlag:
            self.logThetaParticles.append(currLogTheta)
            self.logWeights.append(currRLFV)
            self.gaussProcessParticles.append(self.gaussProcess)
        #print("All Theta Particles:"+str(self.logThetaParticles))
        #print("All Weights Particles:"+str(self.logWeights))
    
    def dropSmallParticles(self):
        maxLogWeight = max(self.logWeights)
        smallRefs = [ref for (ref,lw) in zip(range(len(self.logWeights)),self.logWeights) if lw <= maxLogWeight - self.relLogWeightTol]
        for iind in reversed(smallRefs):
            del self.logWeights[iind]
            del self.logThetaParticles[iind]
            del self.gaussProcessParticles[iind]
     
    def saveCurrRun(self):
        
        #print(self.allParams)
        #print(self.currParams)
        
        if len(self.allParams)>0:
            boolArray = [nm.all(nm.absolute(self.currParams - t)<=self.samePointTol) for t in self.allParams]
        else:
            boolArray = [False];
        if any(boolArray):    
            print("Repeated saved point: not a problem probably but CHAT TO MICHAEL")
            #Found a repeat evaluation, replace appropriately
            #print("found repeat value")
            repInd = boolArray.index(True)
            if self.currBad==self.allBads[repInd]:
                #Both points are good or both are bad, so average the parameters costs and uncertainties 
                self.allEvals[repInd] += 1
                self.allParams[repInd] = (self.allParams[repInd]*(self.allEvals[repInd] - 1) + self.currParams)/self.allEvals[repInd]
                self.allCosts[repInd] = (self.allCosts[repInd]*(self.allEvals[repInd] - 1) + self.currCost)/self.allEvals[repInd]
                self.allUncers[repInd] = math.sqrt((self.allUncers[repInd]*(self.allEvals[repInd] - 1))**2 + self.currUncer**2)/self.allEvals[repInd]
                self.allNuggets[repInd] = self.nuggetFunc(self.allCosts[repInd],self.allUncers[repInd])
            else :
                if self.allBads[repInd]:
                    #This means last bad is False while b = True, so just replace the data   
                    self.allEvals[repInd] = 1     
                    self.allParams[repInd] = self.currParams
                    self.allCosts[repInd] = self.currCost
                    self.allUncers[repInd] = self.currUncer
                    self.allNuggets[repInd] = self.nuggetFunc(self.currCost,self.currUncer)
                    self.allBads[repInd] = self.currBad
        else:
            #Non duplicate point so add to the list
            #print("no repeat value")
            self.allEvals.append(1)
            self.allParams.append(self.currParams)
            self.allCosts.append(self.currCost)
            self.allUncers.append(self.currUncer)
            self.allNuggets.append(self.nuggetFunc(self.currCost,self.currUncer))
            self.allBads.append(self.currBad)
            
    
    def runExperiment(self):
        
        postParams = self.realParamsFunc(self.nextParams)
        self.interface.sendParameters(postParams)
        
    def getExperiment(self):
        
        preCost, preUncer, preBad = self.interface.getCost()
        self.currCost = self.normCostFunc(preCost)
        self.currUncer = self.normUncerFunc(preUncer)
        self.currBad = self.normBadFunc(preBad)
    
    def getExpRunExpSaveData(self):
        
        self.getExperiment()
        self.runExperiment()
        
        #Save last loop data
        self.saveCurrRun()
        self.saveLearnerMatlab()
        self.saveEverything()
            
        #Set currParams to what is currently running on experiment
        self.currParams = self.nextParams
        
    def findAllMinima(self):
        
        self.minimaTol = 1e-3
        
        self.bestIndex = nm.argmin(self.allCosts)
        self.bestParams = self.allParams[self.bestIndex]
        self.bestCost = self.allCosts[self.bestIndex]
        
        self.allMinima = []
        self.allMinCosts = []
        self.allMinReps = []
        self.bestMinima = None
        self.bestJacobian = None
        #self.bestHessian = None
        self.bestMinCost = float('inf')
        
        self.learnerMatDict
        
        for p in reversed(self.allParams):
            
            res = so.minimize(self.predictJustCost, p, bounds = self.optBounds)
            currMinima = res.x
            currMinCost = res.fun
            
            distArray = nm.array([nl.norm(currMinima - t) for t in self.allMinima]) 
            boolArray = distArray<self.minimaTol
            
            if nm.any(boolArray):    
                #Found a repeat evaluation, replace appropriately
                minInd = int(nm.flatnonzero(boolArray)[0])
                if currMinCost < self.allMinCosts[minInd]:
                    self.allMinima[minInd] = currMinima
                    self.allMinCosts[minInd] = currMinCost
                    self.allMinReps[minInd] += 1
            else:
                #Non duplicate point so add to the list
                self.allMinima.append(currMinima)
                self.allMinCosts.append(currMinCost)
                self.allMinReps.append(1)
            
            if currMinCost<self.bestMinCost:
                self.bestMinima = currMinima
                self.bestMinCost = currMinCost
                self.bestJacobian = res.jac
        
        self.numMinima = len(self.allMinCosts)
        self.realBest = self.realParamsFunc(self.bestParams)
        self.allRealMinima = [self.realParamsFunc(t) for t in self.allMinima]
        self.bestRealMinima = self.realParamsFunc(self.bestMinima)
        self.realJacobian = self.bestJacobian * nm.array([[self.realDerivFunc(ind) for ind in range(self.numParams)] for _ in range(self.numParams)])
        #self.realHessian =  self.bestHessian * nm.array([[self.realDerivFunc(ind)*self.realDerivFunc(jnd) for ind in range(self.numParams)] for jnd in range(self.numParams)])
        
    def calculateMatlabFileExtras(self):
        
        print('Performing final processing of Learner (this may take a little time)...')
        
        self.makeGaussianProcess()
        
        self.findAllMinima()
        
        (xvec,costvecs,uncervecs) = self.get1DCrossSectionsAboutBestScaledCost(100)
        
        print('Writing results to file LearnerTracking.mat...')
        
        self.learnerMatDict['numMinima'] = self.numMinima
        self.learnerMatDict['bestIndex'] = self.bestIndex 
        self.learnerMatDict['bestParams'] = self.bestParams
        self.learnerMatDict['bestCost'] = self.bestCost
        self.learnerMatDict['allMinima'] = self.allMinima
        self.learnerMatDict['allMinCosts'] = self.allMinCosts
        self.learnerMatDict['allMinReps'] = self.allMinReps
        self.learnerMatDict['bestMinima'] = self.bestMinima
        self.learnerMatDict['bestJacobian'] = self.bestJacobian
        self.learnerMatDict['bestMinCost'] = self.bestMinCost
        self.learnerMatDict['crossXvec'] = xvec
        self.learnerMatDict['crossCostVecs'] = costvecs
        self.learnerMatDict['crossUncerVecs'] = uncervecs
        
        si.savemat('LearnerTracking.mat', self.learnerMatDict)
        
    def saveLearnerMatlab(self):
        
        si.savemat('LearnerTracking.mat', self.learnerMatDict)

    def get1DCrossSectionsAboutBestScaledCost(self, pts):
        
        xvec = nm.linspace(0,1,pts)
        
        costvecs = []
        uncervecs = []
        
        for iind in range(self.numParams):
            sampParams = nm.array([self.bestParams for _ in range(pts)])
            sampParams[:, iind] = xvec
            costNUnc = nm.array([self.predictCostAndUncer(sampParams[jind,:]) for jind in range(pts)])
            costs = costNUnc[:,0]
            uncers = costNUnc[:,1]
            costvecs.append(costs)
            uncervecs.append(uncers)
       
        return (xvec,costvecs,uncervecs)
    
    def get2DCrossSectionsAboutBestScaledCost(self, pts, dims):
        
        dimA = dims[0]
        dimB = dims[1]
        
        xvec = nm.linspace(0,1,pts)
        
        costMat = nm.zeros((pts,pts))
        uncerMat = nm.zeros((pts,pts))
        
        sampParams = nm.array(self.bestParams)
        
        for x1 in range(pts):
            for x2 in range(pts):
                sampParams[dimA] = xvec[x1]
                sampParams[dimB] = xvec[x2]
                (costs,uncers) = self.predictCostAndUncer(sampParams)
                costMat[x1,x2] = costs
                uncerMat[x1,x2] = uncers
       
        return (xvec,costMat,uncerMat)
     
        
class GlobalLearner(LearnerController):

    def __init__(self, interface, numPlannedRuns, numParams, minBoundary, maxBoundary, minCost, maxCost, trainingNum, trainingParams, trainingCosts, trainingUncers, trainingBads, initParams, configDict, leashLength=None, sweepLength=None, thetaParticleNumber=None, thetaSearchNumber=None, paramsSearchNumber=None, corrLength=None):
        
        super(GlobalLearner,self).__init__(interface, numPlannedRuns, numParams, minBoundary, maxBoundary, minCost, maxCost, trainingNum, trainingParams, trainingCosts, trainingUncers, trainingBads, initParams, configDict, thetaParticleNumber, thetaSearchNumber, paramsSearchNumber, corrLength)
        
        if sweepLength==None:
            self.sweepLength=5
        else:
            self.sweepLength = int(self.numParams)+1
            
        if leashLength==None:
            self.leashLength=1.0
        else:
            self.leashLength = float(leashLength)
        
        self.sweepFunc = lambda x : ((x - 2) % self.sweepLength ) / (self.sweepLength - 1.0)
        self.leashFunc = lambda x : self.leashLength
    
    def runOptimization(self):
        
        self.numExpCalls = 0
        
        #Very First run just try and get the best cost
        self.runExperiment()
        
        while (self.numExpCalls < self.numPlannedRuns):
            #currParams stores what is running on the experiment now
            #nextParams now needs to found and replaced
            
            print("Determining next parameters...")
            
            self.numExpCalls += 1
            
            self.makeGaussianProcess()
            print("BestTheta:"+str(self.bestTheta))
            print("WeightEntropy:"+str(self.weightEntropy))
            
            self.currLeash = float(self.leashFunc(self.numExpCalls))
            self.histLeash.append(self.currLeash)
            print("CurrLeash:"+str(self.currLeash))
            
            self.findLeashedBoundsAndSearchParams()
            print("BestParams:" + str(self.bestParams))
            #print("allParams:"+ str(self.allParams))
            #print("MinBoundary:" + str(self.currMinBoundary))
            #print("MaxBoundary:" + str(self.currMaxBoundary))
            #print("LocalParams:" + str(self.localParams))
            #print("SearchParams:"+ str(self.searchParams))
            
            self.currBias = float(self.sweepFunc(self.numExpCalls))
            self.histBias.append(self.currBias)
            print("CurrBias:"+str(self.currBias))
            
            self.findNextParamsBiased()
            print("NextParams:"+str(self.nextParams))
            
            print("Getting result from experiment.")
            
            self.getExpRunExpSaveData()
        
        #Save last run data
        self.getExperiment() 
        self.saveCurrRun()
        self.saveLearnerMatlab()
        self.saveEverything()
    
class AutoLearner(LearnerController):

    def __init__(self, interface, numPlannedRuns, numParams, minBoundary, maxBoundary, minCost, maxCost, trainingNum, trainingParams, trainingCosts, trainingUncers, trainingBads, initParams, configDict, thetaSearchFac=None, paramsSearchFac=None, runsInCycle=None, leashInit = None, leashMax = None, leashMin=None):
        
        super(AutoLearner,self).__init__(interface, numPlannedRuns, numParams, minBoundary, maxBoundary, minCost, maxCost, trainingNum, trainingParams, trainingCosts, trainingUncers, trainingBads, initParams, configDict, thetaSearchFac, paramsSearchFac)
        
        if runsInCycle is None:
            self.runsInCycle = int(self.numParams+1) 
        else:
            self.runsInCycle = int(runsInCycle)
            
        if runsInCycle < 2:
            sys.exit("Learning cycle for simple learner must be greater than 1.") 
            
        if leashInit is None:
            self.currLeashSize = 1.0/8.0
        
        self.currBiasBase = 1.0
    
    def runOptimization(self):
        
        self.numExpCalls = 0
        
        #Very First run just try and get the best cost
        self.runExperiment()
        
        while (self.numExpCalls < self.numPlannedRuns):
            
            self.numExpCalls += 1
            
            self.makeGaussianProcess()
            
            self.findLeashedBoundsAndSearchParams()
            
            if (self.numExpCalls-1)%(self.runsInCycle) == 0:
                #Just do a simple optimal search for the next best point
                self.findNextParamsOptimal()
            else:
                #Do a randomised balanced search 
                #Make new gaussian process based on it's own prediction of current run coming true
                self.makeAnticipatingGaussianProcess()
                
                self.currBias = float(nr.random()*self.currBiasBase)
                
                self.findNextParamsBiased()
            
            self.getExpRunExpSaveData()
        
        #Save last run data
        self.getExperiment()
        self.saveCurrRun()
        self.saveLearnerMatlab()
        self.saveEverything()

class CopiedLearner(LearnerController):
    
    def __init__(self, learnerCont):
        
        #Translation of scaled parameters back to real values
        self.realParamsFunc = learnerCont.realParamsFunc
        #Translate internal cost back to real value
        self.realCostFunc = learnerCont.realParamsFunc
        
        #Storage arrays for all parameters
        self.allParams = learnerCont.allParams
        self.allCosts = learnerCont.allCosts
        self.allUncers = learnerCont.allUncers
        self.allNuggets = learnerCont.allNuggets
        self.allBads = learnerCont.allBads
        self.allEvals = learnerCont.allEvals
        
        #Storage of best parameters parameters etc
        self.bestParams = learnerCont.bestParams
        
        self.gaussProcess = learnerCont.gaussProcess
  