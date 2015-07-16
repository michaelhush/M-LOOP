'''
Created on 16 Jul 2015

@author: michaelhush
'''

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

