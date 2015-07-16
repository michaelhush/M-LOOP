'''
Created on 16 Jul 2015

@author: michaelhush
'''

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
