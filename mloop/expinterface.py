'''
Created on 16 Jul 2015

@author: michaelhush
'''

class ExpInterface():
    
    def __init__(self, timeDelay=None, fileAppend=None):
        
        #Configurations
        if (timeDelay==None):
            self.timeDelay = 1
        else:
            self.timeDelay = timeDelay
        if fileAppend==None:
            self.fileAppend = ""
        else:
            self.fileAppend = fileAppend
        
        #Counters
        self.numRuns = 0
        
        self.expOutFileName = 'ExpOutput.mat'
        self.expInFileName = 'ExpInput.txt'
        
        #Best history of relevant properties for immediate access for future training
        self.histParams = []
        self.histCosts = []
        self.histUncers = []
        self.histBads = []
        
        #Best Parameters and cost
        self.bestParams = nm.array([0]);
        self.bestCost = float('nan');

        self.matFileDict = {'parameters':[],'runNumber':[]}
        self.lastParams = None
        
        self.repExp = 0
        
    def sendParameters(self,currParams):
        
        self.lastParams = nm.copy(currParams)
        self.repExp = 0
        
        self.numRuns += 1
        print("Starting experiment number: " + str(self.numRuns))
        
        #Write file with parameters to trigger experiment to run
        with open(self.expInFileName,'w') as expInputFile:
            for param in list(self.lastParams.ravel()):
                expInputFile.write(str(param) + " ")
        
        return
        
    def getCost(self):
        
        #Check directory for files 
        print("Waiting for experiment...")
        
        readFileFlag = False
        while not readFileFlag:
            while not os.path.isfile(self.expOutFileName):
                #Give a time delay in seconds before we check the for output file again
                time.sleep(self.timeDelay)
            
            #Give some time to make sure the file is written to disk
            time.sleep(self.timeDelay)
            
            try:
                #Get cost from file
                tempDict = si.loadmat(self.expOutFileName)
                os.remove(self.expOutFileName)
                readFileFlag = True
            except FileNotFoundError:
                print("FileNotFoundError thrown when trying to read ExpOutput.mat. Trying again...")
            
        
        '''
        ExpOutput.mat file is a matlab file with many points of data in it
        '''
        
        if self.numRuns==1:
            #First time for a run.
            self.matFileDict['parameters'].append(self.lastParams)
            self.matFileDict['runNumber'].append(self.numRuns)
            
            if 'parameters' in tempDict:
                sys.exit("You can't have an object in ExpOutput.mat called parameters. Call it something else.")
            
            if 'runNumber' in tempDict:
                sys.exit("You can't have an object in ExpOutput.mat called runNumber. Call it something else.")
                
            if not ('bad' in tempDict):
                sys.exit("You're missing bad in your output file, you have to provide it.")
            
            if not ('cost' in tempDict):
                sys.exit("You're missing cost in your output file, you have to provide it.")
            
            if not ('uncer' in tempDict):
                sys.exit("You're missing uncer in your output file, you have to provide it.")
            
            for key in tempDict:
                self.matFileDict[key] = []
                self.matFileDict[key].append(tempDict[key])
            
        else:
            
            self.matFileDict['parameters'].append(self.lastParams)
            self.matFileDict['runNumber'].append(self.numRuns)
            
            try:
                #Save everything in the dictionary into our running array
                for key in tempDict:
                    self.matFileDict[key].append(tempDict[key])
                    
            except KeyError:
                print("Key was missing from your matlab file. You probably added an extra variable that wasn't in the first one you submitted. Don't do that.")
                raise
        
        currCost = float(tempDict['cost'])
        currUncer = float(tempDict['uncer'])
        currBad = bool(tempDict['bad'])
        
        #Save history locally as well easy access
        self.histParams.append(self.lastParams)
        self.histCosts.append(currCost)
        self.histUncers.append(currUncer)
        self.histBads.append(currBad)
        
        #Write history to file
        self.saveHistory()
        
        if currBad and self.repExp < 1:
            
            print("Bad run, repeating experiment")
            
            self.repExp += 1
            
            #Write file with parameters to trigger experiment to run
            with open(self.expInFileName,'w') as expInputFile:
                for param in list(self.lastParams.ravel()):
                    expInputFile.write(str(param) + " ")
                    
            return self.getCost()
        
        if (self.numRuns==1):
            self.bestCost = currCost
            self.bestParams = self.lastParams
        else:
            if (not currBad) and (currCost < self.bestCost):
                self.bestCost = currCost
                self.bestParams = self.lastParams
        
         
        #Return cost
        return currCost,currUncer,currBad
        
    def sendParametersGetCost(self,currParams):
        
        self.sendParameters(currParams)
        return self.getCost()
        
            
    def saveHistory(self):
        """
        Adds to a file currently tracking the experiment
        """
        
        si.savemat('ExpTracking'+self.fileAppend+'.mat', self.matFileDict)
    

class TestingExpInterface():
    
    def __init__(self, fileAppend=None):
        
        #Configurations
        if fileAppend==None:
            self.fileAppend = ""
        else:
            self.fileAppend = fileAppend
        
        #Counters
        self.numRuns = 0
        
        self.histParams = []
        self.histCosts = []
        self.histUncers = []
        self.histBads = []
        
        #Best Parameters and cost
        self.bestParams = nm.array([0]);
        self.bestCost = float('nan');
    
        self.lastParams = None
    
    def sendParameters(self,currParams):
    
        self.lastParams = currParams
        self.numRuns += 1
        print("Parameters sent to experiment. "+str(self.numRuns))
    
    def getCost(self):
        
        noP = self.lastParams.size
        costArray = -nm.sinc(self.lastParams)/noP
        currCost = nm.sum(costArray)
        currUncer =10**(-2.0*currCost - 3.7)
        currCost = currCost + nr.normal()*currUncer
        currBad = False
        
        #distToMin = math.sqrt(nm.sum(nm.square(currParams - minR)))
        #currCost = distToMin 
                
        #Save history for this run
        self.histCosts.append(currCost)
        self.histUncers.append(currUncer)
        self.histParams.append(self.lastParams)
        self.histBads.append(currBad)
                
        if (self.numRuns==1):
            self.bestCost = currCost
            self.bestParams = self.lastParams
        else:
            if (currCost < self.bestCost):
                self.bestCost = currCost
                self.bestParams = self.lastParams
            
        #Write history to file
        self.saveHistory()
        
        #Return cost
        return currCost, currUncer, currBad

    def saveHistory(self):
        """
        Adds to a file currently tracking the experiment
        """
        
        si.savemat('ExpTracking'+self.fileAppend+'.mat', self.__dict__)

    def sendParametersGetCost(self,currParams):
        
        self.sendParameters(currParams)
        return self.getCost()
