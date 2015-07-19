'''
Start of main routine
---------------------
'''

#Clean up directory


import sys
import numpy as nm
import numpy.random as nr
import os
import dill
import cProfile

import mloop as ml

#import scipy.optimize as so
#import scipy.io as si
#import time
#import os
#import datetime
#import math
#from test.test_logging import ConfigDictTest

#Clean up directory
if os.path.isfile("ExpHalt.txt"):
    os.remove("ExpHalt.txt")
if os.path.isfile("ExpOutput.mat"):
    os.remove("ExpOutput.mat")
if os.path.isfile("ExpInput.txt"):
    os.remove("ExpInput.txt")

#Check config file actually exists    
if not(os.path.isfile("ExpConfig.txt")):
    sys.exit("ExpConfig.txt does not exist.");

    
with open("ExpConfig.txt",'r') as initInputFile:
    #Extract info from file
    configString = ""
    numConfigLines = 0
    for line in initInputFile:
        temp = line.strip("\n").strip()
        if temp != '':
            configString += temp +","
            numConfigLines += 1

    
    configString += "NumberOfConfigLines=" + str(numConfigLines)

configDict = eval("dict("+configString+")")

print("Initial configuration file parsed as:")
print(configDict)

try:
    
    if 'TestInterface' in configDict and bool(configDict['TestInterface']):
        #For testing 
        interface = ml.TestingExpInterface()
    else:
        interface = ml.ExpInterface()
    
    if 'ProfileLearner' in configDict and bool(configDict['ProfileLearner']):
        profileLearner = True
    else:
        profileLearner = False
    
    if configDict['ControllerType']=='NelderMead':
        
        numPlannedRuns = int(configDict['NumberOfRuns'])
        
        numParams = int(configDict['NumberOfParameters'])
        
        boundaryConditions = str(configDict['BoundaryConditions'])
        
        if boundaryConditions=='Hard':
        
            minBoundary = nm.array(configDict['MinimumBoundaries'], dtype=float)
            maxBoundary = nm.array(configDict['MaximumBoundaries'], dtype=float)
        
        elif boundaryConditions=='None':
            
            minBoundary = nm.array([float('-inf') for i in range(numParams)], dtype=float)
            maxBoundary = nm.array([float('inf') for i in range(numParams)], dtype=float)
            
        else:
            
            sys.exit('Boundary conditions for NelderMead not correctly specified.')
            
        initCondSetup = str(configDict['InitialConditionsSetup'])
        
        if initCondSetup=='Explicit':
            
            initParams = nm.array(configDict['InitialParameters'], dtype=float)
            initSimplexDisp = nm.array(configDict['InitialSimplexDisplacements'], dtype=float)
        
        elif initCondSetup=='Scaled':
            
            if ('MinimumInitialParameters' in configDict):
            
                minSimplex = nm.array(configDict['MinimumInitialParameters'], dtype=float)
                maxSimplex = nm.array(configDict['MaximumInitialParameters'], dtype=float)
                
            else:
                print("No bounds for initial simplex given. Using Boundaries instead")
                minSimplex = minBoundary
                maxSimplex = maxBoundary
            
            if (nm.all((maxSimplex-minSimplex)>0.0)==False):
                sys.exit("Maximum simplex values are not larger than minimum values.")
  
            
            initScaleFac = float(configDict['SimplexScaleSize'])
            
            if ((initScaleFac<=0)|(initScaleFac>=1)) :
                sys.exit("Scale factor is not between 0 and 1")
            
            diffSimplex = maxSimplex - minSimplex
            initParams = nm.array(configDict['InitialParameters'], dtype=float)
            initSimplexDisp = initScaleFac*diffSimplex 
            
            
        elif initCondSetup=='Random':
            
            if ('MinimumInitialParameters' in configDict):
            
                minRandom = nm.array(configDict['MinimumInitialParameters'], dtype=float)
                maxRandom = nm.array(configDict['MaximumInitialParameters'], dtype=float)
            
            else:
                print("No bounds for initial random simplex given. Using Boundaries instead")
                minRandom = minBoundary
                maxRandom = maxBoundary
            
            if (nm.all((maxRandom-minRandom)>0.0)==False):
                sys.exit("Maximum random values are not larger than minimum values.")
            
            initScaleFac = float(configDict['SimplexScaleSize'])
            
            if ((initScaleFac<=0)|(initScaleFac>=1)) :
                sys.exit("Scale factor is not between 0 and 1")
            
            diffRandom = maxRandom-minRandom
            initParams = minRandom + (1.0-initScaleFac)*nr.rand(numParams)*diffRandom
            initSimplexDisp = initScaleFac*diffRandom
            
        else :
            
            sys.exit('No correct InitialConditionsSetup given.') 
            
        controller = ml.NelderController(interface, numPlannedRuns, numParams, minBoundary, maxBoundary, initParams, initSimplexDisp, configDict)
        
        controller.runOptimization()
    
    elif configDict['ControllerType']=='Random':
        
        numPlannedRuns = int(configDict['NumberOfRuns'])
        
        numParams = int(configDict['NumberOfParameters'])
        
        minBoundary = nm.array(configDict['MinimumBoundaries'], dtype=float)
        maxBoundary = nm.array(configDict['MaximumBoundaries'], dtype=float)
    
        controller = ml.RandomController(interface, numPlannedRuns, numParams, minBoundary, maxBoundary, configDict)
        
        controller.runOptimization()
        
    
    elif (configDict['ControllerType']=='GlobalLearner'): # or (configDict['ControllerType']=='AutoLearner')
        
        numParams = int(configDict['NumberOfParameters'])
        
        if 'CorrelationLengths' in configDict:
            corrLength = nm.array(configDict['CorrelationLengths'], dtype=float)
            print("CorrelationLengths set to user value")
        else:
            corrLength = None
            
        
        if configDict['InitialTrainingSource']=='Random':
            
            if ('MinimumInitialParameters' in configDict):
            
                minRandom = nm.array(configDict['MinimumInitialParameters'], dtype=float)
                maxRandom = nm.array(configDict['MaximumInitialParameters'], dtype=float)
            
            else:
            
                print("No bounds for initial random simplex given. Using Boundaries instead")
                minRandom = nm.array(configDict['MinimumBoundaries'], dtype=float)
                maxRandom = nm.array(configDict['MaximumBoundaries'], dtype=float)
            
            numPlannedRuns = int(configDict['NumberOfTrainingRuns'])
            
            initContr = ml.RandomController(interface, numPlannedRuns, numParams, minRandom, maxRandom, configDict)
            
            initContr.runOptimization()
            
            initParams = initContr.nextParams()
            
            initInterface = interface
        
        elif configDict['InitialTrainingSource']=='Simplex':
            
            minBoundary = nm.array(configDict['MinimumBoundaries'], dtype=float)
            maxBoundary = nm.array(configDict['MaximumBoundaries'], dtype=float)
            
            if ('MinimumInitialParameters' in configDict):
            
                minSimplex = nm.array(configDict['MinimumInitialParameters'], dtype=float)
                maxSimplex = nm.array(configDict['MaximumInitialParameters'], dtype=float)
                
            else:
                print("No bounds for initial simplex given. Using Boundaries instead")
                minSimplex = minBoundary
                maxSimplex = maxBoundary
            
            if (nm.all((maxSimplex-minSimplex)>0.0)==False):
                sys.exit("Maximum simplex values are not larger than minimum values.")
            
            initScaleFac = float(configDict['SimplexScaleSize'])
            
            if ((initScaleFac<=0)|(initScaleFac>=1)) :
                sys.exit("Scale factor is not between 0 and 1")
            
            diffSimplex = maxSimplex - minSimplex
            initParams = nm.array(configDict['InitialParameters'], dtype=float)
            initSimplexDisp = initScaleFac*diffSimplex 
            
            if ('NumberOfTrainingRuns' in configDict):
                numPlannedRuns = int(configDict['NumberOfTrainingRuns'])
            else:
                print("NumberOfTrainingRuns not provided setting to default for simplex NumberOrParameters + 1")
                numPlannedRuns = numParams + 1
            
            initContr = ml.NelderController(interface, numPlannedRuns, numParams, minBoundary, maxBoundary, initParams, initSimplexDisp, configDict)
            
            initContr.runOptimization()
            
            initInterface = interface
            
            initParams = initContr.nextParams()
            
        elif configDict['InitialTrainingSource']=='FromFile':
            
            if ('FileName' in configDict):
                fileName = str(configDict['FileName'])
                
            else:
                
                print('FileName not provided. Setting to default ControllerArchive.pkl')
                fileName = 'ControllerArchive.pkl'
            
            with open(fileName,'rb') as archiveFile:
                initContr = dill.load(archiveFile)
            
            initInterface = initContr.interface
            
            initParams = initInterface.bestParams
            
        else: 
            
            sys.exit('No recognized InitialLearningDataSource given. Read README')
        
        trainingParams = list(initInterface.histParams)
        trainingCosts = list(initInterface.histCosts)
        trainingUncers = list(initInterface.histUncers)
        trainingBads = list(initInterface.histBads)
        trainingNum = len(trainingCosts)
        
        minBoundary = nm.array(configDict['MinimumBoundaries'], dtype=float)
        maxBoundary = nm.array(configDict['MaximumBoundaries'], dtype=float)
        minCost = nm.array(configDict['MinimumCost'])
        maxCost = nm.array(configDict['MaximumCost'])
        numPlannedRuns = int(configDict['NumberOfRuns'])
        
        if 'NumberOfThetaSearches' in configDict:
            thetaSearchs = int(configDict['NumberOfThetaSearches'])
            print("NumberOfThetaSearches set to user value: " + str(thetaSearchs))
        else:
            thetaSearchs = None
        
        if 'NumberOfParameterSearches' in configDict:
            paramsSearchs = int(configDict['NumberOfParameterSearches'])
            print("NumberOfParameterSearches set to user value: " + str(paramsSearchs))
        else:
            paramsSearchs = None
        
        if 'NumberOfParticles' in configDict:
            particleNumber = int(configDict['NumberOfParticles'])
            print("NumberOfParticles set to user value: " + str(particleNumber))
        else:
            particleNumber = None
        
        
        if configDict['ControllerType']=='GlobalLearner':
            
            if 'RunsInSweep' in configDict:
                sweepRuns = int(configDict['RunsInSweep'])
                print("RunsInSweep set to user value: " + str(sweepRuns))
            else:
                sweepRuns = None
            
            if 'LeashSize' in configDict:
                leashSize = float(configDict['LeashSize'])
                print("LeashSize set to user value: " + str(leashSize))
            else:
                leashSize = None
            
            controller = ml.GlobalLearner(interface, numPlannedRuns, numParams, minBoundary, maxBoundary, minCost, maxCost, trainingNum, trainingParams, trainingCosts, trainingUncers, trainingBads, initParams, configDict, leashSize, sweepRuns, particleNumber, thetaSearchs, paramsSearchs, corrLength)
        
        else:
            
            sys.exit('No recognized InitialLearningDataSource given. Read README')
        
        try:
            if profileLearner:
                cProfile.run('controller.runOptimization()')
            else:
                controller.runOptimization()
        except KeyboardInterrupt:
            print('Program halted with ctrl-c. Safely wrapping things up...')
            controller.saveEverything()
        
        controller.calculateMatlabFileExtras()
        controller.saveEverything()
        
    else:
        
        sys.exit('No recognized ControllerType given. Read README.')
    
except KeyError:
    print("The following necessary configuration property was missing from the file:")
    raise

print('Maximum number of experimental runs reach. Ending optimization and halting experiment.')

with open('ExpHalt.txt','w') as expHaltFile:
        expHaltFile.write('Stop dammit!!!')