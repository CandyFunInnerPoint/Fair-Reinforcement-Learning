import torch
from collections import OrderedDict

class AgentBase(object):
    def __init__(self):
        pass

    def read_algoDictDict_from_files(self, algoDictFileNameList):
        algoDictDict = OrderedDict()
        for algoDictFileName in algoDictFileNameList:
            checkPoint = torch.load(algoDictFileName)
            for key, value in checkPoint.items():
#                print(key)
                algoInfo = key.split("_")
                algoType = algoInfo[0]
                trainEpisodeIndex = algoInfo[1]
                trainSubEpisodeIndex = algoInfo[2]
                nnTrainStepIndex = algoInfo[3]
#                numNnTrainStepsLogging = algoInfo[3]
                newKey = trainEpisodeIndex + "_" + trainSubEpisodeIndex + "_" + nnTrainStepIndex# + "_" + numNnTrainStepsLogging
                if newKey not in algoDictDict:
                    algoDictDict[newKey] = OrderedDict()
                algoDictDict[newKey][algoType] = value
        return algoDictDict
