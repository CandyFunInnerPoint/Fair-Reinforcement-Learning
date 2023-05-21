from Node import Node
from Path import Path
from Config import Config
import numpy as np
import os
import json, csv
import pandas as pd

import torch

class MemMgmt(Node, Path):
    def __init__(self):
        self.savedPath = None

    def read_config(self, identifier, pipeNameIptDict, pipeNameOptDict, diskMode):
        pass


    def run(self):
        pass

    def read_data_from_file(self):
        pass

    def write_data_to_file(self):
        pass



class MemMgmtOptTwoClassNormalDist(MemMgmt):
    def __init__(self):
        super(MemMgmtOptTwoClassNormalDist, self).__init__()

    def read_config(self, identifier, rootPath, optConfigCurDir, optDataCurDir, iptConfigHistDir, iptDataHistDir, pipeNameOptDict, diskMode, inputDim):
        self.identifier = identifier

        iptConfigFileName, self.optConfigCurDir, self.iptConfigHistDir, self.optDataCurDir = Path.make_path(rootPath, self.__class__.__name__, identifier, optConfigCurDir, iptConfigHistDir, optDataCurDir)
        self.iptDataHistDir = iptDataHistDir

        with open(iptConfigFileName) as json_file:
            config = json.load(json_file)
            

            self.pipeNameOptDict = pipeNameOptDict

            self.diskMode = diskMode
            self.inputDim = inputDim

            if self.diskMode == "ReadFromDisk":
                self.exdIptDataHistDir = iptDataHistDir + self.__class__.__name__ + self.identifier + "/"#config["exdRelIptDataHistDir"]


            Config.write_config(config, self.optConfigCurDir + self.__class__.__name__ + self.identifier + ".json")
 
    def run(self):
        if self.diskMode == "ReadFromDisk":
            normalClass1Mean, normalClass1Cov = self.read_para_from_file("normalClass1")
            normalClass2Mean, normalClass2Cov = self.read_para_from_file("normalClass2")
            
            self.pipeNameOptDict["normalClass1Mean"] = normalClass1Mean
            self.pipeNameOptDict["normalClass1Cov"] = normalClass1Cov
            self.pipeNameOptDict["normalClass2Mean"] = normalClass2Mean
            self.pipeNameOptDict["normalClass2Cov"] = normalClass2Cov

        elif self.diskMode == "WriteToDisk":
            normalClass1Mean = self.pipeNameOptDict["normalClass1Mean"]
            normalClass1Cov = self.pipeNameOptDict["normalClass1Cov"]
            normalClass2Mean = self.pipeNameOptDict["normalClass2Mean"]
            normalClass2Cov = self.pipeNameOptDict["normalClass2Cov"]

            self.write_para_to_file(normalClass1Mean, normalClass1Cov, "normalClass1")
            self.write_para_to_file(normalClass2Mean, normalClass2Cov, "normalClass2")

    def read_para_from_file(self, name):
        fileName = self.exdIptDataHistDir + name + ".csv"
        para = np.loadtxt(fileName, delimiter = ",")
        mean = para[:self.inputDim]
        cov = para[self.inputDim:].reshape((self.inputDim,self.inputDim))
        return mean, cov

    def write_para_to_file(self, mean, cov, name):
        fileName = self.optDataCurDir + name + ".csv"
        para = np.concatenate((mean.reshape(-1,1), cov.reshape(-1, 1)), axis = 0)
        np.savetxt(fileName, para, delimiter = ",")


class MemMgmtOptTwoClassNormalSampler(MemMgmt):
    def __init__(self):
        super(MemMgmtOptTwoClassNormalSampler, self).__init__()

    def read_config(self, identifier, rootPath, optConfigCurDir, optDataCurDir, iptConfigHistDir, iptDataHistDir, device, pipeNameOptDict, diskMode, inputDim):
        self.identifier = identifier

        iptConfigFileName, self.optConfigCurDir, self.iptConfigHistDir, self.optDataCurDir = Path.make_path(rootPath, self.__class__.__name__, identifier, optConfigCurDir, iptConfigHistDir, optDataCurDir)
        self.iptDataHistDir = iptDataHistDir

        with open(iptConfigFileName) as json_file:
            config = json.load(json_file)
            

            self.device = device
 
            self.pipeNameOptDict = pipeNameOptDict

            self.diskMode = diskMode
            self.inputDim = inputDim

            if self.diskMode == "ReadFromDisk":
                self.exdIptDataHistDir = iptDataHistDir + self.__class__.__name__ + self.identifier + "/"#config["exdRelIptDataHistDir"]

            Config.write_config(config, self.optConfigCurDir + self.__class__.__name__ + self.identifier + ".json")

    def run(self):
        if self.diskMode == "ReadFromDisk":
            dataInputState, dataTargetState = self.read_data_from_file("dataState")
            dataInputReward, dataTargetReward = self.read_data_from_file("dataReward")

            dataInputState = torch.from_numpy(dataInputState).float().to(self.device)
            dataTargetState = torch.from_numpy(dataTargetState).float().to(self.device)
            dataInputReward = torch.from_numpy(dataInputReward).float().to(self.device)
            dataTargetReward = torch.from_numpy(dataTargetReward).float().to(self.device)

            self.pipeNameOptDict["dataInputState"] = dataInputState
            self.pipeNameOptDict["dataTargetState"] = dataTargetState
            self.pipeNameOptDict["dataInputReward"] = dataInputReward
            self.pipeNameOptDict["dataTargetReward"] = dataTargetReward

        elif self.diskMode == "WriteToDisk":
            dataInputState = self.pipeNameOptDict["dataInputState"]
            dataTargetState = self.pipeNameOptDict["dataTargetState"]
            dataInputReward = self.pipeNameOptDict["dataInputReward"] 
            dataTargetReward = self.pipeNameOptDict["dataTargetReward"] 

            dataInputState = dataInputState.cpu().numpy()
            dataTargetState = dataTargetState.cpu().numpy()
            dataInputReward = dataInputReward.cpu().numpy()
            dataTargetReward = dataTargetReward.cpu().numpy()

            self.write_data_to_file(dataInputState, dataTargetState, "dataState")
            self.write_data_to_file(dataInputReward, dataTargetReward, "dataReward")
             

    def read_data_from_file(self, name):
        fileName = self.exdIptDataHistDir + name + ".csv"
        data = np.loadtxt(fileName, delimiter = ",")
        dataInput = data[:,:self.inputDim]
        dataTarget = data[:,self.inputDim].reshape((-1,1))
        return dataInput, dataTarget


    def write_data_to_file(self, dataInput, dataTarget, name):
        fileName = self.optDataCurDir + name + ".csv"
        data = np.concatenate((dataInput, dataTarget), axis = 1)
        np.savetxt(fileName, data, delimiter = ",")#, sep = ",", format = "%10.5f")

class MemMgmtOptInitNnModelParaLossLevelGen(MemMgmt):
    def __init__(self):
        super(MemMgmtOptInitNnModelParaLossLevelGen, self).__init__()

    def read_config(self, identifier, rootPath, optConfigCurDir, optDataCurDir, iptConfigHistDir, iptDataHistDir, device, pipeNameOptDict, diskMode, inputDim):
        self.identifier = identifier

        iptConfigFileName, self.optConfigCurDir, self.iptConfigHistDir, self.optDataCurDir = Path.make_path(rootPath, self.__class__.__name__, identifier, optConfigCurDir, iptConfigHistDir, optDataCurDir)
        self.iptDataHistDir = iptDataHistDir

        with open(iptConfigFileName) as json_file:
            config = json.load(json_file)
            self.device = device
 
            self.pipeNameOptDict = pipeNameOptDict

            self.diskMode = diskMode
            self.inputDim = inputDim

            if self.diskMode == "ReadFromDisk":
                self.exdIptDataHistDir = iptDataHistDir + self.__class__.__name__ + self.identifier + "/"#config["exdRelIptDataHistDir"]
                self.nnModelParaFileName = iptDataHistDir + config["exdRelNnModelParaFileName"] 
                self.lossLevelFileName = iptDataHistDir + config["exdRelLossLevelFileName"] 

            Config.write_config(config, self.optConfigCurDir + self.__class__.__name__ + self.identifier + ".json")

    def run(self):
        if self.diskMode == "ReadFromDisk":
            nnModelParaList = self.read_nnModelParaList_from_file()
            lossLevel = self.read_lossLevel_from_file()

            self.pipeNameOptDict["nnModelParaList"] = nnModelParaList
            self.pipeNameOptDict["lossLevel"] = lossLevel

        elif self.diskMode == "WriteToDisk":
            nnModelParaList = self.pipeNameOptDict["nnModelParaList"]
            lossLevel = self.pipeNameOptDict["lossLevel"]

            self.write_nnModelParaList_to_file(nnModelParaList)
            self.write_lossLevel_to_file(lossLevel)

    def read_nnModelParaList_from_file(self):
        nnModelParaList = torch.load(self.nnModelParaFileName)
        return nnModelParaList

    def read_lossLevel_from_file(self):
        lossLevel = torch.load(self.lossLevelFileName)
        print("lossLevel, ", lossLevel)

        return lossLevel

    def write_nnModelParaList_to_file(self, nnModelParaList):
        torch.save(nnModelParaList, self.optDataCurDir + "nnModelPara")

    def write_lossLevel_to_file(self, lossLevel):
        torch.save(lossLevel, self.optDataCurDir + "lossLevel")

class MemMgmtOptActorCriticsBackpropTrain(MemMgmt):
    def __init__(self):
        super(MemMgmtOptActorCriticsBackpropTrain, self).__init__()

    def read_config(self, identifier, rootPath, optConfigCurDir, optDataCurDir, iptConfigHistDir, iptDataHistPathList, pipeNameOptDict, diskMode):
        self.identifier = identifier

        iptConfigFileName, self.optConfigCurDir, self.iptConfigHistDir, self.optDataCurDir = Path.make_path(rootPath, self.__class__.__name__, identifier, optConfigCurDir, iptConfigHistDir, optDataCurDir)

        self.iptDataHistPathList = iptDataHistPathList

        with open(iptConfigFileName) as json_file:
            config = json.load(json_file)

            self.pipeNameOptDict = pipeNameOptDict

            self.diskMode = diskMode

            if self.diskMode == "ReadFromDisk":
                pass

            Config.write_config(config, self.optConfigCurDir + self.__class__.__name__ + self.identifier + ".json")

    def run(self):
        if self.diskMode == "ReadFromDisk":
            algoDictFileNameList = self.iptDataHistPathList###self.read_algoDictFileNameList_from_files()#"algoFromTrain")
            self.pipeNameOptDict["algoDictFileNameList"] = algoDictFileNameList

        elif self.diskMode == "WriteToDisk":
            pass

    def read_algoDictFileNameList_from_files(self):#, endAlgoFileKeyName):

        algoDictFileNameList = []
        for fileName in os.listdir(self.algoDictFileDir):#self.exdIptDataHistDir + endAlgoFileKeyName + "/"):
            result = fileName.split("_")
            logStartTrainEpisodeIndex = int(result[0])
            logStartNnTrainStepIndex = int(result[1])
            logEndTrainEpisodeIndex = int(result[2])
            logEndNnTrainStepIndex = int(result[3])
            numNnTrainStepsLogging = int(result[4])
            algoDictFileNameList.append(self.algoDictFileDir + fileName)#self.exdIptDataHistDir + endAlgoFileKeyName + "/" + fileName)

        return algoDictFileNameList


class MemMgmtOptActorCriticsBackpropEval(MemMgmt):
    def __init__(self):
        super(MemMgmtOptActorCriticsBackpropEval, self).__init__()

    def read_config(self, identifier, rootPath, optConfigCurDir, optDataCurDir, iptConfigHistDir, iptDataHistDir, pipeNameOptDict, diskMode):
        self.identifier = identifier

        iptConfigFileName, self.optConfigCurDir, self.iptConfigHistDir, self.optDataCurDir = Path.make_path(rootPath, self.__class__.__name__, identifier, optConfigCurDir, iptConfigHistDir, optDataCurDir)
        self.iptDataHistDir = iptDataHistDir

        with open(iptConfigFileName) as json_file:
            config = json.load(json_file)

            self.pipeNameOptDict = pipeNameOptDict

            self.diskMode = diskMode

            if self.diskMode == "ReadFromDisk":
                self.metricPdFileDir = iptDataHistDir + config["exdRelMetricPdFileDir"]

            Config.write_config(config, self.optConfigCurDir + self.__class__.__name__ + self.identifier + ".json")

    def run(self):
        if self.diskMode == "ReadFromDisk":
            metricPd = self.read_metricPd_from_files()#"evalDataLog")

            self.pipeNameOptDict["metricPd"] = metricPd

        elif self.diskMode == "WriteToDisk":
            pass

    def read_metricPd_from_files(self):#, endProcFileKeyName):

        metricProcPdList = []
        for fileName in os.listdir(self.metricPdFileDir):#dirr):
            metricProcPd = pd.read_csv(self.metricPdFileDir + fileName, header = 0)
            metricProcPdList.append(metricProcPd)
        metricPd = pd.concat(metricProcPdList, axis = 0)
        return metricPd

class MemMgmtOptSgdEval(MemMgmt):
    def __init__(self):
        super(MemMgmtOptSgdEval, self).__init__()

    def read_config(self, identifier, rootPath, optConfigCurDir, optDataCurDir, iptConfigHistDir, iptDataHistDir, pipeNameOptDict, diskMode):
        self.identifier = identifier

        iptConfigFileName, self.optConfigCurDir, self.iptConfigHistDir, self.optDataCurDir = Path.make_path(rootPath, self.__class__.__name__, identifier, optConfigCurDir, iptConfigHistDir, optDataCurDir)
        self.iptDataHistDir = iptDataHistDir

        with open(iptConfigFileName) as json_file:
            config = json.load(json_file)

            self.pipeNameOptDict = pipeNameOptDict

            self.diskMode = diskMode

            if self.diskMode == "ReadFromDisk":
                self.metricPdFileDir = iptDataHistDir + config["exdRelMetricPdFileDir"]

            Config.write_config(config, self.optConfigCurDir + self.__class__.__name__ + self.identifier + ".json")

    def run(self):
        if self.diskMode == "ReadFromDisk":
            metricPd = self.read_metricPd_from_files()#"evalDataLog")

            self.pipeNameOptDict["metricPd"] = metricPd

        elif self.diskMode == "WriteToDisk":
            pass

    def read_metricPd_from_files(self):#, endProcFileKeyName):

        metricProcPdList = []
        for fileName in os.listdir(self.metricPdFileDir):#dirr):
            metricProcPd = pd.read_csv(self.metricPdFileDir + fileName, header = 0)
            metricProcPdList.append(metricProcPd)
        metricPd = pd.concat(metricProcPdList, axis = 0)
        return metricPd


class MemMgmtOptRLFairTrain(MemMgmt):
    def __init__(self):
        super(MemMgmtOptRLFairTrain, self).__init__()

    def read_config(self, identifier, rootPath, optConfigCurDir, optDataCurDir, iptConfigHistDir, iptDataHistDirList, pipeNameOptDict, diskMode):
        self.identifier = identifier

        iptConfigFileName, self.optConfigCurDir, self.iptConfigHistDir, self.optDataCurDir = Path.make_path(rootPath, self.__class__.__name__, identifier, optConfigCurDir, iptConfigHistDir, optDataCurDir)
        self.iptDataHistDirList = iptDataHistDirList
 
        with open(iptConfigFileName) as json_file:
            config = json.load(json_file)

            self.pipeNameOptDict = pipeNameOptDict

            self.diskMode = diskMode

            if self.diskMode == "ReadFromDisk":
                pass

            Config.write_config(config, self.optConfigCurDir + self.__class__.__name__ + self.identifier + ".json")

    def run(self):
        if self.diskMode == "ReadFromDisk":
            trainDataLogPd = self.read_trainDataLogPd_from_files()#"evalDataLog")
            optObjConstrPd = self.read_optObjConstrPd_from_files()

            self.pipeNameOptDict["trainDataLogPd"] = trainDataLogPd
            self.pipeNameOptDict["optObjConstrPd"] = optObjConstrPd

        elif self.diskMode == "WriteToDisk":
            pass

    def read_trainDataLogPd_from_files(self):#, endProcFileKeyName):
        iptDataHistPathList = [iptDataHistDir + "trainDataLog" for iptDataHistDir in self.iptDataHistDirList]

        trainDataLogProcPdList = []
        for path in iptDataHistPathList: #os.listdir(self.metricPdFileDir):#dirr):
            trainDataLogProcPd = pd.read_csv(path, header = 0)
            trainDataLogProcPdList.append(trainDataLogProcPd)
        trainDataLogPd = pd.concat(trainDataLogProcPdList, axis = 0)

        trainDataLogPd.to_csv(self.optDataCurDir + self.__class__.__name__ + self.identifier + "trainDataLogPd.csv", index = False)
        return trainDataLogPd

    def read_optObjConstrPd_from_files(self):#, endProcFileKeyName):
        iptDataHistPathList = [iptDataHistDir + "AgentTrain/optObjConstrForUpdateLog" for iptDataHistDir in self.iptDataHistDirList]

        optObjConstrProcPdList = []
        for path in iptDataHistPathList: #os.listdir(self.metricPdFileDir):#dirr):
            optObjConstrProcPd = pd.read_csv(path, header = 0)
            optObjConstrProcPdList.append(optObjConstrProcPd)
        optObjConstrPd = pd.concat(optObjConstrProcPdList, axis = 0)

        optObjConstrPd.to_csv(self.optDataCurDir + self.__class__.__name__ + self.identifier + "optObjConstrPd.csv", index = False)
        return optObjConstrPd

class MemMgmtOptRLFairEval(MemMgmt):
    def __init__(self):
        super(MemMgmtOptRLFairEval, self).__init__()

    def read_config(self, identifier, rootPath, optConfigCurDir, optDataCurDir, iptConfigHistDir, iptDataHistDir, pipeNameOptDict, diskMode):
        self.identifier = identifier

        iptConfigFileName, self.optConfigCurDir, self.iptConfigHistDir, self.optDataCurDir = Path.make_path(rootPath, self.__class__.__name__, identifier, optConfigCurDir, iptConfigHistDir, optDataCurDir)
        self.iptDataHistDir = iptDataHistDir
 
        with open(iptConfigFileName) as json_file:
            config = json.load(json_file)

            self.pipeNameOptDict = pipeNameOptDict

            self.diskMode = diskMode

            if self.diskMode == "ReadFromDisk":
                pass

            Config.write_config(config, self.optConfigCurDir + self.__class__.__name__ + self.identifier + ".json")

    def run(self):
        if self.diskMode == "ReadFromDisk":
            evalDataLogPd = self.read_evalDataLogPd_from_files()#"evalDataLog")

            self.pipeNameOptDict["evalDataLogPd"] = evalDataLogPd

        elif self.diskMode == "WriteToDisk":
            pass

    def read_evalDataLogPd_from_files(self):#, endProcFileKeyName):

        evalDataLogProcPdList = []

        for path in os.listdir(self.iptDataHistDir):
            evalDataLogProcPd = pd.read_csv(self.iptDataHistDir + path, header = 0)
            evalDataLogProcPdList.append(evalDataLogProcPd)
        evalDataLogPd = pd.concat(evalDataLogProcPdList, axis = 0)

        evalDataLogPd.to_csv(self.optDataCurDir + self.__class__.__name__ + self.identifier + ".csv", index = False)
        return evalDataLogPd


