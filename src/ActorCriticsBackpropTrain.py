import json, sys, os
from os import path

import math

import pandas as pd


from AgentHg import AgentHg
from AgentAggOrigGp import AgentAggOrigGp
from AgentAggSgtGp import AgentAggSgtGp
from AgentMypFair import AgentMypFair
from AgentMypUnfair import AgentMypUnfair
from AgentDpOrigGp import AgentDpOrigGp
from AgentDpSgtGp import AgentDpSgtGp
from AgentDpCvOrigGp import AgentDpCvOrigGp
from AgentDpCvSgtGp import AgentDpCvSgtGp
from AgentEqOptOrigGp import AgentEqOptOrigGp
from AgentEqOptSgtGp import AgentEqOptSgtGp
from AgentEqOptCvOrigGp import AgentEqOptCvOrigGp
from AgentEqOptCvSgtGp import AgentEqOptCvSgtGp


from EnvTrainSim import EnvTrainSim
from EnvTrainReal import EnvTrainReal
from EnvTrainSimExplicit import EnvTrainSimExplicit
from EnvTrainRealExplicit import EnvTrainRealExplicit
from EnvTrainSimGenerative import EnvTrainSimGenerative
from EnvTrainRealGenerative import EnvTrainRealGenerative

from DataLogger import DataLogger

import torch

from Config import Config
from Node import Node
from Path import Path

from torch.profiler import profile, record_function, ProfilerActivity

import random

#from torch.profiler import schedule

class ActorCriticsBackpropTrain(Node):
    def __init__(self):
        self.path = None
        self.useConfigMode = None
        self.device = None

        self.agent = None
        self.envTrain = None

        self.dataDim = None

        self.trainDataInputState = None
        self.trainDataTargetState = None
        self.trainDataInputReward = None
        self.trainDataTargetReward = None

        self.gamma = None
        self.lrCritic = None
        self.lrActor = None

        self.numTrainEpisode = None
        self.numNnTrainSteps = None
       
        self.trainReplayMemory = None
        self.trainReplayMemoryCapacity = None
        self.replayMemoryBatchSize = None
        self.replayMemoryWarmStartNumSteps = None
        self.numNnTrainStepsSampling = None

        self.nnBatchSize = None
        
        self.nnHiddenDim = None
        self.nnTargetDim = None

    def read_config(self, identifier, rootDir, exprimentHyperConfigDir, optConfigCurDir, optDataCurDir, iptConfigHistDir, iptDataHistDir, curLocalTimeInMicro, device, pipeNameOptDict, dataDim):
        iptConfigFileName, self.optConfigCurDir, self.iptConfigHistDir, self.optDataCurDir = Path.make_path(rootDir, self.__class__.__name__, identifier, optConfigCurDir, iptConfigHistDir, optDataCurDir)
        self.iptDataHistDir = iptDataHistDir

        self.exprimentHyperConfigDir = exprimentHyperConfigDir
        self.curLocalTimeInMicro = curLocalTimeInMicro

#        self.hyperConfigIndex = hyperConfigIndex

        if self.curLocalTimeInMicro != None:
            with open(self.exprimentHyperConfigDir + self.__class__.__name__ + "Hyper" + str(self.curLocalTimeInMicro) + ".json") as json_file:
                config = json.load(json_file)
                self.paretoType = config["paretoType"]
                self.evalType = config["evalType"]
                self.envType = config["envType"]
                self.agentName = config["agentName"]

                self.numInterPolicyUpdateEpisodes = config["numInterPolicyUpdateEpisodes"]
                self.trialTrainIndex = config["trialTrainIndex"]

        with open(iptConfigFileName) as json_file:
            config = json.load(json_file)

                #rl training algo
            self.gamma = config["gamma"]
            self.lrCritic = config["lrCritic"]
            self.lrActor = config["lrActor"]

            
            self.numTrainSubEpisode = config["numTrainSubEpisode"]
            self.numNnTrainSteps = config["numNnTrainSteps"]

            
            self.numInterPolicyUpdateEvalEpisodes = config["numInterPolicyUpdateEvalEpisodes"]
            self.numInterPolicyEvalEpisodes = config["numInterPolicyEvalEpisodes"]


            self.trainReplayMemoryCapacity = config["trainReplayMemoryCapacity"]
            self.replayMemoryBatchSize = config["replayMemoryBatchSize"]
            self.replayMemoryWarmStartNumSteps = config["replayMemoryWarmStartNumSteps"]
            self.numNnTrainStepsSampling = config["numNnTrainStepsSampling"]
            


            #nn algo
            self.nnBatchSize = config["nnBatchSize"]
            
            self.nnHiddenDim = config["nnHiddenDim"]
            self.nnTargetDim = config["nnTargetDim"]
            self.nnLayerDim = config["nnLayerDim"]

            

            self.identifier = identifier


            self.pipeNameOptDict = pipeNameOptDict
                
            self.trainLoggerCapacity = config["trainLoggerCapacity"]
            self.trainLoggerFileName = config["trainLoggerFileName"]

            self.numPolicyUpdates = config["numPolicyUpdates"]
            self.numStartPolicyUpdates = config["numStartPolicyUpdates"]

            self.numTrainEpisode = config["numTrainEpisode"]

            if self.curLocalTimeInMicro == None:#hyperConfigIndex == None:
                self.evalType = config["evalType"]
                self.envType = config["envType"]


                self.numInterPolicyUpdateEpisodes = config["numInterPolicyUpdateEpisodes"]


            self.grpProbPath = config["grpProbPath"]
            groupProbList = config["groupProbList"]

            self.useTrueModel = config["useTrueModel"]

            self.device = device


            self.dataDim = dataDim

            self.compute_globaUpdateEpisodeIndex_estConstrForUpdate()

            print("agentName: ", self.agentName)

            if self.envType == "Sim":
                self.groupProbList = groupProbList
                self.agent = globals()[self.agentName]()
                self.agent.read_config("", rootDir, self.exprimentHyperConfigDir, self.optConfigCurDir, self.optDataCurDir, self.iptConfigHistDir, self.iptDataHistDir, self.curLocalTimeInMicro, self.device, self.numTrainEpisode, self.numInterPolicyUpdateEpisodes, self.numInterPolicyUpdateEvalEpisodes, self.numInterPolicyEvalEpisodes, self.numNnTrainSteps, True, self.evalType, self.envType, self.groupProbList, self.trialTrainIndex, self.useTrueModel, self.updateEpisodeIndexList, self.globalAgentEndEstEpisodeIndexList, self.estConstrForUpdate)#, self.nnBatchSize, self.dataDim, self.nnHiddenDim, self.nnTargetDim, self.nnLayerDim)#, self.nStates, self.nActions)

                self.envTrain = EnvTrainSim()
                self.envTrain.read_config("", rootDir, self.optConfigCurDir, self.optDataCurDir, self.iptConfigHistDir, self.iptDataHistDir, self.device, self.numTrainEpisode, self.numInterPolicyUpdateEvalEpisodes, self.numNnTrainSteps, self.groupProbList)#, self.nnBatchSize, self.dataDim, self.nnHiddenDim, self.nnTargetDim, self.nnLayerDim, self.numNnTrainSteps)
            elif self.envType == "Real":
                self.groupProbList = self.load_from_grpProbPath(self.grpProbPath)

                self.agent = globals()[self.agentName]()
                self.agent.read_config("", rootDir, self.exprimentHyperConfigDir, self.optConfigCurDir, self.optDataCurDir, self.iptConfigHistDir, self.iptDataHistDir, self.curLocalTimeInMicro, self.device, self.numTrainEpisode, self.numInterPolicyUpdateEpisodes, self.numInterPolicyUpdateEvalEpisodes, self.numInterPolicyEvalEpisodes, self.numNnTrainSteps, True, self.evalType, self.envType, self.groupProbList, self.trialTrainIndex, self.useTrueModel, self.updateEpisodeIndexList, self.globalAgentEndEstEpisodeIndexList, self.estConstrForUpdate)#, self.nnBatchSize, self.dataDim, self.nnHiddenDim, self.nnTargetDim, self.nnLayerDim)#, self.nStates, self.nActions)

                self.envTrain = EnvTrainReal()
                self.envTrain.read_config("", rootDir, self.optConfigCurDir, self.optDataCurDir, self.iptConfigHistDir, self.iptDataHistDir, self.device, self.numTrainEpisode, self.numInterPolicyUpdateEvalEpisodes, self.numNnTrainSteps, self.groupProbList)#, self.nnBatchSize, self.dataDim, self.nnHiddenDim, self.nnTargetDim, self.nnLayerDim, self.numNnTrainSteps)
            elif self.envType == "SimExplicit":
                self.groupProbList = groupProbList
                self.agent = globals()[self.agentName]()
                self.agent.read_config("", rootDir, self.exprimentHyperConfigDir, self.optConfigCurDir, self.optDataCurDir, self.iptConfigHistDir, self.iptDataHistDir, self.curLocalTimeInMicro, self.device, self.numTrainEpisode, self.numInterPolicyUpdateEpisodes, self.numInterPolicyUpdateEvalEpisodes, self.numInterPolicyEvalEpisodes, self.numNnTrainSteps, True, self.evalType, self.envType, self.groupProbList, self.trialTrainIndex, self.useTrueModel, self.updateEpisodeIndexList, self.globalAgentEndEstEpisodeIndexList, self.estConstrForUpdate)#, self.nnBatchSize, self.dataDim, self.nnHiddenDim, self.nnTargetDim, self.nnLayerDim)#, self.nStates, self.nActions)

                self.envTrain = EnvTrainSimExplicit()
                self.envTrain.read_config("", rootDir, self.optConfigCurDir, self.optDataCurDir, self.iptConfigHistDir, self.iptDataHistDir, self.device, self.numTrainEpisode, self.numInterPolicyUpdateEvalEpisodes, self.numNnTrainSteps, self.groupProbList, self.evalType)#, self.nnBatchSize, self.dataDim, self.nnHiddenDim, self.nnTargetDim, self.nnLayerDim, self.numNnTrainSteps)

                if self.useTrueModel:
                    initProbList, transProbList, rewardList = self.envTrain.get_initProbList_transProbList_rewardList()
                    self.agent.set_initProbList_transProbList_rewardList(initProbList, transProbList, rewardList)
            elif self.envType == "RealExplicit":
                self.groupProbList = self.load_from_grpProbPath(self.grpProbPath)
                self.agent = globals()[self.agentName]()
                self.agent.read_config("", rootDir, self.exprimentHyperConfigDir, self.optConfigCurDir, self.optDataCurDir, self.iptConfigHistDir, self.iptDataHistDir, self.curLocalTimeInMicro, self.device, self.numTrainEpisode, self.numInterPolicyUpdateEpisodes, self.numInterPolicyUpdateEvalEpisodes, self.numInterPolicyEvalEpisodes, self.numNnTrainSteps, True, self.evalType, self.envType, self.groupProbList, self.trialTrainIndex, self.useTrueModel, self.updateEpisodeIndexList, self.globalAgentEndEstEpisodeIndexList, self.estConstrForUpdate)#, self.nnBatchSize, self.dataDim, self.nnHiddenDim, self.nnTargetDim, self.nnLayerDim)#, self.nStates, self.nActions)

                self.envTrain = EnvTrainRealExplicit()
                self.envTrain.read_config("", rootDir, self.optConfigCurDir, self.optDataCurDir, self.iptConfigHistDir, self.iptDataHistDir, self.device, self.numTrainEpisode, self.numInterPolicyUpdateEvalEpisodes, self.numNnTrainSteps, self.groupProbList, self.evalType)#, self.nnBatchSize, self.dataDim, self.nnHiddenDim, self.nnTargetDim, self.nnLayerDim, self.numNnTrainSteps)
                if self.useTrueModel:
                    initProbList, transProbList, rewardList = self.envTrain.get_initProbList_transProbList_rewardList()
                    self.agent.set_initProbList_transProbList_rewardList(initProbList, transProbList, rewardList)
            elif self.envType == "SimGenerative":
                self.groupProbList = groupProbList
                self.agent = globals()[self.agentName]()
                self.agent.read_config("", rootDir, self.exprimentHyperConfigDir, self.optConfigCurDir, self.optDataCurDir, self.iptConfigHistDir, self.iptDataHistDir, self.curLocalTimeInMicro, self.device, self.numTrainEpisode, self.numInterPolicyUpdateEpisodes, self.numInterPolicyUpdateEvalEpisodes, self.numInterPolicyEvalEpisodes, self.numNnTrainSteps, True, self.evalType, self.envType, self.groupProbList, self.trialTrainIndex, self.useTrueModel, self.updateEpisodeIndexList, self.globalAgentEndEstEpisodeIndexList, self.estConstrForUpdate)#, self.nnBatchSize, self.dataDim, self.nnHiddenDim, self.nnTargetDim, self.nnLayerDim)#, self.nStates, self.nActions)

                self.envTrain = EnvTrainSimGenerative()
                self.envTrain.read_config("", rootDir, self.optConfigCurDir, self.optDataCurDir, self.iptConfigHistDir, self.iptDataHistDir, self.device, self.numTrainEpisode, self.numInterPolicyUpdateEvalEpisodes, self.numNnTrainSteps, self.groupProbList, self.evalType)#, self.nnBatchSize, self.dataDim, self.nnHiddenDim, self.nnTargetDim, self.nnLayerDim, self.numNnTrainSteps)

                if self.useTrueModel:
                    initProbList, transProbList, rewardList = self.envTrain.get_initProbList_transProbList_rewardList()
                    self.agent.set_initProbList_transProbList_rewardList(initProbList, transProbList, rewardList)
            elif self.envType == "RealGenerative":
                self.groupProbList = self.load_from_grpProbPath(self.grpProbPath)
                self.agent = globals()[self.agentName]()
                self.agent.read_config("", rootDir, self.exprimentHyperConfigDir, self.optConfigCurDir, self.optDataCurDir, self.iptConfigHistDir, self.iptDataHistDir, self.curLocalTimeInMicro, self.device, self.numTrainEpisode, self.numInterPolicyUpdateEpisodes, self.numInterPolicyUpdateEvalEpisodes, self.numInterPolicyEvalEpisodes, self.numNnTrainSteps, True, self.evalType, self.envType, self.groupProbList, self.trialTrainIndex, self.useTrueModel, self.updateEpisodeIndexList, self.globalAgentEndEstEpisodeIndexList, self.estConstrForUpdate)#, self.nnBatchSize, self.dataDim, self.nnHiddenDim, self.nnTargetDim, self.nnLayerDim)#, self.nStates, self.nActions)

                self.envTrain = EnvTrainRealGenerative()
                self.envTrain.read_config("", rootDir, self.optConfigCurDir, self.optDataCurDir, self.iptConfigHistDir, self.iptDataHistDir, self.device, self.numTrainEpisode, self.numInterPolicyUpdateEvalEpisodes, self.numNnTrainSteps, self.groupProbList, self.evalType)#, self.nnBatchSize, self.dataDim, self.nnHiddenDim, self.nnTargetDim, self.nnLayerDim, self.numNnTrainSteps)
                if self.useTrueModel:
                    initProbList, transProbList, rewardList = self.envTrain.get_initProbList_transProbList_rewardList()
                    self.agent.set_initProbList_transProbList_rewardList(initProbList, transProbList, rewardList)


            self.methodType = self.agent.methodType
            self.methodConst = self.agent.methodConst
            self.methodSampleSize = self.agent.methodSampleSize
            self.methodMix = self.agent.methodMix


            self.methodName = self.agent.algoType### + "_" + self.envType

            self.agent.set_methodName(self.methodName)


            trainHeader = ["evalTypeEnvTypeParetoType", "paretoType", "methodName", "methodConst", "methodSampleSize", "methodMix", "curLocalTimeInMicro", "trialTrainIndex", "trainEpisodeIndex", "trainNnTrainStepIndex", "policyUpdateAvgReward", "estConstr"]
            self.trainLogger = DataLogger(self.trainLoggerCapacity, self.optDataCurDir + self.trainLoggerFileName, trainHeader) 

            Config.write_config(config, self.optConfigCurDir + self.__class__.__name__ + self.identifier + ".json")

    def compute_globaUpdateEpisodeIndex_estConstrForUpdate(self):
        logBaseVal = 2
        numRawUpdatePoint = int(math.log(self.numTrainEpisode) / math.log(logBaseVal))
        intervalInLogSpace = numRawUpdatePoint * 1.0 / self.numPolicyUpdates
        self.updateEpisodeIndexList = [logBaseVal ** int(intervalInLogSpace * policyUpdateIndex) for policyUpdateIndex in range(self.numStartPolicyUpdates, self.numPolicyUpdates)]
        self.globalAgentEndEstEpisodeIndexList =[]
        self.globalEnvEndEstEpisodeIndexList =[]
        self.globalEnvEndEstEpisodeIndexList.append(self.numInterPolicyUpdateEvalEpisodes)
        for i in range(1, len(self.updateEpisodeIndexList)):
            self.globalAgentEndEstEpisodeIndexList.append(self.globalEnvEndEstEpisodeIndexList[i-1] + self.updateEpisodeIndexList[i])
            self.globalEnvEndEstEpisodeIndexList.append(self.globalAgentEndEstEpisodeIndexList[i-1] + self.numInterPolicyUpdateEvalEpisodes)

        startScale = math.sqrt(self.updateEpisodeIndexList[0])
        self.estConstrForUpdate = [round(1.0/math.sqrt(updateEpisodeIndex) * startScale,3) for updateEpisodeIndex in self.updateEpisodeIndexList]

        print("numRawUpdatePoint, intervalInLogSpace, startScale: ", numRawUpdatePoint, intervalInLogSpace, startScale)
        print("updateEpisodeIndexList: ", self.updateEpisodeIndexList)
        print("globalAgentEndEstEpisodeIndexList: ", self.globalAgentEndEstEpisodeIndexList)
        print("globalEnvEndEstEpisodeIndexList: ", self.globalEnvEndEstEpisodeIndexList)
        print("estConstrForUpdate: ", self.estConstrForUpdate)

    def load_from_grpProbPath(self, path):
        df = pd.read_csv(path)
        whiteFreq = df.iloc[0]["Non- Hispanic white"] 
        blackFreq = df.iloc[0]["Black"]
        totalFreq = whiteFreq + blackFreq
        grpProb = [whiteFreq * 1.0 / totalFreq, blackFreq * 1.0 / totalFreq]
        return grpProb

    def run(self):#, trainDataInputState, trainDataTargetState, trainDataInputReward, trainDataTargetReward):

        self.envTrain.begin_est()

        for trainEpisodeIndex in range(self.globalEnvEndEstEpisodeIndexList[-1]):#self.numTrainEpisode):
            nnTrainStepIndex = 0 
            self.envTrain.epiInit()


            while nnTrainStepIndex < self.numNnTrainSteps:

                cbData = self.envTrain.getCbData()

                action = self.agent.on_cbData(cbData, trainEpisodeIndex, nnTrainStepIndex)

            #2. env feedback 
                state, reward = self.envTrain.feedback(action, nnTrainStepIndex)

                self.agent.on_state_action_reward(state, action, reward, trainEpisodeIndex, nnTrainStepIndex)
                nnTrainStepIndex += 1
                

            if (trainEpisodeIndex + 1) in self.globalAgentEndEstEpisodeIndexList:
                self.envTrain.begin_est()#trainEpisodeIndex)#state)
                self.agent.end_est()


            if (trainEpisodeIndex + 1) in self.globalEnvEndEstEpisodeIndexList:
                policyEvalIndex = self.globalEnvEndEstEpisodeIndexList.index(trainEpisodeIndex + 1)
                print("policyEvalIndex: ", policyEvalIndex)
                if self.evalType == "Dp":
                    policyAvgRet, estConstr = self.envTrain.end_est_dp()#trainEpisodeIndex)#state)
                elif self.evalType == "EqOpt":
                    policyAvgRet, estConstr = self.envTrain.end_est_eqOpt()#trainEpisodeIndex)
                elif self.evalType == "Agg":
                    policyAvgRet, estConstr = self.envTrain.end_est_agg()
                elif self.evalType == "DpCv":
                    policyAvgRet, estConstr = self.envTrain.end_est_dpCv()#trainEpisodeIndex)#state)
                elif self.evalType == "EqOptCv":
                    policyAvgRet, estConstr = self.envTrain.end_est_eqOptCv()#trainEpisodeIndex)
                self.agent.begin_est()

                if self.methodName in ["DpOrigGp", "DpCvOrigGp", "EqOptOrigGp", "EqOptCvOrigGp", "AggOrigGp"]:
                    methodMix = self.estConstrForUpdate[policyEvalIndex]
                else:
                    methodMix = self.agent.methodMix

                trainList = [self.evalType + self.envType + self.paretoType, self.paretoType, self.methodName, self.methodConst, self.methodSampleSize, methodMix, self.curLocalTimeInMicro, self.trialTrainIndex, trainEpisodeIndex, self.numNnTrainSteps + 1, policyAvgRet, estConstr]
                print("policyAvgRet, estConstr: ", policyAvgRet, estConstr)
                self.trainLogger.write_log_to_file(False, trainList)
        self.trainLogger.write_log_to_file(True, None)
        torch.save(self.agent.algoUpdateLogDictDict, self.agent.algoDir + "occuMeasNpForUpdate")#_" + str(self.logStartTrainEpisodeIndex) + "_" + str(trainEpisodeIndex))# + "_" + str(self.numTrainEpisodeLogging))
        torch.save(self.agent.algoEvalLogDictDict, self.agent.algoDir + "occuMeasNpForEval")#_" + str(self.logStartTrainEpisodeIndex) + "_" + str(trainEpisodeIndex))# + "_" + str(self.numTrainEpisodeLogging))
        self.agent.optObjConstrForUpdateLogger.write_log_to_file(True, None)


