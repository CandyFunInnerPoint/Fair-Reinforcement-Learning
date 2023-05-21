import json, sys, os
from os import path
from time import strftime, gmtime
import csv
import torch
from DataStruct import Obs, Action, CbData, Reward, State

from AgentBase import AgentBase

from collections import OrderedDict

from DataLogger import DataLogger

from Config import Config
from Path import Path

import numpy as np

import cvxpy as cp

#import qcqp 

import math

import gurobipy as gp 
from gurobipy import GRB

from Constant import * 

class GroupAgentTrain(object):
    def __init__(self):

        self.numScores = None
        self.minScore = None

        self.rewardEst = None

        self.scoreIsQualScoreSgt1IsQualSgt1DecSgt1Freq = None
        self.rewardMultScoreIsQualDecFreq = None
        self.groupFreq = None

        self.estTransProb = None
        self.estReward = None

        self.scoreIndex = None
        self.decision = None
        self.isQual = None       
        
        self.occuMeasNpForUpdate = None
        self.occuMeasCvx = None

        self.status = None

    def init(self, groupIndex, evalType, numScores, minScore, optRoundEps, numNnTrainSteps, status):
        self.groupIndex = groupIndex
        self.evalType = evalType

        self.numScores = numScores
        self.minScore = minScore
        self.optRoundEps = optRoundEps
        self.numNnTrainSteps = numNnTrainSteps
        self.initScoreIsQualFreq = np.ones(shape = (self.numScores, 2)) * optRoundEps
        self.scoreIsQualScoreSgt1IsQualSgt1DecSgt1Freq = np.ones(shape = (self.numScores, 2, self.numScores, 2, 2)) * optRoundEps
        self.rewardMultScoreIsQualDecFreq = np.zeros(shape = (self.numScores, 2, 2))
        self.groupFreq = 0

        self.estInitProb = np.ones(shape = (self.numScores, 2)) * optRoundEps
        self.estTransProb = np.ones(shape = (self.numScores, 2, self.numScores, 2, 2)) * optRoundEps
        self.estReward = np.ndarray(shape = (self.numScores, 2, 2))

        self.occuMeasNpForUpdate = np.ndarray(shape = (self.numScores, 2, 2, self.numNnTrainSteps), dtype = float)
        for timeStep in range(self.numNnTrainSteps):
            occuMeasTmNpUnNrm = np.random.rand(self.numScores, 2, 2) 
            occuMeasTmNpNrm = occuMeasTmNpUnNrm / np.sum(occuMeasTmNpUnNrm) 
            self.occuMeasNpForUpdate[:,:,:,timeStep] = occuMeasTmNpNrm

        self.occuMeasNpForUpdateEval = np.ndarray(shape = (self.numScores, 2, 2, self.numNnTrainSteps), dtype = float)
        for timeStep in range(self.numNnTrainSteps):
            occuMeasTmNpUnNrm = np.random.rand(self.numScores, 2, 2) 
            occuMeasTmNpNrm = occuMeasTmNpUnNrm / np.sum(occuMeasTmNpUnNrm) 
            self.occuMeasNpForUpdateEval[:,:,:,timeStep] = occuMeasTmNpNrm

        self.occuMeasNpForEval = np.ndarray(shape = (self.numScores, 2, 2, self.numNnTrainSteps), dtype = float)
        for timeStep in range(self.numNnTrainSteps):
            occuMeasTmNpUnNrm = np.random.rand(self.numScores, 2, 2) 
            occuMeasTmNpNrm = occuMeasTmNpUnNrm / np.sum(occuMeasTmNpUnNrm) 
            self.occuMeasNpForEval[:,:,:,timeStep] = occuMeasTmNpNrm

        self.occuMeasCvx = []
        for score in range(numScores):
            occuMeasScore = []
            for isQual in range(2):
                occuMeasScore.append(cp.Variable(shape = (2, self.numNnTrainSteps)))
            self.occuMeasCvx.append(occuMeasScore)

        self.status = status


    def comp_policy(self, scoreIndex, nnTrainStepIndex):

        occuMeas = self.occuMeasNpForUpdateEval[scoreIndex,:,:,nnTrainStepIndex]


        dec = np.sum(occuMeas, axis = 0)#.reshape(2) 
        policyProb = dec * 1.0 / np.sum(dec) #np.sum(probScoreDec, axis = 0)

        return policyProb

    def update_initScoreIsQual_freq(self, scoreIndex, isQual):
        self.initScoreIsQualFreq[scoreIndex][isQual] += 1

    #update stats
    def update_scoreIsQualScoreSgt1IsQualSgt1DecSgt1_freq(self, scoreIndex, isQual):
        if self.scoreIndex != None and self.decision != None:
            self.scoreIsQualScoreSgt1IsQualSgt1DecSgt1Freq[scoreIndex][isQual][self.scoreIndex][self.isQual][self.decision] += 1

    #update stats
    def update_rewardMultScoreIsQualDec_freq(self, reward, scoreIndex, isQual, dec):
        self.rewardMultScoreIsQualDecFreq[scoreIndex][isQual][dec] += reward

    def update_group_freq(self):
        self.groupFreq += 1


    def update_estInitProb(self):
        for scoreIndex in range(self.numScores):
            for isQualIndex in range(2):
                self.estInitProb[scoreIndex][isQualIndex] = self.initScoreIsQualFreq[scoreIndex][isQualIndex] * 1.0 / np.sum(self.initScoreIsQualFreq)

    def update_estTransProb(self):
        for scoreIndex in range(self.numScores):
            for isQualIndex in range(2):
                for scoreSgt1Index in range(self.numScores):
                    for isQualSgt1Index in range(2):
                        for decSgt1Index in range(2):
                            self.estTransProb[scoreIndex][isQualIndex][scoreSgt1Index][isQualSgt1Index][decSgt1Index] = self.scoreIsQualScoreSgt1IsQualSgt1DecSgt1Freq[scoreIndex][isQualIndex][scoreSgt1Index][isQualSgt1Index][decSgt1Index] * 1.0 / np.sum(self.scoreIsQualScoreSgt1IsQualSgt1DecSgt1Freq, axis = (0,1))[scoreSgt1Index][isQualSgt1Index][decSgt1Index]  #max(self.scoreIsQualDecFreq[scoreSgt1Index][isQualSgt1Index][decSgt1Index], 1)

    def update_estReward(self):
        for scoreIndex in range(self.numScores):
            for isQualIndex in range(2):
                for decIndex in range(2):
                    self.estReward[scoreIndex][isQualIndex][decIndex] = self.rewardMultScoreIsQualDecFreq[scoreIndex][isQualIndex][decIndex] * 1.0 / np.sum(self.scoreIsQualScoreSgt1IsQualSgt1DecSgt1Freq, axis = (0,1))[scoreIndex][isQualIndex][decIndex]#max(self.scoreIsQualDecFreq[scoreIndex][isQualIndex][decIndex], 1)# + bonus[[self.score2ScoreIndex(score)][isQual][dec]
    #update sample
    def update_scoreDec(self, scoreIndex, decision):
        self.scoreIndex = scoreIndex
        self.decision = decision

    #update sample
    def update_isQual(self, isQual):
        self.isQual = isQual


class AgentEqOptSgtGp(AgentBase):#AgentMbLnrFwdGatFwdLnrBwdGatBwd
    def __init__(self):
        self.path = None
        self.useConfigMode = None
        self.device = None


        self.nnModel = None

        self.nnInputDim = None
        self.nnHiddenDim = None
        self.nnTargetDim = None

        self.gamma = None

        self.nnLr = None

        self.logStartTrainEpisodeIndex = None

        self.objValEval = None
        self.objValUpdate = None

        self.status = None

    def read_config(self, identifier, rootDir, exprimentHyperConfigDir, optConfigCurDir, optDataCurDir, iptConfigHistDir, iptDataHistDir, curLocalTimeInMicro, device, numTrainEpisode, numInterPolicyUpdateEpisodes, numInterPolicyUpdateEvalEpisodes, numInterPolicyEvalEpisodes, numNnTrainSteps, isTrainPhase, evalType, envType, groupProbList, trialTrainIndex, useTrueModel, updateEpisodeIndexList, globalAgentEndEstEpisodeIndexList, estConstrForUpdate):#nnBatchSize, nnInputDim, nnHiddenDim, nnTargetDim, nnLayerDim):#, nStates, nActions):
        iptConfigFileName, self.optConfigCurDir, self.iptConfigHistDir, self.optDataCurDir = Path.make_path(rootDir, self.__class__.__name__, identifier, optConfigCurDir, iptConfigHistDir, optDataCurDir)
        self.iptDataHistDir = iptDataHistDir

        self.optDataCurDir = optDataCurDir + "AgentTrain/"
        if not os.path.exists(self.optDataCurDir):
            os.makedirs(self.optDataCurDir)

        self.exprimentHyperConfigDir = exprimentHyperConfigDir
        self.curLocalTimeInMicro = curLocalTimeInMicro

        if self.curLocalTimeInMicro != None:
            with open(self.exprimentHyperConfigDir + self.__class__.__name__ + "Hyper" + str(self.curLocalTimeInMicro) + ".json") as json_file:
                config = json.load(json_file)
                self.algoType = config["algoType"]
                self.methodType = config["methodType"]
                self.methodConst = config["methodConst"]
                self.methodSampleSize = config["methodSampleSize"]
                self.methodMix = config["methodMix"]
                self.paretoType = config["paretoType"]

        with open(iptConfigFileName) as json_file:
            config = json.load(json_file)


            self.numScores = config["numScores"]
            self.minScore = config["minScore"]
            self.numGroups = config["numGroups"]

            self.timeLimitEqOptSgt = config["timeLimitEqOptSgt"]
            self.feasEps = config["feasEps"]
            self.optValEps = config["optValEps"]
            self.optRoundEps = config["optRoundEps"]
            self.optRoundEpsTransProb = config["optRoundEpsTransProb"]
            self.mipGap = config["mipGap"]

            self.algoLoggerCapacity = config["algoLoggerCapacity"]
            self.optObjConstrForUpdateLoggerFileName = config["optObjConstrForUpdateLoggerFileName"]           

            self.hasTransProbConstr = config["hasTransProbConstr"]
            self.isComputeEval = config["isComputeEval"]
            

            self.numInterPolicyUpdateEpisodes = numInterPolicyUpdateEpisodes
            self.numInterPolicyUpdateEvalEpisodes = numInterPolicyUpdateEvalEpisodes
            self.numInterPolicyEvalEpisodes = numInterPolicyEvalEpisodes
            
            self.envType = envType
            self.evalType = evalType

            self.device = device

            self.identifier = identifier

            self.numTrainEpisode = numTrainEpisode
            self.numNnTrainSteps = numNnTrainSteps

            self.isTrainPhase = isTrainPhase

            self.groupProbList = groupProbList

            self.trialTrainIndex = trialTrainIndex

            self.useTrueModel = useTrueModel

            self.evalOptIndex = 0
            self.updateOptIndex = 0

            self.status = AGENT_UPDATE
            self.groupList = []
            for groupIndex in range(self.numGroups):
                group = GroupAgentTrain()
                group.init(groupIndex, self.evalType, self.numScores, self.minScore, self.optRoundEps, self.numNnTrainSteps, self.status)
                self.groupList.append(group)

            self.updateEpisodeIndexList = updateEpisodeIndexList
            self.globalAgentEndEstEpisodeIndexList = globalAgentEndEstEpisodeIndexList
            self.estConstrForUpdate = estConstrForUpdate

            self.algoEvalLogDictDict = OrderedDict()
            self.algoUpdateLogDictDict = OrderedDict()

            optObjConstrForUpdateHeader = ["evalTypeEnvTypeParetoType", "paretoType", "methodName", "methodConst", "methodSampleSize", "methodMix", "curLocalTimeInMicro", "trialTrainIndex", "trainEpisodeIndex", "optObjVal", "optConstrVal", "optGap"]

            self.optObjConstrForUpdateLogger = DataLogger(self.algoLoggerCapacity, self.optDataCurDir + self.optObjConstrForUpdateLoggerFileName, optObjConstrForUpdateHeader) 

            Config.write_config(config, self.optConfigCurDir + self.__class__.__name__ + self.identifier + ".json")

            self.algoDir = self.optDataCurDir + "algoFromTrain/"
            if not os.path.exists(self.algoDir):
                os.makedirs(self.algoDir)

    def set_initProbList_transProbList_rewardList(self, initProbList, transProbList, rewardList):
        self.initProbList = initProbList
        self.transProbList = transProbList
        self.rewardList = rewardList

    def set_methodName(self, methodName):
        self.methodName = methodName

    def on_cbData(self, cbData, trainEpisodeIndex, nnTrainStepIndex):
        obs = self.preprocess(cbData, trainEpisodeIndex, nnTrainStepIndex)
        action = self.select_action(obs, trainEpisodeIndex, nnTrainStepIndex)
        self.postprocess(obs, action, trainEpisodeIndex, nnTrainStepIndex)


        return action


    def preprocess(self, cbData, trainEpisodeIndex, nnTrainStepIndex):#, gateActNet, stepSizeScaleActNet, state):

        obs = Obs()
        obs.scoreIndexList = cbData.scoreIndexList

        return obs

    def select_action(self, obs, trainEpisodeIndex, nnTrainStepIndex):
        policyProbList = []
        for groupIndex in range(self.numGroups):
            policyProb = self.groupList[groupIndex].comp_policy(obs.scoreIndexList[groupIndex], nnTrainStepIndex)
            policyProbList.append(policyProb)
        action = Action()
        action.decisionList = []
        for groupIndex in range(self.numGroups):
            decision = np.argmax(np.random.multinomial(1, policyProbList[groupIndex]))
            action.decisionList.append(decision)
        return action#.detach()

    def postprocess(self, obs, action, trainEpisodeIndex, nnTrainStepIndex):
        pass
    def on_state_action_reward(self, state, action, reward, trainEpisodeIndex, nnTrainStepIndex):
        if self.status == AGENT_UPDATE:
            for groupIndex in range(self.numGroups):
                if nnTrainStepIndex == 0:
                    self.groupList[groupIndex].update_initScoreIsQual_freq(state.scoreIndexList[groupIndex], state.isQualList[groupIndex])
                self.groupList[groupIndex].update_scoreIsQualScoreSgt1IsQualSgt1DecSgt1_freq(state.scoreIndexList[groupIndex], state.isQualList[groupIndex])
                self.groupList[groupIndex].update_rewardMultScoreIsQualDec_freq(reward.rewardValList[groupIndex], state.scoreIndexList[groupIndex], state.isQualList[groupIndex], action.decisionList[groupIndex])
                self.groupList[groupIndex].update_scoreDec(state.scoreIndexList[groupIndex], action.decisionList[groupIndex])
                self.groupList[groupIndex].update_isQual(state.isQualList[groupIndex])
        
        if (trainEpisodeIndex + 1) in self.globalAgentEndEstEpisodeIndexList and (nnTrainStepIndex+1) % self.numNnTrainSteps == 0:
            policyUpdateIndex = self.globalAgentEndEstEpisodeIndexList.index(trainEpisodeIndex + 1)

            self.compute_occuMeas_update(trainEpisodeIndex, policyUpdateIndex)   
            print("the index of the optimization for updateOpt: ", self.updateOptIndex)
            self.updateOptIndex += 1

    def compute_occuMeas_eval(self, trainEpisodeIndex, policyEvalIndex):#, trainEpisodeIndex, nnTrainStepIndex):
        
        #2. get estimates for both groups
        for groupIndex in range(self.numGroups):
            if self.useTrueModel:
                self.groupList[groupIndex].estInitProb = self.initProbList[groupIndex]
                self.groupList[groupIndex].estTransProb = self.transProbList[groupIndex]
                self.groupList[groupIndex].estReward = self.rewardList[groupIndex]
            else:
                self.groupList[groupIndex].update_estInitProb()
                self.groupList[groupIndex].update_estTransProb()
                self.groupList[groupIndex].update_estReward()

        if self.algoType == "dpOrigCvx":
            self.solve_dpOrigCvx_opt_eval(trainEpisodeIndex, policyEvalIndex)#estGroupProb, estDpConstr)#, estTransProbFirstGroup, estTransProbSecondGroup, estRewardFirstGroup, estRewardSecondGroup, estDpConstr)
        elif self.algoType == "dpSgtCvx":
            self.solve_dpSgtCvx_opt_eval(trainEpisodeIndex, policyEvalIndex)
        elif self.algoType == "dpOrigGp":
            self.solve_dpOrigGp_opt_eval(trainEpisodeIndex, policyEvalIndex)#estGroupProb, estDpConstr)#, estTransProbFirstGroup, estTransProbSecondGroup, estRewardFirstGroup, estRewardSecondGroup, estDpConstr)
        elif self.algoType == "dpSgtGp":
            self.solve_dpSgtGp_opt_eval(trainEpisodeIndex, policyEvalIndex)
        elif self.algoType == "eqOptOrigGp":
            self.solve_eqOptOrigGp_opt_eval(trainEpisodeIndex, policyEvalIndex)
        elif self.algoType == "eqOptSgtGp":#eqOptSgt
            self.solve_eqOptSgtGp_opt_eval(trainEpisodeIndex, policyEvalIndex)#estGroupProb, estEqOptConstr)#, estTransProbFirstGroup, estTransProbSecondGroup, estRewardFirstGroup, estRewardSecondGroup, estEqOptConstr)
        elif self.algoType == "eqOptSdp":
            self.solve_eqOptSdp_opt_eval(trainEpisodeIndex, policyEvalIndex)
        elif self.algoType == "aggOrigGp":
            self.solve_aggOrigGp_opt_eval(trainEpisodeIndex, policyEvalIndex)
        elif self.algoType == "aggSgtGp":
            self.solve_aggSgtGp_opt_eval(trainEpisodeIndex, policyEvalIndex)

    def compute_occuMeas_update(self, trainEpisodeIndex, policyUpdateIndex):#, trainEpisodeIndex, nnTrainStepIndex):
        
        #2. get estimates for both groups
        for groupIndex in range(self.numGroups):
            if self.useTrueModel:
                self.groupList[groupIndex].estInitProb = self.initProbList[groupIndex]
                self.groupList[groupIndex].estTransProb = self.transProbList[groupIndex]
                self.groupList[groupIndex].estReward = self.rewardList[groupIndex]
            else:
                self.groupList[groupIndex].update_estInitProb()
                self.groupList[groupIndex].update_estTransProb()
                self.groupList[groupIndex].update_estReward()

        if self.algoType == "EqOptSgtGp":#eqOptSgt
            self.solve_eqOptSgtGp_opt_update(trainEpisodeIndex, policyUpdateIndex)#estGroupProb, estEqOptConstr)#, estTransProbFirstGroup, estTransProbSecondGroup, estRewardFirstGroup, estRewardSecondGroup, estEqOptConstr)

    def solve_eqOptSgtGp_opt_eval(self, trainEpisodeIndex, policyEvalIndex):#, estEqOptConstr):#estGroupProb, estTransProbFirstGroup, estTransProbSecondGroup, estRewardFirstGroup, estRewardSecondGroup, estEqOptConstr):
        firstGroupIndex = 0
        secondGroupIndex = 1

        if True:
            qcqp = gp.Model("qcqp")

            #variables
            occuMeas = [[[[[None for timeStep in range(self.numNnTrainSteps)] for decIndex in range(2)] for isQualIndex in range(2)] for scoreIndex in range(self.numScores)] for groupIndex in range(self.numGroups)]
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for decIndex in range(2):
                            for timeStep in range(self.numNnTrainSteps):
                                occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep] = qcqp.addVar(vtype = GRB.CONTINUOUS, name = "rho_" + str(groupIndex) + "_" + str(scoreIndex) + "_" + str(isQualIndex) + "_" + str(decIndex) + "_" + str(timeStep))


            prod_exprSumScoreFirstGroupIsQual1Dec1_exprSumScoreDecSecondGroupIsQual1 = [None for timeStep in range(self.numNnTrainSteps)]
            prod_exprSumScoreSecondGroupIsQual1Dec1_exprSumScoreDecFirstGroupIsQual1 = [None for timeStep in range(self.numNnTrainSteps)]

             
            for timeStep in range(self.numNnTrainSteps):#-1, -1, -1):
                prod_exprSumScoreFirstGroupIsQual1Dec1_exprSumScoreDecSecondGroupIsQual1[timeStep] = qcqp.addVar(vtype = GRB.CONTINUOUS, name = "gamma_" + str(timeStep))
                prod_exprSumScoreSecondGroupIsQual1Dec1_exprSumScoreDecFirstGroupIsQual1[timeStep] = qcqp.addVar(vtype = GRB.CONTINUOUS, name = "eta_" + str(timeStep))

            #objective
            objSum = 0
            for groupIndex in range(self.numGroups):
                expr = 0
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for decIndex in range(2):
                            for timeStep in range(self.numNnTrainSteps):
                                expr = expr + occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep] * self.groupList[groupIndex].estReward[scoreIndex][isQualIndex][decIndex]
                objSum = objSum + self.groupProbList[groupIndex] * expr

            #fairness constr
            for timeStep in range(self.numNnTrainSteps):#-1, -1, -1):
                exprLhs = 0

                exprSumScoreFirstGroupIsQual1Dec1 = 0
                for scoreIndex in range(self.numScores):
                    exprSumScoreFirstGroupIsQual1Dec1 = exprSumScoreFirstGroupIsQual1Dec1 + occuMeas[firstGroupIndex][scoreIndex][1][1][timeStep]

                exprSumScoreDecFirstGroupIsQual1 = 0
                for scoreIndex in range(self.numScores):
                    for decIndex in range(2):
                        exprSumScoreDecFirstGroupIsQual1 = exprSumScoreDecFirstGroupIsQual1 + occuMeas[firstGroupIndex][scoreIndex][1][decIndex][timeStep]

                exprSumScoreSecondGroupIsQual1Dec1 = 0
                for scoreIndex in range(self.numScores):
                    exprSumScoreSecondGroupIsQual1Dec1 = exprSumScoreSecondGroupIsQual1Dec1 + occuMeas[secondGroupIndex][scoreIndex][1][1][timeStep]

                exprSumScoreDecSecondGroupIsQual1 = 0
                for scoreIndex in range(self.numScores):
                    for decIndex in range(2):
                        exprSumScoreDecSecondGroupIsQual1 = exprSumScoreDecSecondGroupIsQual1 + occuMeas[secondGroupIndex][scoreIndex][1][decIndex][timeStep]

                qcqp.addConstr(prod_exprSumScoreFirstGroupIsQual1Dec1_exprSumScoreDecSecondGroupIsQual1[timeStep] == exprSumScoreFirstGroupIsQual1Dec1 * exprSumScoreDecSecondGroupIsQual1)
                qcqp.addConstr(prod_exprSumScoreSecondGroupIsQual1Dec1_exprSumScoreDecFirstGroupIsQual1[timeStep] == exprSumScoreSecondGroupIsQual1Dec1 * exprSumScoreDecFirstGroupIsQual1)



            for timeStep in range(self.numNnTrainSteps):
                exprLhs = 0
                exprLhs = exprLhs + prod_exprSumScoreFirstGroupIsQual1Dec1_exprSumScoreDecSecondGroupIsQual1[timeStep] - prod_exprSumScoreSecondGroupIsQual1Dec1_exprSumScoreDecFirstGroupIsQual1[timeStep]
                objSum = objSum - self.methodConst * exprLhs * exprLhs ###+ self.methodConst * (exprRhsUnscale)

            qcqp.setObjective(objSum, GRB.MAXIMIZE)

            #conditional independence constraint
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for timeStep in range(self.numNnTrainSteps):#self.numNnTrainSteps-1, -1, -1):
                        exprScoreXIsQual1Dec1 = occuMeas[groupIndex][scoreIndex][1][1][timeStep]

                        exprSumDecScoreXIsQual1 = 0
                        for decIndex in range(2):
                            exprSumDecScoreXIsQual1 = exprSumDecScoreXIsQual1 + occuMeas[groupIndex][scoreIndex][1][decIndex][timeStep]

                        exprScoreXIsQual0Dec1 = occuMeas[groupIndex][scoreIndex][0][1][timeStep]

                        exprSumDecScoreXIsQual0 = 0
                        for decIndex in range(2):
                            exprSumDecScoreXIsQual0 = exprSumDecScoreXIsQual0 + occuMeas[groupIndex][scoreIndex][0][decIndex][timeStep]

                        expr = exprScoreXIsQual1Dec1 * exprSumDecScoreXIsQual0 - exprScoreXIsQual0Dec1 * exprSumDecScoreXIsQual1


                        qcqp.addConstr(expr <= self.optRoundEps)
                        qcqp.addConstr(expr >= - self.optRoundEps)

            #trans dyn constr
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for timeStep in range(self.numNnTrainSteps-1, 0, -1):
                            exprLeft = 0
                            for decSgt1Index in range(2):
                                exprLeft = exprLeft + occuMeas[groupIndex][scoreIndex][isQualIndex][decSgt1Index][timeStep]
                            exprRight = 0
                            for scoreSgt1Index in range(self.numScores):
                                for isQualSgt1Index in range(2):
                                    for decSgt1Index in range(2):
                                        exprRight = exprRight + occuMeas[groupIndex][scoreSgt1Index][isQualSgt1Index][decSgt1Index][timeStep-1] * self.groupList[groupIndex].estTransProb[scoreIndex][isQualIndex][scoreSgt1Index][isQualSgt1Index][decSgt1Index]
                            qcqp.addConstr(exprLeft <= exprRight + self.optRoundEpsTransProb)
                            qcqp.addConstr(exprLeft >= exprRight - self.optRoundEpsTransProb)

            #prob box constr
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for decIndex in range(2):
                            for timeStep in range(self.numNnTrainSteps):#-1, -1, -1):
                                qcqp.addConstr(occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep] >= self.optRoundEps)
                                qcqp.addConstr(occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep] <= 1)

            #prob sum 1 constr
            for groupIndex in range(self.numGroups):
                for timeStep in range(self.numNnTrainSteps):
                    expr = 0
                    for scoreIndex in range(self.numScores):
                        for isQualIndex in range(2):
                            for decIndex in range(2):
                                expr = expr + occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep]
                    qcqp.addConstr(expr <= 1 + self.optRoundEps)
                    qcqp.addConstr(expr >= 1 - self.optRoundEps)

             #init prob constraint
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        expr = 0
                        for decIndex in range(2):
                            expr = expr + occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][0]
                        qcqp.addConstr(expr <= self.groupList[groupIndex].estInitProb[scoreIndex][isQualIndex] + self.optRoundEps)      
                        qcqp.addConstr(expr >= self.groupList[groupIndex].estInitProb[scoreIndex][isQualIndex] - self.optRoundEps)

            qcqp.setParam(GRB.param.Method, 2)
            qcqp.setParam(GRB.param.OptimalityTol, self.optValEps)
            qcqp.setParam(GRB.param.FeasibilityTol, self.feasEps)
            qcqp.setParam(GRB.param.NumericFocus, 3)
            qcqp.setParam(GRB.param.NonConvex, 2)
            qcqp.setParam(GRB.param.MIPGap, self.mipGap)
            qcqp.setParam(GRB.param.TimeLimit, self.timeLimitEqOptSgt)
            qcqp.setParam(GRB.param.DualReductions, 0)

            qcqp.optimize()

            sol = qcqp.getVars()
            status = qcqp.Status

        try:
#            print(sol[0].X)
            if qcqp.Status == GRB.OPTIMAL or qcqp.Status == GRB.TIME_LIMIT:
                count = 0
                for groupIndex in range(self.numGroups):
                    for scoreIndex in range(self.numScores):
                        for isQualIndex in range(2):
                            for decIndex in range(2):
                                for timeStep in range(self.numNnTrainSteps):
                                    value = sol[count].X
                                    count += 1
                                    self.groupList[groupIndex].occuMeasNpForEval[scoreIndex][isQualIndex][decIndex][timeStep] = occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep].X###value #"rho_" + str(groupIndex) + "_" + str(scoreIndex) + "_" + str(isQualIndex) + "_" + str(decIndex) + "_" + str(timeStep)]
            obj = qcqp.getObjective()
            objValEval = obj.getValue()
        except AttributeError:
            objValEval = self.objValEval
        self.algoEvalLogDictDict[trainEpisodeIndex] = OrderedDict()
        for groupIndex in range(self.numGroups):
            self.algoEvalLogDictDict[trainEpisodeIndex][groupIndex] = self.groupList[groupIndex].occuMeasNpForEval

        
        
        avgFairConstrVal = 0
        for timeStep in range(self.numNnTrainSteps):#-1, -1, -1):
            sumScoreFirstGroupIsQual1Dec1 = 0
            for scoreIndex in range(self.numScores):
                sumScoreFirstGroupIsQual1Dec1 = sumScoreFirstGroupIsQual1Dec1 + self.groupList[firstGroupIndex].occuMeasNpForEval[scoreIndex][1][1][timeStep]

            sumScoreDecFirstGroupIsQual1 = 0
            for scoreIndex in range(self.numScores):
                for decIndex in range(2):
                    sumScoreDecFirstGroupIsQual1 = sumScoreDecFirstGroupIsQual1 + self.groupList[firstGroupIndex].occuMeasNpForEval[scoreIndex][1][decIndex][timeStep]

            sumScoreSecondGroupIsQual1Dec1 = 0
            for scoreIndex in range(self.numScores):
                sumScoreSecondGroupIsQual1Dec1 = sumScoreSecondGroupIsQual1Dec1 + self.groupList[secondGroupIndex].occuMeasNpForEval[scoreIndex][1][1][timeStep]

            sumScoreDecSecondGroupIsQual1 = 0
            for scoreIndex in range(self.numScores):
                for decIndex in range(2):
                    sumScoreDecSecondGroupIsQual1 = sumScoreDecSecondGroupIsQual1 + self.groupList[secondGroupIndex].occuMeasNpForEval[scoreIndex][1][decIndex][timeStep]

            fairConstrVal = abs(sumScoreFirstGroupIsQual1Dec1 / sumScoreDecFirstGroupIsQual1 - sumScoreSecondGroupIsQual1Dec1 / sumScoreDecSecondGroupIsQual1)
        
            avgFairConstrVal += abs(fairConstrVal)
        avgFairConstrVal /= self.numNnTrainSteps

    def solve_eqOptSgtGp_opt_update(self, trainEpisodeIndex, policyUpdateIndex):#, estEqOptConstr):#estGroupProb, estTransProbFirstGroup, estTransProbSecondGroup, estRewardFirstGroup, estRewardSecondGroup, estEqOptConstr):
        firstGroupIndex = 0
        secondGroupIndex = 1

        if True:

            qcqp = gp.Model("qcqp")

            #variables
            occuMeas = [[[[[None for timeStep in range(self.numNnTrainSteps)] for decIndex in range(2)] for isQualIndex in range(2)] for scoreIndex in range(self.numScores)] for groupIndex in range(self.numGroups)]
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for decIndex in range(2):
                            for timeStep in range(self.numNnTrainSteps):
                                occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep] = qcqp.addVar(vtype = GRB.CONTINUOUS, name = "rho_" + str(groupIndex) + "_" + str(scoreIndex) + "_" + str(isQualIndex) + "_" + str(decIndex) + "_" + str(timeStep))

            prod_exprSumScoreFirstGroupIsQual1Dec1_exprSumScoreDecSecondGroupIsQual1 = [None for timeStep in range(self.numNnTrainSteps)]
            prod_exprSumScoreSecondGroupIsQual1Dec1_exprSumScoreDecFirstGroupIsQual1 = [None for timeStep in range(self.numNnTrainSteps)]
###            prod_exprSumScoreDecFirstGroupIsQual1_exprSumScoreDecSecondGroupIsQual1 = [None for timeStep in range(self.numNnTrainSteps)]

             
            for timeStep in range(self.numNnTrainSteps):#-1, -1, -1):
                prod_exprSumScoreFirstGroupIsQual1Dec1_exprSumScoreDecSecondGroupIsQual1[timeStep] = qcqp.addVar(vtype = GRB.CONTINUOUS, name = "gamma_" + str(timeStep))
                prod_exprSumScoreSecondGroupIsQual1Dec1_exprSumScoreDecFirstGroupIsQual1[timeStep] = qcqp.addVar(vtype = GRB.CONTINUOUS, name = "eta_" + str(timeStep))
            #objective
            objSum = 0
            for groupIndex in range(self.numGroups):
                expr = 0
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for decIndex in range(2):
                            for timeStep in range(self.numNnTrainSteps):
                                expr = expr + occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep] * self.groupList[groupIndex].estReward[scoreIndex][isQualIndex][decIndex]
                objSum = objSum + self.groupProbList[groupIndex] * expr
            #fairness constr
            for timeStep in range(self.numNnTrainSteps):#-1, -1, -1):
                exprLhs = 0
                exprSumScoreFirstGroupIsQual1Dec1 = 0
                for scoreIndex in range(self.numScores):
                    exprSumScoreFirstGroupIsQual1Dec1 = exprSumScoreFirstGroupIsQual1Dec1 + occuMeas[firstGroupIndex][scoreIndex][1][1][timeStep]

                exprSumScoreDecFirstGroupIsQual1 = 0
                for scoreIndex in range(self.numScores):
                    for decIndex in range(2):
                        exprSumScoreDecFirstGroupIsQual1 = exprSumScoreDecFirstGroupIsQual1 + occuMeas[firstGroupIndex][scoreIndex][1][decIndex][timeStep]

                exprSumScoreSecondGroupIsQual1Dec1 = 0
                for scoreIndex in range(self.numScores):
                    exprSumScoreSecondGroupIsQual1Dec1 = exprSumScoreSecondGroupIsQual1Dec1 + occuMeas[secondGroupIndex][scoreIndex][1][1][timeStep]

                exprSumScoreDecSecondGroupIsQual1 = 0
                for scoreIndex in range(self.numScores):
                    for decIndex in range(2):
                        exprSumScoreDecSecondGroupIsQual1 = exprSumScoreDecSecondGroupIsQual1 + occuMeas[secondGroupIndex][scoreIndex][1][decIndex][timeStep]

                qcqp.addConstr(prod_exprSumScoreFirstGroupIsQual1Dec1_exprSumScoreDecSecondGroupIsQual1[timeStep] == exprSumScoreFirstGroupIsQual1Dec1 * exprSumScoreDecSecondGroupIsQual1)
                qcqp.addConstr(prod_exprSumScoreSecondGroupIsQual1Dec1_exprSumScoreDecFirstGroupIsQual1[timeStep] == exprSumScoreSecondGroupIsQual1Dec1 * exprSumScoreDecFirstGroupIsQual1)


            for timeStep in range(self.numNnTrainSteps):
                exprLhs = 0
                exprLhs = exprLhs + prod_exprSumScoreFirstGroupIsQual1Dec1_exprSumScoreDecSecondGroupIsQual1[timeStep] - prod_exprSumScoreSecondGroupIsQual1Dec1_exprSumScoreDecFirstGroupIsQual1[timeStep]


                objSum = objSum - self.methodConst * exprLhs * exprLhs ###+ self.methodConst * exprRhsUnscale
            qcqp.setObjective(objSum, GRB.MAXIMIZE)

            #conditional independence constraint
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for timeStep in range(self.numNnTrainSteps):#self.numNnTrainSteps-1, -1, -1):
                        exprScoreXIsQual1Dec1 = occuMeas[groupIndex][scoreIndex][1][1][timeStep]

                        exprSumDecScoreXIsQual1 = 0
                        for decIndex in range(2):
                            exprSumDecScoreXIsQual1 = exprSumDecScoreXIsQual1 + occuMeas[groupIndex][scoreIndex][1][decIndex][timeStep]

                        exprScoreXIsQual0Dec1 = occuMeas[groupIndex][scoreIndex][0][1][timeStep]

                        exprSumDecScoreXIsQual0 = 0
                        for decIndex in range(2):
                            exprSumDecScoreXIsQual0 = exprSumDecScoreXIsQual0 + occuMeas[groupIndex][scoreIndex][0][decIndex][timeStep]

                        expr = exprScoreXIsQual1Dec1 * exprSumDecScoreXIsQual0 - exprScoreXIsQual0Dec1 * exprSumDecScoreXIsQual1


                        qcqp.addConstr(expr <= self.optRoundEps)
                        qcqp.addConstr(expr >= - self.optRoundEps)

            #trans dyn constr
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for timeStep in range(self.numNnTrainSteps-1, 0, -1):
                            exprLeft = 0
                            for decSgt1Index in range(2):
                                exprLeft = exprLeft + occuMeas[groupIndex][scoreIndex][isQualIndex][decSgt1Index][timeStep]
                            exprRight = 0
                            for scoreSgt1Index in range(self.numScores):
                                for isQualSgt1Index in range(2):
                                    for decSgt1Index in range(2):
                                        exprRight = exprRight + occuMeas[groupIndex][scoreSgt1Index][isQualSgt1Index][decSgt1Index][timeStep-1] * self.groupList[groupIndex].estTransProb[scoreIndex][isQualIndex][scoreSgt1Index][isQualSgt1Index][decSgt1Index]
    #                    objSum = objSum - self.eqOptSgtTransDynConstr * (exprLeft - exprRight) ** 2
                            qcqp.addConstr(exprLeft <= exprRight + self.optRoundEpsTransProb)
                            qcqp.addConstr(exprLeft >= exprRight - self.optRoundEpsTransProb)

            #prob box constr
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for decIndex in range(2):
                            for timeStep in range(self.numNnTrainSteps):#-1, -1, -1):
                                qcqp.addConstr(occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep] >= self.optRoundEps)
                                qcqp.addConstr(occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep] <= 1)

            #prob sum 1 constr
            for groupIndex in range(self.numGroups):
                for timeStep in range(self.numNnTrainSteps):
                    expr = 0
                    for scoreIndex in range(self.numScores):
                        for isQualIndex in range(2):
                            for decIndex in range(2):
                                expr = expr + occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep]
                    qcqp.addConstr(expr <= 1 + self.optRoundEps)
                    qcqp.addConstr(expr >= 1 - self.optRoundEps)

            #init prob constraint
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        expr = 0
                        for decIndex in range(2):
                            expr = expr + occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][0]
                        qcqp.addConstr(expr <= self.groupList[groupIndex].estInitProb[scoreIndex][isQualIndex] + self.optRoundEps)
                        qcqp.addConstr(expr >= self.groupList[groupIndex].estInitProb[scoreIndex][isQualIndex] - self.optRoundEps)

            qcqp.setParam(GRB.param.Method, 2)
            qcqp.setParam(GRB.param.OptimalityTol, self.optValEps)
            qcqp.setParam(GRB.param.FeasibilityTol, self.feasEps)
            qcqp.setParam(GRB.param.NumericFocus, 3)
            qcqp.setParam(GRB.param.NonConvex, 2)
            qcqp.setParam(GRB.param.MIPGap, self.mipGap)
            qcqp.setParam(GRB.param.TimeLimit, self.timeLimitEqOptSgt)
            qcqp.setParam(GRB.param.DualReductions, 0)

            qcqp.optimize()
            sol = qcqp.getVars()

            status = qcqp.Status

        try:
            if qcqp.Status == GRB.OPTIMAL or qcqp.Status == GRB.TIME_LIMIT:
                count = 0
                for groupIndex in range(self.numGroups):
                    for scoreIndex in range(self.numScores):
                        for isQualIndex in range(2):
                            for decIndex in range(2):
                                for timeStep in range(self.numNnTrainSteps):
                                    value = sol[count].X
                                    count += 1
                                    self.groupList[groupIndex].occuMeasNpForUpdateEval[scoreIndex][isQualIndex][decIndex][timeStep] = occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep].X###value #"rho_" + str(groupIndex) + "_" + str(scoreIndex) + "_" + str(isQualIndex) + "_" + str(decIndex) + "_" + str(timeStep)]
            obj = qcqp.getObjective()
            objValUpdate = obj.getValue()
        except AttributeError:
            objValUpdate = self.objValUpdate
        self.algoUpdateLogDictDict[trainEpisodeIndex] = OrderedDict()
        for groupIndex in range(self.numGroups):
            self.algoUpdateLogDictDict[trainEpisodeIndex][groupIndex] = self.groupList[groupIndex].occuMeasNpForUpdateEval

        avgFairConstrVal = 0
        for timeStep in range(self.numNnTrainSteps):#-1, -1, -1):
            sumScoreFirstGroupIsQual1Dec1 = 0
            for scoreIndex in range(self.numScores):
                sumScoreFirstGroupIsQual1Dec1 = sumScoreFirstGroupIsQual1Dec1 + self.groupList[firstGroupIndex].occuMeasNpForUpdateEval[scoreIndex][1][1][timeStep]

            sumScoreDecFirstGroupIsQual1 = 0
            for scoreIndex in range(self.numScores):
                for decIndex in range(2):
                    sumScoreDecFirstGroupIsQual1 = sumScoreDecFirstGroupIsQual1 + self.groupList[firstGroupIndex].occuMeasNpForUpdateEval[scoreIndex][1][decIndex][timeStep]

            sumScoreSecondGroupIsQual1Dec1 = 0
            for scoreIndex in range(self.numScores):
                sumScoreSecondGroupIsQual1Dec1 = sumScoreSecondGroupIsQual1Dec1 + self.groupList[secondGroupIndex].occuMeasNpForUpdateEval[scoreIndex][1][1][timeStep]

            sumScoreDecSecondGroupIsQual1 = 0
            for scoreIndex in range(self.numScores):
                for decIndex in range(2):
                    sumScoreDecSecondGroupIsQual1 = sumScoreDecSecondGroupIsQual1 + self.groupList[secondGroupIndex].occuMeasNpForUpdateEval[scoreIndex][1][decIndex][timeStep]

            fairConstrVal = abs(sumScoreFirstGroupIsQual1Dec1 / sumScoreDecFirstGroupIsQual1 - sumScoreSecondGroupIsQual1Dec1 / sumScoreDecSecondGroupIsQual1)
        
            avgFairConstrVal += abs(fairConstrVal)
        avgFairConstrVal /= self.numNnTrainSteps

        optObjConstrForUpdateList = [self.evalType + self.envType + self.paretoType, self.paretoType, self.methodName, self.methodConst, self.methodSampleSize, self.methodMix, self.curLocalTimeInMicro, self.trialTrainIndex, trainEpisodeIndex, objValUpdate, avgFairConstrVal, qcqp.MIPGap]
        self.optObjConstrForUpdateLogger.write_log_to_file(False, optObjConstrForUpdateList)
        print("objValUpdate, avgFairConstrVal: ", objValUpdate, avgFairConstrVal)


        self.objValUpdate = objValUpdate

    def begin_est(self):

        self.status = AGENT_UPDATE

    def end_est(self):
        self.status = AGENT_UPDATE_EVAL
