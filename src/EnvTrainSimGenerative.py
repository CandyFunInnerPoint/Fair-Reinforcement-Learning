import json, sys, os
from os import path
from time import strftime, gmtime

import torch

from DataStruct import CbData, Reward, State

from Config import Config
from Path import Path

from torch.profiler import profile, record_function, ProfilerActivity

import random

import numpy as np

from Constant import * 

class GroupEnv(object):
    def __init__(self):

        self.minScore = None
        self.numScores = None

        self.transProbIsQualCondIsQualLag1Val1DecLag1Val1ListList = None
        self.transProbIsQualCondIsQualLag1Val0DecLag1Val1ListList = None
        self.transProbIsQualCondIsQualLag1Val1DecLag1Val0ListList = None

        self.transProbScoreCondScoreLag1IsQualVal1ListList = None
        self.transProbScoreCondScoreLag1IsQualVal0ListList = None

        self.rewardCoefIsQual1Dec1 = None
        self.rewardCoefIsQual0Dec1 = None
        self.rewardCoefDec0 = None

        self.sumScoreIsQualAct1PerTm = None
        self.sumScoreAct1IsQual1PerTm = None
        self.sumScoreActIsQual1PerTm = None
        self.sumScoreAct1IsQual1 = None
        self.sumScoreActIsQual1 = None

        self.scoreIndex = None
        self.isQual = None

    def init(self, numScores, minScore, numNnTrainSteps, isQualDnIsQualLag1Val1DecLag1Val1WgtDisList, isQualEqIsQualLag1Val1DecLag1Val1WgtDis, isQualUpIsQualLag1Val1DecLag1Val1WgtDisList, isQualDnIsQualLag1Val0DecLag1Val1WgtDisList, isQualEqIsQualLag1Val0DecLag1Val1WgtDis, isQualUpIsQualLag1Val0DecLag1Val1WgtDisList, isQualDnIsQualLag1Val0DecLag1Val0WgtDisList, isQualEqIsQualLag1Val0DecLag1Val0WgtDis, isQualUpIsQualLag1Val0DecLag1Val0WgtDisList, isQualDnIsQualLag1Val1DecLag1Val0WgtDisList, isQualEqIsQualLag1Val1DecLag1Val0WgtDis, isQualUpIsQualLag1Val1DecLag1Val0WgtDisList, scoreScoreLag1IsQualVal1DnWgtDisList, scoreScoreLag1IsQualVal1EqWgtDis, scoreScoreLag1IsQualVal1UpWgtDisList, scoreScoreLag1IsQualVal0DnWgtDisList, scoreScoreLag1IsQualVal0EqWgtDis, scoreScoreLag1IsQualVal0UpWgtDisList, rewardCoefIsQual1Dec1, rewardCoefIsQual0Dec1, rewardCoefDec0, initProbTrue, transProbTrue, rewardTrue):

        self.minScore = minScore
        self.numScores = numScores
        self.numNnTrainSteps = numNnTrainSteps

        self.transProbIsQualCondIsQualLag1Val1DecLag1Val1ListList = []
        for i in range(2):
            probWgtList = isQualDnIsQualLag1Val1DecLag1Val1WgtDisList[1:(i+1)][::-1] + [isQualEqIsQualLag1Val1DecLag1Val1WgtDis] + isQualUpIsQualLag1Val1DecLag1Val1WgtDisList[1:(2-i)]
            nrmProbWgtList = [probWgt * 1.0 / sum(probWgtList) for probWgt in probWgtList]
            self.transProbIsQualCondIsQualLag1Val1DecLag1Val1ListList.append(nrmProbWgtList)

        self.transProbIsQualCondIsQualLag1Val0DecLag1Val1ListList = []
        for i in range(2):
            probWgtList = isQualDnIsQualLag1Val0DecLag1Val1WgtDisList[1:(i+1)][::-1] + [isQualEqIsQualLag1Val0DecLag1Val1WgtDis] + isQualUpIsQualLag1Val0DecLag1Val1WgtDisList[1:(2-i)]
            nrmProbWgtList = [probWgt * 1.0 / sum(probWgtList) for probWgt in probWgtList]
            self.transProbIsQualCondIsQualLag1Val0DecLag1Val1ListList.append(nrmProbWgtList)


        self.transProbIsQualCondIsQualLag1Val1DecLag1Val0ListList = []
        for i in range(2):
            probWgtList = isQualDnIsQualLag1Val1DecLag1Val0WgtDisList[1:(i+1)][::-1] + [isQualEqIsQualLag1Val1DecLag1Val0WgtDis] + isQualUpIsQualLag1Val1DecLag1Val0WgtDisList[1:(2-i)]
            nrmProbWgtList = [probWgt * 1.0 / sum(probWgtList) for probWgt in probWgtList]
            self.transProbIsQualCondIsQualLag1Val1DecLag1Val0ListList.append(nrmProbWgtList)

        self.transProbIsQualCondIsQualLag1Val0DecLag1Val0ListList = []
        for i in range(2):
            probWgtList = isQualDnIsQualLag1Val0DecLag1Val0WgtDisList[1:(i+1)][::-1] + [isQualEqIsQualLag1Val0DecLag1Val0WgtDis] + isQualUpIsQualLag1Val0DecLag1Val0WgtDisList[1:(2-i)]
            nrmProbWgtList = [probWgt * 1.0 / sum(probWgtList) for probWgt in probWgtList]
            self.transProbIsQualCondIsQualLag1Val0DecLag1Val0ListList.append(nrmProbWgtList)

        self.transProbScoreCondScoreLag1IsQualVal1ListList = []
        for i in range(self.numScores):
            probWgtList = scoreScoreLag1IsQualVal1DnWgtDisList[1:(i+1)][::-1] + [scoreScoreLag1IsQualVal1EqWgtDis] + scoreScoreLag1IsQualVal1UpWgtDisList[1:(self.numScores-i)]
            nrmProbWgtList = [probWgt * 1.0 / sum(probWgtList) for probWgt in probWgtList]
            self.transProbScoreCondScoreLag1IsQualVal1ListList.append(nrmProbWgtList)


        self.transProbScoreCondScoreLag1IsQualVal0ListList = []
        for i in range(self.numScores):
            probWgtList = scoreScoreLag1IsQualVal0DnWgtDisList[1:(i+1)][::-1] + [scoreScoreLag1IsQualVal0EqWgtDis] + scoreScoreLag1IsQualVal0UpWgtDisList[1:(self.numScores-i)]
            nrmProbWgtList = [probWgt * 1.0 / sum(probWgtList) for probWgt in probWgtList]
            self.transProbScoreCondScoreLag1IsQualVal0ListList.append(nrmProbWgtList)

        
        self.rewardCoefIsQual1Dec1 = rewardCoefIsQual1Dec1#np.random.random_sample()
        self.rewardCoefIsQual0Dec1 = rewardCoefIsQual0Dec1#np.random.random_sample()
        self.rewardCoefDec0 = rewardCoefDec0

        self.initProbTrue = initProbTrue
        self.transProbTrue = transProbTrue
        self.rewardTrue = rewardTrue
 
        self.sumScoreIsQualAct1PerTm = np.zeros(shape = (self.numNnTrainSteps))
        self.sumScoreAct1IsQual1PerTm = np.zeros(shape = (self.numNnTrainSteps))
        self.sumScoreActIsQual1PerTm = np.zeros(shape = (self.numNnTrainSteps))
        self.sumScoreAct1IsQual1 = 0
        self.sumScoreActIsQual1 = 0

    def create_uniformTransProb(self):
        transProbCondScoreLag1IsQualLag1DecLag1 = np.ones(shape = (self.numScores, 2)) * 1.0 / self.numScores / 2
        self.transProb = np.tile(transProbCondScoreLag1IsQualLag1DecLag1.reshape(self.numScores, 2, 1, 1, 1), (1, 1, self.numScores, 2, 2))

    def epiInit(self):
        scoreIndexIsQual2DArr = self.initProbTrue.reshape(self.numScores * 2)
        index = np.argmax(np.random.multinomial(1, scoreIndexIsQual2DArr))
        self.scoreIndex = int(index / 2)
        self.isQual = index - self.scoreIndex * 2


    def create_isQual(self, decision):
        if decision == 1:
            if self.isQual == 1:
                isQual = np.argmax(np.random.multinomial(1, self.transProbIsQualCondIsQualLag1Val1DecLag1Val1ListList[self.isQual]))
            else:
                isQual = np.argmax(np.random.multinomial(1, self.transProbIsQualCondIsQualLag1Val0DecLag1Val1ListList[self.isQual]))
        else:
            isQual = np.argmax(np.random.multinomial(1, self.transProbIsQualCondIsQualLag1Val1DecLag1Val0ListList[self.isQual]))
        return isQual

    def create_scoreIndex(self, isQual):
#        print(decision)
        if isQual == 1:
            scoreIndex = np.argmax(np.random.multinomial(1, self.transProbScoreCondScoreLag1IsQualVal1ListList[self.scoreIndex]))
        else:
            scoreIndex = np.argmax(np.random.multinomial(1, self.transProbScoreCondScoreLag1IsQualVal0ListList[self.scoreIndex]))
        return scoreIndex

    def create_rewardVal(self, decision):#how the decision impacting the distribution
        if decision == 1:
            if self.isQual == 1:
                rewardVal = self.rewardCoefIsQual1Dec1 * self.scoreIndex #* self.scoreIndex2Score(self.scoreIndex) + self.rewardCoefDecNeg
            else:
                rewardVal = self.rewardCoefIsQual0Dec1
        else:
            rewardVal = self.rewardCoefDec0
        return rewardVal


    def update_sumScoreIsQualAct1PerTm(self, decision, timeIndex):
        if decision == 1:
            self.sumScoreIsQualAct1PerTm[timeIndex] += 1


    def update_sumScoreAct1IsQual1PerTm(self, decision, timeIndex):
        if self.isQual == 1 and decision == 1:
            self.sumScoreAct1IsQual1PerTm[timeIndex] += 1

    def update_sumScoreActIsQual1PerTm(self, timeIndex):
        if self.isQual == 1:
            self.sumScoreActIsQual1PerTm[timeIndex] += 1

    def update_sumScoreAct1IsQual1(self, decision):
        if self.isQual == 1 and decision == 1:
            self.sumScoreAct1IsQual1 += 1

    def update_sumScoreActIsQual1(self):
        if self.isQual == 1:
            self.sumScoreActIsQual1 += 1

    def set_isQual(self, isQual):
        self.isQual = isQual

    def set_scoreIndex(self, scoreIndex):
        self.scoreIndex = scoreIndex

    def scoreIndex2Score(self, scoreIndex):
        return scoreIndex + self.minScore

class EnvTrainSimGenerative(object):
    def __init__(self):
        self.path = None
        self.useConfigMode = None
        self.identifier = None
        self.device = None

        self.nnModel = None

        self.nnInputDim = None
        self.nnHiddenDim = None
        self.nnTargetDim = None

        self.nStates = None
        self.nActions = None

        self.numNnTrainSteps = None

        self.stepIndex = None

        #rl mem var
        self.initState = None
        self.state = None

        self.rewardFunc = None

        self.initDataInputState = None
        self.initDataTargetState = None
        self.initDataInputReward = None
        self.initDataTargetReward = None

        self.fbDataInputState = None
        self.fbDataTargetState = None
        self.fbDataInputReward = None
        self.fbDataTargetReward = None

        self.groupList = None
        self.groupProbList = None

        self.policyUpdateEvalCumReward = None

        self.status = None

    def read_config(self, identifier, rootDir, optConfigCurDir, optDataCurDir, iptConfigHistDir, iptDataHistDir, device, numEpisode, numInterPolicyUpdateEvalEpisodes, numNnTrainSteps, groupProbList, evalType):#, nnBatchSize, nnInputDim, nnHiddenDim, nnTargetDim, nnLayerDim, numNnTrainSteps):
        iptConfigFileName, self.optConfigCurDir, self.iptConfigHistDir, self.optDataCurDir = Path.make_path(rootDir, self.__class__.__name__, identifier, optConfigCurDir, iptConfigHistDir, optDataCurDir)
        self.iptDataHistDir = iptDataHistDir

        
        with open(iptConfigFileName) as json_file:
            config = json.load(json_file)
 
            nnModelName = config["nnModelName"]

            self.rewardFuncName = config["rewardFuncName"]

            self.numScores = config["numScores"]
            self.minScore = config["minScore"]
            self.isQualDnIsQualLag1Val1DecLag1Val1WgtDisList = config["isQualDnIsQualLag1Val1DecLag1Val1WgtDisList"]
            self.isQualEqIsQualLag1Val1DecLag1Val1WgtDis = config["isQualEqIsQualLag1Val1DecLag1Val1WgtDis"]
            self.isQualUpIsQualLag1Val1DecLag1Val1WgtDisList = config["isQualUpIsQualLag1Val1DecLag1Val1WgtDisList"]
            
            self.isQualDnIsQualLag1Val1DecLag1Val0WgtDisList = config["isQualDnIsQualLag1Val1DecLag1Val0WgtDisList"]
            self.isQualEqIsQualLag1Val1DecLag1Val0WgtDis = config["isQualEqIsQualLag1Val1DecLag1Val0WgtDis"]
            self.isQualUpIsQualLag1Val1DecLag1Val0WgtDisList = config["isQualUpIsQualLag1Val1DecLag1Val0WgtDisList"]

            self.isQualDnIsQualLag1Val0DecLag1Val1WgtDisList = config["isQualDnIsQualLag1Val0DecLag1Val1WgtDisList"]
            self.isQualEqIsQualLag1Val0DecLag1Val1WgtDis = config["isQualEqIsQualLag1Val0DecLag1Val1WgtDis"]
            self.isQualUpIsQualLag1Val0DecLag1Val1WgtDisList = config["isQualUpIsQualLag1Val0DecLag1Val1WgtDisList"]

            self.isQualDnIsQualLag1Val0DecLag1Val0WgtDisList = config["isQualDnIsQualLag1Val0DecLag1Val0WgtDisList"]
            self.isQualEqIsQualLag1Val0DecLag1Val0WgtDis = config["isQualEqIsQualLag1Val0DecLag1Val0WgtDis"]
            self.isQualUpIsQualLag1Val0DecLag1Val0WgtDisList = config["isQualUpIsQualLag1Val0DecLag1Val0WgtDisList"]

            self.scoreScoreLag1IsQualVal1DnWgtDisList = config["scoreScoreLag1IsQualVal1DnWgtDisList"]
            self.scoreScoreLag1IsQualVal1EqWgtDis = config["scoreScoreLag1IsQualVal1EqWgtDis"]
            self.scoreScoreLag1IsQualVal1UpWgtDisList = config["scoreScoreLag1IsQualVal1UpWgtDisList"]
            self.scoreScoreLag1IsQualVal0DnWgtDisList = config["scoreScoreLag1IsQualVal0DnWgtDisList"]
            self.scoreScoreLag1IsQualVal0EqWgtDis = config["scoreScoreLag1IsQualVal0EqWgtDis"]
            self.scoreScoreLag1IsQualVal0UpWgtDisList = config["scoreScoreLag1IsQualVal0UpWgtDisList"]
 
            self.rewardCoefIsQual1Dec1List = config["rewardCoefIsQual1Dec1List"]
            self.rewardCoefIsQual0Dec1List = config["rewardCoefIsQual0Dec1List"]
            self.rewardCoefDec0List = config["rewardCoefDec0List"]

          

            self.identifier = identifier
            self.device = device

            self.numEpisode = numEpisode

            self.numInterPolicyUpdateEvalEpisodes = numInterPolicyUpdateEvalEpisodes
            self.numNnTrainSteps = numNnTrainSteps

            self.stepIndexCrossEpisode = 0

            self.groupProbList = groupProbList

            self.evalType = evalType

            self.status = ENV_INIT

            self.policyUpdateEvalCumReward = 0

            if self.evalType == "Dp":
                initProbTrueList, transProbTrueList, rewardTrueList = self.create_explicitModel_dp()
            elif self.evalType == "EqOpt":
                initProbTrueList, transProbTrueList, rewardTrueList = self.create_explicitModel_eqOpt()
            elif self.evalType == "Agg":
                initProbTrueList, transProbTrueList, rewardTrueList = self.create_explicitModel_agg()
            elif self.evalType == "DpCv":
                initProbTrueList, transProbTrueList, rewardTrueList = self.create_explicitModel_dp()
            elif self.evalType == "EqOptCv":
                initProbTrueList, transProbTrueList, rewardTrueList = self.create_explicitModel_eqOpt()

            self.groupList = []
            self.numGroups = len(self.groupProbList)
            for i in range(self.numGroups):
                group = GroupEnv()
                group.init(self.numScores, self.minScore, self.numNnTrainSteps, self.isQualDnIsQualLag1Val1DecLag1Val1WgtDisList[i], self.isQualEqIsQualLag1Val1DecLag1Val1WgtDis[i], self.isQualUpIsQualLag1Val1DecLag1Val1WgtDisList[i], self.isQualDnIsQualLag1Val0DecLag1Val1WgtDisList[i], self.isQualEqIsQualLag1Val0DecLag1Val1WgtDis[i], self.isQualUpIsQualLag1Val0DecLag1Val1WgtDisList[i], self.isQualDnIsQualLag1Val1DecLag1Val0WgtDisList[i], self.isQualEqIsQualLag1Val1DecLag1Val0WgtDis[i], self.isQualUpIsQualLag1Val1DecLag1Val0WgtDisList[i], self.isQualDnIsQualLag1Val0DecLag1Val0WgtDisList[i], self.isQualEqIsQualLag1Val0DecLag1Val0WgtDis[i], self.isQualUpIsQualLag1Val0DecLag1Val0WgtDisList[i], self.scoreScoreLag1IsQualVal1DnWgtDisList[i], self.scoreScoreLag1IsQualVal1EqWgtDis[i], self.scoreScoreLag1IsQualVal1UpWgtDisList[i], self.scoreScoreLag1IsQualVal0DnWgtDisList[i], self.scoreScoreLag1IsQualVal0EqWgtDis[i], self.scoreScoreLag1IsQualVal0UpWgtDisList[i], self.rewardCoefIsQual1Dec1List[i], self.rewardCoefIsQual0Dec1List[i], self.rewardCoefDec0List[i], initProbTrueList[i], transProbTrueList[i], rewardTrueList[i])
                self.groupList.append(group)


            Config.write_config(config, self.optConfigCurDir + self.__class__.__name__ + self.identifier + ".json")

    def create_explicitModel_dp(self):

        rewardTrueList = []
        rewardTrueList.append(np.tile(np.array([[(2-j) * (self.numScores - i) for j in range(2)] for i in range(self.numScores)]).reshape(self.numScores, 1, 2),(1, 2, 1)))
        rewardTrueList.append(np.tile(np.array([[(j+1) * (i+1) for j in range(2)] for i in range(self.numScores)]).reshape(self.numScores, 1, 2),(1, 2, 1)))

        initProbTrueList = []
        npInitProbPerGroupArr= np.array([[i * self.numScores + j for j in range(2)] for i in range(self.numScores)])
        initProbPerGroupArr = npInitProbPerGroupArr / np.sum(npInitProbPerGroupArr)
        initProbTrueList.append(initProbPerGroupArr)
        npInitProbPerGroupArr= np.array([[(self.numScores-i) * self.numScores + 1-j for j in range(2)] for i in range(self.numScores)])
        initProbPerGroupArr = npInitProbPerGroupArr / np.sum(npInitProbPerGroupArr)
        initProbTrueList.append(initProbPerGroupArr)

        #uniform transProb
        transProbTrueList = []
        scoreIsQualArr = np.ones(shape = (self.numScores, 2)) * 1.0 / self.numScores / 2
        transProbTrueList.append(np.tile(scoreIsQualArr.reshape(self.numScores, 2, 1, 1, 1), (1, 1, self.numScores, 2, 2)))
        scoreIsQualArr = np.ones(shape = (self.numScores, 2)) * 1.0 / self.numScores / 2
        transProbTrueList.append(np.tile(scoreIsQualArr.reshape(self.numScores, 2, 1, 1, 1), (1, 1, self.numScores, 2, 2)))


        return initProbTrueList, transProbTrueList, rewardTrueList

    def create_explicitModel_eqOpt(self):
        rewardTrueList = []

        rewardTrueList.append(np.tile(np.array([[(2-j) * (self.numScores - i) for j in range(2)] for i in range(self.numScores)]).reshape(self.numScores, 1, 2),(1, 2, 1)))
        rewardTrueList.append(np.tile(np.array([[(j+1) * (i+1) for j in range(2)] for i in range(self.numScores)]).reshape(self.numScores, 1, 2),(1, 2, 1)))

        initProbTrueList = []
        initProbPerGroupArr = np.ones(shape = (self.numScores, 2)) * 1.0 / self.numScores / 2
        initProbTrueList.append(initProbPerGroupArr)
        initProbPerGroupArr = np.ones(shape = (self.numScores, 2)) * 1.0 / self.numScores / 2
        initProbTrueList.append(initProbPerGroupArr)

        transProbTrueList = []
        scoreIsQualArr = np.ones(shape = (self.numScores, 2)) * 1.0 / self.numScores / 2
        transProbTrueList.append(np.tile(scoreIsQualArr.reshape(self.numScores, 2, 1, 1, 1), (1, 1, self.numScores, 2, 2)))
        scoreIsQualArr = np.ones(shape = (self.numScores, 2)) * 1.0 / self.numScores / 2
        transProbTrueList.append(np.tile(scoreIsQualArr.reshape(self.numScores, 2, 1, 1, 1), (1, 1, self.numScores, 2, 2)))

        return initProbTrueList, transProbTrueList, rewardTrueList

    def create_explicitModel_agg(self):
        rewardTrueList = []

        rewardTrueList.append(np.tile(np.array([[(2-j) * (self.numScores - i) for j in range(2)] for i in range(self.numScores)]).reshape(self.numScores, 1, 2),(1, 2, 1)))
        rewardTrueList.append(np.tile(np.array([[(j+1) * (i+1) for j in range(2)] for i in range(self.numScores)]).reshape(self.numScores, 1, 2),(1, 2, 1)))

        initProbTrueList = []
        initProbPerGroupArr = np.ones(shape = (self.numScores, 2)) * 1.0 / self.numScores / 2
        initProbTrueList.append(initProbPerGroupArr)
        initProbPerGroupArr = np.ones(shape = (self.numScores, 2)) * 1.0 / self.numScores / 2
        initProbTrueList.append(initProbPerGroupArr)

        transProbTrueList = []
        scoreIsQualArr = np.ones(shape = (self.numScores, 2)) * 1.0 / self.numScores / 2
        transProbTrueList.append(np.tile(scoreIsQualArr.reshape(self.numScores, 2, 1, 1, 1), (1, 1, self.numScores, 2, 2)))
        scoreIsQualArr = np.ones(shape = (self.numScores, 2)) * 1.0 / self.numScores / 2
        transProbTrueList.append(np.tile(scoreIsQualArr.reshape(self.numScores, 2, 1, 1, 1), (1, 1, self.numScores, 2, 2)))

        return initProbTrueList, transProbTrueList, rewardTrueList

    def get_initProbList_transProbList_rewardList(self):
        initProbList = []
        transProbList = []
        rewardList = []
        for groupIndex in range(self.numGroups):
            initProbList.append(self.groupList[groupIndex].initProbTrue)
            transProbList.append(self.groupList[groupIndex].transProbTrue)
            rewardList.append(self.groupList[groupIndex].rewardTrue)
        return initProbList, transProbList, rewardList

    def load_init_data(self, dataInputState, dataTargetState, dataInputReward, dataTargetReward):
        self.initDataInputState = dataInputState
        self.initDataTargetState = dataTargetState
        self.initDataInputReward = dataInputReward
        self.initDataTargetReward = dataTargetReward

    def load_fb_data(self, dataInputState, dataTargetState, dataInputReward, dataTargetReward):
        self.fbDataInputState = dataInputState
        self.fbDataTargetState = dataTargetState
        self.fbDataInputReward = dataInputReward
        self.fbDataTargetReward = dataTargetReward

    def epiInit(self):
        for groupIndex in range(self.numGroups):
            self.groupList[groupIndex].epiInit()

        
        scoreIndexList = []
        for groupIndex in range(self.numGroups):
            scoreIndexList.append(self.groupList[groupIndex].scoreIndex)

        cbData = CbData()
        cbData.set_scoreIndexList(scoreIndexList)


        self.cbData = cbData


    def getCbData(self):
        return self.cbData

    def feedback(self, action, nnTrainStepIndex):
        #part 1: forward dynamics for the group of the action, update this group stats
        memScoreIndexList = []
        memIsQualList = []
        for groupIndex in range(self.numGroups):
            memScoreIndexList.append(self.groupList[groupIndex].scoreIndex)
            memIsQualList.append(self.groupList[groupIndex].isQual)

        state = State() 
        state.set(memScoreIndexList, memIsQualList)
        
        memRewardValList = []
        for groupIndex in range(self.numGroups):
            memRewardVal = self.groupList[groupIndex].create_rewardVal(action.decisionList[groupIndex])#.groupActionList[groupIndex])
            memRewardValList.append(memRewardVal)

        reward = Reward()
        reward.set_groupProb(self.groupProbList)
        reward.set_rewardVal(memRewardValList)


        if self.status == ENV_UPDATE_EVAL:
            if self.evalType == "Dp":
                for groupIndex in range(self.numGroups):
                    self.groupList[groupIndex].update_sumScoreIsQualAct1PerTm(action.decisionList[groupIndex], nnTrainStepIndex)

            elif self.evalType == "EqOpt":
                for groupIndex in range(self.numGroups):
                    self.groupList[groupIndex].update_sumScoreAct1IsQual1PerTm(action.decisionList[groupIndex], nnTrainStepIndex)
                    self.groupList[groupIndex].update_sumScoreActIsQual1PerTm(nnTrainStepIndex)
        
            elif self.evalType == "Agg":
                for groupIndex in range(self.numGroups):
                    self.groupList[groupIndex].update_sumScoreAct1IsQual1(action.decisionList[groupIndex])
                    self.groupList[groupIndex].update_sumScoreActIsQual1()

            if self.evalType == "DpCv":
                for groupIndex in range(self.numGroups):
                    self.groupList[groupIndex].update_sumScoreIsQualAct1PerTm(action.decisionList[groupIndex], nnTrainStepIndex)

            elif self.evalType == "EqOptCv":
                for groupIndex in range(self.numGroups):
                    self.groupList[groupIndex].update_sumScoreAct1IsQual1PerTm(action.decisionList[groupIndex], nnTrainStepIndex)
                    self.groupList[groupIndex].update_sumScoreActIsQual1PerTm(nnTrainStepIndex)

            rewardValCrossGroup = reward.comp_rewardCrossGroup()
            self.policyUpdateEvalCumReward += rewardValCrossGroup

        isQualList = []
        scoreIndexList = []
        for groupIndex in range(self.numGroups):
            isQual = self.groupList[groupIndex].create_isQual(action.decisionList[groupIndex])
            scoreIndex = self.groupList[groupIndex].create_scoreIndex(isQual)

            isQualList.append(isQual)
            scoreIndexList.append(scoreIndex)

        for groupIndex in range(self.numGroups):
            self.groupList[groupIndex].set_isQual(isQualList[groupIndex])
            self.groupList[groupIndex].set_scoreIndex(scoreIndexList[groupIndex])

        cbData = CbData()
        cbData.set_scoreIndexList(scoreIndexList)
        self.cbData = cbData


        return state, reward

    def begin_est(self):
        self.status = ENV_UPDATE_EVAL

    def end_est_dp(self):#, trainEpisodeIndex):
        firstGroupIndex = 0
        secondGroupIndex = 1
        if True:
            estDp = 0
            for timeIndex in range(self.numNnTrainSteps):
                constrPerTm = abs(self.groupList[firstGroupIndex].sumScoreIsQualAct1PerTm[timeIndex] * 1.0 / self.numInterPolicyUpdateEvalEpisodes - self.groupList[secondGroupIndex].sumScoreIsQualAct1PerTm[timeIndex] * 1.0 / self.numInterPolicyUpdateEvalEpisodes)
                estDp += constrPerTm
                
            estDp /= self.numNnTrainSteps### * self.numScores#* self.numInterPolicyUpdateEvalEpisodes * 1.0
   
            policyAvgRet = self.policyUpdateEvalCumReward / self.numInterPolicyUpdateEvalEpisodes

            self.groupList[firstGroupIndex].sumScoreIsQualAct1PerTm.fill(0)
            self.groupList[secondGroupIndex].sumScoreIsQualAct1PerTm.fill(0)
            self.policyUpdateEvalCumReward = 0

            self.status = ENV_INIT

            return policyAvgRet, estDp

    def end_est_eqOpt(self):#, trainEpisodeIndex):
        firstGroupIndex = 0
        secondGroupIndex = 1

        if True:
            estEqOpt = 0
            for timeIndex in range(self.numNnTrainSteps):
                constrPerTm = abs(self.groupList[firstGroupIndex].sumScoreAct1IsQual1PerTm[timeIndex] * 1.0 / max(self.groupList[firstGroupIndex].sumScoreActIsQual1PerTm[timeIndex], 1e-6) - self.groupList[secondGroupIndex].sumScoreAct1IsQual1PerTm[timeIndex] * 1.0 / max(self.groupList[secondGroupIndex].sumScoreActIsQual1PerTm[timeIndex], 1e-6))
                estEqOpt += constrPerTm
            estEqOpt /= self.numNnTrainSteps * 1.0 ##self.numInterPolicyUpdateEvalEpisodes * 1.0

            policyAvgRet = self.policyUpdateEvalCumReward / self.numInterPolicyUpdateEvalEpisodes

            self.groupList[firstGroupIndex].sumScoreAct1IsQual1PerTm.fill(0)
            self.groupList[secondGroupIndex].sumScoreAct1IsQual1PerTm.fill(0)
            self.groupList[firstGroupIndex].sumScoreActIsQual1PerTm.fill(0)
            self.groupList[secondGroupIndex].sumScoreActIsQual1PerTm.fill(0)

            self.policyUpdateEvalCumReward = 0

            self.status = ENV_INIT

            return policyAvgRet, estEqOpt

    def end_est_agg(self):#, trainEpisodeIndex):
        firstGroupIndex = 0
        secondGroupIndex = 1

        if True:
            estAgg = 0
            constr = abs(self.groupList[firstGroupIndex].sumScoreAct1IsQual1 * 1.0 / max(self.groupList[firstGroupIndex].sumScoreActIsQual1, 1e-6) - self.groupList[secondGroupIndex].sumScoreAct1IsQual1 * 1.0 / max(self.groupList[secondGroupIndex].sumScoreActIsQual1, 1e-6))
            estAgg = constr#+= constrPerTm

            policyAvgRet = self.policyUpdateEvalCumReward / self.numInterPolicyUpdateEvalEpisodes

            self.groupList[firstGroupIndex].sumScoreAct1IsQual1 = 0#.fill(0)
            self.groupList[secondGroupIndex].sumScoreAct1IsQual1 = 0#.fill(0)
            self.groupList[firstGroupIndex].sumScoreActIsQual1 = 0#.fill(0)
            self.groupList[secondGroupIndex].sumScoreActIsQual1 = 0#.fill(0)

            self.policyUpdateEvalCumReward = 0

            self.status = ENV_INIT

            return policyAvgRet, estAgg

    def end_est_dpCv(self):#, trainEpisodeIndex):
        firstGroupIndex = 0
        secondGroupIndex = 1
        if True:
            estDp = 0
            for timeIndex in range(self.numNnTrainSteps):
                constrPerTm = abs(self.groupList[firstGroupIndex].sumScoreIsQualAct1PerTm[timeIndex] * 1.0 / self.numInterPolicyUpdateEvalEpisodes - self.groupList[secondGroupIndex].sumScoreIsQualAct1PerTm[timeIndex] * 1.0 / self.numInterPolicyUpdateEvalEpisodes)
                estDp += constrPerTm
            estDp /= self.numNnTrainSteps### * self.numScores#* self.numInterPolicyUpdateEvalEpisodes * 1.0
   
            policyAvgRet = (self.policyUpdateEvalCumReward / self.numInterPolicyUpdateEvalEpisodes) ** 2

            self.groupList[firstGroupIndex].sumScoreIsQualAct1PerTm.fill(0)
            self.groupList[secondGroupIndex].sumScoreIsQualAct1PerTm.fill(0)

            self.policyUpdateEvalCumReward = 0

            self.status = ENV_INIT

            return policyAvgRet, estDp

    def end_est_eqOptCv(self):#, trainEpisodeIndex):
        firstGroupIndex = 0
        secondGroupIndex = 1

        if True:
            estEqOpt = 0
            for timeIndex in range(self.numNnTrainSteps):
                constrPerTm = abs(self.groupList[firstGroupIndex].sumScoreAct1IsQual1PerTm[timeIndex] * 1.0 / max(self.groupList[firstGroupIndex].sumScoreActIsQual1PerTm[timeIndex], 1e-6) - self.groupList[secondGroupIndex].sumScoreAct1IsQual1PerTm[timeIndex] * 1.0 / max(self.groupList[secondGroupIndex].sumScoreActIsQual1PerTm[timeIndex], 1e-6))
                estEqOpt += constrPerTm
            estEqOpt /= self.numNnTrainSteps * 1.0 ##self.numInterPolicyUpdateEvalEpisodes * 1.0

            policyAvgRet = (self.policyUpdateEvalCumReward / self.numInterPolicyUpdateEvalEpisodes) ** 2

            self.groupList[firstGroupIndex].sumScoreAct1IsQual1PerTm.fill(0)
            self.groupList[secondGroupIndex].sumScoreAct1IsQual1PerTm.fill(0)
            self.groupList[firstGroupIndex].sumScoreActIsQual1PerTm.fill(0)
            self.groupList[secondGroupIndex].sumScoreActIsQual1PerTm.fill(0)

            self.policyUpdateEvalCumReward = 0

            self.status = ENV_INIT

            return policyAvgRet, estEqOpt


