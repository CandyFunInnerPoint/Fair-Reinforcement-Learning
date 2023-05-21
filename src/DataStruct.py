#from NnModel import NActionsNnModel, NStatesNnModel, ActionNnModel, StateNnModel
import torch

class Obs(object):
    def __init__(self):
        self.groupIndex = None
        self.scoreIndex = None


class Action(object):
    def __init__(self):
        self.decisionList = None

class Reward(object):
    def __init__(self):
        self.groupProbList = None
        self.rewardValList = None
    
    def set_groupProb(self, groupProbList):
        self.groupProbList = groupProbList

    def set_rewardVal(self, rewardValList):
        self.rewardValList = rewardValList

    def comp_rewardCrossGroup(self):
        rewardCrossGroup = 0
        for groupIndex in range(len(self.groupProbList)):
            rewardCrossGroup += self.groupProbList[groupIndex] * self.rewardValList[groupIndex]
        return rewardCrossGroup

class State(object):
    def __init__(self):
        self.scoreIndexList = None
        self.isQualList = None

    def set(self, scoreIndexList, isQualList):
        self.scoreIndexList = scoreIndexList
        self.isQualList = isQualList

#can be used for both stateBatch and state
class CbData(object):
    def __init__(self):
#        self.groupIndex = None
        self.scoreIndexList = None


    def set_scoreIndexList(self, scoreIndexList):
        self.scoreIndexList = scoreIndexList



