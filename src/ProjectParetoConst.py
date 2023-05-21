import subprocess
from subprocess import PIPE
from collections import OrderedDict
from datetime import datetime
import json, os, csv
from Config import Config
import shutil

class ProjectParetoConst(object):
    def __init__(self):
        self.numTrialsTrain = None
        self.algoTypeList = None
        self.origConstList = None
        self.lagConstList = None
        self.evalTypeList = None
        self.envTypeList = None

        self.projDir = None

    def read_config(self, identifier, projDir):
        with open("../config/" + __class__.__name__ + ".json") as json_file:
            config = json.load(json_file)
            self.numTrialsTrain = config["numTrialsTrain"]
            self.algoTypeList = config["algoTypeList"]
#            self.origConstList = config["origConstList"]
            self.lagConstList = config["lagConstList"]
            self.mypConstList = config["mypConstList"]
            self.sampleSizeScalar = config["sampleSizeScalar"]
            self.evalTypeList = config["evalTypeList"]
            self.envTypeList = config["envTypeList"]

            self.projDir = projDir
            self.identifier = identifier 

            self.optConfigCurDir, self.optDataCurDir = self.make_cur_path()

            self.exprimentHyperConfigDir = self.optConfigCurDir + "hyperConfig/"
            if not os.path.exists(self.exprimentHyperConfigDir):
                os.makedirs(self.exprimentHyperConfigDir)
            self.exprimentIdPath = self.optDataCurDir + self.__class__.__name__ + self.identifier + ".csv"

            Config.write_config(config, self.optConfigCurDir + self.__class__.__name__ + self.identifier + ".json")

    def write_hyperConfig(self):
        exprRunDictList = []
        for algoType in self.algoTypeList:
            if True:
                for evalType in self.evalTypeList:
                    isQualRuleDp = (evalType == "Dp" and algoType in ["DpOrigGp"]) 
                    isQualRuleEqOpt = (evalType == "EqOpt" and algoType in ["EqOptOrigGp"]) 
                    isQualRuleAgg = (evalType == "Agg" and algoType in ["AggOrigGp"]) 
                    isQualRuleDpCv = (evalType == "DpCv" and algoType in ["DpCvOrigGp"]) 
                    isQualRuleEqOptCv = (evalType == "EqOptCv" and algoType in ["EqOptCvOrigGp"]) 
                    isQualRule = isQualRuleDp or isQualRuleEqOpt or isQualRuleAgg or isQualRuleDpCv or isQualRuleEqOptCv
                   
                    if not isQualRule:
                        continue

                    for envType in self.envTypeList:
                        for trialIndex in range(self.numTrialsTrain):
                            exprRunDict = OrderedDict()
                            exprRunDict["algoType"] = algoType
                            exprRunDict["paretoType"] = "Const"
                            exprRunDict["methodType"] = "orig"
                            exprRunDict["methodConst"] = "NA" #origConst
                            exprRunDict["methodSampleSize"] = self.sampleSizeScalar
                            exprRunDict["methodMix"] = "NA"
                            exprRunDict["evalType"] = evalType
                            exprRunDict["envType"] = envType
                            exprRunDict["trialIndex"] = trialIndex
                            exprRunDictList.append(exprRunDict)

            #paraeto const lag
            for lagConst in self.lagConstList:
                for evalType in self.evalTypeList:
                    isQualRuleDp = (evalType == "Dp" and algoType in ["DpLagGp"]) 
                    isQualRuleEqOpt = (evalType == "EqOpt" and algoType in ["EqOptLagGp"]) 
                    isQualRuleAgg = (evalType == "Agg" and algoType in ["AggLagGp"]) 
                    isQualRuleDpCv = (evalType == "DpCv" and algoType in ["DpCvLagGp"]) 
                    isQualRuleEqOptCv = (evalType == "EqOptCv" and algoType in ["EqOptCvLagGp"]) 
                    isQualRule = isQualRuleDp or isQualRuleEqOpt or isQualRuleAgg or isQualRuleDpCv or isQualRuleEqOptCv
                   
                    if not isQualRule:
                        continue

                    for envType in self.envTypeList:
                        for trialIndex in range(self.numTrialsTrain):
                            exprRunDict = OrderedDict()
                            exprRunDict["algoType"] = algoType
                            exprRunDict["paretoType"] = "Const"
                            exprRunDict["methodType"] = "lag"
                            exprRunDict["methodConst"] = lagConst
                            exprRunDict["methodSampleSize"] = self.sampleSizeScalar
                            exprRunDict["methodMix"] = "NA"
                            exprRunDict["evalType"] = evalType
                            exprRunDict["envType"] = envType
                            exprRunDict["trialIndex"] = trialIndex
                            exprRunDictList.append(exprRunDict)

            #paraeto const myp
            for mypConst in self.mypConstList:
                for evalType in self.evalTypeList:
                    isQualRuleDp = (evalType == "Dp" and algoType in ["Myp"]) 
                    isQualRuleEqOpt = (evalType == "EqOpt" and algoType in ["Myp"]) 
                    isQualRuleAgg = (evalType == "Agg" and algoType in ["Myp"]) 
                    isQualRuleDpCv = (evalType == "DpCv" and algoType in ["Myp"]) 
                    isQualRuleEqOptCv = (evalType == "EqOptCv" and algoType in ["Myp"]) 

                    isQualRule = isQualRuleDp or isQualRuleEqOpt or isQualRuleAgg or isQualRuleDpCv or isQualRuleEqOptCv
                   
                    if not isQualRule:
                        continue

                    for envType in self.envTypeList:
                        for trialIndex in range(self.numTrialsTrain):
                            exprRunDict = OrderedDict()
                            exprRunDict["algoType"] = algoType
                            exprRunDict["paretoType"] = "Const"
                            exprRunDict["methodType"] = "myp"
                            exprRunDict["methodConst"] = mypConst
                            exprRunDict["methodSampleSize"] = self.sampleSizeScalar
                            exprRunDict["methodMix"] = "NA"
                            exprRunDict["evalType"] = evalType
                            exprRunDict["envType"] = envType
                            exprRunDict["trialIndex"] = trialIndex
                            exprRunDictList.append(exprRunDict)

        self.curLocalTimeInMicroForTrainList = []
        for exprRunDict in exprRunDictList:
            algoType = exprRunDict["algoType"]
            paretoType = exprRunDict["paretoType"]
            methodType = exprRunDict["methodType"]#orig or lag
            methodConst = exprRunDict["methodConst"]
            methodSampleSize = exprRunDict["methodSampleSize"]
            methodMix = exprRunDict["methodMix"]
            evalType = exprRunDict["evalType"]
            envType = exprRunDict["envType"]
            trialIndex = exprRunDict["trialIndex"]

            agentHyperJsonDict = OrderedDict()
            actorCriticsBackpropHyperJsonDict = OrderedDict()

            agentHyperJsonDict["algoType"] = algoType
            agentHyperJsonDict["methodType"] = methodType
            agentHyperJsonDict["methodConst"] = methodConst
            agentHyperJsonDict["methodSampleSize"] = methodSampleSize
            agentHyperJsonDict["methodMix"] = methodMix
            agentHyperJsonDict["paretoType"] = paretoType

            actorCriticsBackpropHyperJsonDict["paretoType"] = paretoType
            actorCriticsBackpropHyperJsonDict["evalType"] = evalType
            actorCriticsBackpropHyperJsonDict["envType"] = envType
            actorCriticsBackpropHyperJsonDict["numInterPolicyUpdateEpisodes"] = methodSampleSize
            actorCriticsBackpropHyperJsonDict["trialTrainIndex"] = trialIndex
            actorCriticsBackpropHyperJsonDict["agentName"] = "Agent" + algoType

            curLocalTimeInMicroForTrain = datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S-%f')

            with open(self.exprimentHyperConfigDir + "Agent" + algoType + "Hyper" + str(curLocalTimeInMicroForTrain) + ".json", "w") as jsonWriter:
                json.dump(agentHyperJsonDict, jsonWriter)

            with open(self.exprimentHyperConfigDir + "ActorCriticsBackpropTrainHyper" + str(curLocalTimeInMicroForTrain) + ".json", "w") as jsonWriter:
                json.dump(actorCriticsBackpropHyperJsonDict, jsonWriter)
        
            self.curLocalTimeInMicroForTrainList.append(curLocalTimeInMicroForTrain)

        with open(self.exprimentIdPath, "w") as csvFile:
            csvWriter = csv.writer(csvFile)
            for curLocalTimeInMicroForTrain in self.curLocalTimeInMicroForTrainList:
                csvWriter.writerow([curLocalTimeInMicroForTrain])

    def train(self):
        print(self.curLocalTimeInMicroForTrainList)
        self.curLocalTimeInMicroForTrainList = []
        
        with open(self.exprimentIdPath) as csvFile:
            csvReader = csv.reader(csvFile)
            for row in csvReader:
                self.curLocalTimeInMicroForTrainList.append(row[0])
        for curLocalTimeInMicroForTrain in self.curLocalTimeInMicroForTrainList:
            runDir = self.projDir + "hist/PipelineActorCriticsBackpropTrain/" + str(curLocalTimeInMicroForTrain) + "/"
            infeasibilityErrorCode = 1
            while infeasibilityErrorCode == 1:#result = 1 implies error
                if os.path.exists(runDir):
                    shutil.rmtree(runDir)
                subprocess.call(["python", "PipelineActorCriticsBackpropTrain.py", self.exprimentHyperConfigDir, curLocalTimeInMicroForTrain])
                infeasiblityErrorCodePath = runDir + "data/ActorCriticsBackpropTrain/AgentTrain/" + "infeasibilityErrorCode.csv"
                if os.path.exists(infeasiblityErrorCodePath):
                    with open(infeasiblityErrorCodePath) as csvFile:
                        csvReader = csv.reader(csvFile)
                        infeasibilityErrorCode = int(next(csvReader)[0])
                        print(infeasibilityErrorCode)
                else:
                    break

    def eval(self):
        print(self.curLocalTimeInMicroForTrainList)
        self.curLocalTimeInMicroForTrainList = []
        
        with open(self.exprimentIdPath) as csvFile:
            csvReader = csv.reader(csvFile)
            for row in csvReader:
                self.curLocalTimeInMicroForTrainList.append(row[0])
        self.curLocalTimeInMicroForEval = datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S-%f')
        proc = subprocess.Popen(["python", "PipelineActorCriticsBackpropEval.py"] + [self.exprimentHyperConfigDir, self.curLocalTimeInMicroForEval] + self.curLocalTimeInMicroForTrainList)
        proc.wait()


    def plot(self):
        print(self.curLocalTimeInMicroForTrainList)
        self.curLocalTimeInMicroForTrainList = []
        
        with open(self.exprimentIdPath) as csvFile:
            csvReader = csv.reader(csvFile)
            for row in csvReader:
                self.curLocalTimeInMicroForTrainList.append(row[0])
        subprocess.Popen(["python", "PipelineRLFairVis.py"] + [self.exprimentHyperConfigDir, self.curLocalTimeInMicroForEval] + self.curLocalTimeInMicroForTrainList)

    def make_cur_path(self):
        while True:
            curLocalTimeInMicro = datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S-%f') #time.strftime('%Y-%m-%d-%H-%M-%S-%f', time.localtime(time.time()))# * 10**7
            if not os.path.exists(self.projDir + 'hist/' + self.__class__.__name__ + self.identifier + "/" + str(curLocalTimeInMicro) + "/"):
                break;
        optConfigCurDir = self.projDir + "hist/" + self.__class__.__name__ + self.identifier + "/" + str(curLocalTimeInMicro) + "/config/" #+ '_' + str(tempId) + "/"
        optDataCurDir = self.projDir + "hist/" + self.__class__.__name__ + self.identifier + "/" + str(curLocalTimeInMicro) + "/data/"
        os.makedirs(optConfigCurDir)
        os.makedirs(optDataCurDir)
        return optConfigCurDir, optDataCurDir

if __name__ == "__main__":
    project = ProjectParetoConst()
    projDir = "../../"
    identifier = ""
    project.read_config(identifier, projDir)
    project.write_hyperConfig()
    project.train()
    project.eval()
    project.plot()
