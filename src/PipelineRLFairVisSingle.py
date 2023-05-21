from Config import Config
import json, os, sys
from datetime import datetime
from Experiment import Experiment
from Pipeline import Pipeline
from MemMgmtSingle import MemMgmtOptRLFairTrain, MemMgmtOptRLFairEval
import torch
from VisRLFairTrainSingle import VisRLFairTrainSingle
from VisRLFairEval import VisRLFairEval
from collections import OrderedDict

class PipelineRLFairVis(Pipeline):
    def __init__(self):
        self.path = None
        
        self.device = None

        self.data = None

        self.rootDir = None

    def create_nodes(self, identifier, projDir, rootDir):
        with open(rootDir + "config/" + __class__.__name__ + identifier  + ".json") as json_file:
            config = json.load(json_file)
            self.pipeTrainClassNameId = config["pipeTrainClassNameId"]
            self.pipeEvalClassNameId = config["pipeEvalClassNameId"]
            self.histLocalTimeInMicro = config["histLocalTimeInMicro"]
            self.memMgmtOptRLFairTrainDiskMode = config["memMgmtOptRLFairTrainDiskMode"]
            self.memMgmtOptRLFairEvalDiskMode = config["memMgmtOptRLFairEvalDiskMode"]

            self.startNodeClassName = config["startNodeClassName"]
            self.startNodeObjName = config["startNodeObjName"]

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self.projDir = projDir
            self.rootDir = rootDir
            self.identifier = identifier
            self.visTrain = VisRLFairTrainSingle()
            self.visEval = VisRLFairEval()

            self.memMgmtOptRLFairTrain = MemMgmtOptRLFairTrain()
            self.memMgmtOptRLFairEval = MemMgmtOptRLFairEval()

            self.pipeNameOptDict = OrderedDict()

            self.optConfigCurDir, self.optDataCurDir = self.make_cur_path()

            self.iptConfigHistDir = projDir + 'hist/' + self.__class__.__name__ + self.identifier + "/" + str(self.histLocalTimeInMicro) + "/config/" if self.histLocalTimeInMicro != None else None 

            self.iptDataHistDir = None#projDir + 'hist/' + self.pipeTrainClassNameId + "/" + self.pipeEvalHistLocalTimeInMicro + "/data/" if self.pipeEvalHistLocalTimeInMicro != None else None
            self.iptTrainDataHistDirList = None
            self.memMgmtOptRLFairTrain.read_config("", rootDir, self.optConfigCurDir, self.optDataCurDir, self.iptConfigHistDir, self.iptTrainDataHistDirList, self.pipeNameOptDict, self.memMgmtOptRLFairTrainDiskMode)

            self.visTrain.read_config("", rootDir, self.optConfigCurDir, self.optDataCurDir, self.iptConfigHistDir, self.iptDataHistDir, self.device, self.pipeNameOptDict)

            Config.write_config(config, self.optConfigCurDir + self.__class__.__name__ + self.identifier + ".json")

    def create_graph(self):
        #connect
        self.graphTraverseList = [self.memMgmtOptRLFairTrain, self.visTrain]#, self.visEval]

        self.nameTraverseIndexDict = OrderedDict()
        self.nameTraverseIndexDict[self.memMgmtOptRLFairTrain.__class__.__name__ + "_" + self.memMgmtOptRLFairTrain.identifier] = 0
        self.nameTraverseIndexDict[self.visTrain.__class__.__name__ + "_" + self.visTrain.identifier] = 1

        self.startNodeIndex = self.nameTraverseIndexDict[self.startNodeClassName + "_" + self.startNodeObjName]


    def make_cur_path(self):#, identifier ,rootDir):
        while True:
            curLocalTimeInMicro = datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S-%f')#time.time() * 10**7
            if not os.path.exists(self.projDir + 'hist/' + self.__class__.__name__ + self.identifier + "/" + str(curLocalTimeInMicro) + "/"):
                break;
#            tempId += 1
        optConfigCurDir = self.projDir + "hist/" + self.__class__.__name__ + self.identifier + "/" + str(curLocalTimeInMicro) + "/config/" #+ '_' + str(tempId) + "/"
        optDataCurDir = self.projDir + "hist/" + self.__class__.__name__ + self.identifier + "/" + str(curLocalTimeInMicro) + "/data/"
        os.makedirs(optConfigCurDir)
        os.makedirs(optDataCurDir)
        return optConfigCurDir, optDataCurDir

    def run(self):
        for node in self.graphTraverseList[self.startNodeIndex:]:
            node.run()

        


if __name__ == "__main__":
    pipeline = PipelineRLFairVis()
    projDir = "../../"
    rootDir = "../"
    pipeline.create_nodes("", projDir, rootDir)
    pipeline.create_graph()
    pipeline.run()

