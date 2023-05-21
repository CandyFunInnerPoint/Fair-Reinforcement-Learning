from Config import Config
import json, os, sys
from datetime import datetime
#import time
from Pipeline import Pipeline
from ActorCriticsBackpropTrain import ActorCriticsBackpropTrain
from MemMgmt import MemMgmtOptTwoClassNormalDist, MemMgmtOptTwoClassNormalSampler, MemMgmtOptInitNnModelParaLossLevelGen#MemMgmtOptInitTrainNnModelParaLossLevelGen

import torch

from collections import OrderedDict

class PipelineActorCriticsBackpropTrain(Pipeline):
    def __init__(self):
        
        self.device = None

        self.rootDir = None

    def create_nodes(self, identifier, projDir, rootDir):
        with open(rootDir + "config/" + __class__.__name__ + identifier  + ".json") as json_file:
            config = json.load(json_file)
            self.pipeDataGenHistLocalTimeInMicro = config["pipeDataGenHistLocalTimeInMicro"]
            self.histLocalTimeInMicro = config["histLocalTimeInMicro"]
            self.memMgmtOptDataSamplerDiskMode = config["memMgmtOptDataSamplerDiskMode"]
            self.memMgmtOptInitNnModelParaLossLevelGenDiskMode = config["memMgmtOptInitNnModelParaLossLevelGenDiskMode"]
            self.dataDim = config["dataDim"]

            self.startNodeClassName = config["startNodeClassName"]
            self.startNodeObjName = config["startNodeObjName"]

            self.exprimentHyperConfigDir = sys.argv[1] if len(sys.argv) > 0 else None
            self.curLocalTimeInMicro = sys.argv[2] if len(sys.argv) > 0 else None

            print("curLocalTimeInMicro: ", self.curLocalTimeInMicro)

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self.projDir = projDir
            self.rootDir = rootDir
            self.identifier = identifier
            self.actorCriticsBackpropTrain = ActorCriticsBackpropTrain()
 
            self.memMgmtOptDataSampler = MemMgmtOptTwoClassNormalSampler()
            self.memMgmtOptInitNnModelParaLossLevelGen = MemMgmtOptInitNnModelParaLossLevelGen()
            self.pipeNameOptDict = OrderedDict()

            self.optConfigCurDir, self.optDataCurDir = self.make_cur_path(self.curLocalTimeInMicro)
            self.iptConfigHistDir = projDir + '/hist/' + self.__class__.__name__ + self.identifier + "/" + self.histLocalTimeInMicro + "/config/" if self.histLocalTimeInMicro != None else None
            self.actorCriticsBackpropTrain.read_config("", rootDir, self.exprimentHyperConfigDir, self.optConfigCurDir, self.optDataCurDir, self.iptConfigHistDir, "", self.curLocalTimeInMicro, self.device, self.pipeNameOptDict, self.dataDim)


            Config.write_config(config, self.optConfigCurDir + self.__class__.__name__ + self.identifier + ".json")

    def create_graph(self):
        self.graphTraverseList = [self.actorCriticsBackpropTrain]
        self.nameTraverseIndexDict = OrderedDict()
        self.nameTraverseIndexDict[self.actorCriticsBackpropTrain.__class__.__name__ + "_" + self.actorCriticsBackpropTrain.identifier] = 0

        self.startNodeIndex = self.nameTraverseIndexDict[self.startNodeClassName + "_" + self.startNodeObjName]

#        return config

    def make_cur_path(self, curLocalTimeInMicro):#, identifier ,rootDir):
        while True:
            if curLocalTimeInMicro == None:
                curLocalTimeInMicro = datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S-%f') #time.strftime('%Y-%m-%d-%H-%M-%S-%f', time.localtime(time.time()))# * 10**7
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
    pipeline = PipelineActorCriticsBackpropTrain()
    projDir = "../../"
    rootDir = "../"
    pipeline.create_nodes("", projDir, rootDir)
    pipeline.create_graph()
    pipeline.run()
    

    

