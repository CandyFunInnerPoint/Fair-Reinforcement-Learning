from Config import Config
import json, os
from datetime import datetime
from Experiment import Experiment
from Pipeline import Pipeline
from MemMgmt import MemMgmtOptActorCriticsBackpropEval
import torch
from VisActorCriticsBackprop import VisActorCriticsBackprop
from collections import OrderedDict

class PipelineActorCriticsBackpropVis(Pipeline):
    def __init__(self):
        self.path = None
        
        self.device = None

        self.data = None

        self.rootDir = None

    def create_nodes(self, identifier, projDir, rootDir):
        with open(rootDir + "config/" + __class__.__name__ + identifier  + ".json") as json_file:
            config = json.load(json_file)
            self.pipeEvalClassNameId = config["pipeEvalClassNameId"]
            self.pipeEvalHistLocalTimeInMicro = config["pipeEvalHistLocalTimeInMicro"]
            self.histLocalTimeInMicro = config["histLocalTimeInMicro"]
            self.memMgmtOptActorCriticsBackpropEvalDiskMode = config["memMgmtOptActorCriticsBackpropEvalDiskMode"]

            self.startNodeClassName = config["startNodeClassName"]
            self.startNodeObjName = config["startNodeObjName"]

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self.projDir = projDir
            self.rootDir = rootDir
            self.identifier = identifier
            self.vis = VisActorCriticsBackprop()

            self.memMgmtOptActorCriticsBackpropEval = MemMgmtOptActorCriticsBackpropEval()

            self.pipeNameOptDict = OrderedDict()

            self.optConfigCurDir, self.optDataCurDir = self.make_cur_path()

            self.iptConfigHistDir = projDir + 'hist/' + self.__class__.__name__ + self.identifier + "/" + str(self.histLocalTimeInMicro) + "/config/" if self.histLocalTimeInMicro != None else None

            self.iptDataHistDir = projDir + 'hist/' + self.pipeEvalClassNameId + "/" + self.pipeEvalHistLocalTimeInMicro + "/data/" if self.pipeEvalHistLocalTimeInMicro != None else None

            self.vis.read_config("", rootDir, self.optConfigCurDir, self.optDataCurDir, self.iptConfigHistDir, self.iptDataHistDir, self.device, self.pipeNameOptDict)


            self.memMgmtOptActorCriticsBackpropEval.read_config("", rootDir, self.optConfigCurDir, self.optDataCurDir, self.iptConfigHistDir, self.iptDataHistDir, self.pipeNameOptDict, self.memMgmtOptActorCriticsBackpropEvalDiskMode)

            Config.write_config(config, self.optConfigCurDir + self.__class__.__name__ + self.identifier + ".json")

    def create_graph(self):
        #connect
        self.graphTraverseList = [self.memMgmtOptActorCriticsBackpropEval, self.vis]
        self.nameTraverseIndexDict = OrderedDict()
        self.nameTraverseIndexDict[self.memMgmtOptActorCriticsBackpropEval.__class__.__name__ + "_" + self.memMgmtOptActorCriticsBackpropEval.identifier] = 0
        self.nameTraverseIndexDict[self.vis.__class__.__name__ + "_" + self.vis.identifier] = 1

        self.startNodeIndex = self.nameTraverseIndexDict[self.startNodeClassName + "_" + self.startNodeObjName]


    def make_cur_path(self):#, identifier ,rootDir):
        while True:
            curLocalTimeInMicro = datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S-%f')#time.time() * 10**7
            if not os.path.exists(self.projDir + 'hist/' + self.__class__.__name__ + self.identifier + "/" + str(curLocalTimeInMicro) + "/"):
                break;
        optConfigCurDir = self.projDir + "hist/" + self.__class__.__name__ + self.identifier + "/" + str(curLocalTimeInMicro) + "/config/" #+ '_' + str(tempId) + "/"
        optDataCurDir = self.projDir + "hist/" + self.__class__.__name__ + self.identifier + "/" + str(curLocalTimeInMicro) + "/data/"
        os.makedirs(optConfigCurDir)
        os.makedirs(optDataCurDir)
        return optConfigCurDir, optDataCurDir


    def run(self):
        for node in self.graphTraverseList[self.startNodeIndex:]:
            node.run()

        


if __name__ == "__main__":
    pipeline = PipelineActorCriticsBackpropVis()
    projDir = "../../"
    rootDir = "../"
    pipeline.create_nodes("", projDir, rootDir)
    pipeline.create_graph()
    pipeline.run()

