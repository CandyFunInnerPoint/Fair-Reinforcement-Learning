import subprocess
from subprocess import PIPE
from collections import OrderedDict
from datetime import datetime
import json, os, csv
from Config import Config

class ProjectPlot(object):
    def __init__(self):

        self.projDir = None

    def read_config(self, identifier, projDir):
        with open("../config/" + __class__.__name__ + ".json") as json_file:
            config = json.load(json_file)

            self.pipeProjectHistLocalTimeInMicro = config["pipeProjectHistLocalTimeInMicro"]

            self.projDir = projDir
            self.identifier = identifier 

            self.iptDataHistDir = projDir + '/hist/Project/' + self.pipeProjectHistLocalTimeInMicro + "/data/" if self.pipeProjectHistLocalTimeInMicro != None else None

            self.exprimentIdPath = self.iptDataHistDir + "Project.csv"

    def plot(self):
        self.curLocalTimeInMicroList = []
        
        with open(self.exprimentIdPath) as csvFile:
            csvReader = csv.reader(csvFile)
            for row in csvReader:
                self.hyperConfigIndexList.append(int(row[0]))
                self.curLocalTimeInMicroList.append(row[1])
        subprocess.Popen(["python", "PipelineRLFairVis.py"] + self.curLocalTimeInMicroList)


if __name__ == "__main__":
    project = ProjectPlot()
    projDir = "../../"
    identifier = ""
    project.read_config(identifier, projDir)
    project.plot()
