import json
import subprocess

class HyperProject(object):
    def __init__(self):
        self.hyperProjDir = None
        self.identifier = None

    def read_config(self, identifier, hyperProjDir):
        with open("../config/" + __class__.__name__ + ".json") as json_file:
            config = json.load(json_file)
            self.paretoMode = config["paretoMode"]

            self.hyperProjDir = hyperProjDir
            self.identifier = identifier 

    def run(self):
        if self.paretoMode == "paretoConst":
            subprocess.call(["python", "ProjectParetoConst.py"])
        elif self.paretoMode == "paretoSampleSize":
            subprocess.call(["python", "ProjectParetoSampleSize.py"])
        elif self.paretoMode == "paretoMix":
            subprocess.call(["python", "ProjectParetoMix.py"])

if __name__ == "__main__":
    hyperProject = HyperProject()
    hyperProjDir = "../../"
    identifer = ""
    hyperProject.read_config(identifer, hyperProjDir)
    hyperProject.run()
