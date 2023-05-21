import pandas as pd
import matplotlib.pyplot as plt
import json

import os
from Node import Node
from Path import Path
from Config import Config

from collections import OrderedDict

class VisActorCriticsBackprop(Node):
    def __init__(self):
        metricProcNoEnv = None

    def read_config(self, identifier, rootDir, optConfigCurDir, optDataCurDir, iptConfigHistDir, iptDataHistDir, device, pipeNameOptDict):
        iptConfigFileName, self.optConfigCurDir, self.iptConfigHistDir, self.optDataCurDir = Path.make_path(rootDir, self.__class__.__name__, identifier, optConfigCurDir, iptConfigHistDir, optDataCurDir)
        self.iptDataHistDir = iptDataHistDir


        with open(iptConfigFileName) as json_file:
            config = json.load(json_file)
 
            self.identifier = identifier

            self.pipeNameOptDict = pipeNameOptDict

    def run(self):
        metricPd = self.pipeNameOptDict["metricPd"]

        pd.set_option("max_columns", None)
        pd.set_option("max_rows", None)


        xAxisName = "algoTrainEpiIndexNnTrainStepIndex"
        
        dirr = self.optDataCurDir + "plotInCrossEpiByGroup" + "/"
        if not os.path.exists(dirr):
            os.makedirs(dirr)
        
        yAxisNameList = ["estQValEvalEnv", "estLoss", "cumEpiRewardEvalEnv"]
        stdMul = 0.5

        legendNameCol = "evalNnTrainStepIndex"

        for evalEpisodeIndex in metricPd.groupby("evalEpisodeIndex").count().index:
            metricPdPerPlot = metricPd[metricPd["evalEpisodeIndex"] == evalEpisodeIndex]
            metricPdPerPlot[xAxisName] = metricPdPerPlot["algoTrainEpisodeIndex"].astype(str) + "_" + metricPdPerPlot["algoTrainSubEpisodeIndex"].astype(str) + "_" + metricPdPerPlot["algoTrainNnTrainStepIndex"].astype(str)
            metricPdPerPlot[xAxisName] = metricPdPerPlot[xAxisName].astype(int)
            metricPdPerPlot.drop(columns = ["evalEpisodeIndex", "evalSubEpisodeIndex", "algoTrainEpisodeIndex", "algoTrainSubEpisodeIndex", "algoTrainNnTrainStepIndex"], inplace = True)
            seasonGrouped = metricPdPerPlot.groupby(["evalNnTrainStepIndex"])

            algoGroupedGivenSeasonMeanDict = OrderedDict()
            algoGroupedGivenSeasonStdDict = OrderedDict()

            for groupName in seasonGrouped.groups.keys():
                seasonGroup = seasonGrouped.get_group(groupName)
                seasonGroup.reset_index(drop = True, inplace = True)
                algoGroupedGivenSeason = seasonGroup.groupby([xAxisName])

                algoGroupedGivenSeasonMean = algoGroupedGivenSeason.mean()
                algoGroupedGivenSeasonMean.reset_index(inplace = True)
                algoGroupedGivenSeasonMean.sort_values(by = [xAxisName], inplace = True)
                algoGroupedGivenSeasonMean.drop(columns = ["evalNnTrainStepIndex"], inplace = True)
                algoGroupedGivenSeasonStd = algoGroupedGivenSeason.std()
                algoGroupedGivenSeasonStd.reset_index(inplace = True)
                algoGroupedGivenSeasonStd.sort_values(by = [xAxisName], inplace = True)
                algoGroupedGivenSeasonStd.drop(columns = ["evalNnTrainStepIndex"], inplace = True)

                algoGroupedGivenSeasonMeanDict[groupName] = algoGroupedGivenSeasonMean
                algoGroupedGivenSeasonStdDict[groupName] = algoGroupedGivenSeasonStd

        

            for yAxisName in yAxisNameList:
                fileName = dirr + str(evalEpisodeIndex) + "_" + yAxisName + "_" + xAxisName + "_" + legendNameCol + ".png"
                print(fileName)
                plt.figure()
            
                for legendName in algoGroupedGivenSeasonMeanDict.keys():
                    xAxis = algoGroupedGivenSeasonMeanDict[legendName][xAxisName]
                    yAxisMean = algoGroupedGivenSeasonMeanDict[legendName][yAxisName]
                    yAxisStd = algoGroupedGivenSeasonStdDict[legendName][yAxisName]
                    yAxisLb = yAxisMean - stdMul * yAxisStd
                    yAxisUb = yAxisMean + stdMul * yAxisStd

                    plt.plot(xAxis, yAxisMean, label = legendNameCol + str(legendName))
                    plt.fill_between(xAxis, yAxisLb, yAxisUb, color = "blue", alpha = 0.1)

                plt.xlabel(xAxisName)
                plt.ylabel(yAxisName)
                plt.title(str(evalEpisodeIndex) + "_" + yAxisName + "_" + xAxisName)
                plt.legend()
                plt.savefig(fileName)

if __name__ == "__main__":
    vis = Vis()
    rootDir = "../"
    config = vis.read_config(rootDir)
    vis.run()

