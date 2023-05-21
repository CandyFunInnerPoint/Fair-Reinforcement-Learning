import pandas as pd
import matplotlib.pyplot as plt
import json

import os
from Node import Node
from Path import Path
from Config import Config

from collections import OrderedDict

class VisRLFairTrain(Node):
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
        trainDataLogPd = self.pipeNameOptDict["trainDataLogPd"]
        optObjConstrPd = self.pipeNameOptDict["optObjConstrPd"]

        #1
        self.plot_metric1_metric2_pareto_method(trainDataLogPd, "evalTypeEnvTypeParetoType", "policyUpdateAvgReward", "estConstr", "methodName", "paretoType", False)
        self.plot_metric1_metric2_pareto_method(trainDataLogPd, "evalTypeEnvTypeParetoType", "policyUpdateAvgReward", "estConstr", "methodName", "paretoType", True)
        self.plot_metric1_metric2_pareto_method(optObjConstrPd, "evalTypeEnvTypeParetoType", "optObjVal", "optConstrVal", "methodName", "paretoType", False)
        self.plot_metric1_metric2_pareto_method(optObjConstrPd, "evalTypeEnvTypeParetoType", "optObjVal", "optConstrVal", "methodName", "paretoType", True)

        #2
        self.plot_metric_seqIndex_method(trainDataLogPd, "evalTypeEnvTypeParetoType", "trainEpisodeIndex", ["policyUpdateAvgReward", "estConstr"], "methodName", "paretoType", [False, False])
        self.plot_metric_seqIndex_method(trainDataLogPd, "evalTypeEnvTypeParetoType", "trainEpisodeIndex", ["policyUpdateAvgReward", "estConstr"], "methodName", "paretoType", [True, True])

        self.plot_metric_seqIndex_method(optObjConstrPd, "evalTypeEnvTypeParetoType", "trainEpisodeIndex", ["optObjVal", "optConstrVal", "optGap"], "methodName", "paretoType", [False, False, False])
        self.plot_metric_seqIndex_method(optObjConstrPd, "evalTypeEnvTypeParetoType", "trainEpisodeIndex", ["optObjVal", "optConstrVal", "optGap"], "methodName", "paretoType", [True, True, True])


    def plot_metric1_metric2_pareto_method(self, trainDataLogPd, plotIdVarName, xAxisName, yAxisName, legendNameCol, paretoType, useLogScale):
        pd.set_option("max_columns", None)
        pd.set_option("max_rows", None)
        stdMul = 0.2

        dirr = self.optDataCurDir + "plotInCrossEpiByGroup" + "/"
        if not os.path.exists(dirr):
            os.makedirs(dirr)
        

        for plotIdVarNameIndex in trainDataLogPd.groupby(plotIdVarName).count().index:
            trainDataLogPdPerPlot = trainDataLogPd[trainDataLogPd[plotIdVarName] == plotIdVarNameIndex]
            trainDataLogPdPerPlot = trainDataLogPdPerPlot[~ (trainDataLogPdPerPlot["methodName"].isin(["DpOrigGp", "DpCvOrigGp", "EqOptOrigGp", "EqOptCvOrigGp"]) & (trainDataLogPdPerPlot["methodMix"] > 0.1))]
            trainDataLogPdPerPlot.drop(columns = [plotIdVarName], inplace = True)
            trainDataLogPdPerPlotCurveGroups = trainDataLogPdPerPlot.groupby([legendNameCol])

            curveGroupXAxisGroupsMeanDict = OrderedDict()
            curveGroupXAxisGroupsStdDict = OrderedDict()
            for curveGroupName in trainDataLogPdPerPlotCurveGroups.groups.keys():
                curveGroup = trainDataLogPdPerPlotCurveGroups.get_group(curveGroupName)
                curveGroup.reset_index(drop = True, inplace = True)
                if curveGroup[paretoType].iloc[0] == "Const":
                    curveGroupXAxisGroups = curveGroup.groupby(["methodConst"])
                elif curveGroup[paretoType].iloc[0] == "SampleSize":
                    curveGroupXAxisGroups = curveGroup.groupby(["methodSampleSize"])
                elif curveGroup[paretoType].iloc[0] == "Mix":
                    curveGroupXAxisGroups = curveGroup.groupby(["methodMix"])
                print(curveGroup[paretoType].iloc[0])
                curveGroupXAxisGroupsMean = curveGroupXAxisGroups.mean()
                curveGroupXAxisGroupsMean.reset_index(inplace = True)
                if curveGroup[paretoType].iloc[0] == "Const":
                    curveGroupXAxisGroupsMean.sort_values(by = ["methodConst"], inplace = True)
                elif curveGroup[paretoType].iloc[0] == "SampleSize":
                    curveGroupXAxisGroupsMean.sort_values(by = ["methodSampleSize"], inplace = True)
                elif curveGroup[paretoType].iloc[0] == "Mix":
                    curveGroupXAxisGroupsMean.sort_values(by = ["methodMix"], inplace = True)

                curveGroupXAxisGroupsStd = curveGroupXAxisGroups.std()
                curveGroupXAxisGroupsStd.reset_index(inplace = True)
                if curveGroup[paretoType].iloc[0] == "Const":
                    curveGroupXAxisGroupsStd.sort_values(by = ["methodConst"], inplace = True)
                elif curveGroup[paretoType].iloc[0] == "SampleSize":
                    curveGroupXAxisGroupsStd.sort_values(by = ["methodSampleSize"], inplace = True)
                elif curveGroup[paretoType].iloc[0] == "Mix":
                    curveGroupXAxisGroupsStd.sort_values(by = ["methodMix"], inplace = True)

                curveGroupXAxisGroupsMeanDict[curveGroupName] = curveGroupXAxisGroupsMean
                curveGroupXAxisGroupsStdDict[curveGroupName] = curveGroupXAxisGroupsStd

            if useLogScale:
                fileName = dirr + str(plotIdVarNameIndex) + "_log_" + yAxisName + "_" + xAxisName + ".png"
            else:
                fileName = dirr + str(plotIdVarNameIndex) + "_linear_" + yAxisName + "_" + xAxisName + ".png"
            print(fileName)
            plt.figure()
            
            for legendName in curveGroupXAxisGroupsMeanDict.keys():
                xAxis = curveGroupXAxisGroupsMeanDict[legendName][xAxisName]
                yAxisMean = curveGroupXAxisGroupsMeanDict[legendName][yAxisName]
                yAxisStd = curveGroupXAxisGroupsStdDict[legendName][yAxisName]

                plt.plot(xAxis, yAxisMean, 'o', label = str(legendName))

            if useLogScale:
                plt.yscale("log")
            plt.xlabel(xAxisName)
            plt.ylabel(yAxisName)
            plt.title(str(plotIdVarNameIndex))
            plt.legend()
            plt.savefig(fileName)

    def my_func(self, df, methodName, methodMix, curveName):
        if df[methodName] in ["DpOrigGp", "DpCvOrigGp", "EqOptOrigGp", "EqOptCvOrigGp", "AggOrigGp"]: 
            df[curveName] = df[methodName]
        else:
            df[curveName] = df[methodName] + str(df[methodMix])#.astype(str)
        return df[curveName]

    def plot_metric_seqIndex_method(self, trainDataLogPd, plotIdVarName, xAxisName, yAxisNameList, methodName, paretoType, useLogScaleList):
        pd.set_option("max_columns", None)
        pd.set_option("max_rows", None)
       
        dirr = self.optDataCurDir + "plotInCrossEpiByGroup" + "/"
        if not os.path.exists(dirr):
            os.makedirs(dirr)
        stdMul = 0.2

        for plotIdVarNameIndex in trainDataLogPd.groupby(plotIdVarName).count().index:
            trainDataLogPdPerPlot = trainDataLogPd[trainDataLogPd[plotIdVarName] == plotIdVarNameIndex]
            trainDataLogPdPerPlot = trainDataLogPdPerPlot[~ (trainDataLogPdPerPlot["methodName"].isin(["DpSgtGp", "DpCvSgtGp", "EqOptSgtGp", "EqOptCvSgtGp"]) & (trainDataLogPdPerPlot["methodMix"] != 0))]
            trainDataLogPdPerPlot = trainDataLogPdPerPlot[~ (trainDataLogPdPerPlot["methodName"].isin(["Hg"]) & ((trainDataLogPdPerPlot["methodMix"] > 0.10001) | (trainDataLogPdPerPlot["methodMix"] < 0.09999)))]
            trainDataLogPdPerPlot.drop(columns = [plotIdVarName], inplace = True)

            if trainDataLogPdPerPlot[paretoType].iloc[0] == "Const":
                trainDataLogPdPerPlot["curveName"] = trainDataLogPdPerPlot[methodName] + trainDataLogPdPerPlot["methodConst"].astype(str)
            elif trainDataLogPdPerPlot[paretoType].iloc[0] == "SampleSize":
                trainDataLogPdPerPlot["curveName"] = trainDataLogPdPerPlot[methodName] + trainDataLogPdPerPlot["methodSampleSize"].astype(str)
            elif trainDataLogPdPerPlot[paretoType].iloc[0] == "Mix":
                trainDataLogPdPerPlot["curveName"] = trainDataLogPdPerPlot.apply(lambda df: self.my_func(df, methodName, "methodMix", "curveName"), axis = 1)

            trainDataLogPdPerPlotCurveGroups = trainDataLogPdPerPlot.groupby(["curveName"])

            curveGroupXAxisGroupsMeanDict = OrderedDict()
            curveGroupXAxisGroupsStdDict = OrderedDict()

            for curveGroupName in trainDataLogPdPerPlotCurveGroups.groups.keys():
                curveGroup = trainDataLogPdPerPlotCurveGroups.get_group(curveGroupName)
                curveGroup.reset_index(drop = True, inplace = True)
                curveGroupXAxisGroups = curveGroup.groupby([xAxisName])

                curveGroupXAxisGroupsMean = curveGroupXAxisGroups.mean()
                curveGroupXAxisGroupsMean.reset_index(inplace = True)
                curveGroupXAxisGroupsMean.sort_values(by = [xAxisName], inplace = True)
                print(curveGroupXAxisGroupsMean.head())
                curveGroupXAxisGroupsStd = curveGroupXAxisGroups.std()
                curveGroupXAxisGroupsStd.reset_index(inplace = True)
                curveGroupXAxisGroupsStd.sort_values(by = [xAxisName], inplace = True)

                curveGroupXAxisGroupsMeanDict[curveGroupName] = curveGroupXAxisGroupsMean
                curveGroupXAxisGroupsStdDict[curveGroupName] = curveGroupXAxisGroupsStd

        

            for index in range(len(yAxisNameList)):
                yAxisName = yAxisNameList[index]
                if useLogScaleList[index]:
                    fileName = dirr + str(plotIdVarNameIndex) + "_log_" + yAxisName + "_" + xAxisName + ".png"
                else:
                    fileName = dirr + str(plotIdVarNameIndex) + "_linear_" + yAxisName + "_" + xAxisName + ".png"
                print(fileName)
                plt.figure()
            
                for legendName in curveGroupXAxisGroupsMeanDict.keys():
                    xAxis = curveGroupXAxisGroupsMeanDict[legendName][xAxisName]
                    yAxisMean = curveGroupXAxisGroupsMeanDict[legendName][yAxisName]
                    yAxisStd = curveGroupXAxisGroupsStdDict[legendName][yAxisName]
                    yAxisLb = yAxisMean - stdMul * yAxisStd
                    yAxisUb = yAxisMean + stdMul * yAxisStd

                    plt.plot(xAxis, yAxisMean, 'o', label = str(legendName))
                    plt.fill_between(xAxis, yAxisLb, yAxisUb, color = "blue", alpha = 0.1)

                if useLogScaleList[index]:
                    plt.yscale("log")
                plt.xlabel(xAxisName)
                plt.ylabel(yAxisName)
                plt.title(str(plotIdVarNameIndex))
                plt.legend()
                plt.savefig(fileName)

    def plot_metric_pareto_method(self, trainDataLogPd, plotIdVarName, yAxisNameList, methodName, paretoType, useLogScaleList):
        pd.set_option("max_columns", None)
        pd.set_option("max_rows", None)
       
        dirr = self.optDataCurDir + "plotInCrossEpiByGroup" + "/"
        if not os.path.exists(dirr):
            os.makedirs(dirr)
        stdMul = 0.2

        for plotIdVarNameIndex in trainDataLogPd.groupby(plotIdVarName).count().index:
            trainDataLogPdPerPlot = trainDataLogPd[trainDataLogPd[plotIdVarName] == plotIdVarNameIndex]

            if trainDataLogPdPerPlot[paretoType].iloc[0] == "Const":
                continue
            trainDataLogPdPerPlot.drop(columns = [plotIdVarName], inplace = True)

            trainDataLogPdPerPlotCurveGroups = trainDataLogPdPerPlot.groupby(methodName)

            curveGroupXAxisGroupsMeanDict = OrderedDict()
            curveGroupXAxisGroupsStdDict = OrderedDict()
            for curveGroupName in trainDataLogPdPerPlotCurveGroups.groups.keys():
                curveGroup = trainDataLogPdPerPlotCurveGroups.get_group(curveGroupName)
                curveGroup.reset_index(drop = True, inplace = True)

                xAxisName = "methodSampleSize"

                curveGroupXAxisGroups = curveGroup.groupby([xAxisName])
                curveGroupXAxisGroupsMean = curveGroupXAxisGroups.mean()
                curveGroupXAxisGroupsMean.reset_index(inplace = True)
                curveGroupXAxisGroupsMean.sort_values(by = [xAxisName], inplace = True)
                print(curveGroupXAxisGroupsMean.head())
                curveGroupXAxisGroupsStd = curveGroupXAxisGroups.std()
                curveGroupXAxisGroupsStd.reset_index(inplace = True)
                curveGroupXAxisGroupsStd.sort_values(by = [xAxisName], inplace = True)

                curveGroupXAxisGroupsMeanDict[curveGroupName] = curveGroupXAxisGroupsMean
                curveGroupXAxisGroupsStdDict[curveGroupName] = curveGroupXAxisGroupsStd

        

            for index in range(len(yAxisNameList)):
                yAxisName = yAxisNameList[index]
                if useLogScaleList[index]:
                    fileName = dirr + str(plotIdVarNameIndex) + "_log_" + yAxisName + "_" + xAxisName + ".png"
                else:
                    fileName = dirr + str(plotIdVarNameIndex) + "_linear_" + yAxisName + "_" + xAxisName + ".png"
                print(fileName)
                plt.figure()
            
                for legendName in curveGroupXAxisGroupsMeanDict.keys():
                    xAxis = curveGroupXAxisGroupsMeanDict[legendName][xAxisName]
                    yAxisMean = curveGroupXAxisGroupsMeanDict[legendName][yAxisName]
                    yAxisStd = curveGroupXAxisGroupsStdDict[legendName][yAxisName]
                    yAxisLb = yAxisMean - stdMul * yAxisStd
                    yAxisUb = yAxisMean + stdMul * yAxisStd

                    plt.plot(xAxis, yAxisMean, 'o', label = str(legendName))
                    plt.fill_between(xAxis, yAxisLb, yAxisUb, color = "blue", alpha = 0.1)

                if useLogScaleList[index]:
                    plt.yscale("log")
                plt.xlabel(xAxisName)
                plt.ylabel(yAxisName)
                if useLogScaleList[index]:
                    plt.title(str(plotIdVarNameIndex) + "_log_" + yAxisName + "_" + xAxisName)
                else:
                    plt.title(str(plotIdVarNameIndex) + "_linear_" + yAxisName + "_" + xAxisName)
                plt.legend()
                plt.savefig(fileName)



if __name__ == "__main__":
    vis = VisRLFair()
    rootDir = "../"
    config = vis.read_config(rootDir)
    vis.run()

