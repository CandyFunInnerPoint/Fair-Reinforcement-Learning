import pandas as pd
import matplotlib.pyplot as plt
import json

import os
from Node import Node
from Path import Path
from Config import Config

from collections import OrderedDict

class VisRLFairTrainSingle(Node):
    def __init__(self):
        metricProcNoEnv = None

    def read_config(self, identifier, rootDir, optConfigCurDir, optDataCurDir, iptConfigHistDir, iptDataHistDir, device, pipeNameOptDict):
        iptConfigFileName, self.optConfigCurDir, self.iptConfigHistDir, self.optDataCurDir = Path.make_path(rootDir, self.__class__.__name__, identifier, optConfigCurDir, iptConfigHistDir, optDataCurDir)
        self.iptDataHistDir = iptDataHistDir


        with open(iptConfigFileName) as json_file:
            config = json.load(json_file)
 
            self.identifier = identifier

            self.pipeNameOptDict = pipeNameOptDict


    def my_func2(self, df, methodName):
        if df[methodName] == "Myp": 
            df[methodName] = "Hg"
        elif df[methodName] == "DpOrigGp": 
            df[methodName] = "methodDp"
        elif df[methodName] == "EqOptOrigGp": 
            df[methodName] = "methodEqOpt"
        elif df[methodName] == "DpLagGp": 
            df[methodName] = "SeqUnfairDp"
        elif df[methodName] == "EqOptLagGp": 
            df[methodName] = "SeqUnfairEqOpt"
        elif df[methodName] == "Myp": 
            df[methodName] = "Hg"
        return df[methodName]

    def run(self):
        trainDataLogPd = self.pipeNameOptDict["trainDataLogPd"]

        trainDataLogPd["methodName"] = trainDataLogPd.apply(lambda trainDataLogPd: self.my_func2(trainDataLogPd, "methodName"), axis = 1)

        #1
        self.plot_metric1_metric2_pareto_method(trainDataLogPd, "evalTypeEnvTypeParetoType", "policyUpdateAvgReward", "estConstr", "methodName", "paretoType", False)
        self.plot_metric1_metric2_pareto_method(trainDataLogPd, "evalTypeEnvTypeParetoType", "policyUpdateAvgReward", "estConstr", "methodName", "paretoType", True)

        #2
        self.plot_metric_seqIndex_method(trainDataLogPd, "evalTypeEnvTypeParetoType", "trainEpisodeIndex", ["policyUpdateAvgReward", "estConstr"], "methodName", "paretoType", [False, False])
        self.plot_metric_seqIndex_method(trainDataLogPd, "evalTypeEnvTypeParetoType", "trainEpisodeIndex", ["policyUpdateAvgReward", "estConstr"], "methodName", "paretoType", [True, True])


    def plot_metric1_metric2_pareto_method(self, trainDataLogPd, plotIdVarName, xAxisName, yAxisName, legendNameCol, paretoType, useLogScale):
        pd.set_option("max_columns", None)
        pd.set_option("max_rows", None)
        stdMul = 0.2

        dirr = self.optDataCurDir + "plotInCrossEpiByGroup" + "/"
        if not os.path.exists(dirr):
            os.makedirs(dirr)
        

        for plotIdVarNameIndex in trainDataLogPd.groupby(plotIdVarName).count().index:
            trainDataLogPdPerPlot = trainDataLogPd[trainDataLogPd[plotIdVarName] == plotIdVarNameIndex]
            trainDataLogPdPerPlot = trainDataLogPdPerPlot[~ (trainDataLogPdPerPlot["methodName"].isin(["methodDp", "DpCvOrigGp", "methodEqOpt", "EqOptCvOrigGp"]) & (trainDataLogPdPerPlot["methodMix"] > 0.1))]
            trainDataLogPdPerPlot = trainDataLogPdPerPlot[~ trainDataLogPdPerPlot["methodName"].isin(["Hg"])]
            trainDataLogPdPerPlot = trainDataLogPdPerPlot[~ (trainDataLogPdPerPlot["methodName"].isin(["SeqUnfairDp"]) & (trainDataLogPdPerPlot["trainEpisodeIndex"] < 660279))] 
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
                fileName = dirr + str(plotIdVarNameIndex) + "_log_" + yAxisName + "_" + xAxisName + ".pdf"
            else:
                fileName = dirr + str(plotIdVarNameIndex) + "_linear_" + yAxisName + "_" + xAxisName + ".pdf"
            print(fileName)
            plt.figure()
            
            for legendName in curveGroupXAxisGroupsMeanDict.keys():
                xAxis = curveGroupXAxisGroupsMeanDict[legendName][xAxisName]
                yAxisMean = curveGroupXAxisGroupsMeanDict[legendName][yAxisName]
                yAxisStd = curveGroupXAxisGroupsStdDict[legendName][yAxisName]
                text = curveGroupXAxisGroupsMeanDict[legendName]["methodMix"]
                if legendName in ["SeqUnfairDp", "SeqUnfairEqOpt"]:
                    legendName1 = "SeqUnfair"
                    legendName2 = "PenalizedSeq"
                    plt.plot(xAxis[:1], yAxisMean[:1], 'o', label = str(legendName1))
                    plt.plot(xAxis[1:], yAxisMean[1:], 'o', label = str(legendName2))
                    self.add_value_label(xAxis, yAxisMean, text)
                else:
                    plt.plot(xAxis, yAxisMean, 'o', label = str(legendName))
                    plt.plot(xAxis[:1], yAxisMean[:1], 'x', label = str(legendName) + "Final")

            if useLogScale:
                plt.yscale("log")
            plt.xlabel(xAxisName,fontsize = 16)
            plt.ylabel(yAxisName,fontsize = 16)

            plt.rcParams['pdf.fonttype'] = 42
            plt.legend(prop={'size': 16})
            plt.savefig(fileName, format = "pdf")

    def add_value_label(self, x_list,y_list, z_list):

        #for dp sim
        for i in range(1, len(x_list)+1):
            if z_list[i-1] == 1000:
                y_list[i-1] += 0.04#0.01
                x_list[i-1] -= 0#0.01
            elif z_list[i-1] == 10:
                x_list[i-1] += 0.01#0.01
            plt.text(x_list[i-1],y_list[i-1],int(z_list[i-1]),fontsize = 16)

    def my_func(self, df, methodName, methodMix, curveName):
        if df[methodName] in ["methodDp", "DpCvOrigGp", "methodEqOpt", "EqOptCvOrigGp", "AggOrigGp"]: 
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
            trainDataLogPdPerPlot = trainDataLogPdPerPlot[~ trainDataLogPdPerPlot["methodName"].isin(["Hg"])]
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
                    fileName = dirr + str(plotIdVarNameIndex) + "_log_" + yAxisName + "_" + xAxisName + ".pdf"
                else:
                    fileName = dirr + str(plotIdVarNameIndex) + "_linear_" + yAxisName + "_" + xAxisName + ".pdf"
                print(fileName)
                plt.figure()
            
                for legendName in curveGroupXAxisGroupsMeanDict.keys():
                    xAxis = curveGroupXAxisGroupsMeanDict[legendName][xAxisName]
                    yAxisMean = curveGroupXAxisGroupsMeanDict[legendName][yAxisName]
                    yAxisStd = curveGroupXAxisGroupsStdDict[legendName][yAxisName]
                    yAxisLb = yAxisMean - stdMul * yAxisStd
                    yAxisUb = yAxisMean + stdMul * yAxisStd

                    if legendName in ["SeqUnfairDp0.0", "SeqUnfairEqOpt0.0"]:
                        legendName = "SeqUnfair"
                    elif legendName in ["SeqUnfairDp1.0", "SeqUnfairEqOpt1.0"]:
                        legendName = "PenalizedSeq$\lambda$=1"
                    elif legendName in ["SeqUnfairDp10.0", "SeqUnfairEqOpt10.0"]:
                        legendName = "PenalizedSeq$\lambda$=10"
                    elif legendName in ["SeqUnfairDp1000.0", "SeqUnfairEqOpt1000.0"]:
                        legendName = "PenalizedSeq$\lambda$=$10^3$"

                    plt.plot(xAxis[1:], yAxisMean[1:], 'o', label = str(legendName))
                    plt.fill_between(xAxis[1:], yAxisLb[1:], yAxisUb[1:], color = "blue", alpha = 0.1)


                if useLogScaleList[index]:
                    plt.yscale("log")
                plt.xlabel(xAxisName,fontsize = 16)
                plt.ylabel(yAxisName,fontsize = 16)

                plt.rcParams['pdf.fonttype'] = 42
                plt.legend(prop={'size': 12})
                plt.savefig(fileName, format = "pdf")
  
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
                    fileName = dirr + str(plotIdVarNameIndex) + "_log_" + yAxisName + "_" + xAxisName + ".pdf"
                else:
                    fileName = dirr + str(plotIdVarNameIndex) + "_linear_" + yAxisName + "_" + xAxisName + ".pdf"
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

