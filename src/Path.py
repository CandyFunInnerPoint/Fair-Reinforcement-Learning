import os

class Path(object):
    def __init__(self):
        pass

    @staticmethod
    def make_path(rootDir, className, identifier, optConfigCurDir, iptConfigHistDir, optDataCurDir):
        if iptConfigHistDir == None:
            iptConfigFileName = rootDir + "config/" + className + identifier + ".json" #identifier + ".json"
        else:
            iptConfigFileName = iptConfigHistDir + className + "/" + className + identifier + ".json" #identifier + ".json"
            iptConfigHistDir = iptConfigHistDir + className + "/" #identifier + "/"

##        print(optConfigCurPath)
        optConfigCurDir = optConfigCurDir + className + "/" #identifier + "/"
        if not os.path.exists(optConfigCurDir):
            os.makedirs(optConfigCurDir)

        optDataCurDir = optDataCurDir + className + "/"
        if not os.path.exists(optDataCurDir):
            os.makedirs(optDataCurDir)

##        print(optConfigCurPath)
#        optDataCurPath = optDataCurPath + self.__class__.__name__ + identifier + "/" 


#      iptDataHistPath = iptDataHistPath + self.__class__.__name__ + identifier if iptDataHistPath != None else iptDataHistPath
        return iptConfigFileName, optConfigCurDir, iptConfigHistDir, optDataCurDir


