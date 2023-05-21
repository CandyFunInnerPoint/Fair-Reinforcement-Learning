import json, os
#from time import strftime, gmtime

class Config(object):
    def __init__(self):
        pass

    def read_config(self):#, rootPath, parentPath):
        pass

###    def make_path_generate(self, identifier, parentPath):
###        curLocalTime = strftime("%Y%m%d_%H%M%S", gmtime())
###        tempId = 0
###        path = parentPath + self.__class__.__name__ + "_" + identifier + '/' + curLocalTime + '_' + str(tempId) + "/"
###
###        while os.path.exists(path):
###            tempId += 1
###        os.makedirs(path)
###        return path

#    def make_path_generate(self, parentPath):
#        curLocalTime = strftime("%Y%m%d_%H%M%S", gmtime())
#        tempId = 0
#        path = parentPath + self.__class__.__name__ + '/' + curLocalTime + '_' + str(tempId) + "/"
#        while os.path.exists(path):
#            tempId += 1
#        os.makedirs(path)
#        return path

###    def make_path_readFromDisk(self, identifier, parentPath, localPath):
###        path = parentPath + self.__class__.__name__ + "/" + localPath + "/"
###        return path

#    def write_config(self, config):
#        path = self.rootPath + "/hist/" + self.pipeCurType + "/" + self.pipeCurId + "/config/" + __class__.__name__ + "_" + self.identifier + ".json"
#        with open(path, "w") as jsonFile:
#            jsonFile.write(json.dumps(config, indent = 4))

    @staticmethod
    def write_config(config, configPath):
#        print(configPath)
#        path = self.optConfigCurPath + __class__.__name__ + self.identifier + ".json"
#        os.makedirs(configPath)
#        if not os.path.exists(configPath):
#            os.makedirs(configPath)
        with open(configPath, "w") as jsonFile:
            jsonFile.write(json.dumps(config, indent = 4))
