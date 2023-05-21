
class Node(object):
    def __init__(self):
#        self.childNodeList = None
        self.inputDict = None

    def read_config(self):
        pass

#    def add_chilren(self, childNodeList):
#        self.childNodeList = childNodeList

    def is_input_prepared(self):
        for inputtName, inputt in self.inputDict.items():
            if inputt is None:
                return False
        return True

    def run(self):
        pass
#        for childNode in self.childNodeList:
#            childNode.run()



