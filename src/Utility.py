class Utility(object):
    def __init__(self):
        pass

    @staticmethod
    def clipTwoSides(val, thres):
        if val > thres:
            return thres
        elif val < -thres:
            return -thres
        else:
            return val
