import csv

class DataLogger(object):
    def __init__(self, capacity, filePath, header):
        self.buffer = []
        self.capacity = capacity
        self.filePath = filePath
        with open(self.filePath, "w") as csvFile:
            csvWriter = csv.writer(csvFile)
            csvWriter.writerow(header)

    def write_log_to_file(self, isFinish, loggerList):
        if not isFinish:
            self.buffer.append(loggerList)
        if len(self.buffer) == self.capacity or isFinish:
            with open(self.filePath, "a") as csvFile:
                csvWriter = csv.writer(csvFile)
                for item in self.buffer:
                    csvWriter.writerow(item)
                self.buffer = []
