class BashMonitor:
    def __init__(self, stdout, stderr):
        pass
    
    def isFinished(self):
        return True
        
class QSubMonitor:
    def __init__(self, stdout, stderr):
        self.jobId = stdout
        print("stdout: " + stdout)
    
    def isFinished(self):
        return True
        
def GetMonitor(submitCmd, stdout, stderr):
    if(submitCmd == "bash"):
        return BashMonitor(stdout, stderr)
    elif(submitCmd == "qsub"):
        return QSubMonitor(stdout, stderr)
    else:
        raise Exception("Unsupported submit command: `" + submitCmd + "`")

