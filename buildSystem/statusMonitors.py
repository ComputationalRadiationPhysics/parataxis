from execHelpers import execCmd

class BaseMonitor:
    isWaiting = False
    isFinished = False
    
    def update(self):
        pass

class BashMonitor(BaseMonitor):
    def __init__(self, stdout, stderr):
        self.isFinished = True
  
        
class QSubMonitor(BaseMonitor):
    def __init__(self, stdout, stderr):
        if(len(stdout) != 1):
            raise Exception("Unexpected output from qsub command: " + stdout)
        self.jobId = stdout[0]
        self.update()
    
    def update(self):
        if(self.jobId != None):
            res = execCmd("qstat " + self.jobId, True)
            if(res.result == 0):
                if(len(res.stdout) < 2):
                    raise Exception("Unexpected output from qstat: " + res.stdout)
                jobStatus = res.stdout[-1].split()
                if(len(jobStatus) != 6):
                    raise Exception("Unexpected output from qstat: " + res.stdout)
                jobState = jobStatus[4]
                if(jobState == "C" or jobState == "E"):
                    self.isWaiting = False
                    self.isFinished = True
                elif(jobState == "R" or jobState == "T"):
                    self.isWaiting = False
                    self.isFinished = False
                elif(jobState == "W" or jobState == "Q"):
                    self.isWaiting = True
                    self.isFinished = False
                else:
                    raise Exception("Invalid job state: " + jobState)
            elif("Unknown Job Id" in res.stderr):
                self.isWaiting = False
                self.isFinished = True
            else:
                raise Exception("Failed to execute qstat: " + res.stdout + "\n" + res.stderr)
                
                
            
        
def GetMonitor(submitCmd, stdout, stderr):
    if(submitCmd == "bash"):
        return BashMonitor(stdout, stderr)
    elif(submitCmd == "qsub"):
        return QSubMonitor(stdout, stderr)
    else:
        raise Exception("Unsupported submit command: `" + submitCmd + "`")

