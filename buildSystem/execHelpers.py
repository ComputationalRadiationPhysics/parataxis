import subprocess
import collections

def execCmds(cmds, silent = False):
    """Executes the given commands one by one and returns a tuple consisting of the return code, stdout and stderr"""
    ReturnType = collections.namedtuple('ExecReturn', ['result', 'stdout', 'stderr'])
    output = ""
    error = ""
    for cmd in cmds:
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        while True:
            outputLine = proc.stdout.readline()
            errorLine = proc.stderr.readline()
            if outputLine:
                outputLine = outputLine.decode()
                if(not silent):
                    print(outputLine, end='')
                output += outputLine
                
            if errorLine:
                errorLine = errorLine.decode()
                if(not silent):
                    print(errorLine, end='')
                error += errorLine
            if(not outputLine) and (not errorLine):
                break
        retCode = proc.poll()
        if(retCode != 0):
            if(not silent):
                print("Executing `" + cmd + "` failed with code " + str(retCode))
            return ReturnType(retCode, output, error)
    return ReturnType(0, output, error)

def execCmd(cmd, silent = False):
    return execCmds([cmd], silent)
