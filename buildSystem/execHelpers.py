import subprocess
import collections

def execCmds(cmds):
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
                print(outputLine, end='')
                output += outputLine
                
                errorLine = errorLine.decode()
                print(errorLine, end='')
                error += errorLine
            else:
                break
        retCode = proc.poll()
        if(retCode != 0):
            print("Executing `" + cmd + "` failed with code " + str(retCode))
            return ReturnType(retCode, output, error)
    return ReturnType(0, output, error)

def execCmd(cmd):
    return execCmds([cmd])
