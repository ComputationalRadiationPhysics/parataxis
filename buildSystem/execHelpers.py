import subprocess

def execCmds(cmds):
    """Executes the given commands one by one and returns a tuple consisting of the return code and the output"""
    output = ""
    for cmd in cmds:
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        while True:
            outputLine = proc.stdout.readline()
            if outputLine:
                outputLine = outputLine.decode()
                print(outputLine, end='')
                output += outputLine
            else:
                break
        retCode = proc.poll()
        if(retCode != 0):
            print("Executing `" + cmd + "` failed with code " + str(retCode))
            return (retCode, output)
    return (0, output)

def execCmd(cmd):
    return execCmds([cmd])
