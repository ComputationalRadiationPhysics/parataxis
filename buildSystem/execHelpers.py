import subprocess
import collections
from contextlib import contextmanager
import os

def readAndOutputLine(fileHandle, output, silent):
    """Read and print a line from the file
    
    If anything is read, append to output (list) and print it if silent is not false
    Returns whether EOF was reached
    """
    outputLine = fileHandle.readline()
    if outputLine:
        outputLine = outputLine.decode('UTF-8').rstrip()
        if(not silent):
            print(outputLine)
        output.append(outputLine)
        return False
    else:
        return True

ExecReturn = collections.namedtuple('ExecReturn', ['result', 'stdout', 'stderr'])

def execCmds(cmds, silent = False):
    """Execute the given commands one by one and return the result
    
    Return a tuple consisting of return code(result), stdout and stderr
    Execution is aborted if any command fail
    Distinct commands (list entries) do not share an environment
    """
    output = []
    error = []
    retCode = 0
    for cmd in cmds:
        proc = subprocess.Popen(["bash", "-e"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE)
        proc.stdin.write(str.encode(cmd))
        proc.stdin.flush()
        proc.stdin.close()
        retCode = None
        while retCode == None:
            outEOF = readAndOutputLine(proc.stdout, output, silent)
            errEOF = readAndOutputLine(proc.stderr, error, silent)
            if(outEOF and errEOF):
                retCode = proc.poll()
        if(retCode != 0):
            if(not silent):
                print("Executing `" + cmd + "` failed with code " + str(retCode))
            break
    return ExecReturn(retCode, output, error)

def execCmd(cmd, silent = False):
    """Execute the given command and returns a (named) tuple consisting of the return code, stdout and stderr.
    
    If silent is True then no output is printed. Else stdout and stderr are printed as received
    Executing multiple commands (separated by new lines) is allowed and they share the same environment
    Execution is aborted if any command fails
    """
    return execCmds([cmd], silent)
    
@contextmanager
def cd(newdir):
    """Change the current directory and switch back at and of `with` block"""
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)

