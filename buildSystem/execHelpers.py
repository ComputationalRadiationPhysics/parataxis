import subprocess
import collections
from contextlib import contextmanager
import os
import threading
import sys

def __readAndOutput(fileHandle, output, silent):
    """Read and print a content from the file(handle)
    
    If anything is read, append to output (list) and print it if silent is not false
    """
    for line in fileHandle:
        if line:
            line = line.decode('UTF-8').rstrip()
            if(not silent):
                print(line)
            output.append(line)

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
        # Start threads to avoid deadlocking on many output
        stdoutThread = threading.Thread(target=__readAndOutput, args=(proc.stdout, output, silent))
        stdoutThread.setDaemon(True)
        stdoutThread.start()
        stderrThread = threading.Thread(target=__readAndOutput, args=(proc.stderr, error, silent))
        stderrThread.setDaemon(True)
        stderrThread.start()
        # Write commands
        proc.stdin.write(str.encode(cmd))
        proc.stdin.flush()
        proc.stdin.close()
        # Wait till finish
        retCode = proc.wait()
        stdoutThread.join(10)
        stderrThread.join(10)
        # Flush remaining output, so different commands output does not get mangled
        sys.stdout.flush()
        sys.stderr.flush()
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
def cd(newDir):
    """Change the current directory and switch back at and of `with` block"""
    prevDir = os.getcwd()
    newDir2 = os.path.expanduser(newDir)
    if not os.path.isdir(newDir2):
        if os.path.isabs(newDir2):
            absNewDir = newDir2
        else:
            absNewDir = os.path.abspath(newDir2)
        raise Exception("Cannot change to " + newDir + ": " + absNewDir + " does not exist")
    os.chdir(newDir2)
    try:
        yield
    finally:
        os.chdir(prevDir)

