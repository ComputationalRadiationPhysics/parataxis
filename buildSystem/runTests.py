#!/usr/bin/python3 -u

import argparse
import sys
import os
import multiprocessing
from distutils.util import strtobool
from execHelpers import execCmd
import Example

def getExampleFolders(exampleNameOrFolder, getAll):
    """Return list of absolute paths to the examples
    
       getAll == False -- The example specified by name or folder
       else            -- All examples in the folder
    """
    if(getAll):
        if(not os.path.isdir(exampleNameOrFolder)):
            print("Path to example does not exist: " + exampleNameOrFolder)
            sys.exit(1)
        exampleDirs = os.listdir(exampleNameOrFolder)
        parentDir = exampleNameOrFolder
    elif(os.path.isdir(exampleNameOrFolder)):
        exampleDirs = ['exampleNameOrFolder']
        parentDir = ""
    else:
        exampleFolder = "examples/" + exampleNameOrFolder
        if(os.path.isdir(exampleFolder)):
            exampleDirs = [exampleFolder]
            parentDir = ""
        else:
            print("Path to example or example does not exist: " + exampleNameOrFolder)
            sys.exit(1)
    result = [os.path.abspath(parentDir + "/" + dir) for dir in exampleDirs]
    return result
    
def writeListToFile(lstIn, file):
    """Write a list into an open file"""
    for el in lstIn:
        file.write("%s\n" % el)

def writeStatusToFiles(result, folder):
    """Write status of an example to files
    
    Write stdout.log, stderr.log, returnCode.txt and cmakeSettings.log (if CMakeCache.txt exists)
    """
    if(not os.path.isdir(folder)):
        return
    with open(folder + "/stdout.log", "w") as file:
        writeListToFile(result.stdout, file)
    with open(folder + "/stderr.log", "w") as file:
        writeListToFile(result.stderr, file)
    with open(folder + "/returnCode.txt", "w") as file:
        file.write("%d" % result.result)
    cmakeSettingsPath = os.path.abspath(folder + "/cmakeSettings.log")
    if(os.path.isfile(cmakeSettingsPath)):
        os.remove(cmakeSettingsPath)
    if(os.path.isfile(folder + "/CMakeCache.txt")):
        execCmd('cd "' + folder + '" && cmake -L . &> "' + cmakeSettingsPath + '"', True)

def doCompile(compilation, args):
    """Actually compile a compilation and do some state keeping (write status files)"""
    result = compilation.configAndCompile(*args)
    if result != None:
        writeStatusToFiles(result, compilation.getBuildPath())
    return result == None or result.result == 0

def compileWorker(input, output):
    """Worker function that processes compilations from input queue and writes result (bool) to output queue"""
    for compilation, args in iter(input.get, 'STOP'):
        try:
            result = doCompile(compilation, args)
            output.put(result)
        except Exception as e:
            print("Error during compilation: " + str(e))
            output.put(False)
        
def processCompilations(compilations, srcDir, dryRun, numParallel):
    """Compile the compilations passed
    
    srcDdir     -- Path to folder with CMakeLists.txt
    dryRun      -- Show only commands
    numParallel -- Execute in parallel using N processes
    Return number of errors
    """
    numErrors = 0
    if(numParallel < 2):
        for c in compilations:
            result = doCompile(c, (srcDir, dryRun, False))
            if(not result):
                numErrors += 1
    else:
        taskQueue = multiprocessing.Queue()
        doneQueue = multiprocessing.Queue()

        for c in compilations:
            taskQueue.put((c, (srcDir, dryRun, True)))

        for i in range(numParallel):
            multiprocessing.Process(target=compileWorker, args=(taskQueue, doneQueue)).start()
            taskQueue.put('STOP')
            
        for i in range(len(compilations)):
            res = doneQueue.get() # Potentially blocks
            if(not res):
                numErrors += 1
    return numErrors

def main(argv):
    srcDir = os.path.abspath(os.path.dirname(__file__) + "../..")
    if not os.path.isfile(srcDir + "/CMakeLists.txt"):
        print("Unexpected file location. CMakeLists.txt not found in " + srcDir)
        return 1
    exampleFolder = srcDir + '/examples'
    # Parse the command line.
    parser = argparse.ArgumentParser(description="Compile and run tests", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--example', default=exampleFolder, help='Example name, path to one example or example folder (if --all is used)')
    parser.add_argument('-o', '--output', default="testBuilds", help='Path to directory with build directories. If a build directory does not exist, the example will be build automatically')
    parser.add_argument('-a', '--all', action='store_true', help='Process all examples in the folder')
    parser.add_argument('-j', type=int, default=1, const=-1, nargs='?', help='Compile in parallel using N processes', metavar='N')
    parser.add_argument('-d', '--dry-run', action='store_true', help='Just print commands and exit')
    parser.add_argument('-t', '--test', action='append', nargs='+', help='Compile and execute only tests with given names')
    parser.add_argument('--compile-only', action='store_true', help='Run only compile tests (do not run compiled programs)')
    options = parser.parse_args(argv)
    if options.j < 0:
        options.j = multiprocessing.cpu_count()
    if options.example == exampleFolder and not options.all and len(argv) > 0:
        sys.stdout.write("Path to default example folder given.\n\nShall I process all examples in that folder? [y/n]")
        options.all = strtobool(input().lower())
    if len(argv) == 0:
        sys.stdout.write("No parameters given. Taking default:\nLoad all examples from " + options.example + ", compile and run them.\n\nContinue? [y/n]")
        if not strtobool(input().lower()):
            return 1
        options.all = True
    
    print("Getting examples...")
    exampleDirs = getExampleFolders(options.example, options.all)
    if(len(exampleDirs) == 0):
        print("No examples found")
        return 1
    print("Loading examples...")
    examples = Example.loadExamples(exampleDirs)
    if(examples == None):
        return 1
    compilations = Example.getCompilations(examples, options.output, options.test)
    if options.j > 1:
        print("Compiling examples using", options.j, "processes...")
    else:
        print("Compiling examples...")
    numErrors = processCompilations(compilations, srcDir, options.dry_run, options.j)
    if(numErrors > 0):
        print(str(numErrors) + " compile errors occured!")
        return 1
    if(options.compile_only):
        return 0
    print("Running examples...")
    runtimeTests = Example.getRuntimeTests(examples, options.test)
    for test in runtimeTests:
        if test.execute(srcDir, options.output, options.dry_run) != 0:
            numErrors += 1
    if(numErrors > 0):
        print(str(numErrors) + " run errors occured!")
        return 1
    return 0
    
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

