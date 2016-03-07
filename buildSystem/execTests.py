#!/usr/bin/python3 -u

import argparse
import yaml
from contextlib import contextmanager
import os
import sys
import shutil
from execHelpers import execCmds, execCmd
import statusMonitors
import time

@contextmanager
def cd(newdir):
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)


def runTests(exampleDir, buildBaseDir, submitCmd, submitTemplate):
    #exampleDir = "/home/grund59/Dokumente/XRayTracing/src/examples/DoubleSlit"
    #buildBaseDir = "/home/grund59/Tests"
    #submitCmd = "qsub"
    #submitTemplate = "submit/hypnos-hzdr/k80_profile.tpl"

    selfDir = os.path.dirname(os.path.realpath(__file__))
    srcDir = os.path.dirname(selfDir)
    compileScript = srcDir + "/compileExampleSet.sh"
    if(not os.path.isfile(compileScript)):
        print("Compile script at " + compileScript + " does not exist")
        return 42

    with open(exampleDir + "/documentation.yml", 'r') as stream:
        documentation = yaml.safe_load(stream)
        
    shortName = documentation["example"]["short"]
    
    errorCode = 0
        
    for test in documentation["tests"]:
        print("Executing test \"" + test["name"] + "\"...")
        buildDir = buildBaseDir + "/build_" + shortName + "_cmakePreset_" + str(test["cmakeflag"])
        compileCmd = compileScript + " -f -t " + str(test["cmakeflag"]) + " \"" + exampleDir + "\" \"" + buildDir + "\""
        print("Compiling via " + compileCmd)
        res = execCmd(compileCmd)
        if(res.result != 0):
            print("Compiling failed!")
            errorCode = 1
            continue
        print("Changing to build directory " + buildDir)
        with(cd(buildDir)):
            print("Executing pre-run commands...")
            res = execCmds(test['pre-run'])
            if(res.result != 0):
                errorCode = 2
                break
            outputDir = "out_" + test["name"]
            if(os.path.isdir(outputDir)):
                shutil.rmtree(outputDir)
            tbgCmd = "tbg -s \"" + submitCmd + "\" -c submit/" + test["cfgFile"] + " -t " + submitTemplate + " " + outputDir
            print("Submitting to queue: " + tbgCmd)
            res = execCmd(tbgCmd)
            if(res.result != 0):
                print("Submit or execution failed!")
                errorCode = 3
                break
            
            monitor = statusMonitors.GetMonitor(submitCmd, res.stdout, res.stderr)
            if(monitor.isWaiting):
                print("Waiting for program to be executed")
                while (monitor.isWaiting):
                    time.sleep(5)
                    monitor.update()
            if(not monitor.isFinished):
                print("Waiting for program to be finished")
                while (not monitor.isFinished):
                    time.sleep(5)
                    monitor.update()
            if(monitor.isWaiting) or (not monitor.isFinished):
                errorCode = 4
                break
            
            outputFile = outputDir + "/simOutput/output"
            if(not os.path.isfile(outputFile)):
                print("Note: " + outputFile + " not found. Check your template so this file is created!\n Skipping check")
            else:
                simulationFinished = False
                with open(outputFile, 'r') as inF:
                    for line in inF:
                        if 'full simulation time' in line:
                            simulationFinished = True
                            break
                if(not simulationFinished):
                    errorCode = 5
                    print("Test does not seem to have finished successfully!")
                    continue
                    
            print("Executing post-run commands...")
            res = execCmds(test['post-run'])
            if(res.result != 0):
                errorCode = 2
                break                
           
        print("Test \"" + test["name"] + "\" finished successfully!")
                        
    return errorCode
            
def main():
  # Parse the command line.
  parser = argparse.ArgumentParser(description="Runs tests", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('-e', '--example', help=('Path to one example'), required=True)
  parser.add_argument('-o', '--output', default="testBuilds", help=('Path to directory with build directories. If a build directory does not exist, the example will be build automatically'))
  parser.add_argument('-s', '--submitCmd', default="qsub", help=('Command used to execute the program'))
  parser.add_argument('-t', '--template', default="submit/hypnos-hzdr/k80_profile.tpl", help=('Submit template used'))
  options = parser.parse_args()
  return runTests(options.example, options.output, options.submitCmd, options.template)  

if __name__ == '__main__':
    sys.exit(main())
    
