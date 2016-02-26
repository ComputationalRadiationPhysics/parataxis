#!/usr/bin/python3 -u

import argparse
import yaml
import subprocess
from contextlib import contextmanager
import os
import sys
import shutil

@contextmanager
def cd(newdir):
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)
        
def execCmds(cmds):
    for cmd in cmds:
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        while True:
            output = proc.stdout.readline()
            if output:
                print(output.decode(), end='')
            else:
                break
        retCode = proc.poll()
        if(retCode != 0):
            print("Executing \"" + cmd + "\" failed with code " + str(retCode))
            return retCode
    return 0

def execCmd(cmd):
    return execCmds([cmd])

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
        if(execCmd(compileCmd) != 0):
            print("Compiling failed!")
            errorCode = 1
            continue
        print("Changing to build directory " + buildDir)
        with(cd(buildDir)):
            print("Executing pre-run commands...")
            if(execCmds(test['pre-run']) != 0):
                errorCode = 2
                break
            outputDir = "out_" + test["name"]
            if(os.path.isdir(outputDir)):
                shutil.rmtree(outputDir)
            tbgCmd = "tbg -s \"" + submitCmd + "\" -c submit/" + test["cfgFile"] + " -t " + submitTemplate + " " + outputDir
            print("Submitting to queue: " + tbgCmd)
            if(execCmd(tbgCmd) != 0):
                print("Submit or execution failed!")
                errorCode = 3
                break
            
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
    
