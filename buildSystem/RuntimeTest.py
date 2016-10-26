# Copyright 2015-2016 Alexander Grund
#
# This file is part of ParaTAXIS.
#
# ParaTAXIS is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ParaTAXIS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ParaTAXIS.  If not, see <http://www.gnu.org/licenses/>.

import os
import time
import re
import shutil
import itertools
import statusMonitors
from termHelpers import cprint
from execHelpers import execCmd, cd
from Compilation import Compilation

def flattenList(lst):
    """Generates a simple list out of a list (possibly) of lists"""
    for x in lst:
        if hasattr(x, '__iter__') and not isinstance(x, str):
            for y in flattenList(x):
                yield y
        else:
            yield x

class RuntimeTest:
    """Represents a specific configuration of an example that is executed and potentially validated
    
    """
    def __init__(self, example, testDocu, profileFile = None):
        """Create a new runtime test for a given example.
    
        Takes the example and the dictionary defining the test.
        Requires "name", "cmakeFlag", "cfgFile" and optionally "description", "post-run
        """
        for key in ["name", "cmakeFlag", "cfgFile"]:
            if not key in testDocu:
                raise Exception("Did not found required key '" + key + "' for runtime test '" + testDocu.get('name', '<noName>') + "' (" + example.getMetaData()["name"] + ")")
        self.example = example
        self.name = testDocu['name']
        self.profileFile = profileFile
        # Check for non-alphanumeric chars
        if re.match("[^A-Za-z0-9]", self.name) != None:
            raise Exception("Name for runtime test is invalid (only alphanumeric chars are allowed): " + self.name)
        
        self.description = testDocu.get('description')
        self.cmakeFlag = testDocu['cmakeFlag']
        self.cfgFile = testDocu['cfgFile']
        self.dependency = testDocu.get('dependency', None)
        self.preRunCmds = list(flattenList(testDocu.get('pre-run', [])))
        self.postRunCmds = list(flattenList(testDocu.get('post-run', [])))
        
        self.lastResult = False
        self.lastOutputPath = None
        
    def getConfig(self):
        """Return the tuple (exampleName, cmakePreset, profileFile) that identifies this RuntimeTest"""
        return (self.example.getMetaData()["name"], self.cmakeFlag, self.profileFile)
        
    def getDependency(self):
        """Return dependent runtime test name (from same example)"""
        return self.dependency

    def findCompilation(self, outputDir = None):
        """Return the compilation instance required for this test.
        
        If outputDir is set, it will also be set in the compilation and a new compilation will be created if none is found.
        Otherwise the compilations buildPath will not be checked and None will be returned if no matching one is found
        """
        if outputDir != None:
            outputDir = os.path.abspath(outputDir)

        for c in self.example.getCompilations():
            if(self.getConfig() == c.getConfig()):
                if(outputDir != None and c.getParentBuildPath() != outputDir):
                    c.setParentBuildPath(outputDir)
                return c
        if outputDir == None:
            return None
        else:
            c = Compilation(self.example, self.cmakeFlag, outputDir, self.profileFile)
            self.example.addCompilation(c)
            return c
    
    def wait(self, timeout = -1):
        """Wait for the test to finish.
        
        Only usefull for batch submission and after starting the test.
        timeout -- number of seconds to wait for completion
        """
        if self.monitor == None:
            return True
        startTime = time.time()
        if(self.monitor.isWaiting):
            cprint("Waiting for " + self.name + " to be executed", "yellow")
            while (self.monitor.isWaiting and not self.__isTimeout(startTime, timeout)):
                time.sleep(5)
                self.monitor.update()
        if(not self.monitor.isFinished and not self.__isTimeout(startTime, timeout)):
            cprint("Waiting for " + self.name + " to be finished", "yellow")
            while (not self.monitor.isFinished and not self.__isTimeout(startTime, timeout)):
                time.sleep(5)
                self.monitor.update()
        if(self.monitor.isWaiting) or (not self.monitor.isFinished):
            return False        
        self.monitor = None
        return True
    
    def checkFinished(self):
        """Poll if the test has finished.
        
        Return codes:
        0 = Finished successfully
        1 = Still running
        2 = Not running but not finished (error, aborted, ...)
        """
        if not self.wait(0):
            return 1
        
        outputFile = self.getSimOutputPath() + "/output"
        if(not os.path.isfile(outputFile)):
            cprint("Note: " + outputFile + " not found. Test was probably not started or template is wrong", "red")
            return 3
        else:
            simulationFinished = False
            with open(outputFile, 'r') as inF:
                for line in inF:
                    if 'full simulation time' in line:
                        simulationFinished = True
                    elif 'Unhandled exception occurred' in line:
                        cprint("Test " + self.name + " has thrown an exception. Info: " + line, "red")
                        return 2
            if(not simulationFinished):
                cprint("Test " + self.name + " does not seem to have finished successfully!", "red")
                return 2
        return 0
            
    def execute(self, srcDir, parentBuildPath, dryRun, verbose):
        """Execute the test and returns 0 on success.
        
        srcDir          -- Path to dir containing the CMakeLists.txt
        parentBuildPath -- Parent path which should contain build folders
        dryRun          -- Just print commands

        Return 0 on success
        """
        result = self.startTest(srcDir, parentBuildPath, dryRun, verbose)
        if result:
            return result
        return self.finishTest(dryRun, verbose)
        
    def startTest(self, srcDir, parentBuildPath, dryRun, verbose):
        """Start the test (submit to batch system or actually execute it)
        
        Requires 'TBG_SUBMIT' and 'TBG_TPLFILE' to be set in the environment
        srcDir          -- Path to dir containing the CMakeLists.txt
                           (only used if compiled version could not be found)
        parentBuildPath -- Parent path which should contain build folders
        dryRun          -- Just print commands

        Return 0 on success
        """        
        
        self.monitor = None
        self.lastOutputPath = None
        self.lastResult = False
        
        if not self.__checkTBGCfg():
            return 1
            
        # Print newline for separation
        print("")
        
        compilation = self.findCompilation(parentBuildPath)
        if(compilation.lastResult == None):
            cprint("Did not find pre-compiled program for " + self.name+ ". Compiling...", "yellow")
            result = compilation.configAndCompile(srcDir, dryRun, verbose, False)
        else:
            result = compilation.lastResult
        if(result != None and result.result != 0):
            return result.result
        
        outputFolderName = self.example.getMetaData()["short"] + "_" + self.name
        self.lastOutputPath = os.path.abspath(os.path.join(parentBuildPath, "output", outputFolderName))
                
        preRunResult = self.__execCmds(True, compilation, dryRun, verbose)
        if not preRunResult == None:
            return preRunResult

        cprint("Changing to install directory " + compilation.getInstallPath(), "yellow")
        with(cd(compilation.getInstallPath() if not dryRun else ".")):
            result = self.__submit(compilation, self.lastOutputPath, dryRun, verbose)
            
        if result:
            return result
        return 0
    
    def finishTest(self, dryRun, verbose):
        """Finish test after starting it (wait for completion and execute post-build commands)
        
        dryRun -- Just show commands

        Return 0 on success
        """
        if self.lastOutputPath == None:
            if(verbose):
                cprint("Output path not set, maybe this was not run yet", "red")
            return 1
        compilation = self.findCompilation()
        if compilation == None:
            if(verbose):
                cprint("Did not find a compilation.", "red")
            return 1

        if(not self.wait()):
            if(verbose):
                cprint("Unknown error during wait for program finish", "red")
            return 2
        
        if not dryRun:
            result = self.checkFinished()
            if result:
                if(verbose):
                    cprint("Program did not finish successfully", "red")
                return result
        
        postRunResult = self.__execCmds(False, compilation, dryRun, verbose)
        if not postRunResult == None:
            return postRunResult
           
        cprint("Test \"" + self.name + "\" finished successfully!", "green")
        self.lastResult = True
        return 0
        
    def getOutputPath(self, checkStarted = True):
        assert (not checkStarted or self.lastOutputPath != None), "Runtime test was not started"            
        return self.lastOutputPath
        
    def getSimOutputPath(self):        
        return self.getOutputPath() + "/simOutput"
        
    def getSetupCmd(self, compilation, dryRun):
        """Return command required to setup environment. Includes terminating newline if non-empty"""
        assert os.path.isabs(self.getOutputPath()), "Need absolute path"
        cfgFilePath = os.path.join(compilation.getInstallPath(), "submit", self.cfgFile)
        if dryRun and not os.path.isfile(cfgFilePath):
            content = 'TBG_gridSize="-g 42 42 42"\nTBG_steps="-s 42"\n'
        else:
            with open(cfgFilePath, 'r') as cfgFile:
                content = cfgFile.read()
        gridSize = re.search("TBG_gridSize=\"-g (\\d+( \\d+)*)\"", content)
        assert gridSize != None, "Gridsize not found. Please define it in your cfg file using TBG_gridSize"
        gridSize = gridSize.group(1)
        numTimesteps = re.search("TBG_steps=\"-s (\\d+)\"", content)
        assert numTimesteps != None, "Timesteps not found. Please define it in your cfg file using TBG_step"
        numTimesteps = numTimesteps.group(1)
        cmd = compilation.getSetupCmd()
        variables = [('NAME',           self.name),
                     ('OUTPUT_PATH',    self.getOutputPath()),
                     ('SIMOUTPUT_PATH', self.getSimOutputPath()),
                     ('GRID_SIZE',      gridSize),
                     ('TIMESTEPS',      numTimesteps)
                    ]
        for (name, value) in variables:
            cmd += 'export TEST_' + name + '="' + value + '"\n'
        return cmd

    def __checkTBGCfg(self):
        """Check if required TBG variables are set in the environment"""
        for var in ('TBG_SUBMIT', 'TBG_TPLFILE'):
            if(not os.environ.get(var)):
                cprint("Missing " + var + " definition", "red")
                return False
        return True
    
    def __submit(self, compilation, outputDir, dryRun = False, verbose = False):
        """Submit test to tbg"""
        if(os.path.isdir(outputDir)):
            shutil.rmtree(outputDir)
        tbgCmd = "tbg -t -s";
        if(self.profileFile):
            tbgCmd += " -o 'TBG_profile_file='" + self.profileFile + "'"
        tbgCmd += " -c submit/" + self.cfgFile + " " + outputDir
        cprint("Submitting " + self.getConfig()[0] + "/" + self.name + " to queue", "yellow")
        if dryRun or verbose:
            tbgVars = "TBG_SUBMIT="+os.environ.get("TBG_SUBMIT") + " TBG_TPLFILE="+os.environ.get("TBG_TPLFILE")
            print(tbgVars + " " + tbgCmd)
        if dryRun:
            self.monitor = None
        else:
            res = execCmd(self.getSetupCmd(compilation, dryRun) + tbgCmd)
            if(res.result != 0):
                cprint("Submit or execution failed!", "red")
                return res.result

            self.monitor = statusMonitors.GetMonitor(os.environ['TBG_SUBMIT'], res.stdout, res.stderr)
        return 0
    
    def __execCmds(self, preRunOrPostRun, compilation, dryRun, verbose):
        """Execute pre- or post-run commands'
        
           Return error result or None on success
        """
        if preRunOrPostRun:
            cmds = self.preRunCmds
            runDir = compilation.getInstallPath()
            descr = "pre-run"
        else:
            cmds = self.postRunCmds
            runDir = self.lastOutputPath
            descr = "post-run"
        if cmds:
            with(cd(runDir if not dryRun else ".")):
                cprint("Executing " + descr + " commands for " + self.name+ "...", "yellow")
                envSetupCmd = self.getSetupCmd(compilation, dryRun)
                if verbose:
                    print("Environment: " + envSetupCmd)
                
                for cmd in cmds:
                    if dryRun or verbose:
                        print(cmd)
                    if not dryRun:
                        result = execCmd(envSetupCmd + cmd)
                        if(result.result != 0):
                            return result.result
        return None

    def __str__(self):
        (exName, cmakeCfg, profileFile) = self.getConfig()
        result = "Runtime test " + exName + "/" + self.name + ", cmakeCfg: " + str(cmakeCfg)
        if profileFile:
            result += ", profileFile: " + profileFile
        return result

    @staticmethod
    def __isTimeout(startTime, timeout):
        """Check if timeout occured (for timeout >=0)"""
        return timeout >= 0 and time.time() - startTime >= timeout
    
