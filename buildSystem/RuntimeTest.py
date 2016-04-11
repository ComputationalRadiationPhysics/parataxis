import os
import time
import re
import statusMonitors
from termHelpers import cprint
from execHelpers import execCmd, cd
from Compilation import Compilation

class RuntimeTest:
    """Represents a specific configuration of an example that is executed and potentially validated
    
    """
    def __init__(self, example, testDocu, profileFile = None):
        """Create a new runtime test for a given example.
    
        Takes the example and the dictionary defining the test.
        Requires "name", "cmakeFlag", "cfgFile" and optionally "description", "post-run
        """
        self.example = example
        self.name = testDocu['name']
        self.profileFile = profileFile
        # Check for non-alphanumeric chars
        if re.match("\W", self.name) != None:
            raise Exception("Name for runtime test is invalid (only alphanumeric chars are allowed): " + self.name)
        
        self.description = testDocu.get('description')
        self.cmakeFlag = testDocu['cmakeFlag']
        self.cfgFile = testDocu['cfgFile']
        self.postRunCmds = testDocu.get('post-run', [])
        self.lastResult = False
        
    def getConfig(self):
        """Return the tuple (exampleName, cmakePreset, profileFile) that identifies this RuntimeTest"""
        return (self.example.getMetaData()["name"], self.cmakeFlag, self.profileFile)

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
            return Compilation(self.example, self.cmakeFlag, outputDir, self.profileFile)
    
    def wait(self, timeout = -1):
        """Wait for the test to finish.
        
        Only usefull for batch submission and after starting the test.
        timeout -- number of seconds to wait for completion
        """
        if self.monitor == None:
            return True
        startTime = time.time()
        if(self.monitor.isWaiting):
            cprint("Waiting for program to be executed", "yellow")
            while (self.monitor.isWaiting and not self.__isTimeout(startTime, timeout)):
                time.sleep(5)
                self.monitor.update()
        if(not self.monitor.isFinished and not self.__isTimeout(startTime, timeout)):
            cprint("Waiting for program to be finished", "yellow")
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
                        break
            if(not simulationFinished):
                cprint("Test does not seem to have finished successfully!", "red")
                return 2
        return 0
            
    def execute(self, srcDir, parentBuildPath, dryRun):
        """Execute the test and returns 0 on success.
        
        srcDir          -- Path to dir containing the CMakeLists.txt
        parentBuildPath -- Parent path which should contain build folders
        dryRun          -- Just print commands
        """
        result = self.startTest(srcDir, parentBuildPath, dryRun)
        if result:
            return result
        return self.finishTest(dryRun)
        
    def startTest(self, srcDir, parentBuildPath, dryRun):
        """Start the test (submit to batch system or actually execute it)
        
        Requires 'TBG_SUBMIT' and 'TBG_TPLFILE' to be set in the environment
        srcDir          -- Path to dir containing the CMakeLists.txt
                           (only used if compiled version could not be found)
        parentBuildPath -- Parent path which should contain build folders
        dryRun          -- Just print commands
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
            cprint("Did not find pre-compiled program. Compiling...", "yellow")
            result = compilation.configAndCompile(srcDir, dryRun, False)
        else:
            result = compilation.lastResult
        if(result != None and result.result != 0):
            return result.result
        
        cprint("Changing to install directory " + compilation.getInstallPath(), "yellow")
        outputDir = "out_" + self.name
        self.lastOutputPath = os.path.abspath(compilation.getInstallPath() + '/' + outputDir)
        
        with(cd(compilation.getInstallPath() if not dryRun else ".")):
            result = self.__submit(compilation, outputDir, dryRun)
            
        if result:
            return result
        return 0
    
    def finishTest(self, dryRun):
        """Finish test after starting it (wait for completion and execute post-build commands)
        
        dryRun -- Just show commands
        """
        if self.lastOutputPath == None:
            return 1
        compilation = self.findCompilation()
        if compilation == None:
            return 1

        if(not self.wait()):
            return 2
        
        if not dryRun:
            result = self.checkFinished()
            if result:
                return result
                    
        with(cd(self.lastOutputPath if not dryRun else ".")):
            cprint("Executing post-run commands...", "yellow")
            envSetupCmd = self.getSetupCmd(compilation)
            
            for cmd in self.postRunCmds:
                if dryRun:
                    print(cmd)
                else:
                    result = execCmd(envSetupCmd + cmd)
                    if(result.result != 0):
                        return result.result
           
        cprint("Test \"" + self.name + "\" finished successfully!", "green")
        self.lastResult = True
        return 0
        
    def getOutputPath(self, checkStarted = True):
        assert (not checkStarted or self.lastOutputPath != None), "Runtime test was not started"            
        return self.lastOutputPath
        
    def getSimOutputPath(self):        
        return self.getOutputPath() + "/simOutput"
        
    def getSetupCmd(self, compilation):
        """Return command required to setup environment. Includes terminating newline if non-empty"""
        assert os.path.isabs(self.getOutputPath()), "Need absolute path"
        cmd = compilation.getSetupCmd()
        variables = [('NAME',           self.name),
                     ('OUTPUT_PATH',    self.getOutputPath()),
                     ('SIMOUTPUT_PATH', self.getSimOutputPath())
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
    
    def __submit(self, compilation, outputDir, dryRun = False):
        """Submit test to tbg"""
        if(os.path.isdir(outputDir)):
            shutil.rmtree(outputDir)
        tbgCmd = "tbg -t -s";
        if(self.profileFile):
            tbgCmd += " -o 'TBG_profile_file='" + self.profileFile + "'"
        tbgCmd += " -c submit/" + self.cfgFile + " " + outputDir
        cprint("Submitting to queue", "yellow")
        if(dryRun):
            print(tbgCmd)
            self.monitor = None
        else:
            res = execCmd(self.getSetupCmd(compilation) + tbgCmd)
            if(res.result != 0):
                cprint("Submit or execution failed!", "red")
                return res.result

            self.monitor = statusMonitors.GetMonitor(os.environ['TBG_SUBMIT'], res.stdout, res.stderr)
        return 0

    @staticmethod
    def __isTimeout(startTime, timeout):
        """Check if timeout occured (for timeout >=0)"""
        return timeout >= 0 and time.time() - startTime >= timeout
    
