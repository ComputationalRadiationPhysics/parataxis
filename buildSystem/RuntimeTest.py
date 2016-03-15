import os
import time
import statusMonitors
from execHelpers import execCmd, cd

class RuntimeTest:
    """Represents a specific configuration of an example that is executed and potentially validated
    
    """
    def __init__(self, example, testDocu):
        """Create a new runtime test for a given example.
    
        Takes the example and the dictionary defining the test.
        Requires "name", "cmakeFlag", "cfgFile" and optionally "description", "env", "post-run
        """
        self.example = example
        self.name = testDocu['name']
        self.description = testDocu.get('description')
        self.cmakeFlag = testDocu['cmakeFlag']
        self.cfgFile = testDocu['cfgFile']
        self.env = testDocu.get('env')
        self.postRunCmds = testDocu.get('post-run', [])
        
    def findCompilation(self, outputDir = None):
        """Return the compilation instance required for this test.
        
        If outputDir is set, it will also be set in the compilation and a new compilation will be created if none is found.
        Otherwise the compilations buildPath will not be checked and None will be returned if no matching one is found
        """
        for c in self.example.getCompilations():
            if((self.cmakeFlag, self.env) == c.getConfig()):
                if(outputDir != None and c.getParentBuildPath() != outputDir):
                    c.setParentBuildPath(outputDir)
                return c
        if outputDir == None:
            return None
        else:
            return Compilation(self.example, self.cmakeFlag, outputDir, self.env)
    
    def wait(self, timeout = -1):
        """Wait for the test to finish.
        
        Only usefull for batch submission and after starting the test.
        timeout -- number of seconds to wait for completion
        """
        if self.monitor == None:
            return True
        startTime = time.time()
        if(self.monitor.isWaiting):
            print("Waiting for program to be executed")
            while (self.monitor.isWaiting and not __isTimeout(startTime, timeout)):
                time.sleep(5)
                self.monitor.update()
        if(not self.monitor.isFinished and not __isTimeout(startTime, timeout)):
            print("Waiting for program to be finished")
            while (not self.monitor.isFinished and not __isTimeout(startTime, timeout)):
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
        
        outputFile = self.lastOutputDir + "/simOutput/output"
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
                print("Test does not seem to have finished successfully!")
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
        self.lastOutputDir = None
        
        if not self.__checkTBGCfg():
            return 1
        
        compilation = self.findCompilation(parentBuildPath)
        if(compilation.lastResult == None):
            print("Did not find pre-compiled program. Compiling...")
            result = compilation.configAndCompile(srcDir, dryRun)
        else:
            result = compilation.lastResult
        if(result != None and result.result != 0):
            return result.result
        
        print("Changing to install directory " + compilation.getInstallPath())
        with(cd(compilation.getInstallPath())):
            outputDir = "out_" + self.name
            self.lastOutputDir = compilation.getInstallPath() + '/' + outputDir
            result = self.__submit(compilation, outputDir, dryRun)
            if result:
                return result
        return 0
    
    def finishTest(self, dryRun):
        """Finish test after starting it (wait for completion and execute post-build commands)
        
        dryRun -- Just show commands
        """
        if self.lastOutputDir == None:
            return 1
        compilation = self.__findCompilation()
        if compilation == None:
            return 1
        with(cd(self.lastOutputDir)):
            if(not self.wait()):
                return 2
            
            if not dryRun:
                result = self.checkFinished()
                if result:
                    return result
                    
            print("Executing post-run commands...")
            envSetupCmd = compilation.getSetupCmd()
            
            for cmd in self.postRunCmds:
                if dryRun:
                    print(cmd)
                else:
                    result = execCmd(envSetupCmd + cmd)
                    if(res.result != 0):
                        return res.result
           
        print("Test \"" + self.name + "\" finished successfully!")
        return 0

    def __checkTBGCfg(self):
        """Check if required TBG variables are set in the environment"""
        for var in ('TBG_SUBMIT', 'TBG_TPLFILE'):
            if(not os.environ.get(var)):
                print("Missing " + var +" definition")
                return False
        return True
    
    def __submit(self, compilation, outputDir, dryRun):
        """Submit test to tbg"""
        if(os.path.isdir(outputDir)):
            shutil.rmtree(outputDir)
        tbgCmd = "tbg -c submit/" + self.cfgFile + " " + outputDir
        print("Submitting to queue")
        if(dryRun):
            print(tbgCmd)
            self.monitor = None
        else:
            res = execCmd(compilation.getSetupCmd() + tbgCmd)
            if(res.result != 0):
                print("Submit or execution failed!")
                return res.result

            self.monitor = statusMonitors.GetMonitor(os.environ['TBG_SUBMIT'], res.stdout, res.stderr)
        return 0
    
    def __isTimeout(startTime, timeout):
        """Check if timeout occured (for timeout >=0)"""
        return timeout >= 0 and time.time() - startTime >= timeout
    
