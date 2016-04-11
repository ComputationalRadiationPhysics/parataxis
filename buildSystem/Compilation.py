import re
import os
from execHelpers import execCmd, ExecReturn

class Compilation:
    """Represents a specific configuration of an example that can be compiled into a given folder"""
    def __init__(self, example, cmakePreset, parentBuildPath = 'build', profileFile = None):
        """Create a new compilation of an example with a cmakePreset
        
        parentBuildPath -- Parent directory where a subfolder for the build will be created. Defaults to 'build'
        profileFile     -- Path to profile file to use (gets sourced into build env). Defaults to None (current env)
        """
        self.example = example
        self.profileFile = profileFile
        self.cmakePreset = cmakePreset
        self.setParentBuildPath(parentBuildPath)
        self.buildFolderName = "build_" + example.getMetaData()['short'] + "_cmake" + str(cmakePreset)
        # Remove non-alphanumeric chars
        self.buildFolderName = re.sub("\W", "", self.buildFolderName)
        
    def getConfig(self):
        """Return the tuple (exampleName, cmakePreset, profileFile) that identifies this Compilation"""
        return (self.example.getMetaData()["name"], self.cmakePreset, self.profileFile)
        
    def setParentBuildPath(self, parentBuildPath):
        """Set the parent directory for builds. Reset also lastResult to None"""
        self.parentBuildPath = os.path.abspath(parentBuildPath)
        self.lastResult = None
    
    def getParentBuildPath(self):
        assert os.path.isabs(self.parentBuildPath)
        return self.parentBuildPath
    
    def getSetupCmd(self):
        """Return command required to setup environment. Includes terminating newline if non-empty"""
        cmd = ''
        if(self.profileFile != None):
            cmd += 'source ' + self.profileFile + '\n'
        variables = [('BASE_BUILD_PATH', self.getParentBuildPath()),
                     ('BUILD_PATH',      self.getBuildPath()),
                     ('INSTALL_PATH',    self.getInstallPath())
                    ]
        for (name, value) in variables:
            cmd += 'export TEST_' + name + '="' + value + '"\n'
        return cmd
        
    def getBuildPath(self):
        """Get the path where this is build"""
        return self.parentBuildPath + "/" + self.buildFolderName
    
    def getInstallPath(self):
        """Get the path to which this is installed"""
        return self.getBuildPath() + '/installed'
    
    def configure(self, pathToCMakeLists, dryRun, verbose, silent):
        """Configure the example via CMake
        
        pathToCMakeLists -- Folder that contains the CMakeLists.txt
        dryRun           -- Only print the commands
        silent           -- Do not print progress to stdout
        Return None for dryRuns or the result tuple from execCmd (result, stdout, stderr)
        """
        buildPath = self.getBuildPath()
        setupCmd = self.getSetupCmd()
        cmd = 'mkdir -p "' + buildPath + '" && cd "' + buildPath + '"\n'
        cmd += "cmake"
        cmd += " " + self.example.getCMakeFlags()[self.cmakePreset].replace(";", "\\;")
        cmd += ' -DXRT_EXTENSION_PATH="' + self.example.getFolder() + '"'
        cmd += ' -DCMAKE_INSTALL_PREFIX="' + self.getInstallPath() + '"'
        cmd += ' "' + pathToCMakeLists + '"'
        if dryRun or verbose:
            print(cmd)
        if dryRun:
            return None
        else:
            return execCmd(setupCmd + cmd, silent)
        
    def compile(self, dryRun, verbose, silent):
        """Compile the example (after configuring)        
        dryRun -- Only print the commands
        silent -- Do not print progress to stdout
        Return None for dryRuns or the result tuple from execCmd (result, stdout, stderr)
        """
        setupCmd = self.getSetupCmd()
        cmd = 'cd "' + self.getBuildPath() + '"\n'
        cmd += 'make install'
        if dryRun or verbose:
            print(cmd)
        if dryRun:
            return None
        else:
            return execCmd(setupCmd + cmd, silent)
        
    def configAndCompile(self, pathToCMakeLists, dryRun, verbose, silent):
        """Compile and configure the example (execute both of them)
        
        pathToCMakeLists -- Folder that contains the CMakeLists.txt
        dryRun           -- Only print the commands
        silent           -- Do not print progress to stdout
        Return None for dryRuns or the result tuple from execCmd (result, stdout, stderr)
        """
        cfgResult = self.configure(pathToCMakeLists, dryRun, verbose, silent)
        if(cfgResult != None and cfgResult.result != 0):
            return cfgResult
        compileResult = self.compile(dryRun, verbose, silent)
        if(compileResult == None):
            return None
        result = ExecReturn(compileResult.result,
                            cfgResult.stdout + compileResult.stdout,
                            cfgResult.stderr + compileResult.stderr)        
        
        self.lastResult = result
        return result

