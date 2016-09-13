import re
import os
from execHelpers import execCmd, ExecReturn

def mergeCompilations(compilations, additionalCompilations):
    """Add all compilations from additionalCompilations to compilations that are not already present and return it
    
    No duplicates are allowed in compilations, but they are allowed in additionalCompilations
    """
    cfgs = []
    # Collect all current configs
    for c in compilations:
        cfg = c.getConfig()
        assert(not cfg in cfgs)
        cfgs.append(cfg)
    # Add new ones
    for c in additionalCompilations:
        cfg = c.getConfig()
        if not cfg in cfgs:
            cfgs.append(cfg)
            compilations.append(c)
    return compilations

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
        self.outputFolderName = example.getMetaData()['short'] + "_cmake" + str(cmakePreset)
        if self.cmakePreset >= len(self.example.getCMakeFlags()):
            raise Exception("Invalid cmakePreset: " + str(self.cmakePreset) + " of " + str(len(self.example.getCMakeFlags())) + " for " + str(self))
        
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
                     ('INSTALL_PATH',    self.getInstallPath()),
                     ('CMAKE_FLAGS',     self.example.getCMakeFlags()[self.cmakePreset])
                    ]
        for (name, value) in variables:
            cmd += 'export TEST_' + name + '="' + value + '"\n'
        return cmd
        
    def getBuildPath(self):
        """Get the path where this is build"""
        return os.path.join(self.parentBuildPath, "build", self.outputFolderName)
    
    def getInstallPath(self):
        """Get the path to which this is installed"""
        return os.path.join(self.parentBuildPath, "installed", self.outputFolderName)
    
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
            result = execCmd(setupCmd + cmd, silent)
            if result != None and result.result == 0:
                # Write used cmake flags to file
                with open(os.path.join(self.getInstallPath(), "cmakeFlags.txt"), 'w') as f:
                    flags = self.example.getCMakeFlags()[self.cmakePreset]
                    flags = flags.replace(";", "\n")
                    flags = flags.replace(" ", "\n")
                    f.write(flags)
            return result
            
        
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

    def __str__(self):
        (exName, cmakeCfg, profileFile) = self.getConfig()
        result = "Compilation of " + exName + ", cmakeCfg: " + str(cmakeCfg)
        if profileFile:
            result += ", profileFile: " + profileFile
        return result

