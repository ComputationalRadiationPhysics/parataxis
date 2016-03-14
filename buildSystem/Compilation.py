import re
from execHelpers import execCmd, ExecReturn

class Compilation:
    """Represents a specific configuration of an example that can be compiled into a given folder"""
    def __init__(self, example, cmakePreset, parentBuildPath = 'build', env = None):
        """Create a new compilation of an example with a cmakePreset
        
        parentBuildPath -- Parent directory where a subfolder for the build will be created. Defaults to 'build'
        env             -- Name of the environment from the example to use for the build. Defaults to None (current env)
        """
        self.example = example
        self.env = env
        self.cmakePreset = cmakePreset
        if(env == None):
            envName = ''
        else:
            envName = '_' + env
        self.setParentBuildPath(parentBuildPath)
        self.buildFolderName = "build_" + example.getMetaData()['short'] + envName + "_cmake" + str(cmakePreset)
        # Remove non-alphanumeric chars
        self.buildFolderName = re.sub("\W", "", self.buildFolderName)
        
    def getConfig(self):
        """Return the tuple (example, cmakePreset, env) that identifies this Compilation"""
        return (self.example, self.cmakePreset, self.env)
        
    def setParentBuildPath(self, parentBuildPath):
        """Set the parent directory for builds. Reset also lastResult to None"""
        self.parentBuildPath = parentBuildPath
        self.lastResult = None
    
    def getParentBuildPath(self):
        return self.parentBuildPath
    
    def getSetupCmd(self):
        """Return the string for setting up the environment (can be empty)"""
        if(self.env == None):
            return ''
        else:
            return self.example.env[self.env].getSetupCmd()
        
    def getBuildPath(self):
        """Get the path where this is build"""
        return self.parentBuildPath + "/" + self.buildFolderName
    
    def getInstallPath(self):
        """Get the path to which this is installed"""
        return self.getBuildPath() + '/installed'
    
    def configure(self, pathToCMakeLists, dryRun, silent):
        """Configure the example via CMake
        
        pathToCMakeLists -- Folder that contains the CMakeLists.txt
        dryRun           -- Only print the commands
        silent           -- Do not print progress to stdout
        Return None for dryRuns or the result tuple from execCmd (result, stdout, stderr)
        """
        buildPath = self.getBuildPath()
        cmd = self.getSetupCmd()
        cmd += 'mkdir -p "' + buildPath + '" && cd "' + buildPath + '"\n'
        cmd += "cmake"
        cmd += " " + self.example.getCMakeFlags()[self.cmakePreset].replace(";", "\\;")
        cmd += ' -DXRT_EXTENSION_PATH="' + self.example.getFolder() + '"'
        cmd += ' -DCMAKE_INSTALL_PREFIX="' + self.getInstallPath() + '"'
        cmd += ' "' + pathToCMakeLists + '"'
        if(dryRun):
            print(cmd)
            return None
        else:
            return execCmd(cmd, silent)
        
    def compile(self, dryRun, silent):
        """Compile the example (after configuring)        
        dryRun -- Only print the commands
        silent -- Do not print progress to stdout
        Return None for dryRuns or the result tuple from execCmd (result, stdout, stderr)
        """
        cmd = self.getSetupCmd()
        cmd += 'cd "' + self.getBuildPath() + '"\n'
        cmd += 'make install'
        if(dryRun):
            print(cmd)
            return None
        else:
            return execCmd(cmd, silent)
        
    def configAndCompile(self, pathToCMakeLists, dryRun, silent):
        """Compile and configure the example (execute both of them)
        
        pathToCMakeLists -- Folder that contains the CMakeLists.txt
        dryRun           -- Only print the commands
        silent           -- Do not print progress to stdout
        Return None for dryRuns or the result tuple from execCmd (result, stdout, stderr)
        """
        cfgResult = self.configure(pathToCMakeLists, dryRun, silent)
        if(cfgResult != None and cfgResult.result != 0):
            return cfgResult
        compileResult = self.compile(dryRun, silent)
        if(compileResult == None):
            return None
        result = ExecReturn(compileResult.result,
                            cfgResult.stdout + compileResult.stdout,
                            cfgResult.stderr + compileResult.stderr)        
        
        self.lastResult = result
        return result

