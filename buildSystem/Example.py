import yaml
import os
import sys
import traceback
import re
from execHelpers import execCmd
from Compilation import Compilation
from RuntimeTest import RuntimeTest

def expandList(lst):
    """Expand a list with range specifiers
    
       Ints are not touched
       Strings with (only) numbers will be converted to ints
       Range entries (like "2-6" or "1:7") will be expanded in entries of that range"""
    if (next((x for x in lst if not isinstance(x, int)), None) == None):
        return lst
    result = []
    for x in lst:
        if(isinstance(x, int)):
            result.append(x)
        else:
            match = re.search('^(\d+)( *- *(\d+))?$', x)
            if(match == None):
                raise Exception("Invalid value: " + str(x))
            elif(match.group(3) == None):
                result.append(int(match.group(1)))
            else:
                result.extend([y for y in range(int(match.group(1)), int(match.group(3)) + 1)])
    return result
    
def loadExamples(exampleDirs, profileFile = None):
    """Load all examples from the list of directories"""
    result = []
    names = []
    shortNames = []
    for exampleDir in exampleDirs:
        try:
            example = Example(exampleDir, profileFile)
            name = example.getMetaData()["name"]
            shortName = example.getMetaData()["short"]
            if name in names:
                print("Duplicate name '" + name + "'. Cannot procceed!")
                return None
            if shortName in shortNames:
                newShortName = None
                for i in range(1000):
                    if not (shortName + str(i)) in shortNames:
                        newShortName = shortName + str(i)
                        break
                if newShortname == None:
                    print("Duplicate short name '" + shortName + "'. Cannot procceed!")
                    return None
                else:
                    print("Duplicate short name '" + shortName + "'. Renaming to '" + newShortname + "'.")
                    shortName = newShortName
                    example.getMetaData()["short"] = shortName

            names.append(name)
            shortNames.append(shortName)
            result.append(example)
        except Exception as e:
            (etype, eVal, bt) = sys.exc_info()
            strException = traceback.format_exception_only(etype, eVal)
            print("Error loading example from " + exampleDir + ": " + strException[-1])
            print(traceback.format_list([traceback.extract_tb(bt)[-1]])[0])
            print(traceback.format_list([traceback.extract_tb(bt)[-2]])[0])
            return None
    return result

def getCompilations(examples, parentBuildPath = None, runtimeTestNames = None):
    """Return list of compilations from the examples
    
    parentBuildPath  -- Path to place the build folders in or none to not change the existing compilations
    runtimeTestNames -- If given, return only required compilation for runtime tests with given name
    """
    result = []
    if runtimeTestNames == None:
        for example in examples:
            result.extend(example.getCompilations(parentBuildPath))
    else:
        cfgs = []
        tests = getRuntimeTests(examples, runtimeTestNames)
        for test in tests:
            c = test.findCompilation(parentBuildPath)
            if not c.getConfig() in cfgs:
                cfgs.append(c.getConfig())
                result.append(c)
    return result

def getRuntimeTests(examples, names = None):
    """Return runtime tests from list of examples
    
    If names is given, return only the ones with the given name
    """
    result = []
    for example in examples:
        if names == None or (len(names) == 1 and names[0] == '+'):
            result.extend(example.getRuntimeTests())
        else:
            for test in example.getRuntimeTests():
                if test.name in names:
                    result.append(test)
    return result
    
class Example:
    """Represents an example with its different configurations and tests"""
    def __init__(self, folder, profileFile = None):
        """Create a new example from a folder. This must contain the documentation.yml
        
        folder      -- Folder containing the example (especially the documentation.yml)
        profileFile -- Profile file to be sourced into build/run env
        """
        docuFile = folder + "/documentation.yml"
        if(not os.path.isfile(docuFile)):
            raise Exception("File " + docuFile + " not found!")
        self.folder = folder
        self.profileFile = profileFile
        with open(docuFile, 'r') as stream:
            docu = yaml.safe_load(stream)
            
        self.metaData = docu['example']
        self.cmakeFlags = self.__queryCMakeFlags()
        self.compilations = self.__createCompilations(docu)
        self.runtimeTests = self.__createRuntimeTests(docu)
    
    def getFolder(self):
        """Return folder of example"""
        return self.folder
    
    def getMetaData(self):
        """Return MetaData as dictionary: name, short, author, description"""
        return self.metaData
    
    def getCMakeFlags(self):
        """Return list of CMake presets (each contains a string of flags)"""
        return self.cmakeFlags
    
    def getCompilations(self, parentBuildPath = None):
        """Return a list of compilations of this example
        
        If parentBuildPath != None then the build paths of the compilations are reset to this
        """
        if(parentBuildPath != None):
            for c in self.compilations:
                c.setParentBuildPath(parentBuildPath)
        return self.compilations
    
    def getRuntimeTests(self):
        """Return list of runtime tests"""
        return self.runtimeTests

    def __queryCMakeFlags(self):
        """Return a list of CMake flag strings as returned by the cmakeFlags shell script"""
        result = execCmd(self.folder + "/cmakeFlags -ll", True)
        if(result.result != 0):
            raise Exception("Could not get cmakeFlags: " + result.stderr.join("\n"))
        else:
            return [x for x in result.stdout if x.startswith('-D')]
    
    def __createCompilations(self, docu):
        """Create the compilations for this example as in the docu dictionary and return as a list"""
        result = []
        configs = []
        for compilation in docu['compile']:
            cmakePresets = expandList(compilation['cmakeFlags'])
            for cmakePreset in cmakePresets:
                if(cmakePreset < 0):
                    raise Exception("Invalid cmakePreset: " + str(cmakePreset))
                if(cmakePreset >= len(self.cmakeFlags)):
                    raise Exception("cmakePreset does not exist: " + str(cmakePreset))
                c = Compilation(self, cmakePreset, profileFile = self.profileFile)
                if(c.getConfig() in configs):
                    print("Skipping duplicate config " + str(c.getConfig()) + " in " + self.metaData['name'])
                else:
                    result.append(c)
                    configs.append(c.getConfig())
        return result
    
    def __createRuntimeTests(self, docu):
        """Create the runtime tests for this example as in the docu dictionary and return as a list"""
        if docu.get('tests') == None:
            return []
        result = []
        for test in docu['tests']:
            result.append(RuntimeTest(self, test, self.profileFile))
        return result
    
