import re

class EnvironmentSetup:
    """Represents how the environment for an example should be set up (modules and variables)"""
    def __init__(self, env):
        """Create a new instance out of a dictionary using "modules" and "variables" if present"""
        self.modules = []
        self.variables = []
        moduleList = env.get("modules")
        if(moduleList != None):
            for module in moduleList:
                # Match 'Foo123', 'Foo/1', 'Foo2/1.p1'
                if(re.match("^\w+(/\w+(\.\w+)*)?$", module) == None):
                    raise Exception("Invalid module name: " + module)
                else:
                    self.modules.append(module)
        variableList = env.get("variables")
        if(variableList != None):
            for variable in variableList:
                # Match 'Foo=Bar', 'Foo='
                decl = re.match("^\w+=(.*)$", variable)
                if(decl == None):
                    raise Exception("Invalid variable declaration: " + variable)
                elif(decl.group(1) != "" and re.match("^(\".*?\")|('.*?')$", decl.group(1)) == None):
                    raise Exception("Only single string assignments are allowed: " + variable)
                else:
                    self.variables.append(variable)
    
    def getSetupCmd(self):
        """Return the command that sets up the environment"""
        result = ''
        for module in self.modules:
            result += "module load " + module + "\n"
        for var in self.variables:
            result += "export " + var + "\n"
        return result
