import re
import os

class ParamParser:
    """Class for parsing param files from picongpu (simplified C++ header parser)"""
    def __init__(self, verbose = False):
        self.verbose = verbose
        self.defines = {}
        
    def log(self, text, level = 0):
        """Log to stdout if verbose is turned on. Level is the indendation level"""
        if self.verbose:
            print("  " * level + text)
            
    def throwError(self, text):
        """Raise an exception with the given text. Current filePath is automatically added"""
        raise Exception(text + " (" + self.filePath + ")")

    def cleanLine(self, line, curScopeInput, level):
        """Return a cleaned line.
        
        Put multiline comments on 1 line
        Remove #pragma/#include
        Replace double whitespaces and whitespaces on either end of the line
        """
        if "/*" in line and not "*/" in line:
            self.log("Multiline comment start: " + line, level)
            line = self.parseMultilineComment(line, curScopeInput)
        # Skip those
        if "#pragma" in line or "#include" in line:
            return ""
        line = line.strip()
        # Collapse multiple whitespaces
        line = re.sub(r"\t|(\s\s+)", " ", line)
        return line
    
    def parseMultilineComment(self, line, curScopeInput):
        """Return the full multiline comment"""
        for newLine in curScopeInput:
            newLine = newLine.strip()
            if newLine.startswith('*') and not newLine.startswith('*/'):
                newLine = newLine[1:].lstrip()
            self.log("Adding to multiline comment: " + newLine)
            line = line + "\n" + newLine
            if "*/" in newLine:
                break
        return line
    
    def parseMultilineAssignment(self, line, curScopeInput, level):
        """Return the full assignment when it is split among multiple lines
        
           Search till semicolon
           Detect descriptions inbetween
           Return (line, descriptions[])
        """
        descriptions = []
        for newLine in curScopeInput:
            newLine = newLine.strip()
            newLine = self.cleanLine(newLine, curScopeInput, level)
            if not newLine:
                continue
            if self.matchDescription(newLine, level):
                descriptions.append(self.curDescription)
                continue
            line += newLine
            if newLine.endswith(";"):
                return line, descriptions
            if ';' in newLine:
                self.throwError("Unexpected ';' in line: " + newLine)
        self.throwError("Did not found end of assignment")
        
    def parseFunction(self, line, curScopeInput, level):
        """Parse something that looks like a function. Adds everything till a closing brace or semicolon"""
        if line.endswith(';'):
            return line
        elif line.endswith('{'):
            openBraces = 1
        else:
            openBraces = 0
        for newLine in curScopeInput:
            newLine = newLine.strip()
            newLine = self.cleanLine(newLine, curScopeInput, level)
            if not newLine:
                continue
            line += newLine
            if newLine.endswith("{"):
                openBraces += 1
            elif newLine == "}":
                openBraces -= 1
                if openBraces == 0:
                    return line
                elif openBraces < 0:
                    self.throwError("Unexpected '}'")
            elif newLine.endswith(";") and openBraces == 0:
                return line
    
    @staticmethod
    def addDescription(node, mainDescr, subDescriptions = None):
        """Add description meta data by combining the main description
        and possible sub descriptions (e.g. from assignments)"""
        if mainDescr:
            descr = [mainDescr]
        else:
            descr = []
        if subDescriptions:
            descr.extend(subDescriptions)
        descr = '\n'.join(descr)
        if descr:
            node['__description'] = descr + node.get('__description', '')
    
    def parseAssignment(self, line, data, curScopeInput, level):
        """Parse a line with an assignment (value or type). Return the name of the target"""
        description = self.curDescription
        self.curDescription = None
        if not line.endswith(";"):
            if ';' in line:
                self.throwError("Unexpected ';' in line: " + line)
            line, descriptions = self.parseMultilineAssignment(line, curScopeInput, level)
        else:
            descriptions = None
        # Apply defines
        for define, value in self.defines.items():
            line = line.replace(define, value)
        using = re.match(r"(using|namespace) (\w+) ?= ?([^;]+) ?;$", line)
        if using:
            typedefType = using.group(1)
            name = using.group(2)
            value = using.group(3)
            self.log("Found " + typedefType + ": " + name + " = " + value, level)
            if typedefType == "using":
                typedefType = "Type"
            else:
                typedefType = "Namespace"
            data[name] = {"value": value, "__type": typedefType}
            self.addDescription(data[name], description, descriptions)
            return name
        variable = re.match(r"((static )?constexpr|BOOST_STATIC_CONSTEXPR|BOOST_CONSTEXPR_OR_CONST) (\w+) (\w+) ?=(.*);", line)
        if variable:
            varType = variable.group(3)
            name = variable.group(4)
            value = variable.group(5).strip()
            self.log("Found variable of type " + varType + ": " + name + " = " + value, level)
            data[name] = {"value": value, "__type": varType}
            self.addDescription(data[name], description, descriptions)
            return name
        self.throwError("Unknown assignment: " + line)

    def matchDescription(self, line, level):
        """Try to match a description on the line and add set self.curDescription
            Return True if matched"""
        # Doc comment
        description = re.match(r"/\*(\*|!)(.*)\*/$", line, re.DOTALL)
        if description:
            self.curDescription = description.group(2).strip()
            self.log("New description: " + self.curDescription, level)
            return True
        else:
            return False
        
    def parseScope(self, data, curScopeInput, level = 0):
        """Parse a given scope till its end"""
        self.curDescription = None
        ignoreLines = False
        ifStack = []
        for line in curScopeInput:
            line = self.cleanLine(line, curScopeInput, level)
            if not line:
                continue
            self.log("Line: " + line, level)
            if line == "#endif":
                ignoreLines = ifStack.pop()
                self.log("Found #endif -> Ignoring = " + str(ignoreLines))
                continue
            if ignoreLines:
                if line.startswith("#if"):
                    self.log("Found #if in ignored block")
                    ifStack.append(True)
                else:
                    self.log("Ignored...")
                continue
            if line == "};" or line == "}":
                self.log("End of scope", level)
                return
            ifndef = re.match("#ifndef (\w+)$", line)
            if ifndef:
                name = ifndef.group(1)
                self.log("Found #ifndef: " + name, level)
                ifStack.append(False)
                ignoreLines = name in self.defines
                continue
            define = re.match(r"#define (\w+) (.*[^\\])$", line)
            if define:
                name = define.group(1)
                value = define.group(2)
                self.log("Found define: " + name + " = " + value, level)
                self.AddDefine(name, value)
                continue
            defineFunc = re.match(r"#define (\w+)\(.*[^\\]$", line)
            if defineFunc:
                name = defineFunc.group(1)
                self.log("Removed preprocessor function: " + name, level)
                continue
            scopeStart = re.match(r"(namespace|struct|class) (\w+) ?(\{|;)$", line)
            if scopeStart:
                newScope = scopeStart.group(2)
                self.log("New scope: " + newScope, level)
                if not newScope in data:
                    data[newScope] = {"__type": "Scope"}
                self.addDescription(data[newScope], self.curDescription)
                self.curDescription = None
                if scopeStart.group(3) == '{':
                    self.parseScope(data[newScope], curScopeInput, level + 1)
                continue
            if self.matchDescription(line, level):
                continue
            if "=" in line:
                asgnName = self.parseAssignment(line, data, curScopeInput, level)
                continue
            valueId = re.match(r"value_identifier\((\w+), ?(\w+), ?([^,])+\);$", line)
            if valueId:
                valType = valueId.group(1)
                name = valueId.group(2)
                value = valueId.group(3)
                self.log("Found value_identifier: " + name + " = " + value + "(" + valType+ ")", level)
                if not "__valID" in data:
                    data["__valID"] = {}
                data["__valID"][name] = {"value": value, "__type": valType}
                self.addDescription(data["__valID"][name], self.curDescription)
                self.curDescription = None
                continue
            using = re.match(r"using ([^\;]+);$", line)
            if using:
                self.log("Ignored using " + using.group(1), level)
                continue
            if re.match(r"template ?<[^>]+>$", line):
                self.log("Ignored template", level)
                continue
            alias = re.match(r"alias\((\w+\));$", line)
            if alias:
                name = alias.group(1)
                self.log("Found alias " + name, level)
                if not "__alias" in data:
                    data["__alias"] = {}
                data["__alias"][name] = {}
                self.addDescription(data["__alias"][name], self.curDescription)
                self.curDescription = None
                continue
            functor = re.match(r"(\w+) operator\(\)\(\) ?\{ ?return ([^;]+); ?}", line)
            if functor:
                fType = functor.group(1)
                value = functor.group(2)
                self.log("Found functor: " + value + "(" + fType + ")", level)
                data['__functorType'] = fType
                data['value'] = value
                continue
            if line.startswith("DINLINE") or line.startswith("HINLINE") or line.startswith("HDINLINE"):
                self.parseFunction(line, curScopeInput, level)
                self.log("Ignored function", level)
                continue
            if line == 'private:' or line == 'protected:' or line == 'public:':
                continue
            if line.startswith('PMACC_ALIGN') or re.match("(const )?\w+ \w+(,\w+)*;", line):
                # Ignore variable declaration
                continue

            self.throwError("Unknown statement: " + line)

    def parse(self, filePath, data):
        self.data = data
        self.curNamespaceName = ""
        data['__type'] = 'Scope'
        self.filePath = filePath
        with open(self.filePath, 'r') as paramFile:
            content = paramFile.read()
            # Replace "escaped newline" by nothing
            content = re.sub(r"\\\n", "", content)
            # Remove comments
            content = re.sub("//.*", "", content)
            content = re.sub(r"/\*[^\*!].*?\*/", "", content, flags = re.DOTALL)
            # Docstrings should end the line -> Place rest on new line
            content = re.sub(r"\*/(.)", r"*/\n\1", content)
            # Remove empty lines
            content = re.sub(r"^\s*\n", "", content)
            # Place opening braces on own lines onto previous line
            content = re.sub(r"\n\s*\{", "{", content)
            # Place closing braces on new line
            content = re.sub(r"([^\s\n])\}", r"\1\n}", content)
            # Typedefs -> using
            content = re.sub(r"typedef ([^;]+) (\w+);", r"using \2 = \1;", content, flags = re.DOTALL)
            # "#   ifdef" -> "#ifdef"
            content = re.sub(r"#\s+", "#", content)
            content = re.sub(r"(\w+)\s+operator\(\)\(\)\s*\{\s*([^\}\n]+)\s*\}", r"\1 operator()(){ \2 }", content, flags = re.DOTALL)
            contentArray = content.split('\n')
            self.parseScope(data, iter(contentArray))
            
    def AddDefine(self, name, value):
        """Add a preprocessor definition"""
        self.defines[name] = value
    
    def ParseFolder(self, paramDir):
        """Parse all param files of a folder into one dictionary"""
        paramFiles = [os.path.join(paramDir, fileName) for fileName in os.listdir(paramDir) if fileName.endswith(".param")]
        data = {}
        for filePath in paramFiles:
            self.parse(filePath, data)
        return data
    
    ######################
    # Evaluation functions
    ######################
    
    def SetCurNamespace(self, namespaceName):
        """Set the namespace name in which all future Get* calls will search"""
        if namespaceName:
            self.curNamespaceName = namespaceName + "::"
        else:
            self.curNamespaceName = ''
    
    def GetNode(self, name):
        if not name.startswith("::"):
            name = self.curNamespaceName + name
        return GetNode(name, self.data)
        
    def GetValue(self, name):
        node = self.GetNode(name)
        return node['value']
        
    def GetNumber(self, name):
        if not name.startswith("::"):
            name = self.curNamespaceName + name
        return GetNumber(name, self.data)
        
    def GetVector(self, name):
        if not name.startswith("::"):
            name = self.curNamespaceName + name
        return GetVector(name, self.data)
        
def GetNode(name, mainScope):
    """Return the node in the global scope by its name (Foo::bar)"""
    if name.startswith("::"):
        name = name[2:]
    names = name.split("::")
    curScope = mainScope
    for curName in names:
        if curName in curScope:
            curScope = curScope[curName]
        else:
            return None
    return curScope

def GetNodeInScope(name, scopeName, mainScope):
    """Return (nodeName, node) in a named scope by its name
    
    Examle: Bar::foobar in Foo::Foo2 might be
            Foo::Foo2::Bar::foobar, Foo::Bar::foobar or ::Bar::foobar
    """
    if name.startswith("::"):
        return GetNode(name[2:], mainScope)
    value = GetNode(scopeName + "::" + name, mainScope)
    while value == None:
        if not "::" in scopeName:
            return (None, None)
        scopeName = scopeName.rsplit("::", 1)[0]
        value = GetNode(scopeName + "::" + name, mainScope)
    return (scopeName + "::" + name, value)

def GetNodeFromMemberScope(name, scopeMemberName, mainScope):
    """Return (nodeName, node) starting the search in the scope of another member"""
    scopeName = scopeMemberName.rsplit("::", 1)[0] if "::" in scopeMemberName else ""
    return GetNodeInScope(name, scopeName, mainScope)

def GetNumber(name, mainScope, node = None, lvl = 0):
    """Get numeric value from a node and/or name. Resolve variable references and simple arithmetics"""
    # Get node if not given
    if not node:
        node = GetNode(name, mainScope)
        if not node:
            return None
    # Detect type
    valType = node['__type']
    if valType.startswith("float"):
        isFloat = True
    elif valType.startswith("int") or valType.startswith("uint") or valType.startswith("unsigned") or valType == "size_t":
        isFloat = False
    else:
        return None
    value = node['value']
    # Check if we already have a number
    try:
        return float(value) if isFloat else int(value)
    except ValueError:
        pass
    # Limit recursion level
    if lvl > 3:
        return None
    # Support simple arithmetics
    # Regexp for floats or ints
    numRegExp = r"(\+|-)?\d+(\.\d+)?(e(\+|-)\d+)?" if isFloat else r"(\+|-)?\d+"
    # Detect 'x + y' style
    expr = re.match("(?P<Op1>" + numRegExp + r"|[\w:]+) ?(?P<Op>[\+\-\*/]) ? (?P<Op2>" + numRegExp + r"|[\w:]+)", value)
    if expr:
        op1 = expr.group("Op1")
        op = expr.group("Op")
        op2 = expr.group("Op2")
        try:
            op1 = float(op1) if isFloat else int(op1)
        except ValueError:
            subName, value = GetNodeFromMemberScope(op1, name, mainScope)
            op1 = GetNumber(subName, mainScope, value, lvl + 1)
        try:
            op2 = float(op2) if isFloat else int(op2)
        except ValueError:
            subName, value = GetNodeFromMemberScope(op2, name, mainScope)
            op2 = GetNumber(subName, mainScope, value, lvl + 1)
        if op == '+':
            return op1 + op2
        if op == '-':
            return op1 - op2
        if op == '*':
            return op1 * op2
        if op == '/':
            return op1 / op2
        return None
    subName, value = GetNodeFromMemberScope(value, name, mainScope)
    return GetNumber(subName, mainScope, value, lvl + 1)

def GetVector(name, mainScope):
    """Get a PMacc compiletime vector type as a python array"""
    value = GetNode(name, mainScope)
    if value['__type'] != "Type":
        return None
    vec = re.match(r"(PMacc::)?math::CT::(\w+)< ?([^,]+)(, ?([^,]+)(, ?([^,]+))) ?>$", value['value'])
    if not vec:
        return None
    result = [vec.group(3)]
    if vec.group(5):
        result.append(vec.group(5))
        if vec.group(7):
            result.append(vec.group(7))
    if "Int" in vec.group(2):
        return [int(x) for x in result]
    else:
        return [float(x) for x in result]
    
def ParseParamFolder(paramDir):
    """Parse all param files of a folder into one dictionary"""
    parser = ParamParser()
    return parser.ParseFolder(paramDir)

