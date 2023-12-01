import re
import random
import openai


class WorldObject:
    def __init__(
            self,
            templates,
            textGenerator,
            objectName,
            objects=None,
            cfg=None,
            customTemplate=None,
            verbose=False):

        if cfg is None:
            cfg = {
                "genTextAmount_min": 15,
                "genTextAmount_max": 30,
                "no_repeat_ngram_size": 8,
                "repetition_penalty": 2.0,
                "MIN_ABC": 4,
                "num_beams": 8,
                "temperature": 1.0,
                "MAX_DEPTH": 5
            }
        self.cfg = cfg

        self.textGenerator = textGenerator

        self.verbose = verbose

        self.objectName = objectName
        self.templates = templates
        if objects is None:
            self.objects = {}
        else:
            self.objects = objects

        # need to intialize this here because of *ugh*
        self.object = {}

        if customTemplate is not None and len(customTemplate.strip()) > 0:
            self.template = customTemplate
        else:
            self.template = templates[objectName]

        self.orig_template = self.template

        self.filledTemplate = self.fillTemplate(self.template)

        self.hiddenStates = None

        if self.verbose:
            print("GOT FILLED TEMPLATE", self.objectName,
                  "\n\n", self.filledTemplate, "\n\n")

        self.object = self.parseTemplate(self.filledTemplate)

    def fillTemplate_GPT3(self, sceneDescription, messages=[]):

        templatePrompt = """

Template Genie is a system that outputs text to fill in a template

This system always returns a single line of text, which best fills in the area
 indicated by the symbol <TEXT> in the template
 
The user may give multiple example templates, but final template always contains
the <TEXT> symbol which must be filled in
 
The system NEVER returns more than one line of text,
 and the text consists only of the standard alphanumeric symbols 0-9a-z and the punctuation .,?!


For example:

user:

Fill in the following template

A list of names

description:
Jack

description:
Jill

Description:
<TEXT>

assistant:

Eve


The system NEVER refers to itself as an AI 
the system NEVER refuses to produce a screenplay, 
the system NEVER does anything other than produce a screenplay matching the user's description
the system NEVER refers to itself as "the system"
the system NEVER says it cannot generate a response

the system NEVER uses ""s ()'s {}'s []'s or nonstandard punctuation

"""

        messages = [
            {"role": "system", "content": templatePrompt},
        ] + messages + [
            {"role": "user", "content": "output text that should replace the <TEXT> symbol"},
            {"role": "user", "content": sceneDescription},
        ]

        #print("GPT3 INPUT", messages)

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )

        result = ''
        for choice in response.choices:
            result += choice.message.content

        return result

    def fillTemplateValid(self, sceneDescription, messages=[]):
        for i in range(3):
            result = self.fillTemplate_GPT3(sceneDescription, messages=[])
            # must be one line
            if len(result.split("\n")) == 1:
                return result

        raise Exception(sceneDescription)

    def gpt3GenerateText_turbo(self, textInput):
        input = textInput+"<TEXT>"
        result = self.fillTemplateValid(input)

        print("FOO\n'"+input+"'\nREFOO\n'"+result+"'\nDEFOO")

        return result

        '''
        #call gpt3 api
        MODEL = "gpt-3.5-turbo"
        response = openai.ChatCompletion.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "This system automatically completes text templates in the most logical way possible"},
                {"role": "user", "content": "Please complete the following template"},
                {"role": "user", "content": textInput}
            ],
            temperature=self.cfg["temperature"],
            max_tokens=self.cfg["genTextAmount_max"]
        )
        #return response['choices'][0]['content']
        result = ''
        for choice in response.choices:
            result += choice.message.content

        return result
        '''

    def gpt3GenerateText(self, textInput):

        if self.verbose:
            print("GPT3 INPUT", textInput)

        # print("about to die",len(textInput))

        # call gpt3 api
        completion = openai.Completion.create(
            # engine="text-davinci-003",
            #engine="text-curie-001",
            engine="gpt-3.5-turbo-instruct",
            prompt=textInput,
            stop="\n",
            max_tokens=self.cfg["genTextAmount_max"],
            frequency_penalty=self.cfg["repetition_penalty"],
            presence_penalty=self.cfg["repetition_penalty"],
            timeout=10
        )['choices'][0]['text']

        if self.verbose:
            print("GPT3 OUTPUT", completion)

        return completion

    def generateTextWithInput(self, textInput, depth=0):

        # make sure pipelien is in cuda
        # if not self.textGenerator["name"].startswith("GPT3"):
        #    self.textGenerator['pipeline'].model = self.textGenerator['pipeline'].model.cuda()

        if depth > self.cfg["MAX_DEPTH"]:
            return "error"

        trailingSpace = ""

        if self.textGenerator["name"] == "GPT3":
            # remove trailing space
            if textInput[-1] == " ":
                textInput = textInput[:-1]
                trailingSpace = " "
            result = self.gpt3GenerateText(textInput)
            lines = result.strip().split("\n")
        elif self.textGenerator["name"] == "GPT3-turbo":
            result = self.gpt3GenerateText_turbo(textInput)
            lines = result.strip().split("\n")
        else:
            input_ids = self.textGenerator['tokenizer'](
                textInput, return_tensors="pt").input_ids
            amt = input_ids.shape[1]
            result = self.textGenerator['pipeline'](
                textInput,
                do_sample=True,
                min_length=amt+self.cfg["genTextAmount_min"],
                max_length=amt+self.cfg["genTextAmount_max"],
                pad_token_id=50256,
                return_full_text=False,
                no_repeat_ngram_size=self.cfg["no_repeat_ngram_size"],
                repetition_penalty=self.cfg["repetition_penalty"],
                num_beams=self.cfg["num_beams"],
                temperature=self.cfg["temperature"]
            )

            lines = result[0]['generated_text'].strip().split("\n")

        # remove len()==0 lines
        lines = [line.strip() for line in lines if len(line.strip()) > 0]
        # make sure we have at least some output
        if len(lines) == 0:
            if self.verbose:
                print('no response', result, textInput)
            return self.generateTextWithInput(textInput, depth=depth+1)
        rv = lines[0]
        # remove non-ascii
        rv = rv.encode("ascii", errors="ignore").decode()

        if rv[:3] == "ick":
            print(textInput, result, rv)
            assert False

        # remove trailing ":"s
        if rv[-1] == ":":
            if self.verbose:
                print('trailing :', result)
            return self.generateTextWithInput(textInput, depth=depth+1)
        # ":"s should actually just never appear
        if ":" in rv:
            if self.verbose:
                print(': present', result)
            return self.generateTextWithInput(textInput, depth=depth+1)
        # anything that's all punctuation is also bad
        # rva = re.sub(r'\W+', '', rv)
        rva = re.sub(r'[^a-zA-Z]+', '', rv)
        if len(rva) < self.cfg["MIN_ABC"]:
            if self.verbose:
                print('non alphanumeric', result, self.cfg["MIN_ABC"])
            return self.generateTextWithInput(textInput, depth=depth+1)

        return rv+trailingSpace

    def fillTemplate(self, template):
        t = 0
        output = ""
        thisMatch = re.search("{[^}]*}", template)
        while thisMatch:
            start, end = thisMatch.span()
            obj_and_prop = template[t+start+1:t+end-1]

            output += template[t:t+start]

            gotProp = self.getObjwithProp(obj_and_prop, output)

            output += str(gotProp)

            if self.verbose == 3:
                print("MATCH", thisMatch, gotProp)

            t = t+end
            thisMatch = re.search("{[^}]*}", template[t:])

        output += template[t:]

        return output

    def parseTemplate(self, template):

        # clean up whitespace
        template = "\n".join([line.strip() for line in template.split("\n")])

        objects = template.split("\n\n")

        # trim blank lines from objects
        objects = ["\n".join([line for line in o.split(
            "\n") if len(line) > 0]) for o in objects]

        if self.verbose:
            print(objects)

        def countABC(s):
            sa = re.sub(r'[^a-zA-Z]+', '', s)
            return len(sa)

        startIndex = None

        for i, o in enumerate(objects):
            for line in o.split("\n"):
                if line == "#":
                    startIndex = i+1
                    break

        if self.verbose:
            print("start index", startIndex)

        objects = objects[startIndex:]

        # remove empty objects
        objects = [o for o in objects if len(o) > 0]

        # remove comments
        objects = [o for o in objects if not o.startswith("#")]

        if startIndex is None:
            thisObject = objects[-1]  # by default choose last object
        else:
            thisObject = random.choice(objects)

        self.chosenObject = thisObject

        output = {}
        propName = "NONE"
        for i, line in enumerate(thisObject.split("\n")):
            line = line.strip()
            # print(i,line)
            if line.endswith(":"):
                # print("here0")
                propName = line[:-1]
            else:
                # print("here1, propName=",propName)
                if propName != "NONE" and len(line) > 0:
                    if propName in output:
                        output[propName] += "\n"+line
                    else:
                        output[propName] = line

        # check for #NOREP pattern
        orig_template = self.orig_template
        if "#NOREP\n" in orig_template:
            lastObject = objects[-1]
            i = orig_template.index("#NOREP\n")+len("#NOREP\n")
            new_template = orig_template[:i]+"\n\n" + \
                lastObject+"\n\n"+orig_template[i:]

            # get rid of excess newlines
            new_template = re.sub("\n\n+\n", "\n\n", new_template)

            self.templates[self.objectName] = new_template

            # print("orig template is",new_template)
            # print("new template is",new_template)

            # I NEED some logic here to prevent the template from growing forever
            maximumTemplateSize = 1024
            e = 0
            while len(new_template) > maximumTemplateSize:
                e += 1
                print("TRIMMING TEMPLATE", self.objectName)
                # get the part after #NOREP
                i = new_template.index("#NOREP\n")+len("#NOREP\n")
                templatebeginning = new_template[:i]
                templateend = new_template[i:]

                if e > 3:
                    print("ERROR: template too big", self.objectName)
                    print(new_template, "===", templatebeginning,
                          "===", templateend, "===", objects)
                    break

                # split templateend into objects on ("\n\n")
                objects = templateend.split("\n\n")
                # remove a random object
                objects.pop(random.randint(0, len(objects)-2))
                # rejoin objects
                new_template = templatebeginning+"\n\n".join(objects)

                # get rid of excess newlines
                new_template = re.sub("\n\n+\n", "\n\n", new_template)

            self.templates[self.objectName] = new_template

        return output

    def getObjwithProp(self, obj_and_prop, output):

        overrides = None
        objType = None
        # handle ":"s
        if ":" in obj_and_prop:
            obj_and_prop, objType, overrides = obj_and_prop.split(":")

        # handle "."s
        propName = None
        if "." in obj_and_prop:
            objectName, propName = obj_and_prop.split(".")
        else:
            objectName = obj_and_prop

        if self.verbose == 2:
            print("checking for object", objectName, "in", self.objects)

        # handle saved objectsGPT
        if objectName in self.objects:
            thisObject = self.objects[objectName]

            if self.verbose == 2:
                print("about to die, looking for property",
                      propName, "in", objectName, "=", thisObject)

            if propName is not None:
                return thisObject.getProperty(propName)
            else:
                return thisObject

        # handle type text
        if objType == "TEXT" or obj_and_prop == "TEXT":
            if self.verbose == 2:
                print("generating text", objType,
                      obj_and_prop, "with template", output)

            if not self.textGenerator["name"].startswith("GPT3"):
                output = output.strip()  # remove trailing " "s
            # output = self.generateTextWithInput(output)
            text = self.generateTextWithInput(output)
            if objectName != "TEXT" and propName is None:
                if self.verbose:
                    print("storing text", objectName, text)
                self.objects[objectName] = text
            return text
        else:
            if self.verbose:
                print("got prop", objectName, propName, objType, overrides)
            thisObject = self.getObject(objectName, objType, overrides)
            if propName is not None:
                return thisObject.getProperty(propName)
            else:
                return thisObject

    def getObject(self, objectName, objType, overrides=None):
        if objectName in self.objects:
            return self.objects[objectName]
        else:
            # handle overrides
            objects = None
            if overrides:
                # parse overrides "a=b,c=d,..."
                objects = {}
                for override in overrides.split(","):
                    k, v = override.split("=")
                    gotV = None
                    if "." in v:
                        i = v.index(".")
                        v0 = v[:i]
                        v1 = v[i+1:]
                        gotV = self.objects[v0].getProperty(v1)
                    else:
                        if v in self.objects:
                            gotV = self.objects[v]
                    if gotV:
                        objects[k] = gotV
                    else:
                        print("this should never happen!", v, self.objects)
            # remove trailing digits
            if objType is None:
                objType = re.sub(r'\d+$', '', objectName)
            # generate object
            thisObject = WorldObject(self.templates, self.textGenerator, objType, objects=objects,
                                     cfg=self.cfg,
                                     verbose=self.verbose)
            # store for future use
            if self.verbose:
                print("storing object", objectName, thisObject)
            self.objects[objectName] = thisObject
            return self.objects[objectName]

    def has(self, propName):
        if propName in self.objects:
            return True
        if propName in self.object:
            return True
        return False

    def getProperty(self, propName):

        # todo, handle multiple "."s
        if "." in propName:
            i = propName.index(".")
            v0 = propName[:i]
            v1 = propName[i+1:]
            if self.verbose == 3:
                print("getting sub-property", v0, v1)
            return self.getProperty[v0].getProperty(v1)

        if self.verbose == 3:
            print("getting property", propName, "from object", self.object)
        if propName in self.objects:
            return self.objects[propName]
        if propName in self.object:
            return self.object[propName]
        print("error in", self.__repr__(), "\nmissing property:", propName)
        raise ValueError("property not found!")

    def __repr__(self):
        '''s = self.filledTemplate.split("\n\n")
        # remove empty lines
        v = ["\n".join([line for line in lines.split(
            "\n") if len(line.strip()) > 0]) for lines in s]
        v = [x for x in v if len(x) > 0]
        r = v[-1]
        return "<world object:%s>\n" % self.objectName+r'''
        return "<world object:%s>\n" % self.objectName+self.chosenObject

    def __str__(self):
        # try:
        if self.has("description"):
            return str(self.getProperty("description")).strip()
        # except:
        else:
            return self.__repr__()


class ListObject:
    def __init__(
        self,
        templates,
        textGenerator,
        objectName,
        n=3,
        thisList=None,
        uniqueKey=None,
        objects=None,
        cfg=None,
        verbose=False
    ):

        self.objectName = objectName
        self.n = n

        uniqueKeys = set()

        if thisList is not None:
            self.thisList = thisList
        else:
            self.thisList = []

            # build up list if not provided
            while len(self.thisList) < n:
                newObject = WorldObject(
                    templates,
                    textGenerator,
                    objectName,
                    objects=objects,
                    cfg=cfg,
                    verbose=verbose)

                if uniqueKey is None:
                    self.thisList += [newObject]
                else:
                    thisKey = str(newObject.getProperty(uniqueKey))
                    if thisKey not in uniqueKeys:
                        self.thisList += [newObject]

        # list for random access
        self.randomOrder = list(range(self.n))
        random.shuffle(self.randomOrder)

    def getProperty(self, propName):
        # item
        if propName.startswith("ITEM"):
            whichItem = int(propName[4:])
            return self.thisList[whichItem]

        if propName == "RANDOM":
            return self.thisList[self.randomOrder[0]]
