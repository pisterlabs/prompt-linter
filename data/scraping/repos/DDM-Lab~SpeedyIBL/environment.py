import re

class Environment(object):

    """ Environment """
    def __init__(self, flags):
        self.__name = flags.environment

        if 'MINIMAP' in self.name:
            params = self.name.split('_')
            if 'V' in params[1]:
                version = int(re.sub("[^0-9]", "", params[1]))
                from env.minimap.minimapworld import MAPWORLD
                self.__env = MAPWORLD(version)
                
            else:
                raise ValueError('Invalid environment string format for CMOTP')
        elif 'PREDATOR' in self.name:
            params = self.name.split('_')
            if 'V' in params[1]:
                version = int(re.sub("[^0-9]", "", params[1]))
                from env.predator.predator import PREDATOR
                self.__env = PREDATOR(version)
                
            else:
                raise ValueError('Invalid environment string format for PREDATOR')
        elif 'FIREMAN' in self.name:
            params = self.name.split('_')
            if 'V' in params[1]:
                version = int(re.sub("[^0-9]", "", params[1]))
                from env.apprenticefiremen.fireman import FIREMEN
                self.__env = FIREMEN(version)
                
            else:
                raise ValueError('Invalid environment string format for PREDATOR')
        elif 'NAVIGATION' in self.name:
            params = self.name.split('_')
            if 'V' in params[1]:
                version = int(re.sub("[^0-9]", "", params[1]))
                from env.navigation.navigation import NAVIGATION
                self.__env = NAVIGATION(version)
                
            else:
                raise ValueError('Invalid environment string format for PREDATOR')
        elif 'MISPACMAN' in self.name:
            from env.misPacman.misPacman import MISPACMAN
            self.__env = MISPACMAN()

        elif 'COMMON' in self.name:
            from env.commonInterest.commonInterest import COMMON
            self.__env = COMMON()
        # else:
        #     from env.openai_gym.openai_gym import OpenAI_Gym
        #     self.__env = OpenAI_Gym(flags)
        #     self.__upper_bound = self.env.upper_bound
        #     self.__lower_bound = -self.env.lower_bound
        self.__fieldnames = self.__env.fieldnames
        self.__dim = self.env.dim
        self.__out = self.env.out

    def getHW(self):
        return self.__env.getHW()

    def getSaliencyCoordinates(self):
        return self.__env.getSaliencyCoordinates()

    def getAgentCoordinates(self):
        return self.__env.getAgentCoordinates()

    def processSaliency(self, saliency, folder):
        self.__env.processSaliency(saliency, folder)

    def processSaliencyCoordinates(self, saliency, folder):
        self.__env.processSaliencyCoordinates(saliency, folder)

    @property
    def upper_bound(self):
        return self.__upper_bound

    @property
    def lower_bound(self):
        return self.__lower_bound

    @property
    def dim(self):
        return self.__dim

    @property
    def out(self):
        return self.__out

    @property
    def name(self):
        return self.__name

    @property
    def env(self):
        return self.__env

    @property
    def fieldnames(self):
        return self.__fieldnames

    def evalReset(self, evalType):
        '''
        Reset for evaulations
        '''
        return self.__env.evalReset(evalType)

    def reset(self):
        '''
        Reset env to original state
        '''
        return self.__env.reset()

    def render(self):
        '''
        Render the environment
        '''
        self.__env.render()

    def step(self, a):
        '''
        :param float: action
        '''
        return self.__env.step(a)

    def stats(self):
        '''
        :return: stats from env
        '''
        return self.__env.stats()
    
    def result(self):
        '''
        :return: stats from env
        '''
        return self.__env.result()
