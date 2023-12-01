import sys
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pandas as pd
import pcse
import datetime


class SWATEnv(gym.Env):
    """Custom environment derived from OpenAI gym.Env and based on the Texas A&M SWAT model and the CropGym environment.
    
    ### Description
    The goal is to manage a crop's growth efficiently, given fertilizer and water inputs, from seedling emergence to harvest, 
    such that a maximum yield can be obtained.

    ### Observation Space (Continuous)
    The observation is a `ndarray` with shape `(14,)` where the elements correspond to the following:

        Type: Box(14)

        Num   Observation                     Min       Max         Unit
        1     precipitation                   0.0       inf         mm
        2     ref. evapotranspiration         0.0       inf         mm
        3     actual evapotranspiration       0.0       inf         mm
        4     soil water content              0.0       inf         mm
        5     daily runoff curve number       0.0       1.0         -
        6     avg air temperature             -inf      inf         °C
        7     daily solar radiation           0.0       inf         MJ/mm^2
        8     denitrification                 0.0       inf         kg/ha
        9     nitrogen uptake                 0.0       inf         kg/ha
        10    num. water stress days          0.0       inf         days
        11    num. temp stress days           0.0       inf         days
        12    num. nitrogen stress days       0.0       inf         days
        13    total plant biomass             0.0       inf         kg/ha
        14    leaf area index                 0.0       inf         -

    ### Action Space (Continuous)
    The action is a `ndarray` with shape `(2,)`, representing the fertilizer amount and irrigation amount applied on the crops at a timestep.

        Type: Box(2)
        Num          Action             Min       Max         Unit
        1       fertilizer amount       0.0       inf         kg/ha
        2       irrigation amount       0.0       inf         mm
    
    ### Reward Function
    Reward for each action is a function of the yield (YLD), fertilizer usage and irrigation applied.
    R = yield - alpha*fert_amnt - beta*water_amnt, where alpha and beta are penalties related to costs of input operation.

    ### Transition Dynamics
    Given an action, the crop transitions to a new state based on the dynamics in the [SWAT 2009 handbook](https://swat.tamu.edu/media/99192/swat2009-theory.pdf)
    and with climatic inputs from [PCSE](https://pcse.readthedocs.io/en/stable/)

    ### Other
    Starting State:
        Each episode begins with the user-specified state or random initialization. 
        Default location is set to Temple, Texas, USA.

    Episode Termination:
        Either when target yield is reached or plant reaches maturity or maximum number of steps 
        (default is equivalent to a full growing season for corn == 120 days) is reached.
    
    ### Usage
    Create the env as follows:
    ```
    from envs.swat_gym import SWATEnv
    env = SWATEnv(max_action=x, seed=None, latitude = 31.0565725, longitude = -97.34756354, elevation = 206)
    ```
    with x being the permissible upper limit for the action space. A seed may also be set for experimentation, along with specific location information.
    """
    
    metadata = {'render.modes': ['human']}

    def __init__(self, max_action=60.0, seed=None, latitude = 31.0565725, longitude = -97.3497522, elevation = 206):
        # gym params
        
        self.action_space = spaces.Box(low=0, high=max_action, shape=(2,), dtype=np.float64)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float64)

        # simulation params
        self.max_duration = 120
        self.start_date = datetime.datetime(2021, 4, 15)
        self.end_date = datetime.datetime(2021, 8, 13)

        # location params 
        self.latitude = latitude
        self.longitude = longitude
        self.elevation = elevation

        # management variables
        self.alpha = 2.43 # fertilizer penalty, i.e. fertilizer unit price $/kg
        self.beta = 0.16   # irrigation penalty, i.e. irrigation water unit price, $/m^3, https://doi.org/10.1073/pnas.2005835117

        # initialize environment
        self.init_weather()
        self.init_crop()
        self.init_soilwater()
        self.init_management()
        self._reset()
        self._seed(seed)

        
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """
        This method is the primary interface between environment and agent.

        Paramters: 
            action: array containing fertilizer and irrigation amounts for current day

        Returns:
            output: (array, float, bool, array)
                    information provided by the environment about its current state:
                    (observation, reward, done, info)
        """

        # update cumulative action
        self.fertilizer = action[0]     # kg/day/ha
        self.irrigation = action[1]     # mm/day/ha
        self.cumulative_fertilizer += self.fertilizer
        self.cumulative_irrigation += self.irrigation

        # update current state
        crop_yield, state = self._get_state(day=self.current_date)

        # compute reward
        reward = self.reward_function(crop_yield, action)

        # decide if episode successfully terminates
        is_harvest_time = self.current_date >= self.end_date #or self.frPHU >= 1.0
        is_terminal_action = self.cumulative_fertilizer >= self.max_fertilizer or self.cumulative_irrigation >= self.max_irrigation
        done = is_harvest_time #or is_terminal_action

        # add extra info
        info = [self.current_date, self.cumulative_fertilizer, self.cumulative_irrigation, self.total_HU, self.cumulative_bio, self.total_crop_yld, self.swc, self.nitrogen_level]

        # Record history
        self.time_hist.append(self.current_date)
        # climate 
        self.precip_history.append(self.precip)
        self.solar_rad_hist.append(self.solar_rad)
        self.temp_hist.append(self.avg_temp)

        # crop 
        self.crop_height_hist.append(self.h_c)
        self.crop_lai_hist.append(self.LAI)
        self.biomass_hist.append(self.plant_biom)
        self.yield_hist.append(crop_yield)
        self.HI_hist.append(self.HI)
        self.frPHU_hist.append(self.frPHU)
        self.act_Et_hist.append(self.act_transpiration)
        self.act_Etc_hist.append(self.act_evapotranspiration)
        self.water_stress_hist.append(self.w_stress)
        self.nitrogen_stress_hist.append(self.n_stress)
        self.temp_stress_hist.append(self.temp_stress)

        # soil and hydrology 
        self.n_uptake_hist.append(self.n_uptake)
        self.nitrogen_level_hist.append(self.nitrogen_level)
        self.soil_water_balance_hist.append(self.swc)
        self.runoff_hist.append(self.runoff)
        self.water_uptake_hist.append(self.water_uptake)

        # management
        self.fert_hist.append(self.fertilizer)
        self.irrig_hist.append(self.irrigation)
        self.reward_hist.append(reward)
        self.done_hist.append(done)
        
        # update date to next day
        self.current_date += datetime.timedelta(days=1)

        return state, reward, done, info

    def reset(self, seed=None):
        """
        This method resets the environment to its initial values.

        Returns:
            observation:    array
                            the initial state of the environment
        """
        # Initialize the RNG if the seed is manually passed
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)
        
        # Initialize the weather generator, crop, soil, and hydrology modules
        self.init_weather()
        self.init_crop()
        self.init_soilwater()
        self.init_management()

        # Reset env history
        self._reset()

        state = [self.precip, self.ref_et, self.act_transpiration, self.swc, self.daily_RCN, self.avg_temp, self.solar_rad, self.denitrif, self.n_uptake, self.w_stress, self.temp_stress, self.n_stress, self.plant_biom, self.LAI]
        reward = 0
        done = False
        info = [self.current_date, self.cumulative_fertilizer, self.cumulative_irrigation, self.total_HU, self.cumulative_bio, self.total_crop_yld, self.swc, self.nitrogen_level]
        return state, reward, done, info

    def render(self, mode='human', close=False):
        """
        This methods provides the option to render the environment's behavior to a window 
        which should be readable to the human eye if mode is set to 'human'.
        """
        pass

    def close(self):
        """
        This method provides the user with the option to perform any necessary cleanup.
        """
        pass

    def init_weather(self, path=None):
        self.current_date = self.start_date
        location = (self.latitude, self.longitude)

        if path is not None:
            self.avg_temp = 0
            self.solar_rad = 0
            self.avg_vapor_pressure = 0
            self.precip = 0
            self.ref_et = 0
            self.soil_evap = 0
        else:
            self.weatherdataprovider = pcse.db.NASAPowerWeatherDataProvider(*location) # PCSE weather data provider
            weather = self._get_weather_day(self.weatherdataprovider, self.start_date - datetime.timedelta(days=1))
            self.avg_temp = weather[0]
            self.solar_rad = weather[1]/1e6 # convert to MegaJoules
            self.avg_vapor_pressure = weather[2]
            self.precip = weather[3]*10 # convert cm to mm
            self.ref_et = weather[4]*10
            self.total_ref_et = 0
            self.soil_evap = weather[5]*10

    def init_crop(self):
        # temperature params
        self.T_base = 8
        self.T_opt = 25

        self.kl = 0.65          # light extinction coefficient  
        self.h_c = 0            # initial canopy height # 5:2.1.14, p323
        self.max_h_c = 2.5      # max canopy height for CORN
        self.LAI = 0            # leaf area index
        self.LAI_mx = 3         # max LAI
        self.fr_LAI_sen = 0.90  # leaf senescence
        self.fr_root = 0        # daily root biomass fraction
        self.max_root_depth = 2000.0 # RDMX in SWAT (mm)
        self.growth_factor = 1  # plant growth factor
        
        self.act_transpiration = 0  # actual plant transpiration
        self.act_evapotranspiration = 0  # actual evapotranspiration
        fr_PHU1 = 0.15
        fr_PHU2 = 0.50
        fr_LAI1 = 0.05
        fr_LAI2 = 0.95

        # variables to compute self.max_fr_lai, fraction of the plant's max LAI: eqn 5:2.1.10-13
        x = np.log(np.abs(fr_PHU1/fr_LAI1 - fr_PHU1))
        self.l2 = (x - np.log(np.abs(fr_PHU2/fr_LAI2 - fr_PHU2)))/(fr_PHU2 - fr_PHU1)
        self.l1 = x + self.l2 * fr_PHU1

        # Biomass production params for CORN
        self.RUE = 3.9          # kg/ha/(MJ/m^2) or 10^-1 gMJ-1, I/O Table A-5, OR https://digitalcommons.unl.edu/cgi/viewcontent.cgi?article=1092&context=agronomyfacpub
        self.HU = 0             # initial fraction heat units
        self.PHU = 1456         # based on Iowa example, ST2009, p309
        self.total_HU = 0       # compute actual PHU
        self.frPHU = 0          # fraction of potential heat units accumulated on given day
        self.max_fr_lai = 0     # canopy height and LAI variable
        self.prev_max_fr_lai = 0
        self.HI = 0                 # daily harvest index
        self.HI_opt = 0.55          # potential harvest index at maturity, assuming ideal conditions
        self.HI_min = 0.30          # min harvest index for corn, Table A-8 ST2012 I/O
        self.plant_biom = 0.005     # initial plant biomass, kg/ha
        self.bioN = 0       # initial nitrogen in plant
        self.potential_n_uptake = 0
        self.cumulative_bio = 0.005
        self.crop_yld = 0.0         # crop yield, kg/ha
        self.total_crop_yld = 0.0   # cumulative crop yield, kg/ha
        self.delta_bio = 0          # initial potential increase in plant biomass

        # nutrient params
        self.denitrif = 0   # N amnt lost to denitrification
        self.n_uptake = 0   # N amnt lost to plant
        # growth constraints
        self.w_stress, self.temp_stress, self.n_stress = 0, 0, 0

    def init_soilwater(self):
        # soil temp
        self.soil_temp = 0.9*self.avg_temp
        self.S_max = self._compute_retention_param()  # also computes Field Capacity
        self.S = 0.9*self.S_max                       # init retention parameter
        self.daily_RCN = 25400/(self.S+254)     # runoff curve number
        self.runoff = 0.0 if self.precip < 0.2*self.S else (self.precip - 0.2*self.S)**2 / (self.precip + 0.8*self.S)
        self.nitrogen_level = 5                 # kg/ha

        # soil moisture/water content
        self.swc = 0                            # soil water content
        # init prev_swc or FFCB : Initial soil water storage expressed as a fraction of field capacity water content
        self.prev_swc = 0 # 0.1*self.FC/7.0 # IO file?
        self.paw, self.prev_paw = 0, 0
        self.sm, self.prev_sm = 0, 0

        self.total_precip = 0
        self.total_runoff = 0
        self.total_soil_evap = 0
        self.total_act_transpiration = 0
        self.total_act_et = 0

    def init_management(self):
        self.fertilizer = 0 # nitrogen to fertilize for current day (kg/ha)
        self.irrigation = 0 # water depth to irrigate for current day (mm/m2)

    def reward_function(self, crop_yield, action):
        # compute reward as function of state and action, here we just consider biomass
        fertilizer, irrigation = action[0], action[1]
        return crop_yield - self.alpha * fertilizer - self.beta * irrigation

    def _get_state(self, day):
        # computes and returns state (array)

        # get current day's weather
        self.weather = self._get_weather_day(self.weatherdataprovider, day)
        # weather variables: 'TEMP', 'IRRAD', 'VAP', 'RAIN', 'ET0', 'ES0'
        self.avg_temp = self.weather[0] + np.random.normal(0, 1)
        self.solar_rad = self.weather[1]/1e6 # convert to MegaJoules
        self.avg_vapor_pressure = self.weather[2]
        self.precip = self.weather[3]*10    # convert cm to mm
        self.ref_et = self.weather[4]*10
        self.soil_evap = self.weather[5]*10

        # update crop variables
        self.HU = self._heat_unit(self.avg_temp)    # heat units for given day
        self.total_HU += self.HU                    # total accumulated heat units since day 1
        self.frPHU = self.total_HU / self.PHU       # fraction of potential heat units on a given day
        # self.PHU = self.total_HU
   
        # compute canopy height and LAI
        self.max_fr_lai, self.h_c = self._canopy_height(self.frPHU)
        prev_LAI = self.LAI
        self.LAI = self._get_LAI(prev_LAI, self.prev_max_fr_lai, self.max_fr_lai, self.growth_factor) 
        self.prev_max_fr_lai = self.max_fr_lai 

        # compute solar radiation intercepted using LAI 
        rad_intercept = self._daily_solar_rad_intercept(self.solar_rad)

        # computes fraction of total biomass in roots and root development
        self.fr_root, z_root = self._root_development(self.frPHU)  

        # compute actual transpiration and evapotranspiration
        self.act_transpiration, self.act_evapotranspiration = self._calc_transpiration(self.ref_et, z_root)
        self.total_act_transpiration += self.act_transpiration
        self.total_act_et += self.act_evapotranspiration
        self.total_ref_et += self.ref_et

        # compute water uptake based on transpiration and root development
        # self.water_uptake = self._water_dynamics(self.act_transpiration, z_root)
        self.water_uptake = self.act_transpiration

        # determine current growth factor, potential N uptake, and growth constraints
        self.growth_factor, self.potential_n_uptake, self.w_stress, self.temp_stress, self.n_stress = self._growth_constraints(self.water_uptake, self.frPHU, rad_intercept)

        # compute surface runoff and curve number
        self.runoff, self.daily_RCN = self._surface_runoff(self.precip, self.ref_et)

        # water balance equation to compute soil water content (mm), ST2009 page 9
        self.swc = max(0.0, self.prev_swc + self.precip - self.runoff - self.act_evapotranspiration) # w_seep, q_gw
        self.prev_swc = self.swc

        # paw = max(0, swc - wp)
         
        self.total_precip += self.precip
        self.total_runoff += self.runoff
        self.total_soil_evap += self.soil_evap
        
        # compute total plant biomass from day 0 to current day
        self.plant_biom = self._total_biomass(self.growth_factor, self.plant_biom, rad_intercept)
        self.cumulative_bio += self.plant_biom

        # estimate crop yield
        self.crop_yld = self._crop_yield(self.frPHU, self.fr_root, self.plant_biom)
        self.total_crop_yld += self.crop_yld

        # soil temperature
        self.soil_temp = self._calc_soil_temp(self.avg_temp, self.precip)

        # nitrogen cycle: addition and removal via rain, fert, denitrification, uptake etc.
        self.nitrogen_level, self.denitrif, self.n_uptake = self._nitrogen_dynamics(z_root, self.potential_n_uptake)

        # combine vars to make state
        state = [self.precip, self.ref_et, self.act_transpiration, self.swc, self.daily_RCN, self.avg_temp, self.solar_rad, self.denitrif, self.n_uptake, self.w_stress, self.temp_stress, self.n_stress, self.plant_biom, self.LAI]
        return self.crop_yld, state
    
    def _get_weather(self, date, duration):
        """ Get weather data for a specific location over a certain period

        params
        ----------
        date: datetime.date, start date for requested observations
        duration: int, number of days of weather observations requested

        output
        -------
        numpy array containing the requested weatherdata
        # """
        weather = self._get_weather_day(self.weatherdataprovider, date)
        return np.array(weather)

    @staticmethod
    def _get_weather_day(weatherdataprovider, date):
        """
        Get weather observations for a single day
        https://pcse.readthedocs.io/en/stable/_modules/pcse/base/weather.html

        Parameters
        ----------
        weatherdataprovider : PCSE weather data provider
        date: datetime.date, date for requested observations

        Returns
        -------
        numpy array containing the requested weather data
        """
        weatherdatacontainer = weatherdataprovider(date)
        weather_vars = ['TEMP', 'IRRAD', 'VAP', 'RAIN', 'ET0', 'ES0']
        weather = [getattr(weatherdatacontainer, attr) for attr in weather_vars]
        return weather

    '''Crop and hydrology module '''
    def _heat_unit(self, avg_temp):
        # phenological development is based on daily heat unit accumulation
        return avg_temp - self.T_base

    ## Potential Growth
    def _daily_solar_rad_intercept(self, solar_rad):
        # SWAT computes the amount of daily solar radiation intercepted by the plant
        # using Beer's Law (see ST2009 5:2.1.1)
        H_phosyn = 0.5*solar_rad*(1 - np.exp(-self.kl * self.LAI))  # megajoules per sq. m
        max_rad_intercept = 0.95  # the maximum fraction of radiation interception that a crop can reach, governed by plant spacing, ref: SIMPLE model
        return min(H_phosyn, max_rad_intercept)

    def _get_LAI(self, prev_lai, prev_max_fr_lai, max_fr_lai, growth_factor):
        # compute leaf area added on given day, ST2009 eqns 5:2.1.16 and 5:2.1.18
        delta_lai = (max_fr_lai - prev_max_fr_lai) * self.LAI_mx * (1 - np.exp(5*(prev_lai - self.LAI_mx)))
        actual_delta_lai = delta_lai*np.sqrt(np.abs(growth_factor)) # eqn 5:3.2.2
        lai = prev_lai + actual_delta_lai    
        # account for leaf senescence becoming dominant process
        # if self.frPHU > self.fr_LAI_sen:
        #     lai = self.LAI_mx * (1-self.frPHU)/(1-self.fr_LAI_sen)
        return lai

    def _root_development(self, fr_PHU):
        # computes fraction of total biomass in roots and root development, ST2009 eqns 5:2.1.21-22
        fr_root = 0.40 - 0.20*fr_PHU
        # depth of root development on a given day (mm) 
        z_root = self.max_root_depth if fr_PHU > 0.40 else 2.5*max(0.01, fr_PHU)*self.max_root_depth
        return fr_root, z_root

    def _canopy_height(self, fr_PHU):
        # compute daily change in canopy height, based on corn params: ST2009, eqn 5:2.1.14
        max_fr_lai = max(0, fr_PHU/(fr_PHU + np.exp(self.l1 - self.l2*fr_PHU)))  # canopy height and LAI variable
        h_c = min(self.max_h_c*np.sqrt(max_fr_lai), self.max_h_c)  # once max h_c is reached, plant stops growing
        return max_fr_lai, h_c

    def _calc_transpiration(self, ref_evapotranspiration, z_root):
        # In the crop coefficient approach the crop evapotranspiration, ETc, is calculated
        # by multiplying the reference crop evapotranspiration, ETo, by a crop coefficient, Kc
        # ST2009 section 2:2.3

        canopy_pet = 0.8*ref_evapotranspiration
        et1 = max(0.000001, ref_evapotranspiration*(1-np.exp(-self.kl * self.LAI))) # PCSE max transpiration
        et2 = canopy_pet*self.LAI/3.0 if self.LAI <= 3.0 else canopy_pet  # max plant transpiration, eqn 2:2.3.5-6
        self.max_transpiration = max(et1, et2) # assuming max transpiration for corn is 0.2 inches per day 
        
        Bw = 10 # water-use distribution parameter used in SWAT
        Et = self.max_transpiration # max plant transpiration on a given day
        z = self.max_root_depth

        # potential water uptake from soil surface to any depth in the root zone: eqn 5:2.2.1
        potential_water_uptake = np.abs((Et/(1-np.exp(-Bw)))*np.abs(1-np.exp(-Bw*z/z_root)))

        # actual amount of transpiration (mm) == water uptake, based on PCSE
        FC = 0.32
        WP = 0.09 # * loam soil type

        # drought stress
        SWFAC1 = 0.0 if self.swc < WP*self.FC else 1.0
        # excess water stress
        SWFAC2 = 1.0 if self.swc < FC*self.FC else 0.0
        w_strss = min(SWFAC1, SWFAC2)
        actual_Et = min(potential_water_uptake, WP*self.swc) #*w_strss # max(0, 0.096*awc) # eqn 5:2.2.7

        # compute actual evapotranspiration
        # https://www.fao.org/3/x0490e/x0490e0b.htm
        if self.current_date < self.start_date+datetime.timedelta(days=30):
            K_c = 0.3
        elif self.current_date < self.start_date+datetime.timedelta(days=90):
            K_c = 1.15
        else:
            K_c = 0.4

        # K_c = 0.3*(num_days) + c
        
        Etc = K_c*ref_evapotranspiration  # ST2009, eqn 2:2.3.5, also by Allen 1998 (https://www.fao.org/3/x0490e/x0490e00.htm): Etc = Kc * w_stress * Eto
        return actual_Et, Etc
    
    def _compute_retention_param(self):
        # init values from HUC14 000010013.mgt file 
        # Watershed HRU:13 Subbasin:1 HRU:13 Luse:CORN Soil:391067 Slope:2-8 7/6/2022 4:09:09 PM EPA-HAWQS v2.0 (12070204010101)
        # max value retention param can achieve on any given day
        CN2 = 79.60     # moisture condition II curve number (average moisture condition) for default 5% slope, hydro soil group A for row crops 
        CN1 = CN2 - (2000 - 20*CN2)/(100 - CN2 + np.exp(2.533 - 0.0636*(100 - CN2))) # for dry (wilting point)
        CN3 = CN2*np.exp(0.00673*(100-CN2))     # moisture condition III curve number for wet (field capacity) condition
        self.FC = CN3
        S_max = 25.4*(1000/CN1 - 10)  
        return S_max

    def _surface_runoff(self, precip, ref_et):
        Ia = 0.2*self.S     # initial abstraction, ST2009, eqn 2:1.1.1         
        
        Q_surf = 0  # accumulated surface runoff/rainfall excess amount (mm)
        if precip > Ia: # runoff occurs
            Q_surf = (precip - Ia)**2 / (precip - Ia + self.S)
        
        # update retention param based on plant evapotranspiration, eqn 2:1.1.9
        S_prev = self.S
        cncoef = 1.0    # Plant ET curve number weighting coefficient 
        self.S = S_prev + ref_et*np.exp(-cncoef*S_prev/self.S_max) - precip + Q_surf
        daily_RCN = 25400/(self.S+254)  # curve number ranges from 0 <= CN <= 100

        return Q_surf, daily_RCN

    def _calc_soil_temp(self, avg_air_temp, precip):
        #  compute soil temperature as a function of air temp, precipitation, and vegetation cover
        # based on Zheng et al (1993), https://www.int-res.com/articles/cr/2/c002p183.pdf
        prev_soil_temp = self.soil_temp
        soil_temp = prev_soil_temp + 0.25*(avg_air_temp - prev_soil_temp) #*np.exp(-self.kl * lai)
        
        if avg_air_temp <= prev_soil_temp:
            soil_temp = prev_soil_temp + 0.25*(avg_air_temp - prev_soil_temp)
        if precip > 0:
            soil_temp -= 0.75

        soil_temp = max(soil_temp, 0.95*avg_air_temp)    # assume soil temp is 5% within avg air temp at most
        return soil_temp
    
    def _water_dynamics(self, actual_Et, z_root):
        # consider water added (precip, irrig) and removed (uptake by plant, runoff) from env
        Bw = 10 # water-use distribution parameter used in SWAT
        Et = self.max_transpiration # max plant transpiration on a given day
        z = self.max_root_depth

        # potential water uptake from soil surface to any depth in the root zone: eqn 5:2.2.1
        w_up = (Et/(1-np.exp(-Bw)))*np.abs(1-np.exp(-Bw*z/z_root))
        return np.abs(w_up)

    def _nitrogen_dynamics(self, z_root, N_demand):
        """ consider nitrogen added to soil via rain, fertilizer, assume no bacteria fixation, manure/residue application """
        rain_nitrate_conc = 1.5     # Concentration of nitrogen in rain [mg N/l] (0.0- 2.0 ppm)
        n_rain = 0.01*rain_nitrate_conc*self.precip       # amount of nitrate added via rainfall (kg N/ha)
        efficiency = 1.0 # nitrogen utilization 
        n_fert = efficiency*self.fertilizer # amount of of N in fertilizer

        # N lost due to denitrification, uptake, (erosion, leaching,and volatilization)
        CDN = 1.4           # rate of denitrification, SWAT I/O 2012 or 1.104  Denitrification exponential rate coefficient in basins.bsn
        z_surf = 10         # soil surface layer depth (mm)
        init_nitrate_conc = 7*np.exp(-z_root/1000)   # initial nitrate levels (mg/kg), ST2009 eqn 3:1.1.1
        org_carbon = 0.0174     # amount (1.74%) of organic carbon in soil layer

        # gamma_temp: nutrient cycling temp factor, ST2009, eqn 3:1.2.1
        gamma_temp = max(0.1, 0.9*(self.soil_temp/(self.soil_temp + np.exp(9.93 - 0.312*self.soil_temp))) + 0.1)  
        gamma_sw = max(0.05, self.swc/self.FC)    # nutrient cycling water factor, always > 0.05
        
        # amount of nitrate lost by denitrification (kg N/ha): ST2009 eqn 3:1.4.1-2
        denit_threshold = 0.935 # Denitrification threshold water content from HUC14 Temple basins.bsn 
        denitrif = init_nitrate_conc*(1-np.exp(-CDN*gamma_temp*org_carbon)) if gamma_sw >= denit_threshold else 0.0

        # nitrate lost via leaching: surface runoff, lateral flow or percolation
        mobile_nitrate = 0.5  # conc. of nitrate in mobile water for top soil layer
        nitrate_content = mobile_nitrate + init_nitrate_conc   # amount of nitrate (kg/ha) in soil layer

        # no3_perc: nitrate moved to the underlying layer by percolation(kg N/ha), ST2009 eqn 4:2.1.8
        daily_leaching_rate = 3.21e-06 # kg/ha
        NPERCO = 0.021      # Nitrogen percolation coefficient i.e. N removed from surface layer via runoff rel. to percolation

        no3_surf = NPERCO * mobile_nitrate * self.runoff    # nitrate removed in  surface runoff (kg N/ha), ST2009 eqn 4:2.1.5
        no3_perc = daily_leaching_rate*(mobile_nitrate) # compute w_perc,ly eqn      

        # plant uptake of nitrogen
        N_UPDIS = 8.828 # Nitrogen uptake distribution parameter, limits max amnt rmvd by runoff
        # potential nitrogen uptake from soil surface to max root depth (kg/ha), eqn 5:2.3.6
        potential_n_uptake_surface = N_demand #(N_demand/(1-np.exp(-N_UPDIS)))*np.abs(1-np.exp(-N_UPDIS*self.max_root_depth/z_root))

        # actual amnt of N removed from soil (kg/ha)
        actual_n_uptake = np.abs(min(potential_n_uptake_surface+N_demand, nitrate_content))     # actual nitrogen uptake (kg/ha), eqn 5:2.3.8   

        nitrogen_level = np.abs(n_rain + n_fert - denitrif - no3_surf - no3_perc - actual_n_uptake)

        return nitrogen_level, denitrif, actual_n_uptake

    def _growth_constraints(self, w_uptake, frPHU, rad_intercept):
        """plant growth may be reduced due to extreme temperatures, insufficient water or nutrients. stress is 0 at optimal conditions, 1 if excessive."""
        # irrig when water content falls belows FC or once triggered by water stress/close to WP
        self.paw = self.prev_paw + self.precip + self.irrigation - self.runoff - self.act_evapotranspiration # - lateral_flow - WP - gw_recharge
        self.prev_paw = self.paw

        # # == soil moisture/soil water content
        self.sm = self.prev_sm + self.precip + self.irrigation - self.runoff - self.act_evapotranspiration # - lateral_flow - gw_recharge
        self.prev_sm = self.sm

        # compute water stress, ST2009 eqn 5:3.1.1
        # w_uptake = self.act_transpiration # # eqn 5:2.2.9 - total plant water uptake == actual amnt of transpiration
        trans_rate = w_uptake/self.max_transpiration #if w_uptake <= self.max_transpiration else 0.0
        w_stress = max(0.0, np.abs(1.0 - trans_rate))
        
        # compute temperature stress, ST2009 eqn 5:3.1.2-5
        x = -0.1054*(self.T_opt - self.avg_temp)**2
        if self.T_base < self.avg_temp and self.avg_temp <= self.T_opt:
            temp_stress = 1 - np.exp(x/(self.avg_temp-self.T_base)**2)
        elif self.T_opt < self.avg_temp and self.avg_temp <= (2*self.T_opt - self.T_base):
            temp_stress = 1 - np.exp(x/(2*self.T_opt - self.avg_temp-self.T_base)**2)
        else:
            temp_stress = 1
        
        # compute nitrogen stress, ST2009 eqn 5:3.1.6-7
        fr_N1 = 0.0470  # normal fraction of N in plant biomass at emergence, see 'corn' values Table A-7, ST2012 I/O
        fr_N2 = 0.0177  # normal fraction of N in plant biomass at 50% maturiy
        fr_N3 = 0.0138  # normal fraction of N in plant biomass at maturity
        d1 = fr_N2-fr_N3
        d2 = fr_N1-fr_N3
        
        # plant nitrogen equation shape coeffiecients: eqn 5:2.3.2-3
        n2 = 2*(np.log(np.abs(0.5/(1-d1/d2) - 0.5)) - np.log(np.abs(1/(1- 0.00001/d2) - 1)))
        n1 = np.log(np.abs(0.5/(1-d1/d2) - 0.5)) + 0.5*n2
        y = frPHU + np.exp(n1 - n2*frPHU)

        # optimal fraction of nitrogen in plant biomass, eqn 5:2.3.1
        frN = (fr_N1 - fr_N3)*np.abs(1- frPHU/y)+fr_N3

        # compute actual mass and optimal mass of nitrogen stored in plant (kg N/ha), eqn 5:2.3.4-5
        self.bioN =  max(0.04*self.plant_biom, self.n_uptake) # TODO: daily bioN ... /max(1, 120-(self.current_date - self.start_date).days) # assuming nitrogen stored is 4% of plant biomass
        opt_bioN = frN * self.plant_biom        # mass of nitrogen (kg/ha) stored in plant biomass on a given day under optimal conditions: eqn 5:2.3.4
        
        # plant nitrogen demand / potential plant nitrogen uptake is determined by eqn 5:2.3.5
        delta_bio = self.RUE * rad_intercept 
        potential_n_uptake = min(opt_bioN-self.bioN, 4*fr_N3*delta_bio)

        # finally compute nitrogen stress
        phi_N = max(200*(self.bioN/opt_bioN - 0.5), self.bioN/opt_bioN)  # *200 scaling factor for nitrogen stress
        n_stress = 1 - phi_N/(phi_N+np.exp(3.535 - 0.02597*phi_N))  # ST2009 eqn 5:3.1.6

        # plant growth factor to measure actual growth taking stress into account, eqn 5:3.2.3
        growth_factor = 1.0 - max(w_stress, temp_stress, n_stress)

        return growth_factor, potential_n_uptake, w_stress, temp_stress, n_stress

    def _total_biomass(self, growth_factor, prev_biomass, H_phosyn):
        # ST2009, eqns 5:2.1.2 and 5:2.1.3
        # compute total plant biomass using:
        #   - plant's Radiation Use Efficiency (RUE) [kg/ha/(MJ/M^2)]
        #   - amnt of intercepted photosynthetically active radiation MJ/M^2
        #   - potential increase in total plant biomass on a given day
        potential_bio = self.RUE * H_phosyn # potential increase in total plant biomass
        self.delta_bio =  growth_factor * potential_bio # actual increase in total plant biomass (kg/ha), eqn 5:3.2.1
        bio = prev_biomass + self.delta_bio
        return bio
    
    def _crop_yield(self, frPHU, fr_root, bio):
        # compute potential harvest index (fraction of aboveground plant dry biomass removed as yield)
        self.HI = self.HI_opt*(100*frPHU/(100*frPHU+np.exp(11.1 - 10*frPHU)))   # ST2009 - eqn 5:2.4.1
        # compute actual yield considering water deficit, section 5:3:3
        gamma_wu = 100*self.total_act_et/self.total_ref_et # water deficiency factor, ST2009 eqn 5:3.3.2
        # compute actual harvest index
        HI_act = (self.HI - self.HI_min)*gamma_wu/(gamma_wu + np.exp(6.13 - 0.883*gamma_wu)) + self.HI_min # ST2009 eqn 5:3.3.1
        bio_ag = (1 - fr_root)*bio   # aboveground biomass (kg/ha), ST2009 eqn 5:2.4.4
        # compute actual yield
        yld = bio*(1 - 1/(1+HI_act)) if HI_act > 1.0 else bio_ag*HI_act   # kg/ha, ST2009 eqn 5:2.4.2-3
        return yld

    def _reset(self):
        # reset state and action variables
        self.time_hist = []

        # climate inputs
        self.precip_history = []
        self.solar_rad_hist = []
        self.temp_hist = []

        # crop params
        self.crop_height_hist = []
        self.crop_lai_hist = []
        self.biomass_hist = []
        self.yield_hist = []
        self.HI_hist = []
        self.frPHU_hist = []
        self.act_Et_hist = []
        self.act_Etc_hist = []
        self.water_stress_hist = []
        self.nitrogen_stress_hist = []
        self.temp_stress_hist =[]

        # soil and hydrology params
        self.n_uptake_hist = []
        self.nitrogen_level_hist = []
        self.soil_water_balance_hist = []
        self.runoff_hist = []
        self.water_uptake_hist = []
        # self.paw_history = []

        # management params
        self.fert_hist = []
        self.irrig_hist = []
        self.cumulative_fertilizer = 0
        self.cumulative_irrigation = 0
        self.max_fertilizer = 300   # kg/ha, max fert applied during the year
        self.max_irrigation = 600 # mm/ha, max irrig applied during the year
        self.reward_hist = []
        self.done_hist = []
    
    def show_history(self):
        # returns simulation history dataframe 
        df = pd.DataFrame()
        df['Time'] = pd.Series(self.time_hist)

        # climate hist
        df['precip'] = pd.Series(self.precip_history)
        df['irrad'] = pd.Series(self.solar_rad_hist)
        df['temp'] = pd.Series(self.temp_hist)

        # crop hist
        df['height'] = pd.Series(self.crop_height_hist)
        df['lai'] = pd.Series(self.crop_lai_hist)
        df['biomass'] = pd.Series(self.biomass_hist)
        df['yield'] = pd.Series(self.yield_hist)
        df['HI'] = pd.Series(self.HI_hist)
        df['frPHU'] = pd.Series(self.frPHU_hist)
        df['transpiration'] = pd.Series(self.act_Et_hist)
        df['evapotrans'] = pd.Series(self.act_Etc_hist)
        df['w_stress'] = pd.Series(self.water_stress_hist)
        df['n_stress'] = pd.Series(self.nitrogen_stress_hist)
        df['t_stress'] = pd.Series(self.temp_stress_hist)

        # soil hist
        df['n_uptake'] = pd.Series(self.n_uptake_hist)
        df['nitrogen_level'] = pd.Series(self.nitrogen_level_hist)
        df['swc'] = pd.Series(self.soil_water_balance_hist)
        df['runoff'] = pd.Series(self.runoff_hist)
        df['w_uptake'] = pd.Series(self.water_uptake_hist)

        # management hist
        df['fertilizer'] = pd.Series(self.fert_hist)
        df['irrig'] = pd.Series(self.irrig_hist)
        df['reward'] = pd.Series(self.reward_hist)
        df['dones'] = pd.Series(self.done_hist)
        return df
