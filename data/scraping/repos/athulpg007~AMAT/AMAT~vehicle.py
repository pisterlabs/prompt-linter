
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from scipy.integrate import cumtrapz
import copy
import random as rd
import os


class Vehicle:
	"""
	The Vehicle class is used to define vehicle parameters, 
	such as mass, aerodynamics, and aeroheating relations.

	Attributes
	----------
	vehicleID : str
		identifier string for vehicle object
	mass : float
		vehicle mass, kg
	beta : float
		vehicle ballistic coefficient, kg/m2
	LD : float
		vehicle lift-to-drag ratio
	A : float
		vehicle reference aerodynamic area, m2
	alpha : float
		vehicle angle-of-attack, currently not implemented, rad
	RN : float
		vehicle nose-radius, m
	planetObj : planet.Planet
		planet object associated with the vehicle, indicates
		the planetary atmosphere in which the vehicle will
		operate in
	Abar : float
		vehicle non-dimensional reference area
	mbar : float
		vehicle non-dimensional mass
	CD : float
		vehicle drag coefficient
	CL : float
		vehicle lift coefficient
	h0_km : float
		initial vehicle altitude at atmospheric interface / other 
		start point, meters
	theta0_deg : float
		initial vehicle longitude at atmospheric interface / other 
		start point, degrees
	phi0_deg : float
		initial vehicle latitude at atmospheric interface / other 
		start point, degrees
	v0_kms : float
		initial vehicle speed (planet-relative) at atmospheric
		interface / other  start point, km/s
	psi0_deg : float
		initial vehicle heading at atmospheric interface / other 
		start point, degrees
	gamma0_deg : float
		initial vehicle flight-path angle at atmospheric interface
		/ other start point, degrees
	drange0_km : float
		initial vehicle downrange at atmospheric interface
		/ other start point, km
	heatLoad0 : float
		initial vehicle heatload at atmospheric interface
		/ other start point, J/cm2
	tol : float
		solver tolerance, currently both abstol and reltol
		are set to this value
	index : int
		array index of the event location if one was detected, 
		terminal index otherwise
	exitflag : int
		flag to indicate and classify event occurence 
		or lack of it			
	tc : numpy.ndarray
		truncated time array, sec
	rc : numpy.ndarray
		truncated radial distance array, m
	thetac : numpy.ndarray
		truncated longitude solution array, rad
	phic : numpy.ndarray
		truncated latitude solution, rad
	vc : numpy.ndarray
		truncated speed solution array, m/s
	psic : numpy.ndarray
		truncated heading angle solution, rad
	gammac : numpy.ndarray
		truncated flight-path angle solution, rad
	drangec : numpy.ndarray
		truncated downrange solution array, m
	t_minc : numpy.ndarray
		truncated time array, minutes
	h_kmc : numpy.ndarray
		truncated altitude array, km
	v_kmsc : numpy.ndarray
		truncated speed array, km/s
	phi_degc : numpy.ndarray
		truncated latitude array, deg
	psi_degc : numpy.ndarray
		truncated heading angle, deg
	theta_degc : numpy.ndarray
		truncated longitude array, deg
	gamma_degc : numpy.ndarray
		truncared flight-path angle array, deg
	drange_kmc : numpy.ndarray
		truncated downrange array, km
	acc_net_g  : numpy.ndarray
		acceleration load solution, Earth G      
	acc_drag_g  : numpy.ndarray
		drag acceleration solution, Earth G      
	dyn_pres_atm : numpy.ndarray
		dynamic pressure solution, Pa
	stag_pres_atm : numpy.ndarray
		stagnation pressure solution, Pa
	q_stag_con : numpy.ndarray
		stagnation-point convective heat rate, W/cm2
	q_stag_rad : numpy.ndarray
		stagnation-point radiative heat rate, W/cm2
	q_stag_total : numpy.ndarray
		stagnation-point radiative tottal heat rate, W/cm2
	heatload : numpy.ndarray
		stagnation point heat load, J/cm2 
	maxRollRate : float
		maximum allowed roll rate, degrees per second
	beta1: float
		beta1 (smaller ballistic coeff.) for drag modulation, kg/m2
	betaRatio : float
		ballistic coefficient ratio for drag modulation 
	target_peri_km : float
		vehicle target periapsis altitude, km
	target_apo_km : float
		vehicle target apoapsis altitude, km
	target_apo_km_tol : float
		vehicle target apoapsis altitude error tolerance, km
		used by guidance algorithm
	Ghdot : float
		Ghdot term for vehicle equilibrium glide phase
	Gq : float
		Gq term for vehicle equilibrium glide phase guidance
	v_switch_kms : float
		speed below which eq. glide phase is terminated
	t_step_array : numpy.ndarray
		time step array for guided aerocapture trajectory, min
	delta_deg_array : numpy.ndarray
		bank angle array for guided aerocapture trajectory, deg
	hdot_array : numpy.ndarray
		altitude rate array for guided aerocapture trajectory, m/s
	hddot_array : numpy.ndarray
		altitude acceleration array for guided aerocapture, m/s2
	qref_array : numpy.ndarray
		reference dynamic pressure array for guided aerocapture, Pa
	q_array : numpy.ndarray
		actual dynamic pressure array for guided aerocapture, Pa
	h_step_array : numpy.ndarray
		altitude array for guided aerocapture, meters
	acc_step_array : numpy.ndarray
		acceleration array for guided aerocapture, Earth G
	acc_drag_array : numpy.ndarray
		acceleration due to drag for guided aerocapture, Earth G 
	density_mes_array : numpy.ndarray
		measured density array during descending leg, kg/m3
	density_mes_int : scipy.interpolate.interpolate.interp1d
		measured density interpolation function
	minAlt : float
		minimum altitude at which density measurement is available, km
	lowAlt_km : float
		lower altitude to which density model is to be extrapolated
		based on available measurements, km
	numPoints_lowAlt : int
		number of points to evaluate extrapolation at below the 
		altitude where measurements are available
	hdot_threshold : float
		threshold altitude rate (m/s) above which density measurement
		is terminated and apoapsis prediction is initiated
	t_min_eg : numpy.ndarray
		time solution of equilibrium glide phase, min
	h_km_eg : numpy.ndarray
		altitude array of equilibrium glide phase, km
	v_kms_eg : numpy.ndarray
		speed solution of equilibrium glide phase, km/s
	theta_deg_eg : numpy.ndarray
		longitude solution of equilibrium glide phase, deg
	phi_deg_eg : numpy.ndarray
		latitude solution of equilibrium glide phase, deg
	psi_deg_eg : numpy.ndarray
		heading angle solution of equilibrium glide phase, deg
	gamma_deg_eg : numpy.ndarray
		flight-path angle solution of eq. glide phase, deg
	drange_km_eg : numpy.ndarray
		downrange solution of eq. glide phase, km
	acc_net_g_eg : numpy.ndarray
		acceleration solution of eq. glide phase, Earth G
	dyn_pres_atm_eg : numpy.ndarray
		dynamic pressure solution of eq. glide phase, atm
	stag_pres_atm_eg : numpy.ndarray
		stagnation pressure solution of eq. glide phase, atm
	q_stag_total_eg : numpy.ndarray
		stag. point total heat rate of eq. glide phase, W/cm2
	heatload_eg : numpy.ndarray
		stag. point heatload solution of eq. glide phase, J/cm2
	t_switch : float
		swtich time from eq. glide to exit phase, min
	h_switch : float
		altitude at which guidance switched to exit phase, km
	v_switch : float
		speed at which guidance switched to exit phase, km/s
	p_switch : float
		bank angle at which guidance switched to exit phase, deg
	t_min_full : numpy.ndarray
		time solution of full 
		(eq. gllide + exit phase), min
	h_km_full : numpy.ndarray
		altitude array of full 
		(eq. gllide + exit phase), km
	v_kms_full : numpy.ndarray
		speed solution of full 
		(eq. gllide + exit phase), km/s
	theta_deg_full : numpy.ndarray
		longitude solution of full 
		(eq. gllide + exit phase), deg
	phi_deg_full : numpy.ndarray
		latitude solution of full 
		(eq. gllide + exit phase), deg
	psi_deg_full : numpy.ndarray
		heading angle solution of full 
		(eq. gllide + exit phase), deg
	gamma_deg_full : numpy.ndarray
		flight-path angle solution of full (eq. gllide + exit phase), deg
	drange_km_full : numpy.ndarray
		downrange solution of full 
		(eq. gllide + exit phase), km
	acc_net_g_full : numpy.ndarray
		acceleration solution of full 
		(eq. gllide + exit phase), Earth G
	dyn_pres_atm_full : numpy.ndarray
		dynamic pressure solution of full 
		(eq. gllide + exit phase), atm
	stag_pres_atm_full : numpy.ndarray
		stagnation pressure solution of full 
		(eq. gllide + exit phase), atm
	q_stag_total_full : numpy.ndarray
		stag. point total heat rate of full 
		(eq. gllide + exit phase), W/cm2
	heatload_full : numpy.ndarray
		stag. point heatload solution of full 
		(eq. gllide + exit phase), J/cm2
	NPOS : int
		NPOS value from GRAM model output 
		is the number of data points (altitude) in each atm. profile
	NMONTE : int
		NMONTE is the number of Monte Carlo atm profiles
		from GRAM model output
	heightCol : int
		column number of altitude values in Monte Carlo density file
	densLowCol : int
		column number of low density value in Monte Carlo density file
	densAvgCol : int
		column number of avg. density value in Monte Carlo density file
	densHighCol : int
		column number of high density value in Monte Carlo density file
	densTotalCol : int
		column number of total density value in Monte Carlo density file
	heightInKmFlag : bool
		set to True if height values are in km
	nominalEFPA : float
		nominal entry-flight-path angle
	EFPA_1sigma_value : float
		1-sigma error for EFPA
	nominalLD : float
		nominal vehicle lift-to-drag ratio
	LD_1sigma_value : float
		1-sigma error for vehicle lift-to-drag ratio
	vehicleCopy : Vehicle.vehicle
		copy of the original vehicle object
	timeStep : float
		guidance cycle time, sec
	dt : float
		max. solver timestep, sec
	maxTimeSecs : float
		maximum propogation time used by guidance algorithm, sec
	t_min_en : numpy.ndarray
		time solution of entry phase (DM), min
	h_km_en : numpy.ndarray
		altitude solution of entry phase (DM), km
	v_kms_en : numpy.ndarray
		velocity solution of entry phase (DM), km/s
	theta_deg_en : numpy.ndarray
		longitude solution of entry phase (DM), deg
	phi_deg_en : numpy.ndarray
		latitude solution of entry phase (DM), deg
	psi_deg_en : numpy.ndarray
		heading angle solution of entry phase (DM), deg
	gamma_deg_en : numpy.ndarray
		FPA solution of entry phase (DM), deg
	drange_km_en : numpy.ndarray
		downrange solution of entry phase (DM), km
	acc_net_g_en : numpy.ndarray
		acceleration solution of entry phase, Earth g
	dyn_pres_atm_en : numpy.ndarray
		dynamic pressures solution of entry phase, atm
	stag_pres_atm_en : numpy.ndarray
		stagnation pressure solution of entry phase, atm
	q_stag_total_en : numpy.ndarray
		heat rate solution of entry phase, W/cm2
	heatload_en : float
		heatload solution of entry phase, J/cm2
	userDefinedCDMach : bool
		if set to True, will use a user defined function for 
		CD(Mach) set by setCDMachFunction(), default=False

	"""

	def __init__(self, vehicleID, mass, beta, LD, A, alpha, RN, planetObj, userDefinedCDMach=False):
		"""
		Initializes vehicle object with properties such as mass, 
		aerodynamics etc.

		Parameters
		----------
		vehicleID : str
			name of the vehicle
		mass : float
			mass of the vehicle, kg
		beta : float
			vehicle ballistic coefficient, kg/m2
		LD : float
			vehicle lift-to-drag ratio
		A : float
			vehicle reference aerodynamic area, m2
		alpha : float
			vehicle angle-of-attack
		RN : float
			vehicle nose-radius
		planetObj : planet.Planet
			planet object associated with the vehicle
		userDefinedCDMach : bool
			if set to True, will use a user defined function for 
			CD(Mach) set by setCDMachFunction(), default=False
		"""

		self.vehicleID = vehicleID 
		self.mass = mass
		self.beta = beta
		self.A = A
		self.LD = LD
		self.alpha = alpha
		self.RN = RN
		self.planetObj = planetObj
		self.userDefinedCDMach = userDefinedCDMach

		# Compute other required non dimensional quantities
		self.Abar = self.A / (self.mass / (self.planetObj.rho0*self.planetObj.RP))
		self.mbar = self.mass/self.mass
		self.CD = self.mass / (self.beta*self.A)
		self.CL = self.LD * self.CD

		self.h0_km = None
		self.theta0_deg = None
		self.phi0_deg = None
		self.v0_kms = None
		self.psi0_deg = None
		self.gamma0_deg = None
		self.drange0_km = None
		self.heatLoad0 = None

		self.h0_km_ref = None
		self.theta0_deg_ref = None
		self.phi0_deg_ref = None
		self.v0_kms_ref = None
		self.psi0_deg_ref = None
		self.gamma0_deg_ref = None
		self.drange0_km_ref = None
		self.heatLoad0_ref = None

		self.CD1 = None
		self.CD_vec = None

		self.index = None
		self.exitflag = None

		self.tc = None
		self.rc = None
		self.thetac = None
		self.phic = None
		self.vc = None
		self.psic = None
		self.gammac = None
		self.drangec = None

		self.t_minc = None
		self.h_kmc = None
		self.v_kmsc = None
		self.phi_degc = None
		self.psi_degc = None
		self.theta_degc = None
		self.gamma_degc = None
		self.drange_kmc = None

		self.acc_net_g = None
		self.acc_drag_g = None

		self.dyn_pres_atm = None
		self.stag_pres_atm = None
		self.q_stag_con = None
		self.q_stag_rad = None
		self.q_stag_total = None
		self.heatload = None

		self.terminal_r = None
		self.terminal_v = None
		self.terminal_g = None

		self.terminal_E = None
		self.terminal_h = None

		self.terminal_a = None
		self.terminal_e = None

		self.rp = None
		self.hp = None
		self.hp_km = None

		self.beta1 = None
		self.betaRatio = None

		self.t_step_array = None
		self.delta_deg_array = None
		self.hdot_array = None
		self.qref_array = None
		self.q_array = None
		self.h_step_array = None

		self.hdotref_array = None
		self.hddoti = None
		self.hddot_array = None
		self.acc_step_array = None
		self.acc_drag_array = None

		self.t_min_eg = None
		self.h_km_eg = None
		self.v_kms_eg = None
		self.theta_deg_eg = None
		self.phi_deg_eg = None
		self.psi_deg_eg = None
		self.gamma_deg_eg = None
		self.drange_km_eg = None

		self.acc_net_g_eg = None
		self.dyn_pres_atm_eg = None
		self.stag_pres_atm_eg = None
		self.q_stag_total_eg = None
		self.heatload_eg = None

		self.t_switch = None
		self.h_switch = None
		self.v_switch = None
		self.p_switch = None

		self.t_min_full = None
		self.h_km_full = None
		self.v_kms_full = None
		self.theta_deg_full = None
		self.phi_deg_full = None
		self.psi_deg_full = None
		self.gamma_deg_full = None
		self.drange_km_full = None

		self.acc_net_g_full = None
		self.dyn_pres_atm_full = None
		self.stag_pres_atm_full = None
		self.q_stag_total_full = None
		self.heatload_full = None

		self.NPOS = None
		self.NMONTE = None
		self.atmfiles = None

		self.heightCol = None
		self.densLowCol = None
		self.densAvgCol = None
		self.densHighCol = None
		self.densTotalCol = None

		self.heightInKmFlag = None

		self.nominalEFPA = None
		self.EFPA_1sigma_value = None

		self.nominalLD = None
		self.LD_1sigma_value = None
		self.vehicleCopy = copy.deepcopy(self)

		self.timeStep = None
		self.dt = None
		self.maxTimeSecs = None

	def setCDMachFunction(self, func):
		"""
		Set function for CD (Mach)

		Parameters
		------------
		func : function object
			vectorized numpy function which returns CD (Mach)
			Note: func must return scalar for scalar input, 
			and array for array input!
		"""

		self.CDMach = func


	def setInitialState(self, h0_km, theta0_deg, phi0_deg, v0_kms, psi0_deg, gamma0_deg, drange0_km, heatLoad0):
		"""
		Set initial vehicle state at atmospheric entry interface

		Parameters
		----------
		h0_km : float
			initial vehicle altitude at atmospheric interface / other 
			start point, meters
		theta0_deg : float
			initial vehicle longitude at atmospheric interface / other 
			start point, degrees
		phi0_deg : float
			initial vehicle latitude at atmospheric interface / other 
			start point, degrees
		v0_kms : float
			initial vehicle speed (planet-relative) at atmospheric interface / other start point, km/s
		psi0_deg : float
			initial vehicle heading at atmospheric interface / other 
			start point, degrees
		gamma0_deg : float
			initial vehicle flight-path angle at atmospheric interface
			/ other start point, degrees
		drange0_km : float
			initial vehicle downrange at atmospheric interface
			/ other start point, km
		heatLoad0 : float
			initial vehicle heatload at atmospheric interface
			/ other start point, J/cm2

		"""
		self.h0_km = h0_km
		self.theta0_deg = theta0_deg
		self.phi0_deg = phi0_deg
		self.v0_kms = v0_kms
		self.psi0_deg = psi0_deg
		self.gamma0_deg = gamma0_deg
		self.drange0_km = drange0_km
		self.heatLoad0 = heatLoad0

		self.h0_km_ref = copy.deepcopy(h0_km)
		self.theta0_deg_ref = copy.deepcopy(theta0_deg)
		self.phi0_deg_ref = copy.deepcopy(phi0_deg)
		self.v0_kms_ref = copy.deepcopy(v0_kms)
		self.psi0_deg_ref = copy.deepcopy(psi0_deg)
		self.gamma0_deg_ref = copy.deepcopy(gamma0_deg)
		self.drange0_km_ref = copy.deepcopy(drange0_km)
		self.heatLoad0_ref = copy.deepcopy(heatLoad0)

	def setSolverParams(self, tol):
		"""
		Set the solver parameters.

		Parameters
		----------
		tol : float
			solver tolerance, currently both abstol and reltol
			are set to this value
		"""

		self.tol = tol


	def qStagConvective(self,r,v):
		"""
		This function defines the convective stagnation-point 
		heating relationships. Edit the parameters in the
		source-code if you wish to modify these values.

		Sources : Sutton-Graves relationships, NASA Neptune Orbiter 
		with Probes Vision Report, Bienstock et al.

		Parameters
		----------
		r : numpy.ndarray
			radial distance solution array of trajectory, m
		v : numpy.ndarray
			planet-relative speed array of trajectory, m/s

		Returns
		----------
		ans : numpy.ndarray
			convective stagnation-point heating rate array, W/cm2
		"""

		if self.planetObj.ID == 'VENUS':
			ans = np.zeros(len(r))
			rho_vec = self.planetObj.rhovectorized(r)
			ans[:] = 1.8960E-8 * (rho_vec[:]/self.RN)**0.5 * v[:]**3.0
			return ans

		elif self.planetObj.ID == 'EARTH':
			ans = np.zeros(len(r))
			rho_vec = self.planetObj.rhovectorized(r)
			ans[:] = 1.7623E-8 * (rho_vec[:]/self.RN)**0.5 * v[:]**3.0
			return ans

		elif self.planetObj.ID == 'MARS':
			ans = np.zeros(len(r))
			rho_vec = self.planetObj.rhovectorized(r)
			ans[:] = 1.8980E-8 * (rho_vec[:]/self.RN)**0.5 * v[:]**3.0
			return ans

		elif self.planetObj.ID == 'JUPITER':
			ans = np.zeros(len(r))
			rho_vec = self.planetObj.rhovectorized(r)
			ans[:] = 0.6556E-8 * (rho_vec[:]/self.RN)**0.5 * v[:]**3.0
			return ans

		elif self.planetObj.ID == 'SATURN':
			ans = np.zeros(len(r))
			rho_vec = self.planetObj.rhovectorized(r)
			ans[:] = 0.6356E-8 * (rho_vec[:]/self.RN)**0.5 * v[:]**3.0
			return ans

		elif self.planetObj.ID == 'TITAN':
			ans = np.zeros(len(r))
			rho_vec = self.planetObj.rhovectorized(r)
			ans[:] = 1.7407E-8 * (rho_vec[:]/self.RN)**0.5 * v[:]**3.0
			return ans

		elif self.planetObj.ID == 'URANUS':
			ans = np.zeros(len(r))
			rho_vec = self.planetObj.rhovectorized(r)
			ans[:] = 2.24008E-7 * rho_vec[:]**0.452130 * v[:]**2.691800 * np.sqrt(0.291/self.RN)
			return ans

		elif self.planetObj.ID == 'NEPTUNE':
			ans = np.zeros(len(r))
			rho_vec = self.planetObj.rhovectorized(r)
			ans[:] = 2.24008E-7 * rho_vec[:]**0.452130 * v[:]**2.691800 * np.sqrt(0.291/self.RN)
			return ans

		else:
			print(" >>> ERR : Invalid planet identifier provided.")
			

	def qStagRadiative(self,r,v):
		"""
		This function defines the radiative stagnation-point 
		heating relationships. Edit the parameters in the
		source-code if you wish to modify these values.

		Radiative heating is currently set to 0 for Mars 
		and Titan, though these may not be negligible
		under certain conditions.

		Sources : 
			Craig and Lyne, 2005; 
			Brandis and Johnston, 2014;
			NASA Vision Neptune orbiter with probes, 
			Contract No. NNH04CC41C

		Parameters
		----------
		r : numpy.ndarray
			radial distance solution array of trajectory, m
		v : numpy.ndarray
			planet-relative speed array of trajectory, m/s

		Returns
		----------
		ans : numpy.ndarray
			radiative stagnation-point heating rate array, W/cm2
		"""

		if self.planetObj.ID == 'VENUS':
			
			ans = np.zeros(len(r))
			rho_vec = self.planetObj.rhovectorized(r)

			# REF : Criag and Lyne, Parametric Study of Venus Aerocapture, 2005

			for i in range(0, len(r)):
				if v[i] < 8000.0:
					ans[i] = 3.33E-34*v[i]**10.0*rho_vec[i]**1.2*self.RN**0.49
				elif v[i] >= 8000.0 and v[i] < 10000.0:
					ans[i] = 1.22E-16*v[i]**5.5*rho_vec[i]**1.2*self.RN**0.49
				else:
					ans[i] = 3.07E-48*v[i]**13.4*rho_vec[i]**1.2*self.RN**0.49

			return ans

		elif self.planetObj.ID == 'EARTH':
			
			ans = np.zeros(len(r))
			rho_vec = self.planetObj.rhovectorized(r)

			#REF : Brandis and Johnston, Characterization of Stagnation-Point 
			# Heat Flux for Earth Entry, 2014

			# Tauber-Sutton radiative heating correlation
			C = 4.736E4
			xx = np.array([9000.0, 10000.0, 11000.0, 12000.0, 13000.0,
			              14000.0, 15000.0, 16000.0])
			yy = np.array([1.5, 35, 151, 359, 660, 1065, 1550, 2040])

			fV = interp1d(xx, yy, kind='linear', fill_value=(0.0, 2040), bounds_error=False)

			for i in range(0, len(r)):
				a = 0.6
				b = 1.22
				ans[i] = C*self.RN**a*rho_vec[i]**b*float(fV(v[i]))

			return ans

		elif self.planetObj.ID == 'MARS':
			ans = np.zeros(len(r))
			rho_vec = self.planetObj.rhovectorized(r)
			#REF : Not available; set qrad = 0
			return ans

		elif self.planetObj.ID == 'JUPITER':
			
			ans = np.zeros(len(r))
			rho_vec = self.planetObj.rhovectorized(r)

			ans[:] = (9.7632379E-40*(2*self.RN)**(-0.17905)*(rho_vec[:])**1.763827469*v[:]**10.993852)*1E3/1E4
			# Source: UPITER ENTRY PROBE FEASIBILITY STUDY FROM THE 
			# ESTEC CDF TEAM HEAT FLUX EVALUATION & TPS DEFINITION
			return ans

		elif self.planetObj.ID == 'SATURN':
			
			ans  = np.zeros(len(r))
			rho_vec = self.planetObj.rhovectorized(r)

			ans[:] = (9.7632379E-40*(2*self.RN)**(-0.17905)*(rho_vec[:])**1.763827469*v[:]**10.993852)*1E3/1E4
			# Source: UPITER ENTRY PROBE FEASIBILITY STUDY FROM THE 
			# ESTEC CDF TEAM HEAT FLUX EVALUATION & TPS DEFINITION
			return ans

		elif self.planetObj.ID == 'TITAN':
			ans = np.zeros(len(r))
			rho_vec = self.planetObj.rhovectorized(r)
			#REF : Not available; set qrad = 0
			return ans

		elif self.planetObj.ID == 'URANUS':
			ans = np.zeros(len(r))
			rho_vec = self.planetObj.rhovectorized(r)

			# REF: NASA Vision Neptune orbiter with probes, 
			# Contract No. NNH04CC41C Final Report, 2005
			ans[:] = 8.125812E-3*rho_vec[:]**0.498140*(v[:]/10000)**15.113*(self.RN/0.291)
			return ans

		elif self.planetObj.ID == 'NEPTUNE':
			
			ans = np.zeros(len(r))
			rho_vec = self.planetObj.rhovectorized(r)

			# REF: NASA Vision Neptune orbiter with probes, 
			# Contract No. NNH04CC41C Final Report, 2005

			ans[:] = 8.125812E-3*rho_vec[:]**0.498140*(v[:]/10000)**15.113*(self.RN/0.291)
			return ans

		else:
			print(" >>> ERR : Invalid planet identifier provided.")

	def qStagTotal(self, r, v):
		"""
		Computes the total heat rate which is the sum of the
		convective and radiative heating rates.
		
		Parameters
		----------
		r : numpy.ndarray
			radial distance solution array of trajectory, m
		v : numpy.ndarray
			planet-relative speed array of trajectory, m/s

		Returns
		----------
		ans : numpy.ndarray
			total stagnation-point heating rate array, W/cm2
		"""

		ans = np.zeros(len(r))
		qStagCon = self.qStagConvective(r, v)
		qStagRad = self.qStagRadiative(r, v)
		ans = qStagCon + qStagRad

		return ans

	def L(self, r, theta, phi, v):
		"""
		Computes the vehicle aerodynamic lift, as a function of the
		vehicle location(r,theta, phi), and velocity. 

		Parameters
		----------
		r : float
			radial position value, scalar, m
		theta : float
			longitude, radians
		phi : float
			latitude, radians
		v : float
			planet-relative speed, m/s

		Returns
		----------
		ans : float
			aerodynamic lift force, N

		"""

		ans = 0.5*self.planetObj.rho(r, theta, phi)*v**2.0*self.A*self.CL
		return ans
	
	def Lvectorized(self, r, theta, phi, v):
		"""
		Vectorized version of the L() function

		Computesthe vehicle aerodynamic lift array
		over the provided trajectory array, as a function of the
		vehicle location array (r[:],theta[:], phi[:]), and velocity. 

		Parameters
		----------
		r : numpy.ndarray
			radial position array, m
		theta : numpy.ndarray
			longitude array, radians
		phi : numpy.ndarray
			latitude array, radians
		v : numpy.ndarray
			planet-relative speed array, m/s

		Returns
		----------
		ans : numpy.ndarray
			aerodynamic lift force array, N

		"""

		ans = np.zeros(len(r))
		rho_vec = self.planetObj.rhovectorized(r)
		ans[:] = 0.5*rho_vec[:]*v[:]**2.0*self.A*self.CL
		return ans

	def D(self, r, theta, phi, v):
		"""
		Computes the vehicle aerodynamic drag, as a function of the
		vehicle location(r,theta, phi), and velocity. 

		Parameters
		----------
		r : float
			radial position value, scalar, m
		theta : float
			longitude, radians
		phi : float
			latitude, radians
		v : float
			planet-relative speed, m/s

		Returns
		----------
		ans : float
			aerodynamic drag force, N

		"""

		if self.userDefinedCDMach is True:
			self.CD1 = self.CDMach(self.computeMachScalar(r, v))

		else:
			self.CD1 = self.CD

		ans = 0.5*self.planetObj.rho(r, theta, phi)*v**2.0*self.A*self.CD1
		return ans

	def Dvectorized(self, r, theta, phi, v):
		"""
		Vectorized version of the D() function

		Computesthe vehicle aerodynamic drag array
		over the provided trajectory array, as a function of the
		vehicle location array (r[:],theta[:], phi[:]), and velocity. 

		Parameters
		----------
		r : numpy.ndarray
			radial position array, m
		theta : numpy.ndarray
			longitude array, radians
		phi : numpy.ndarray
			latitude array, radians
		v : numpy.ndarray
			planet-relative speed array, m/s

		Returns
		----------
		ans : numpy.ndarray
			aerodynamic drag force array, N

		"""

		ans = np.zeros(len(r))
		rho_vec = self.planetObj.rhovectorized(r)
		
		if self.userDefinedCDMach is True:
			self.CD_vec = self.CDMach(self.computeMach(r,v))
		else:
			self.CD_vec = self.CD*np.ones(len(v))

		ans[:] = 0.5*rho_vec[:]*v[:]**2.0*self.A*self.CD_vec[:]

		return ans

	def Lbar(self, rbar, theta, phi, vbar):
		"""
		Computes the non-dimensional vehicle aerodynamic lift, 
		as a function of the vehicle location(r,theta, phi), 
		and velocity. 

		Parameters
		----------
		rbar : float
			non-dimensional radial position
		theta : float
			longitude, radians
		phi : float
			latitude, radians
		vbar : float
			non-dimensional planet-relative speed, m/s

		Returns
		----------
		ans : float
			non-dimensional aerodynamic lift force

		"""

		ans = 0.5*self.planetObj.rhobar(rbar,theta,phi)*vbar**2.0*self.Abar*self.CL
		return ans

	def Dbar(self, rbar, theta, phi, vbar):
		"""
		Computes the non-dimensional vehicle aerodynamic drag, 
		as a function of the vehicle location(r,theta, phi), 
		and velocity. 

		Parameters
		----------
		rbar : float
			non-dimensional radial position
		theta : float
			longitude, radians
		phi : float
			latitude, radians
		vbar : float
			non-dimensional planet-relative speed, m/s

		Returns
		----------
		ans : float
			non-dimensional aerodynamic drag force

		"""
		if self.userDefinedCDMach is True:
			self.CD1 = self.CDMach(self.computeMachScalar(rbar*self.planetObj.RP,vbar*self.planetObj.Vref))
		else:
			self.CD1 = self.CD

		ans = 0.5*self.planetObj.rhobar(rbar,theta,phi)*vbar**2.0*self.Abar*self.CD1
		return ans

	def a_s(self, r, theta, phi, v, delta):
		"""
		Function to return tangential acceleration term a_s;
		
		a_s is the tangential acceleration term along the direction of 
		the velocity vector.

		Current formulation does not include thrust, can be added here 
		later in place of 0.0

		Parameters
		----------
		r : float
			radial position value, scalar, m
		theta : float
			longitude, radians
		phi : float
			latitude, radians
		v : float
			planet-relative speed, m/s
		delta : float
			bank angle, rad


		Returns
		----------
		ans : float
			tangential acceleration term a_s

		"""
		ans = (0.0*np.cos(self.alpha) - self.D(r, theta, phi, v)/self.mass)
		return ans

	def a_svectorized(self, r, theta, phi, v, delta):

		"""
		Vectorized version of the a_s() function

		Parameters
		----------
		r : numpy.ndarray
			radial position array, m
		theta : numpy.ndarray
			longitude array, radians
		phi : numpy.ndarray
			latitude array, radians
		v : numpy.ndarray
			planet-relative speed array, m/s
		delta : float
			bank angle, rad


		Returns
		----------
		ans : numpy.ndarray
			tangential acceleration array, m/s2

		"""

		ans = np.zeros(len(r))
		T = np.zeros(len(r))
		D_vec = self.Dvectorized(r, theta, phi, v)
		ans[:] = (T[:]*np.cos(self.alpha) - D_vec[:]/self.mass)
		return ans

	def a_n(self,r,theta,phi,v,delta):
		"""
		Function to return normal acceleration term a_n;
		
		a_n is the normal acceleration term along perpendicular 
		to the velocity vector, in the plane of the trajectory.

		Current formulation does not include thrust, can be added here 
		later in place of 0.0

		Parameters
		----------
		r : float
			radial position value, scalar, m
		theta : float
			longitude, radians
		phi : float
			latitude, radians
		v : float
			planet-relative speed, m/s
		delta : float
			bank angle, rad


		Returns
		----------
		ans : float
			normal acceleration term a_n

		"""
		ans = (self.L(r, theta, phi, v) + 0.0*np.sin(self.alpha))*np.cos(delta)/self.mass
		return ans

	def a_nvectorized(self,r,theta,phi,v,delta):
		"""
		Vectorized version of the a_n() function

		Parameters
		----------
		r : numpy.ndarray
			radial position array, m
		theta : numpy.ndarray
			longitude array, radians
		phi : numpy.ndarray
			latitude array, radians
		v : numpy.ndarray
			planet-relative speed array, m/s
		delta : float
			bank angle, rad


		Returns
		----------
		ans : numpy.ndarray
			normal acceleration array, m/s2

		"""

		ans = np.zeros(len(r))
		T = np.zeros(len(r))
		L_vec = self.Lvectorized(r, theta, phi, v)
		ans[:] = (L_vec[:] + T[:]*np.sin(self.alpha))*np.cos(delta)/self.mass
		return ans

	def a_w(self, r, theta, phi, v, delta):
		"""
		Function to return binormal acceleration term a_w;
		
		a_n is the binormal acceleration term along perpendicular 
		to the velocity vector, perpendicular to 
		the plane of the trajectory.

		Current formulation does not include thrust, can be added here 
		later in place of 0.0

		Parameters
		----------
		r : float
			radial position value, scalar, m
		theta : float
			longitude, radians
		phi : float
			latitude, radians
		v : float
			planet-relative speed, m/s
		delta : float
			bank angle, rad


		Returns
		----------
		ans : float
			binormal acceleration term a_n

		"""
		ans = (self.L(r, theta, phi, v) + 0.0*np.sin(self.alpha))*np.sin(delta)/self.mass
		return ans

	def a_wvectorized(self, r, theta, phi, v, delta):
		"""
		Vectorized version of the a_w() function

		Parameters
		----------
		r : numpy.ndarray
			radial position array, m
		theta : numpy.ndarray
			longitude array, radians
		phi : numpy.ndarray
			latitude array, radians
		v : numpy.ndarray
			planet-relative speed array, m/s
		delta : float
			bank angle, rad


		Returns
		----------
		ans : numpy.ndarray
			binormal acceleration array, m/s2

		"""

		ans = np.zeros(len(r))
		T = np.zeros(len(r))
		L_vec = self.Lvectorized(r, theta, phi, v)
		ans[:] = (L_vec[:] + T[:]*np.sin(self.alpha))*np.sin(delta)/self.mass
		return ans

	def a_sbar(self, rbar, theta, phi, vbar, delta):
		"""
		Function to return non-dimensional 
		tangential acceleration term a_sbar;
		
		a_sbar is the tangential acceleration term along 
		the direction of the velocity vector.

		Current formulation does not include thrust, can be added here 
		later in place of 0.0

		Parameters
		----------
		rbar : float
			non-dimensional radial position
		theta : float
			longitude, radians
		phi : float
			latitude, radians
		vbar : float
			non-dimensional planet-relative speed, m/s
		delta : float
			bank angle, rad

		Returns
		----------
		ans : float
			non-dimensional tangential acceleration term a_sbar

		"""
		ans = (0.0*np.cos(self.alpha) - self.Dbar(rbar, theta, phi, vbar))/self.mbar
		return ans

	def a_nbar(self, rbar, theta, phi, vbar, delta):
		"""
		Function to return non-dimensional 
		normall acceleration term a_nbar;
		
		a_nbar is the tangential acceleration term along 
		the direction of the velocity vector.

		Current formulation does not include thrust, can be added here 
		later in place of 0.0

		Parameters
		----------
		rbar : float
			non-dimensional radial position
		theta : float
			longitude, radians
		phi : float
			latitude, radians
		vbar : float
			non-dimensional planet-relative speed, m/s
		delta : float
			bank angle, rad

		Returns
		----------
		ans : float
			non-dimensional tangential acceleration term a_nbar

		"""
		ans = (self.Lbar(rbar, theta, phi, vbar) + 0.0*np.sin(self.alpha))*np.cos(delta)/self.mbar
		return ans

	def a_wbar(self, rbar, theta, phi, vbar, delta):
		"""
		Function to return non-dimensional 
		normal acceleration term a_wbar;
		
		a_wbar is the tangential acceleration term along 
		the direction of the velocity vector.

		Current formulation does not include thrust, can be added here 
		later in place of 0.0

		Parameters
		----------
		rbar : float
			non-dimensional radial position
		theta : float
			longitude, radians
		phi : float
			latitude, radians
		vbar : float
			non-dimensional planet-relative speed, m/s
		delta : float
			bank angle, rad

		Returns
		----------
		ans : float
			non-dimensional tangential acceleration term a_wbar

		"""
		ans = (self.Lbar(rbar, theta, phi, vbar) + 0.0*np.sin(self.alpha))*np.sin(delta)/self.mbar
		return ans

	def cfvbar(self, rbar, phi, vbar, psi, gamma):
		"""
		Function to return non dimensional centrifugal 
		acceleration term cfvbar
		
		cfvbar is the non dimensional centrifugal acceleration 
		term cfvbar in the EOM

		Parameters
		----------
		rbar : float
			non-dimensional radial position
		phi : float
			latitude, radians
		vbar : float
			non-dimensional planet-relative speed, m/s
		psi : float
			heading angle, rad
		gamma : float
			flight-path angle, rad

		Returns
		----------
		ans : float
			non dimensional centrifugal acceleration cfvbar

		"""
		ans = self.planetObj.OMEGAbar**2.0*rbar*np.cos(phi)* (np.sin(gamma)*np.cos(phi) - np.cos(gamma)*np.sin(phi)*np.sin(psi))
		return ans

	def cfpsibar(self, rbar, phi, vbar, psi, gamma):
		"""
		Function to return non dimensional centrifugal 
		acceleration term cfpsibar
		
		cfpsibar is the non dimensional centrifugal acceleration 
		term  in the EOM

		Parameters
		----------
		rbar : float
			non-dimensional radial position
		phi : float
			latitude, radians
		vbar : float
			non-dimensional planet-relative speed, m/s
		psi : float
			heading angle, rad
		gamma : float
			flight-path angle, rad

		Returns
		----------
		ans : float
			non dimensional centrifugal acceleration cfpsibar

		"""
		ans = (-1.0*self.planetObj.OMEGAbar**2.0*rbar / (vbar*(np.cos(gamma) + 1e-2)))*np.sin(phi)*np.cos(phi)*np.cos(psi)
		return ans

	def cfgammabar(self, rbar, phi, vbar, psi, gamma):
		"""
		Function to return non dimensional centrifugal 
		acceleration term cfgammabar
		
		cfgammabar is the non dimensional centrifugal acceleration 
		term  in the EOM

		Parameters
		----------
		rbar : float
			non-dimensional radial position
		phi : float
			latitude, radians
		vbar : float
			non-dimensional planet-relative speed, m/s
		psi : float
			heading angle, rad
		gamma : float
			flight-path angle, rad

		Returns
		----------
		ans : float
			non dimensional centrifugal acceleration cfpsibar

		"""
		ans = (self.planetObj.OMEGAbar**2.0*rbar/vbar)*np.cos(phi)*(np.cos(gamma)*np.cos(phi) + np.sin(gamma)*np.sin(phi)*np.sin(psi))
		return ans

	def copsibar(self, rbar, phi, vbar, psi, gamma):
		"""
		Function to return non dimensional Coriolis 
		acceleration term copsibar
		
		Parameters
		----------
		rbar : float
			non-dimensional radial position
		phi : float
			latitude, radians
		vbar : float
			non-dimensional planet-relative speed, m/s
		psi : float
			heading angle, rad
		gamma : float
			flight-path angle, rad

		Returns
		----------
		ans : float
			non dimensional Coriolis acceleration copsibar

		"""
		ans = 2.0*self.planetObj.OMEGAbar*((np.sin(gamma)/(np.cos(gamma) + 1e-2))*np.cos(phi)*np.sin(psi) - np.sin(phi))
		return ans

	def cogammabar(self, rbar, phi, vbar, psi, gamma):
		"""
		Function to return non dimensional Coriolis 
		acceleration term cogammabar
		
		Parameters
		----------
		rbar : float
			non-dimensional radial position
		phi : float
			latitude, radians
		vbar : float
			non-dimensional planet-relative speed, m/s
		psi : float
			heading angle, rad
		gamma : float
			flight-path angle, rad

		Returns
		----------
		ans : float
			non dimensional Coriolis acceleration cogammabar

		"""
		ans = 2.0*self.planetObj.OMEGAbar*np.cos(phi)*np.cos(psi)
		return ans

	def grbar(self, rbar, phi):
		"""
		Returns the non-dimensional gravity radial acceleration term grbar.
		
		grbar is the non dimensional gravity  
		radial acceleration term grbar in the EOM
		
		Parameters
		----------
		rbar : float
			non-dimensional radial position
		phi : float
			latitude, radians
		
		Returns
		----------
		ans : float
			non-dimensional radial acceleration term grbar

		"""

		term1 = -1.0/(rbar**2.0)
		term2 = (1.5*self.planetObj.J2/rbar**4.0) * (3.0*np.sin(phi)*np.sin(phi)-1.0)
		term3 = (2.0*self.planetObj.J3/rbar**5.0) * (5.0*(np.sin(phi))**3.0 - 3.0*np.sin(phi))
		ans = term1 + term2 + term3
		return ans

	def gthetabar(self, rbar, phi):
		"""
		Returns the non-dimensional gravity longitudinal acceleration term.
		
		gthetabar is the non dimensional longitudinal
		gravity  acceleration term grbar in the EOM
		
		Parameters
		----------
		rbar : float
			non-dimensional radial position
		phi : float
			latitude, radians
		
		Returns
		----------
		ans : float
			non-dimensional longitudinal gravity 
			acceleration term gthetabar

		"""
		return 0.0

	def gphibar(self, rbar, phi):
		"""
		Returns the non-dimensional gravity 
		latitudinal acceleration term gphibar.
		
		Parameters
		----------
		rbar : float
			non-dimensional radial position
		phi : float
			latitude, radians
		
		Returns
		----------
		ans : float
			non-dimensional gravity 
			latitudinal acceleration term gphibar

		"""
		term1 = (-3.0*self.planetObj.J2/rbar**4.0) * np.sin(phi)*np.cos(phi)
		term2 = ( 1.5*self.planetObj.J3/rbar**5.0) * np.cos(phi)*(1.0 - 5.0*np.sin(phi)*np.sin(phi))
		ans = term1 + term2
		
		return ans

	def gnbar(self, rbar, phi, gamma, psi):
		"""
		Returns the non-dimensional gravity normal acceleration term gnbar.
		
		Parameters
		----------
		rbar : float
			non-dimensional radial position
		phi : float
			latitude, radians
		gamma : float
			flight-path angle, rad
		psi : float
			heading angle, rad
		
		Returns
		----------
		ans : float
			non-dimensional gravity normal acceleration term gnbar

		"""
		term1 = np.cos(gamma)*self.grbar(rbar, phi)
		term2 = -1.0*np.sin(gamma)*np.cos(psi)*self.gthetabar(rbar, phi)
		term3 = -1.0*np.sin(gamma)*np.sin(psi)*self.gphibar(rbar, phi)
		ans = term1 + term2 + term3
		return ans

	def gsbar(self, rbar, phi, gamma, psi):
		"""
		Returns the non-dimensional gravity tangential
		acceleration term gsbar.
		
		Parameters
		----------
		rbar : float
			non-dimensional radial position
		phi : float
			latitude, radians
		gamma : float
			flight-path angle, rad
		psi : float
			heading angle, rad
		
		Returns
		----------
		ans : float
			non-dimensional gravity tangential acceleration term gsbar

		"""	
		term1 = np.sin(gamma)*self.grbar(rbar, phi)
		term2 = np.cos(gamma)*np.cos(psi)*self.gthetabar(rbar, phi)
		term3 = np.cos(gamma)*np.sin(psi)*self.gphibar(rbar, phi)
		ans = term1 + term2 + term3
		return ans

	def gwbar(self, rbar, phi, gamma, psi):
		"""
		Returns the non-dimensional gravity binormal
		acceleration term gwbar.
		
		Parameters
		----------
		rbar : float
			non-dimensional radial position
		phi : float
			latitude, radians
		gamma : float
			flight-path angle, rad
		psi : float
			heading angle, rad
		
		Returns
		----------
		ans : float
			non-dimensional gravity binormal acceleration term gwbar

		"""
		term1 = -1.0*np.sin(psi)*self.gthetabar(rbar, phi)
		term2 = np.cos(psi)*self.gphibar(rbar, phi)
		ans = term1 + term2
		return ans

	def avoid_singularity(self, x):
		if abs(x + np.pi/2) < 2e-3:
			return -np.pi/2 + 4e-3
		else:
			return x

	def EOM(self, y, t, delta):
		"""
		Define the EoMs to propogate the 3DoF trajectory inside the 
		atmosphere of a an oblate rotating planet.

		Reference 1: Vinh, Chapter 3.
		Reference 2: Lescynzki, MS Thesis, NPS.

		Parameters
		----------
		y : numpy.ndarray
			trajectory state vector
		t : numpy.ndarray
			trajectory time vector
		delta : float
			bank angle, rad
		
		Returns
		----------
		ans : dydt
			derivate vector of state, process equations

		"""

		rbar, theta, phi, vbar, psi, gamma, drangebar = y
		dydt = [ 
				(vbar*np.sin(gamma)),
				(vbar*(np.cos(gamma))*np.cos(psi)) / (rbar*np.cos(phi)),
				(vbar*(np.cos(gamma))*np.sin(psi)) / rbar,
				(self.a_sbar(rbar, theta, phi, vbar, delta)) + self.gsbar(rbar, phi, gamma, psi)
				+ self.cfvbar(rbar, phi, vbar, psi, gamma),
				(self.a_wbar(rbar, theta, phi, vbar, delta) + self.gwbar(rbar, phi, gamma, psi))/(vbar*(np.cos(gamma) + 1e-2))
				- (1.0*vbar/rbar)*np.cos(gamma)*np.cos(psi)*np.tan(phi)
				+ self.cfpsibar(rbar, phi, vbar, psi, gamma)
				+ self.copsibar(rbar, phi, vbar, psi, gamma),
				(self.a_nbar(rbar, theta, phi, vbar, delta) + self.gnbar(rbar,phi,gamma,psi))/vbar
				+ (vbar/rbar)*np.cos(gamma)
				+ self.cfgammabar(rbar, phi, vbar, psi, gamma)
				+ self.cogammabar(rbar, phi, vbar, psi, gamma),
				(vbar*(np.cos(gamma)))
				]
		return dydt

	def EOM2(self, t, y, delta):
		"""
		Define the EoMs to propogate the 3DoF trajectory inside the
		atmosphere of a an oblate rotating planet.

		Reference 1: Vinh, Chapter 3.
		Reference 2: Lescynzki, MS Thesis, NPS.

		Parameters
		----------
		y : numpy.ndarray
			trajectory state vector
		t : numpy.ndarray
			trajectory time vector
		delta : float
			bank angle, rad

		Returns
		----------
		ans : dydt
			derivate vector of state, process equations

		"""

		rbar, theta, phi, vbar, psi, gamma, drangebar = y
		dydt = [
			(vbar * np.sin(gamma)),
			(vbar * (np.cos(gamma)) * np.cos(psi)) / (rbar * np.cos(phi)),
			(vbar * (np.cos(gamma)) * np.sin(psi)) / rbar,
			(self.a_sbar(rbar, theta, phi, vbar, delta)) + self.gsbar(rbar, phi, gamma, psi)
			+ self.cfvbar(rbar, phi, vbar, psi, gamma),
			(self.a_wbar(rbar, theta, phi, vbar, delta) + self.gwbar(rbar, phi, gamma, psi))/(vbar*(np.cos(gamma) + 1e-2))
			- (1.0 * vbar / rbar) * np.cos(gamma) * np.cos(psi) * np.tan(phi)
			+ self.cfpsibar(rbar, phi, vbar, psi, gamma)
			+ self.copsibar(rbar, phi, vbar, psi, gamma),
			(self.a_nbar(rbar, theta, phi, vbar, delta) + self.gnbar(rbar, phi, gamma, psi)) / vbar
			+ (vbar/rbar) * np.cos(gamma)
			+ self.cfgammabar(rbar, phi, vbar, psi, gamma)
			+ self.cogammabar(rbar, phi, vbar, psi, gamma),
			(vbar * (np.cos(gamma)))
		]
		return dydt

	def solveTrajectory(self, rbar0, theta0, phi0, vbar0, psi0,	gamma0, drangebar0, t_sec, dt, delta):
		"""
		Function to propogate a single atmospheric entry trajectory 
		given entry interface / other initial conditions and
		bank angle delta.

		Reference 1: Vinh, Chapter 3.
		Reference 2: Lescynzki, MS Thesis, NPS.

		Parameters
		----------
		rbar0 : float
			non-dimensional radial distance initial condition
		theta0 : float
			longitude initial condition, rad
		phi0 : float
			latatitude initial condition, rad
		vbar0 : float
			non-dimensional planet-relative speed initial condition
		psi0 : float
			heading angle initial condition, rad
		gamma0 : float
			entry flight-path angle initial condition, rad
		drangebar0 : float
			non-dimensional downrange initial condition
		t_sec : float
			time in seconds for which propogation is done
		dt : float
			max. time step size in seconds
		delta : float
			bank angle command, rad

		Returns
		----------
		tbar : numpy.ndarray
			nondimensional time at which solution is computed
		rbar : numpy.ndarray
			nondimensional radial distance solution
		theta : numpy.ndarray
			longitude solution, rad
		phi : numpy.ndarray
			latitude array, rad
		vbar : numpy.ndarray
			nondimensional velocity solution
		psi : numpy.ndarray, rad
			heading angle solution, rad
		gamma : numpy.ndarray
			flight-path angle, rad
		drangebar : numpy.ndarray
			downrange solution, meters
		"""

		# store nondimensional initial conditions in xbar_0
		xbar_0 = [rbar0, theta0, phi0, vbar0, psi0, gamma0, drangebar0]

		# discretize time interval [0,time] in steps of dt
		tbar = np.arange(0, (t_sec+dt)/self.planetObj.tau, dt/self.planetObj.tau)
		
		# use scipy odeint to solve for the entry trajectory using initial 
		# conditions xbar_0 and vehicle parameters in args
		xbar = odeint(self.EOM, xbar_0, tbar, rtol=self.tol, atol=self.tol, args=(delta,))

		# extract solution from odeint into solution variable vectors
		rbar = xbar[:, 0]    # radial distance rbar solution
		theta = xbar[:, 1]   # longitude theta solution
		phi = xbar[:, 2]     # latitude phi solution
		vbar = xbar[:, 3]    # velocity vbar solution
		psi = xbar[:, 4]     # heading angle psi solution
		gamma = xbar[:, 5]      # flight path angle solution
		drangebar = xbar[:, 6]  # downrange solution

		return tbar, rbar, theta, phi, vbar, psi, gamma, drangebar

	def hit_EFPA_90(self, t, y, delta):
		return y[5] + 88*np.pi/180
	hit_EFPA_90.terminal = True

	def solveTrajectory2(self, rbar0, theta0, phi0, vbar0, psi0, gamma0, drangebar0, t_sec, dt, delta):
		"""
		Function to propogate a single atmospheric entry trajectory
		given entry interface / other initial conditions and
		bank angle delta.

		Reference 1: Vinh, Chapter 3.
		Reference 2: Lescynzki, MS Thesis, NPS.

		Parameters
		----------
		rbar0 : float
			non-dimensional radial distance initial condition
		theta0 : float
			longitude initial condition, rad
		phi0 : float
			latatitude initial condition, rad
		vbar0 : float
			non-dimensional planet-relative speed initial condition
		psi0 : float
			heading angle initial condition, rad
		gamma0 : float
			entry flight-path angle initial condition, rad
		drangebar0 : float
			non-dimensional downrange initial condition
		t_sec : float
			time in seconds for which propogation is done
		dt : float
			max. time step size in seconds
		delta : float
			bank angle command, rad

		Returns
		----------
		tbar : numpy.ndarray
			nondimensional time at which solution is computed
		rbar : numpy.ndarray
			nondimensional radial distance solution
		theta : numpy.ndarray
			longitude solution, rad
		phi : numpy.ndarray
			latitude array, rad
		vbar : numpy.ndarray
			nondimensional velocity solution
		psi : numpy.ndarray, rad
			heading angle solution, rad
		gamma : numpy.ndarray
			flight-path angle, rad
		drangebar : numpy.ndarray
			downrange solution, meters
		"""

		# store nondimensional initial conditions in xbar_0
		xbar_0 = [rbar0, theta0, phi0, vbar0, psi0, gamma0, drangebar0]

		# discretize time interval [0,time] in steps of dt
		tbar = np.arange(0, (t_sec + dt) / self.planetObj.tau, dt / self.planetObj.tau)

		# use scipy odeint to solve for the entry trajectory using initial
		# conditions xbar_0 and vehicle parameters in args
		xbar = solve_ivp(self.EOM2, (0, tbar[-1]), xbar_0, t_eval=tbar, rtol=self.tol, atol=self.tol, events=self.hit_EFPA_90, args=(delta,))

		tbar = xbar.t
		# extract solution from odeint into solution variable vectors
		rbar = xbar.y[0, :]  # radial distance rbar solution
		theta = xbar.y[1, :]   # longitude theta solution
		phi = xbar.y[2, :]   # latitude phi solution
		vbar = xbar.y[3, :]  # velocity vbar solution
		psi = xbar.y[4, :]   # heading angle psi solution
		gamma = xbar.y[5, :]   # flight path angle solution
		drangebar = xbar.y[6, :]   # downrange solution

		return tbar, rbar, theta, phi, vbar, psi, gamma, drangebar

	def convertToPlotUnits(self, t, r, v, phi, psi, theta, gamma, drange):
		"""
		Convert state vector components to units appropriate 
		for evolution plots.

		Parameters
		----------
		t : numpy.ndarray
			time array, sec
		r : numpy.ndarray
			radial distance array, m
		v : numpy.ndarray
			speed array, m
		phi : numpy.ndarray
			latitude array, rad
		psi : numpy.ndarray
			heading angle array, rad
		theta : numpy.ndarray
			longitude array, rad
		gamma : numpy.ndarray
			flight path angle array, rad
		drange : numpy.ndarray
			downrange array, meters

		Returns
		----------
		t_min : numpy.ndarray
			time array, minutes
		h_km : numpy.ndarray
			altitude array, km
		v_kms : numpy.ndarray
			speed array, km/s
		phi_deg : numpy.ndarray
			latitude array, deg
		psi_deg : numpy.ndarray
			heading angle array, deg
		theta_deg : numpy.ndarray
			longitude array, deg
		gamma_deg : numpy.ndarray
			flight path angle array, deg
		drange_km : numpy.ndarray
			downrange array, km
		"""
		t_min = t/60.0
		h_km = (r - self.planetObj.RP)*1E-3
		v_kms = v*1E-3
		phi_deg = phi*180/np.pi
		psi_deg = psi*180/np.pi
		theta_deg = theta*180/np.pi
		gamma_deg = gamma*180/np.pi
		drange_km = drange*1.0E-3

		return t_min, h_km, v_kms, phi_deg, psi_deg, theta_deg, gamma_deg, drange_km

	def convertToKPa(self, pres):
		"""
		Convert a pressure solution array from Pa to kPa. 

		Parameters
		----------
		pres : numpy.ndarray
			pressure (static/dynamic/total), Pascal (Pa)
		
		Returns
		----------
		ans : numpy.ndarray
			pressure (static/dynamic/total), kiloPascal (kPa)

		"""
		
		return pres/1000.0

	def convertToPerCm2(self, heatrate):
		"""
		Convert a heat rate from W/m2 to W/cm2. 

		Parameters
		----------
		heatrate : numpy.ndarray
			stagnation-point heat rate, W/m2
		
		Returns
		----------
		ans : numpy.ndarray
			stagnation-point heat rate, W/cm2

		"""
		
		return heatrate/10000.0

	def classifyTrajectory(self, r):
		"""
		This function checks the trajectory for "events" which are 
		used to truncate the trajectory
		
		A "skip out event" is said to occur as when the vehicle 
		altitude did not hit the surface and exceeds the prescribed 
		skip out altitude.
		
		A "time out event" is said to occur if the vehicle did not 
		exceed the skip out altitude, and did not reach the trap in 
		altitude.
		
		A "trap in" event is said to occur when the vehicle altitude 
		falls below the prescribed trap in altitude.
		
		This function checks for these events and returns the array 
		index of the "event" location and an exitflag
		
		exitflag to indicate if an event was detected or not.
		exitflag = 1.0 indicates "skip out event" has occured.
		exitflag = 0.0 indicates no "event" was detected in the trajectory, 
		consider increasing simulation time.
		exitflag = -1.0 indicates "trap in event" has occured.

		Parameters
		----------
		r : numpy.ndarray
			dimensional radial distance solution, meters

		Returns
		----------
		index : int
			array index of the event location if one was detected, 
			terminal index otherwise
		exitflag : int
			flag to indicate and classify event occurence or lack of it			

		
		"""
		# Compute altitude history for radial distance solution
		h = self.planetObj.computeH(r)

		# Compute the maximum altitude over the entire trajectory
		h_max = max(h)
		# Compute the minimum altitude over the entire trajectory
		h_min = min(h)

		# Check for event occurences

		if h_min >= 0 and h_max > self.planetObj.h_skip:
			# Check for a "skip out event" occurence.
			# if trajectory did not hit ground and max 
			# altitude > predefined skip altitude
			# then return the index of the first point after skipping out
			# set exitflag = 1.0 to indicate trajectory skipped out
			index = np.argmax(h > self.planetObj.h_skip)
			exitflag = 1.0

		elif h_min >= self.planetObj.h_trap and h_max < self.planetObj.h_skip:
			# if trajectory did not reach trap in altitude and max altiude 
			# < predefined skip altitude
			# then return the index of the last point ie. full set of data 
			# set exitflag = 0 to indicate trajectory stopped before reaching 
			# trap altitude.
			# if this "event" occurs, the time for which the simulation is run 
			# should be increased until exitflag = 1.0
			index = len(h)
			exitflag = 0.0

		elif h_min <= self.planetObj.h_trap:
			# if trajectory descended below trap in altitude
			# return the index of the first point after entering trap altiude
			# set exitflag = -1 to indicate vehicle hit the surface
			index = np.argmax(h < self.planetObj.h_trap)
			exitflag = -1.0

		else:
			# if anything else happens, set index to max value
			# set exitflag = 0.0, to let user know that no "event" was detected.
			index = len(h)
			exitflag = 0.0

		return index, exitflag

	def truncateTrajectory(self, t, r, theta, phi, v, psi, gamma, drange, index):
		"""
		This function truncates the full trajectory returned by the solver 
		to the first event location.
		
		The full trajectory returned by the solver could have skipped out 
		exceeding the skip out altitude, or could have descended below the 
		trap in altitude or even below the surface.
		
		This function helps ensure that we truncate the trajectory to what 
		we actually need, i.e. till a skip out / trap in event.

		Parameters
		----------
		t : numpy.ndarray
			time array, sec
		r : numpy.ndarray
			radial distance array, m
		v : numpy.ndarray
			speed array, m
		phi : numpy.ndarray
			latitude array, rad
		psi : numpy.ndarray
			heading angle array, rad
		theta : numpy.ndarray
			longitude array, rad
		gamma : numpy.ndarray
			flight path angle array, rad
		drange : numpy.ndarray
			downrange array, meters
		index : int
			array index of detected event location / 
			max index if no event detected

		Returns
		----------
		t : numpy.ndarray
			truncated time array, sec
		r : numpy.ndarray
			truncated radial distance array, m
		v : numpy.ndarray
			truncated speed array, m
		phi : numpy.ndarray
			truncated latitude array, rad
		psi : numpy.ndarray
			truncated heading angle array, rad
		theta : numpy.ndarray
			truncated longitude array, rad
		gamma : numpy.ndarray
			truncated flight path angle array, rad
		drange : numpy.ndarray
			truncated downrange array, meters
		
		"""

		return t[0:index], r[0:index], theta[0:index], phi[0:index], v[0:index], psi[0:index], gamma[0:index], drange[0:index]


	def computeAccelerationLoad(self, tc, rc, thetac, phic, vc, index, delta):
		"""
		This function computes the acceleration load (Earth G's) over 
		the entire trajectory from trajectory data returned by the solver.

		Parameters
		----------
		tc : numpy.ndarray
			truncated time array, sec
		rc : numpy.ndarray
			truncated radial distance array, m
		thetac : numpy.ndarray
			truncated longitude array, rad
		phic : numpy.ndarray
			trucnated latitude array, rad
		vc : numpy.ndarray
			truncated speed array, m
		index : int
			array index of detected event location / 
			max index if no event detected
		delta : float
			bank angle, rad

		Returns
		----------
		acc_net_g : numpy.ndarray
			acceleration load (Earth G's) over the entire trajectory
		"""

		acc_s = np.zeros(index)
		acc_n = np.zeros(index)
		acc_w = np.zeros(index)

		acc_s = self.a_svectorized(rc, thetac, phic, vc, delta)
		acc_n = self.a_nvectorized(rc, thetac, phic, vc, delta)
		acc_w = self.a_wvectorized(rc, thetac, phic, vc, delta)

		acc_net = np.zeros(index)
		acc_net[:] = np.sqrt(acc_s[:]**2.0 + acc_n[:]**2.0 + acc_w[:]*2.0)
		acc_net_g = acc_net / self.planetObj.EARTHG

		return acc_net_g

	def computeAccelerationDrag(self, tc, rc, thetac, phic, vc, index, delta):
		"""
		This function computes the drag acceleration load (Earth G's) over 
		the entire trajectory from trajectory data returned by the solver.

		Parameters
		----------
		tc : numpy.ndarray
			truncated time array, sec
		rc : numpy.ndarray
			truncated radial distance array, m
		thetac : numpy.ndarray
			truncated longitude array, rad
		phic : numpy.ndarray
			trucnated latitude array, rad
		vc : numpy.ndarray
			truncated speed array, m
		index : int
			array index of detected event location / 
			max index if no event detected
		delta : float
			bank angle, rad

		Returns
		----------
		acc_drag_g : numpy.ndarray
			drag acceleration load (Earth G's) over the entire trajectory
		"""
		
		acc_s = np.zeros(index)
		acc_n = np.zeros(index)
		acc_w = np.zeros(index)

		acc_s = self.a_svectorized(rc, thetac, phic, vc, delta)
		acc_n = self.a_nvectorized(rc, thetac, phic, vc, delta)
		acc_w = self.a_wvectorized(rc, thetac, phic, vc, delta)

		acc_drag = np.zeros(index)
		acc_drag[:] = np.sqrt(acc_s[:]**2.0 + 0.0*acc_n[:]**2.0 + 0.0*acc_w[:]*2.0)
		acc_drag_g = acc_drag / self.planetObj.EARTHG

		return acc_drag_g

	def computeDynPres(self, r, v):
		"""
		This function computes the dynamic pressure over the 
		entire trajectory.

		Parameters
		----------
		r : numpy.ndarray
			radial distance array, m
		v : numpy.ndarray
			speed array, m/s
		
		Returns
		----------
		ans : numpy.ndarray
			dynamic pressure, Pa
		
		"""

		ans = np.zeros(len(r))
		rho_vec = self.planetObj.rhovectorized(r)
		ans[:] = 0.5 * rho_vec[:] * v[:]**2
		return ans

	def computeStagPres(self, rc, vc):
		"""
		This function computes the stag. pressure over the 
		entire trajectory.

		Parameters
		----------
		rc : numpy.ndarray
			radial distance array, m
		vc : numpy.ndarray
			speed array, m/s
		
		Returns
		----------
		ans : numpy.ndarray
			stag. pressure, Pa
		
		"""
		
		stat_pres = self.planetObj.pressurevectorized(rc)
		stag_pres = np.zeros(len(rc))
		dyn_pres = self.computeDynPres(rc,vc)
		stag_pres = stat_pres + dyn_pres

		return stag_pres

	def computeMach(self, rc, vc):
		"""
		This function computes the Mach. no over the 
		entire trajectory.

		Parameters
		----------
		rc : numpy.ndarray
			radial distance array, m
		vc : numpy.ndarray
			speed array, m/s
		
		Returns
		----------
		ans : numpy.ndarray
			Mach no.
		
		"""
		stat_pres = self.planetObj.pressurevectorized(rc)
		stat_temp = self.planetObj.temperaturevectorized(rc)
		mach = np.zeros(len(rc))
		sonic_spd = self.planetObj.sonicvectorized(rc)
		mach[:] = vc[:]/sonic_spd[:]

		return mach

	def computeMachScalar(self, r, v):
		"""
		This function computes the Mach. no at a single instance.

		Parameters
		----------
		r : float
			radial distance, m
		v : floar
			speed, m/s
		
		Returns
		----------
		ans : float
			Mach no.
		
		"""
		stat_pres = np.float(self.planetObj.pressure_int(r - self.planetObj.RP))
		stat_temp = np.float(self.planetObj.temp_int(r - self.planetObj.RP))
		sonic_spd = np.float(self.planetObj.sonic_int(r - self.planetObj.RP))
		mach = v/sonic_spd

		return mach

	def computeStagTemp(self, rc, vc):
		"""
		This function computes the stag. temperature over the 
		entire trajectory.

		Parameters
		----------
		rc : numpy.ndarray
			radial distance array, m
		vc : numpy.ndarray
			speed array, m/s
		
		Returns
		----------
		ans : numpy.ndarray
			stag. temperature, K
		
		"""
		stat_temp = self.planetObj.temperaturevectorized(rc)
		mach = np.zeros(len(rc))
		sonic_spd = self.planetObj.sonicvectorized(rc)
		mach[:] = vc[:]/sonic_spd[:]
		stag_temp = np.zeros(len(rc))
		stag_temp[:] = stat_temp[:]*(1+0.5*(self.CPCV-1)*mach[:]**2.0)

		return stag_temp

	def computeHeatingForMultipleRN(self, tc, rc, vc, rn_array):
		"""
		This function computes the max. stag. pont heating rate 
		and heat load for an array of nose radii.

		Parameters
		----------
		tc : numpy.ndarray
			truncated time array, sec
		rc : numpy.ndarray
			truncated radial distance array, m
		vc : numpy.ndarray
			speed array, m/s
		rn_array : numpy.ndarray
			nose radius array, m
		
		Returns
		----------
		q_stag_max : numpy.ndarray
			max. heat rate, W/cm2
		heatload : numpy.ndarray
			max. heatload, J/cm2
		
		"""
		
		# store the current vehicle nose radius in a temporary variable
		temp_var = self.RN

		q_stag_max = np.zeros(len(rn_array))
		heatload = np.zeros(len(rn_array))
		count = 0

		for RN in rn_array:
			self.RN = RN       # reset the nose radius to values from rn_array 

			q_stag = self.qStagTotal(rc,vc)
			q_stag_max[count] = max(q_stag)
			heatload[count] = cumtrapz(q_stag, tc, initial=0)[-1]
			count = count+1

		self.RN = temp_var   # set the nose radius to original value of nose radius

		return q_stag_max, heatload

	def computeEnergy(self, rc,vc):
		"""
		This function computes the total specific mechanical energy 
		of the vehicle over the entire trajectory.

		Parameters
		----------
		rc : numpy.ndarray
			radial distance array, m
		vc : numpy.ndarray
			speed array, m/s
		
		Returns
		----------
		ans : numpy.ndarray
			specific energy, J/kg
		
		"""

		energy = np.zeros(len(rc))
		energy[:] = -1.0*self.planetObj.GM/rc[:] + 0.5*vc[:]**2.0

		return energy

	def computeEnergyScalar(self, r, v):
		"""
		This function computes the total specific mechanical energy 
		of the vehicle at an instance.

		Parameters
		----------
		r : float
			radial distance, m
		v : float
			speed array, m/s
		
		Returns
		----------
		ans : float
			specific energy, J/kg
		
		"""

		energy = -1.0*self.planetObj.GM/r + 0.5*v**2.0

		return energy

	def computeSemiMajorAxisScalar(self, E):
		"""
		This function computes the semi-major axis of the orbit given 
		its total specific mechanical energy.
		
		Parameters
		----------
		E : float
			specific energy, J/kg
		
		Returns
		----------
		ans : float
			semi major axis, km
		"""
		
		a = -1.0*self.planetObj.GM/(2*E)  # compute the semi-major axis

		return a

	def computeAngMomScalar(self, terminal_r, terminal_v, terminal_g):
		"""
		This function computes the specific angular momentum (orbital) 
		of the vehicle at an instance given its current radial distance,
		speed, and flight-path angle.

		Parameters
		----------
		terminal_r : float
			radial distance, meters
		terminal_v : float
			speed, meters/sec
		terminal_g : float
			flight-path angle, rad
		
		
		Returns
		----------
		ans : float
			specific angular momentum, SI units
		"""
		
		# compute the specific angular momentum
		angMom = terminal_r*terminal_v*np.cos(terminal_g) 
		return angMom

	def computeEccScalar(self, h, E):
		"""
		This function computes the eccentricity of the orbit given its 
		specific angular momentum, and total specific mechanical energy.
		
		Parameters
		----------
		h : float
			specific angular momentum, SI units
		E : float
			specifice energy, J/kg
		
		Returns
		----------
		ans : float
			eccentricity value
		
		"""

		# compute the eccentricity of the orbit
		ecc = np.sqrt(1.0 + 2*E*h**2.0/self.planetObj.GM**2.0)
		return ecc

	def propogateEntry(self, t_sec, dt, delta_deg):
		"""
		Propogates the vehicle state for a specified time using 
		initial conditions, vehicle properties, and 
		atmospheric profile data.
		
		Parameters
		----------
		t_sec : float
			propogation time, seconds
		dt : float
			max. time step, seconds
		delta_deg : float
			bank angle command, deg
		
		"""

		# Define entry conditions at entry interface
		# Convert initial state variables from input/plot 
		# units to calculation/SI units

		# Entry altitude above planet surface in meters
		# Entry latitude in radians
		# Entry velocity in meters/sec, relative to planet
		# Entry velocity in meters/sec, relative to planet
		# Entry heading angle in radians
		# Entry flight path angle in radians
		# Entry downrange in m

		h0 = self.h0_km*1.0E3
		theta0 = self.theta0_deg*np.pi/180.0
		phi0 = self.phi0_deg*np.pi/180.0
		v0 = self.v0_kms*1.000E3
		psi0 = self.psi0_deg*np.pi/180.0
		gamma0 = self.gamma0_deg*np.pi/180.0
		drange0 = self.drange0_km*1E3          

		# Define control variables
		# Constant bank angle in radians
		delta = delta_deg*np.pi/180.0

		r0 = self.planetObj.computeR(h0)
		
		# Compute non-dimensional entry conditions	
		rbar0, theta0, phi0, vbar0, psi0, gamma0, drangebar0 = self.planetObj.nonDimState(r0, theta0, phi0, v0, psi0,
																						  gamma0, drange0)
		
		# Solve for the entry trajectory
		tbar,rbar,theta,phi,vbar,psi,gamma,drangebar = self.solveTrajectory(rbar0, theta0, phi0, vbar0, psi0, gamma0,
																			drangebar0, t_sec, dt, delta)
		# Note : solver returns non-dimensional variables
		# Convert to dimensional variables for plotting
		t, r, theta, phi, v, psi, gamma, drange = self.planetObj.dimensionalize(tbar, rbar, theta, phi, vbar, psi,
																				gamma, drangebar)

		# dimensional state variables are in SI units
		# convert to more rational units for plotting
		t_min, h_km, v_kms, phi_deg, psi_deg, theta_deg, gamma_deg, drange_km = \
			self.convertToPlotUnits(t, r, v, phi, psi, theta, gamma, drange)

		# classify trajectory
		self.index, self.exitflag = self.classifyTrajectory(r)

		# truncate trajectory
		self.tc, self.rc, self.thetac, self.phic, self.vc, self.psic, self.gammac, self.drangec = \
											self.truncateTrajectory(t, r, theta, phi, v, psi, gamma, drange, self.index)
		self.t_minc, self.h_kmc, self.v_kmsc, self.phi_degc, self.psi_degc, self.theta_degc, self.gamma_degc, self.drange_kmc = \
			self.truncateTrajectory(t_min, h_km, v_kms, phi_deg, psi_deg, theta_deg, gamma_deg, drange_km, self.index)

		# compute acceleration loads
		self.acc_net_g = self.computeAccelerationLoad(self.tc, self.rc, self.thetac, self.phic, self.vc, self.index, delta)
		# compute drag acceleration 
		self.acc_drag_g = self.computeAccelerationDrag(self.tc, self.rc, self.thetac, self.phic, self.vc, self.index,delta)

		# compute dynamic pressure
		self.dyn_pres_atm = self.computeDynPres(self.rc, self.vc)/1.01325E5
		# compute stagnation pressure
		self.stag_pres_atm = self.computeStagPres(self.rc, self.vc)/1.01325E5

		# compute stagnation point convective and radiative heating rate
		self.q_stag_con = self.qStagConvective(self.rc, self.vc)
		self.q_stag_rad = self.qStagRadiative (self.rc, self.vc)
		# compute total stagnation point heating rate
		self.q_stag_total = self.q_stag_con + self.q_stag_rad
		# compute stagnation point heating load
		self.heatload = cumtrapz(self.q_stag_total, self.tc, initial=self.heatLoad0)

	def propogateEntry2(self, t_sec, dt, delta_deg):
		"""
		Propogates the vehicle state for a specified time using
		initial conditions, vehicle properties, and
		atmospheric profile data.

		Parameters
		----------
		t_sec : float
			propogation time, seconds
		dt : float
			max. time step, seconds
		delta_deg : float
			bank angle command, deg

		"""

		# Define entry conditions at entry interface
		# Convert initial state variables from input/plot
		# units to calculation/SI units

		# Entry altitude above planet surface in meters
		# Entry latitude in radians
		# Entry velocity in meters/sec, relative to planet
		# Entry velocity in meters/sec, relative to planet
		# Entry heading angle in radians
		# Entry flight path angle in radians
		# Entry downrange in m

		h0 = self.h0_km * 1.0E3
		theta0 = self.theta0_deg * np.pi / 180.0
		phi0 = self.phi0_deg * np.pi / 180.0
		v0 = self.v0_kms * 1.000E3
		psi0 = self.psi0_deg * np.pi / 180.0
		gamma0 = self.gamma0_deg * np.pi / 180.0
		drange0 = self.drange0_km * 1E3

		# Define control variables
		# Constant bank angle in radians
		delta = delta_deg * np.pi / 180.0

		r0 = self.planetObj.computeR(h0)

		# Compute non-dimensional entry conditions
		rbar0, theta0, phi0, vbar0, psi0, gamma0, drangebar0 = \
			self.planetObj.nonDimState(r0, theta0, phi0, v0, psi0, gamma0, drange0)

		# Solve for the entry trajectory
		tbar, rbar, theta, phi, vbar, psi, gamma, drangebar = \
			self.solveTrajectory2(rbar0, theta0, phi0, vbar0, psi0, gamma0, drangebar0, t_sec, dt, delta)
		# Note : solver returns non-dimensional variables
		# Convert to dimensional variables for plotting
		t, r, theta, phi, v, psi, gamma, drange = \
			self.planetObj.dimensionalize(tbar, rbar, theta, phi, vbar, psi, gamma, drangebar)

		# dimensional state variables are in SI units
		# convert to more rational units for plotting
		t_min, h_km, v_kms, phi_deg, psi_deg, theta_deg, gamma_deg, drange_km \
			= self.convertToPlotUnits(t, r, v, phi, psi, theta, gamma, drange)

		# classify trajectory
		self.index, self.exitflag = self.classifyTrajectory(r)

		# truncate trajectory
		self.tc, self.rc, self.thetac, self.phic, self.vc, self.psic, self.gammac, self.drangec \
			= self.truncateTrajectory(t, r, theta, phi, v, psi, gamma, drange, self.index)
		self.t_minc, self.h_kmc, self.v_kmsc, self.phi_degc, self.psi_degc, \
		self.theta_degc, self.gamma_degc, self.drange_kmc = \
			self.truncateTrajectory(t_min, h_km, v_kms, phi_deg, psi_deg,
									theta_deg, gamma_deg, drange_km, self.index)
		# compute acceleration loads
		self.acc_net_g = self.computeAccelerationLoad(self.tc, self.rc, self.thetac, self.phic, self.vc, self.index, delta)
		# compute drag acceleration
		self.acc_drag_g = self.computeAccelerationDrag(self.tc, self.rc, self.thetac, self.phic, self.vc, self.index, delta)
		# compute dynamic pressure
		self.dyn_pres_atm = self.computeDynPres(self.rc, self.vc) / 1.01325E5
		# compute stagnation pressure
		self.stag_pres_atm = self.computeStagPres(self.rc, self.vc) / 1.01325E5

		# compute stagnation point convective and radiative heating rate
		self.q_stag_con = self.qStagConvective(self.rc, self.vc)
		self.q_stag_rad = self.qStagRadiative(self.rc, self.vc)
		# compute total stagnation point heating rate
		self.q_stag_total = self.q_stag_con + self.q_stag_rad
		# compute stagnation point heating load
		self.heatload = cumtrapz(self.q_stag_total, self.tc, initial=self.heatLoad0)

	def dummyVehicle(self, density_mes_int):
		"""
		Create a copy of the vehicle object which uses a 
		measured density profile for propogation.

		Parameters
		-----------
		density_mes_int : scipy.interpolate.interpolate.interp1d
			density interpolation function

		Returns
		-----------
		vehicleCopy : vehicle object
			dummy vehicle object

		"""
		planetCopy = copy.deepcopy(self.planetObj)
		planetCopy.density_int = density_mes_int

		vehicleCopy = copy.deepcopy(self)
		vehicleCopy.planetObj = planetCopy

		return vehicleCopy

	def propogateEntry_util(self, h0_km, theta0_deg, phi0_deg, v0_kms,
							gamma0_deg, psi0_deg, drange0_km, heatLoad0,
							t_sec, dt, delta_deg, density_mes_int):
		"""
		Utility propogator routine for prediction of atmospheric exit
		conditions which is then supplied to the apoapis prediction 
		module.

		Propogates the vehicle state for using the measured 
		atmospheric profile during the descending leg.
		
		Parameters 
		-----------
		h0_km : float
			current altitude, km
		theta0_deg : float
			current longitude, deg
		phi0_deg : float
			current latitude, deg
		v0_kms : float
			current speed, km/s
		gamma0_deg : float
			current FPA, deg
		psi0_deg : float
			current heading angle, deg
		drange0_km : float
			current downrange, km
		heatLoad0 : float
			current heatload, J/cm2
		t_sec : float
			propogation time, seconds
		dt : float
			max. time step, seconds
		delta_deg : float
			bank angle command, deg


		Returns
		----------
		t_minc : numpy.ndarray
			time solution array, min 
		h_kmc : numpy.ndarray
			altitude solution array, km
		v_kmsc : numpy.ndarray
			speed solution array, km/s
		phi_degc : numpy.ndarray
			latitude solution array, deg
		psi_degc : numpy.ndarray
			heading angle solution array, deg
		theta_degc : numpy.ndarray
			longitude solution array, deg
		gamma_degc : numpy.ndarray
			FPA solution array, deg
		drange_kmc : numpy.ndarray
			downrange solution array, km
		exitflag : int
			exitflag
		acc_net_g : numpy.ndarray
			acceleration solution array, Earth g
		dyn_pres_atm : numpy.ndarray
			dynamic pressure solution array, atm
		stag_pres_atm : numpy.ndarray
			stagnation pressure array, atm
		q_stag_total : numpy.ndarray
			stagnation point heat rate array
		heatload : numpy.ndarray
			stagnation point heat load
		acc_drag_g : numpy.ndarray
			acceleration due to drag, Earth g

		"""

		# Create a copy of the planet object associated 
		# with the vehicle.
		# Set the density_int attribute to be the
		# density_mes_int which is the measured 
		# density function.
		
		planetCopy = copy.deepcopy(self.planetObj)
		planetCopy.density_int = density_mes_int

		# Create a copy of the vehicle object so it does 
		# not affect the existing vehicle state variables
		vehicleCopy = copy.deepcopy(self)
		vehicleCopy.planetObj = planetCopy


		# Define entry conditions at entry interface
		# Convert initial state variables from input/plot 
		# units to calculation/SI units

		# Entry altitude above planet surface in meters
		# Entry latitude in radians
		# Entry velocity in meters/sec, relative to planet
		# Entry velocity in meters/sec, relative to planet
		# Entry heading angle in radians
		# Entry flight path angle in radians
		# Entry downrange in m


		h0 = h0_km*1.0E3
		theta0 = theta0_deg*np.pi/180.0
		phi0 = phi0_deg*np.pi/180.0
		v0 = v0_kms*1.000E3
		psi0 = psi0_deg*np.pi/180.0
		gamma0 = gamma0_deg*np.pi/180.0
		drange0 = drange0_km*1E3          

		# Define control variables
		# Constant bank angle in radians
		delta = delta_deg*np.pi/180.0

		r0 = vehicleCopy.planetObj.computeR(h0)
		
		# Compute non-dimensional entry conditions	
		rbar0,theta0,phi0,vbar0,psi0,gamma0,drangebar0 = \
			vehicleCopy.planetObj.nonDimState(r0,theta0,phi0,v0,psi0,gamma0,drange0)
		
		# Solve for the entry trajectory
		tbar, rbar, theta, phi, vbar, psi, gamma, drangebar = \
		vehicleCopy.solveTrajectory(rbar0, theta0, phi0, vbar0, psi0, gamma0, drangebar0, t_sec, dt, delta)
		# Note : solver returns non-dimensional variables
		# Convert to dimensional variables for plotting
		t, r, theta, phi, v, psi, gamma, drange = \
			vehicleCopy.planetObj.dimensionalize(tbar,rbar,theta,phi,vbar,psi,gamma, drangebar)

		# dimensional state variables are in SI units
		# convert to more rational units for plotting
		t_min, h_km, v_kms, phi_deg, psi_deg, theta_deg, gamma_deg, drange_km = \
			vehicleCopy.convertToPlotUnits(t, r, v, phi, psi, theta, gamma, drange)

		# classify trajectory
		index, exitflag = vehicleCopy.classifyTrajectory(r)
		# truncate trajectory
		tc,rc,thetac,phic,vc,psic,gammac,drangec = \
			vehicleCopy.truncateTrajectory(t, r, theta, phi, v, psi, gamma, drange, index)
		t_minc, h_kmc, v_kmsc, phi_degc, psi_degc, theta_degc, gamma_degc, drange_kmc = \
			vehicleCopy.truncateTrajectory(t_min, h_km, v_kms, phi_deg, psi_deg, theta_deg, gamma_deg, drange_km, index)

		# compute acceleration loads
		acc_net_g = vehicleCopy.computeAccelerationLoad(tc, rc, thetac, phic, vc, index, delta)
		# compute drag acceleration 
		acc_drag_g = vehicleCopy.computeAccelerationDrag(tc, rc, thetac, phic, vc, index, delta)
		# compute dynamic pressure
		dyn_pres_atm = vehicleCopy.computeDynPres(rc, vc)/1.01325E5
		# compute stagnation pressure
		stag_pres_atm = vehicleCopy.computeStagPres(rc, vc)/1.01325E5

	    # compute stagnation point convective and radiative heating rate
		q_stag_con = vehicleCopy.qStagConvective(rc, vc)
		q_stag_rad = vehicleCopy.qStagRadiative (rc, vc)
		# compute total stagnation point heating rate
		q_stag_total = q_stag_con + q_stag_rad
		# compute stagnation point heating load
		heatload = cumtrapz(q_stag_total , tc, initial=heatLoad0)

		return t_minc, h_kmc, v_kmsc, phi_degc, psi_degc, theta_degc, \
				gamma_degc, drange_kmc, exitflag, acc_net_g, dyn_pres_atm, \
				stag_pres_atm, q_stag_total, heatload, acc_drag_g

	def propogateEntry_util2(self, h0_km, theta0_deg, phi0_deg, v0_kms,
							gamma0_deg, psi0_deg, drange0_km, heatLoad0,
							t_sec, dt, delta_deg, density_mes_int):
		"""
		Utility propogator routine for prediction of atmospheric exit
		conditions which is then supplied to the apoapis prediction
		module.

		Propogates the vehicle state for using the measured
		atmospheric profile during the descending leg.

		Parameters
		-----------
		h0_km : float
			current altitude, km
		theta0_deg : float
			current longitude, deg
		phi0_deg : float
			current latitude, deg
		v0_kms : float
			current speed, km/s
		gamma0_deg : float
			current FPA, deg
		psi0_deg : float
			current heading angle, deg
		drange0_km : float
			current downrange, km
		heatLoad0 : float
			current heatload, J/cm2
		t_sec : float
			propogation time, seconds
		dt : float
			max. time step, seconds
		delta_deg : float
			bank angle command, deg


		Returns
		----------
		t_minc : numpy.ndarray
			time solution array, min
		h_kmc : numpy.ndarray
			altitude solution array, km
		v_kmsc : numpy.ndarray
			speed solution array, km/s
		phi_degc : numpy.ndarray
			latitude solution array, deg
		psi_degc : numpy.ndarray
			heading angle solution array, deg
		theta_degc : numpy.ndarray
			longitude solution array, deg
		gamma_degc : numpy.ndarray
			FPA solution array, deg
		drange_kmc : numpy.ndarray
			downrange solution array, km
		exitflag : int
			exitflag
		acc_net_g : numpy.ndarray
			acceleration solution array, Earth g
		dyn_pres_atm : numpy.ndarray
			dynamic pressure solution array, atm
		stag_pres_atm : numpy.ndarray
			stagnation pressure array, atm
		q_stag_total : numpy.ndarray
			stagnation point heat rate array
		heatload : numpy.ndarray
			stagnation point heat load
		acc_drag_g : numpy.ndarray
			acceleration due to drag, Earth g

		"""

		# Create a copy of the planet object associated
		# with the vehicle.
		# Set the density_int attribute to be the
		# density_mes_int which is the measured
		# density function.

		planetCopy = copy.deepcopy(self.planetObj)
		planetCopy.density_int = density_mes_int

		# Create a copy of the vehicle object so it does
		# not affect the existing vehicle state variables
		vehicleCopy = copy.deepcopy(self)
		vehicleCopy.planetObj = planetCopy

		# Define entry conditions at entry interface
		# Convert initial state variables from input/plot
		# units to calculation/SI units

		# Entry altitude above planet surface in meters
		# Entry latitude in radians
		# Entry velocity in meters/sec, relative to planet
		# Entry velocity in meters/sec, relative to planet
		# Entry heading angle in radians
		# Entry flight path angle in radians
		# Entry downrange in m

		h0 = h0_km * 1.0E3
		theta0 = theta0_deg * np.pi / 180.0
		phi0 = phi0_deg * np.pi / 180.0
		v0 = v0_kms * 1.000E3
		psi0 = psi0_deg * np.pi / 180.0
		gamma0 = gamma0_deg * np.pi / 180.0
		drange0 = drange0_km * 1E3

		# Define control variables
		# Constant bank angle in radians
		delta = delta_deg * np.pi / 180.0

		r0 = vehicleCopy.planetObj.computeR(h0)

		# Compute non-dimensional entry conditions
		rbar0, theta0, phi0, vbar0, psi0, gamma0, drangebar0 = \
			vehicleCopy.planetObj.nonDimState(r0, theta0, phi0, v0, psi0, gamma0, drange0)

		# Solve for the entry trajectory
		tbar, rbar, theta, phi, vbar, psi, gamma, drangebar = \
			vehicleCopy.solveTrajectory2(rbar0, theta0, phi0, vbar0, psi0, gamma0, drangebar0, t_sec, dt, delta)
		# Note : solver returns non-dimensional variables
		# Convert to dimensional variables for plotting
		t, r, theta, phi, v, psi, gamma, drange = \
			vehicleCopy.planetObj.dimensionalize(tbar, rbar, theta, phi, vbar, psi, gamma, drangebar)
		# print(t[-1])
		# dimensional state variables are in SI units
		# convert to more rational units for plotting
		t_min, h_km, v_kms, phi_deg, psi_deg, theta_deg, gamma_deg, drange_km \
			= vehicleCopy.convertToPlotUnits(t, r, v, phi, psi, theta, gamma, drange)
		# classify trajectory
		index, exitflag = vehicleCopy.classifyTrajectory(r)
		# truncate trajectory
		tc, rc, thetac, phic, vc, psic, gammac, drangec \
			= vehicleCopy.truncateTrajectory(t, r, theta, phi, v, psi, gamma, drange, index)
		t_minc, h_kmc, v_kmsc, phi_degc, psi_degc, \
		theta_degc, gamma_degc, drange_kmc = \
			vehicleCopy.truncateTrajectory(t_min, h_km, v_kms, phi_deg, psi_deg, theta_deg, gamma_deg, drange_km, index)
		# compute acceleration loads
		acc_net_g = vehicleCopy.computeAccelerationLoad(tc, rc, thetac, phic, vc, index, delta)
		# compute drag acceleration
		acc_drag_g = vehicleCopy.computeAccelerationDrag(tc, rc, thetac, phic, vc, index, delta)
		# compute dynamic pressure
		dyn_pres_atm = vehicleCopy.computeDynPres(rc, vc) / 1.01325E5
		# compute stagnation pressure
		stag_pres_atm = vehicleCopy.computeStagPres(rc, vc) / 1.01325E5

		# compute stagnation point convective and radiative heating rate
		q_stag_con = vehicleCopy.qStagConvective(rc, vc)
		q_stag_rad = vehicleCopy.qStagRadiative(rc, vc)
		# compute total stagnation point heating rate
		q_stag_total = q_stag_con + q_stag_rad
		# compute stagnation point heating load
		heatload = cumtrapz(q_stag_total, tc, initial=heatLoad0)

		return t_minc, h_kmc, v_kmsc, phi_degc, psi_degc, theta_degc, \
				gamma_degc, drange_kmc, exitflag, acc_net_g, dyn_pres_atm, \
				stag_pres_atm, q_stag_total, heatload, acc_drag_g

	def makeBasicEntryPlots(self):
		"""
		This function creates the evolution plots of the 
		altitude, speed, deceleration, and heat rate

		Parameters
		----------
		None.
		
		Returns
		----------
		1 image with 4 subplots

		"""

		fig = plt.figure()
		fig.set_size_inches([6.5,6.5])

		plt.subplot(2,2,1)
		plt.plot(self.t_minc,self.h_kmc,'r-',linewidth=2.0)
		plt.xlabel("Time, min", fontsize=10)
		plt.ylabel("Altitude, km", fontsize=10)
		plt.xticks(fontsize=10)
		plt.yticks(fontsize=10)
		ax = plt.gca()
		ax.tick_params(direction='in')
		ax.yaxis.set_ticks_position('both')
		ax.xaxis.set_ticks_position('both')

		plt.subplot(2,2,2)
		plt.plot(self.t_minc,self.v_kmsc,'g-',linewidth=2.0)
		plt.xlabel("Time, min", fontsize=10)
		plt.ylabel("Velocity (km/s)", fontsize=10)
		plt.xticks(fontsize=10)
		plt.yticks(fontsize=10)
		ax = plt.gca()
		ax.tick_params(direction='in')
		ax.yaxis.set_ticks_position('both')
		ax.xaxis.set_ticks_position('both')

		plt.subplot(2,2,3)
		plt.plot(self.t_minc,self.acc_net_g,'b-',linewidth=2.0)
		plt.xlabel("Time, min", fontsize=10)
		plt.ylabel("Deceleration (Earth g)", fontsize=10)
		plt.xticks(fontsize=10)
		plt.yticks(fontsize=10)
		ax = plt.gca()
		ax.tick_params(direction='in')
		ax.yaxis.set_ticks_position('both')
		ax.xaxis.set_ticks_position('both')

		plt.subplot(2,2,4)
		plt.plot(self.t_minc,self.q_stag_total,'m-',linewidth=2.0)
		plt.xlabel("Time, min")
		plt.ylabel("Stag. point heat-rate (W/cm2)")
		plt.xticks(fontsize=10)
		plt.yticks(fontsize=10)
		ax = plt.gca()
		ax.tick_params(direction='in')
		ax.yaxis.set_ticks_position('both')
		ax.xaxis.set_ticks_position('both')
	
		plt.show()

	def isCaptured(self, t_sec, dt, delta_deg):
		"""
		This function determines if the vehicle is captured.
		Returns -1 if the vehicle is captured, +1 otherwise.

		Parameters
		----------
		t_sec : float
			propogation time, seconds
		dt : float
			max. time step, seconds
		delta_deg : float
			bank angle command, deg
		
		Returns
		----------
		ans : int

		"""

		self.propogateEntry(t_sec, dt, delta_deg)
		energy = self.computeEnergy(self.rc,self.vc)

		# Check if energy at the terminal point in the 
		# trajectory is negative, ie. if vehicle is captured at
		# the atmospheric exit interface. If true return -1.0, 
		# else return 1.0
		if energy[self.index-1] < 0:
			ans = -1.0
		else:
			ans = 1.0

		return ans


	def hitsTargetApoapsis(self, t_sec, dt, delta_deg,targetApopasisAltitude_km):
		"""
		This function is used to check if the vehicle undershoots 
		or overshoots. Does not include effect of planet rotation
		to compute inertial speed.

		Returns +1 if the vehicle is captured into an orbit with the 
		required target apoapsis alt, -1 otherwise.

		Parameters
		----------
		t_sec : float
			propogation time, seconds
		dt : float
			max. time step, seconds
		delta_deg : float
			bank angle command, deg
		targetApopasisAltitude_km : float
			target apoapsis altitude , km
		
		Returns
		----------
		ans : int
			-1 indicates overshoot, +1 indicates undershoot
		
		"""
		self.propogateEntry(t_sec, dt, delta_deg)

		# Compute the specific energy of the vehicle over the 
		# entire trajectory.
		
		energy = self.computeEnergy(self.rc, self.vc)

		# Compute r,v,gamma at the atmospheric exit state, 
		# terminal point of the truncated trajectory
		self.terminal_r = self.rc[self.index-1]
		self.terminal_v = self.vc[self.index-1]
		self.terminal_g = self.gammac[self.index-1]

		# Compute energy E, angular momentum h at the exit state, 
		# terminal point of the truncated trajectory
		self.terminal_E = self.computeEnergyScalar(self.terminal_r, self.terminal_v)
		self.terminal_h = self.computeAngMomScalar(self.terminal_r, self.terminal_v, self.terminal_g)

		# Compute semi-major axis and eccentricity of the post 
		# atmospheric exit orbit
		self.terminal_a = self.computeSemiMajorAxisScalar(self.terminal_E)
		self.terminal_e = self.computeEccScalar(self.terminal_h, self.terminal_E)

		# Compute apoapsis radius, apoapsis altitude, apoapsis altitude in KM
		self.rp = self.terminal_a*(1.0+self.terminal_e)
		self.hp = self.rp - self.planetObj.RP
		self.hp_km = self.hp / 1.0E3

		terminal_alt = (self.terminal_r - self.planetObj.RP)/1000.0

		# if computed apoapsis altitude exceeds target apoapsis altitude
		# or if orbit is hyperbolic at the exit state, return -1.0
		# else if computed apoapsis altitude falls short of
		# target apoapsis altitude return 1.0
		if self.hp_km >= targetApopasisAltitude_km or self.terminal_a < 0:
			ans = -1.0
		else:
			ans = 1.0

		# if terminal altitude (km) < planet.h_low, then assume undershoot
		if terminal_alt < self.planetObj.h_low/1000.0:
			ans = 1.0
	
		return ans

	def hitsTargetApoapsis2(self, t_sec, dt, delta_deg, targetApopasisAltitude_km):
		"""
		This function is used to check if the vehicle undershoots 
		or overshoots. Includes effect of planet rotation to 
		calculate inertial speed.

		Returns +1 if the vehicle is captured into an orbit with the 
		required target apoapsis alt, -1 otherwise.

		Parameters
		----------
		t_sec : float
			propogation time, seconds
		dt : float
			max. time step, seconds
		delta_deg : float
			bank angle command, deg
		targetApopasisAltitude_km : float
			target apoapsis altitude , km
		
		Returns
		----------
		ans : int
			-1 indicates overshoot, +1 indicates undershoot
		
		"""
		self.propogateEntry2(t_sec, dt, delta_deg)

		# Compute the specific energy of the vehicle over the 
		# entire trajectory.
		
		energy = self.computeEnergy(self.rc,self.vc)

		# Compute r,v,gamma at the atmospheric exit state, 
		# terminal point of the truncated trajectory
		terminal_r = self.rc[self.index-1]
		terminal_v = self.vc[self.index-1]
		terminal_g = self.gammac[self.index-1]

		terminal_theta = self.thetac[self.index-1]
		terminal_phi   = self.phic[self.index-1]
		terminal_psi   = self.psic[self.index-1]

		# Compute planet relative speed in Cartesian XYZ coordinates
		v_pr_x = terminal_v*np.sin(terminal_g)*np.cos(terminal_phi)*np.cos(terminal_theta) +\
				terminal_v*np.cos(terminal_g)*np.cos(terminal_psi)*(-1*np.sin(terminal_theta)) +\
				terminal_v*np.cos(terminal_g)*np.sin(terminal_psi)*(-1*np.sin(terminal_phi) * np.cos(terminal_theta))
		
		v_pr_y = terminal_v*np.sin(terminal_g)*np.cos(terminal_phi)*np.sin(terminal_theta) +\
				 terminal_v*np.cos(terminal_g)*np.cos(terminal_psi)*np.cos(terminal_theta) +\
				 terminal_v*np.cos(terminal_g)*np.sin(terminal_psi)*(-1*np.sin(terminal_phi) * np.sin(terminal_theta))
		
		v_pr_z = terminal_v*np.sin(terminal_g)*np.sin(terminal_phi) + \
					terminal_v*np.cos(terminal_g)*np.sin(terminal_psi) * np.cos(terminal_phi)

		# Compute inertial speed in Cartesian XYZ coordinates
		v_ie_x = v_pr_x + terminal_r*self.planetObj.OMEGA*np.cos(terminal_phi)*np.sin(terminal_theta)*(-1.0)
		v_ie_y = v_pr_y + terminal_r*self.planetObj.OMEGA*np.cos(terminal_phi)*np.cos(terminal_theta)
		v_ie_z = v_pr_z

		# Compute terminal radial vector
		terminal_r_vec = terminal_r*np.array([np.cos(terminal_phi)*np.cos(terminal_theta),
										np.cos(terminal_phi)*np.sin(terminal_theta),
										np.sin(terminal_phi)])
		
		# Compute terminal radial unit vector
		terminal_r_hat_vec = terminal_r_vec / np.linalg.norm(terminal_r_vec)

		# Compute inertial velocity vector
		terminal_v_ie_vec  = np.array([v_ie_x, v_ie_y, v_ie_z])
		
		# Compute inertial velocity unit vector
		terminal_v_ie_hat_vec = terminal_v_ie_vec / np.linalg.norm(terminal_v_ie_vec)
		
		# Compute inertial flight path angle at exit using
		# terminal inertial radial and velocity vectors
		terminal_fpa_ie_deg = 90.0 - (180/np.pi)*np.arccos(np.dot(terminal_r_hat_vec,terminal_v_ie_hat_vec))
		
		terminal_fpa_ie_rad = terminal_fpa_ie_deg*np.pi/180.0

		# Compute inertial velocity magnitude
		v_ie_mag = np.sqrt(v_ie_x**2 + v_ie_y**2 + v_ie_z**2)
		
		# Compute orbit energy using inertial speed
		terminal_E = self.computeEnergyScalar(terminal_r, v_ie_mag)
		terminal_h = self.computeAngMomScalar(terminal_r, v_ie_mag, terminal_fpa_ie_rad)

		# Compute semi-major axis and eccentricity of the post atmospheric exit orbit
		terminal_a = self.computeSemiMajorAxisScalar(terminal_E)
		terminal_e = self.computeEccScalar(terminal_h, terminal_E)

		# Compute apoapsis radius, apoapsis altitude, apoapsis altitude in KM
		rp = terminal_a*(1.0+terminal_e)
		hp = rp - self.planetObj.RP
		hp_km = hp / 1.0E3

		terminal_alt = (terminal_r - self.planetObj.RP)/1000.0

		# if computed apoapsis altitude exceeds target apoapsis altitude
		# or if orbit is hyperbolic at the exit state, return -1.0
		# else if computed apoapsis altitude falls short of
		# target apoapsis altitude return 1.0
		if hp_km >= targetApopasisAltitude_km or terminal_a<0:
			ans = -1.0
		else:
			ans = 1.0

		# if terminal altitude (km) < planet.h_low, then assume undershoot
		if terminal_alt < self.planetObj.h_low/1000.0:
			ans = 1.0
	
		return ans

	def findOverShootLimit(self, t_sec, dt, gamma0_deg_guess_low,
							gamma0_deg_guess_high, gamma_deg_tol, targetApopasisAltitude_km):
		"""
		Computes the overshoot limit entry flight-path angle
		for aerocapture vehicle using bisection algorithm.

		This is shallowest entry flight path angle for which a full lift down
		trajectory gets the vehicle captured into a post atmospheric exit orbit 
		with the desired target apoapsis altitude.

		Note: the overshoot limit entry flight path angle should be computed 
		with an accuracy of at least 10 decimal places to ensure the correct 
		atmospheric trajectory is simulated. 

		A bisection algorithm is used to compute the overshoot limit.

		
		Parameters
		----------
		t_sec : float
			propogation time, seconds
		dt : float
			max. time step, seconds
		gamma0_deg_guess_low : float
			lower bound for the guess of overshoot limit FPA, deg
		gamma0_deg_guess_high : float
			upper bound for the guess of overshoot limit FPA, deg
		gamma_deg_tol : float
			desired accuracy for computation of the overshoot limit, deg
		targetApopasisAltitude_km : float
			target apoapsis altitude , km
		
		Returns
		----------
		overShootLimit : float
			overshoot limit EFPA, deg
		exitflag_os : float
			flag to indicate if a solution could not be found for the 
			overshoot limit

		exitflag_os = 1.0 indicates over shoot limit was found.
		exitflag_os = 0.0 indicates overshoot limit was not found 
		within user specified bounds.
		"""
		
		delta_deg = 180.0 # full lift down bank angle

		temp_var = self.gamma0_deg
		
		# compute the apoapsis altitude flag for the lower bound entry 
		# flight path angle and the upper bound entry flight path angle
		self.gamma0_deg = gamma0_deg_guess_low
		ans1 = self.hitsTargetApoapsis(t_sec, dt, delta_deg, targetApopasisAltitude_km)
		self.gamma0_deg = gamma0_deg_guess_high
		ans2 = self.hitsTargetApoapsis(t_sec, dt, delta_deg, targetApopasisAltitude_km)	
		
		# if product of above flags is negative, then overshoot limit is 
		# within user specified bounds, proceed to bisection.
		if ans1*ans2<0:
			# print("EFPA = "+str(gamma0_deg_guess_low)+ " deg. exceeds target \
			# apoapsis altitude or is not captured.")
			# print("EFPA = "+str(gamma0_deg_guess_high)+" deg. does not hit \
			# target apospasis but is captured.")
			# print("Overshoot limit is within user specified bounds.
			# Beginning bisection search...")
			# print("")
			# bisection algorithm begins here, while abs(ub-lb)>tol continue bisection
			while abs(gamma0_deg_guess_high - gamma0_deg_guess_low)>gamma_deg_tol:
				gamma0_deg_guess_mid = 0.5*(gamma0_deg_guess_low+gamma0_deg_guess_high)
				self.gamma0_deg = gamma0_deg_guess_low
				ans1 = self.hitsTargetApoapsis(t_sec, dt, delta_deg, targetApopasisAltitude_km)
				self.gamma0_deg = gamma0_deg_guess_mid
				ans2 = self.hitsTargetApoapsis(t_sec, dt, delta_deg, targetApopasisAltitude_km)
				self.gamma0_deg = gamma0_deg_guess_high
				ans3 = self.hitsTargetApoapsis(t_sec, dt, delta_deg, targetApopasisAltitude_km)
				# reset upper / lower bounds as appropriate
				if ans1*ans2<0:
					# print('ans1*ans2<0')
					gamma0_deg_guess_high = gamma0_deg_guess_mid
				elif ans2*ans3<0:
					# print('ans2*ans3<0')
					gamma0_deg_guess_low = gamma0_deg_guess_mid
				# print("EFPA = "+str(gamma0_deg_guess_low)+ " deg. exceeds target apoapsis altitude or is not captured.")
				# print("EFPA = "+str(gamma0_deg_guess_high)+" deg. does not hit target
				# apospasis but is captured.")
				# print("")
				# set overShootLimit to upper bound
				# set exitflag to 1.0
				overShootLimit = gamma0_deg_guess_high
				exitflag_os = 1.0
		
		# if product of flags is positive, overshootlimit is outside user specified bounds, 
		# print warning message.
		else:
			print("Overshoot limit is outside user specified bounds.")
			overShootLimit = 0.0
			exitflag_os = 0.0

		self.gamma0_deg = temp_var

		return overShootLimit, exitflag_os

	def findOverShootLimit2(self,t_sec, dt, gamma0_deg_guess_low,
								gamma0_deg_guess_high, gamma_deg_tol, targetApopasisAltitude_km):
		"""
		Computes the overshoot limit entry flight-path angle
		for aerocapture vehicle using bisection algorithm.
		Includes effect of planet rotation on inertial speed.

		This is shallowest entry flight path angle for which a full lift down
		trajectory gets the vehicle captured into a post atmospheric exit orbit 
		with the desired target apoapsis altitude.

		Note: the overshoot limit entry flight path angle should be computed 
		with an accuracy of at least 10 decimal places to ensure the correct 
		atmospheric trajectory is simulated. 

		A bisection algorithm is used to compute the overshoot limit.

		
		Parameters
		----------
		t_sec : float
			propogation time, seconds
		dt : float
			max. time step, seconds
		gamma0_deg_guess_low : float
			lower bound for the guess of overshoot limit FPA, deg
		gamma0_deg_guess_high : float
			upper bound for the guess of overshoot limit FPA, deg
		gamma_deg_tol : float
			desired accuracy for computation of the overshoot limit, deg
		targetApopasisAltitude_km : float
			target apoapsis altitude , km
		
		Returns
		----------
		overShootLimit : float
			overshoot limit EFPA, deg
		exitflag_os : float
			flag to indicate if a solution could not be found for the 
			overshoot limit

		exitflag_os = 1.0 indicates over shoot limit was found.
		exitflag_os = 0.0 indicates overshoot limit was not found 
		within user specified bounds.
		"""
		
		delta_deg = 180.0   # full lift down bank angle

		temp_var = self.gamma0_deg
		
		# compute the apoapsis altitude flag for the lower bound entry 
		# flight path angle and the upper bound entry flight path angle
		self.gamma0_deg = gamma0_deg_guess_low
		ans1 = self.hitsTargetApoapsis2(t_sec, dt, delta_deg, targetApopasisAltitude_km)
		self.gamma0_deg = gamma0_deg_guess_high
		ans2 = self.hitsTargetApoapsis2(t_sec, dt, delta_deg, targetApopasisAltitude_km)	
		
		# if product of above flags is negative, then overshoot limit is 
		# within user specified bounds, proceed to bisection.
		if ans1*ans2 < 0:
			# print("EFPA = "+str(gamma0_deg_guess_low)+ " deg. exceeds target \
			# apoapsis altitude or is not captured.")
			# print("EFPA = "+str(gamma0_deg_guess_high)+" deg. does not hit \
			# target apospasis but is captured.")
			# print("Overshoot limit is within user specified bounds.")
			# Beginning bisection search...")
			# print("")
			# bisection algorithm begins here, while abs(ub-lb)>tol continue bisection
			while abs(gamma0_deg_guess_high - gamma0_deg_guess_low)>gamma_deg_tol:
				gamma0_deg_guess_mid = 0.5*(gamma0_deg_guess_low+gamma0_deg_guess_high)
				self.gamma0_deg = gamma0_deg_guess_low
				ans1 = self.hitsTargetApoapsis2(t_sec, dt, delta_deg, targetApopasisAltitude_km)
				self.gamma0_deg = gamma0_deg_guess_mid
				ans2 = self.hitsTargetApoapsis2(t_sec, dt, delta_deg, targetApopasisAltitude_km)
				self.gamma0_deg = gamma0_deg_guess_high
				ans3 = self.hitsTargetApoapsis2(t_sec, dt, delta_deg, targetApopasisAltitude_km)
				# reset upper / lower bounds as appropriate
				if ans1*ans2 < 0:
					# print('ans1*ans2<0')
					gamma0_deg_guess_high = gamma0_deg_guess_mid
				elif ans2*ans3 < 0:
					# print('ans2*ans3<0')
					gamma0_deg_guess_low = gamma0_deg_guess_mid
				# print("EFPA = "+str(gamma0_deg_guess_low)+ " deg. exceeds target apoapsis altitude or is not captured.")
				# print("EFPA = "+str(gamma0_deg_guess_high)+" deg. does not hit target apospasis but is captured.")
				# print("")
				# set overShootLimit to upper bound
				# set exitflag to 1.0
				overShootLimit = gamma0_deg_guess_high
				exitflag_os = 1.0
		
		# if product of flags is positive, overshootlimit is outside user specified bounds, 
		# print warning message.
		else:
			print("Overshoot limit is outside user specified bounds.")
			overShootLimit = 0.0
			exitflag_os = 0.0

		self.gamma0_deg = temp_var

		return overShootLimit, exitflag_os


	def findUnderShootLimit(self, t_sec, dt, gamma0_deg_guess_low,\
							gamma0_deg_guess_high, gamma_deg_tol, targetApopasisAltitude_km):
		"""
		Computes the undershoot limit entry flight-path angle
		for aerocapture vehicle using bisection algorithm.

		This is steepest entry flight path angle for which a full lift up
		trajectory gets the vehicle captured into a post atmospheric exit orbit 
		with the desired target apoapsis altitude.

		Note: the undershoor limit entry flight path angle should be computed 
		with an accuracy of at least 6 decimal places to ensure the correct 
		atmospheric trajectory is simulated. 

		A bisection algorithm is used to compute the undershoot limit.

		
		Parameters
		----------
		t_sec : float
			propogation time, seconds
		dt : float
			max. time step, seconds
		gamma0_deg_guess_low : float
			lower bound for the guess of overshoot limit FPA, deg
		gamma0_deg_guess_high : float
			upper bound for the guess of overshoot limit FPA, deg
		gamma_deg_tol : float
			desired accuracy for computation of the overshoot limit, deg
		targetApopasisAltitude_km : float
			target apoapsis altitude , km
		
		Returns
		----------
		underShootLimit : float
			overshoot limit EFPA, deg
		exitflag_us : float
			flag to indicate if a solution could not be found for the 
			undershoot limit

		exitflag_us = 1.0 indicates undershoot limit was found.
		exitflag_us = 0.0 indicates overshoot limit was not found 
		within user specified bounds.
		"""
		
		delta_deg = 0.0 # full lift up bank angle

		temp_var = self.gamma0_deg

		# compute the apoapsis altitude flag for the lower bound 
		# entry flight path angle and the upper bound entry flight path angle
		self.gamma0_deg = gamma0_deg_guess_low
		ans1 = self.hitsTargetApoapsis(t_sec, dt, delta_deg, targetApopasisAltitude_km)
		self.gamma0_deg = gamma0_deg_guess_high
		ans2 = self.hitsTargetApoapsis(t_sec, dt, delta_deg, targetApopasisAltitude_km)	
		# if product of above flags is negative, then overshoot limit is 
		# within user specified bounds, proceed to bisection.
		if ans1*ans2<0:
			# print("EFPA = "+str(gamma0_deg_guess_low)+ " deg. exceeds \
			# target apoapsis altitude or is not captured.")
			# print("EFPA = "+str(gamma0_deg_guess_high)+" deg. does not \
			# hit target apospasis but is captured.")
			# print("Overshoot limit is within user specified bounds.
			# Beginning bisection search...")
			# print("")
			# bisection algorithm begins here, while abs(ub-lb)>tol continue bisection
			while abs(gamma0_deg_guess_high - gamma0_deg_guess_low)>gamma_deg_tol:
				gamma0_deg_guess_mid = 0.5*(gamma0_deg_guess_low+gamma0_deg_guess_high)
				self.gamma0_deg = gamma0_deg_guess_low
				ans1 = self.hitsTargetApoapsis(t_sec, dt, delta_deg, targetApopasisAltitude_km)
				self.gamma0_deg = gamma0_deg_guess_mid
				ans2 = self.hitsTargetApoapsis(t_sec, dt, delta_deg, targetApopasisAltitude_km)
				self.gamma0_deg = gamma0_deg_guess_high
				ans3 = self.hitsTargetApoapsis(t_sec, dt, delta_deg, targetApopasisAltitude_km)
				# reset upper / lower bounds as appropriate
				if ans1*ans2<0:
					# print('ans1*ans2<0')
					gamma0_deg_guess_high = gamma0_deg_guess_mid
				elif ans2*ans3<0:
					# print('ans2*ans3<0')
					gamma0_deg_guess_low = gamma0_deg_guess_mid
				#print("EFPA = "+str(gamma0_deg_guess_low)+ " deg. exceeds target apoapsis\
				# altitude or is not captured.")
				#print("EFPA = "+str(gamma0_deg_guess_high)+" deg. does not hit target \
				#apoapsis but is captured.")
				#print("")
				# set overShootLimit to upper bound
				# set exitflag to 1.0
				underShootLimit = gamma0_deg_guess_high
				exitflag_us = 1.0
		
		# if product of flags is positive, overshootlimit is outside user specified bounds, 
		# print warning message.
		else:
			print("Undershoot limit is outside user specified bounds.")
			underShootLimit = 0.0
			exitflag_us = 0.0

		self.gamma0_deg = temp_var

		return underShootLimit, exitflag_us

	def findUnderShootLimit2(self, t_sec, dt, gamma0_deg_guess_low,
							gamma0_deg_guess_high,gamma_deg_tol, targetApopasisAltitude_km):
		"""
		Computes the undershoot limit entry flight-path angle
		for aerocapture vehicle using bisection algorithm.
		Includes effect of planet rotation on inertial speed.

		This is steepest entry flight path angle for which a full lift up
		trajectory gets the vehicle captured into a post atmospheric exit orbit 
		with the desired target apoapsis altitude.

		Note: the undershoor limit entry flight path angle should be computed 
		with an accuracy of at least 6 decimal places to ensure the correct 
		atmospheric trajectory is simulated. 

		A bisection algorithm is used to compute the undershoot limit.

		
		Parameters
		----------
		t_sec : float
			propogation time, seconds
		dt : float
			max. time step, seconds
		gamma0_deg_guess_low : float
			lower bound for the guess of overshoot limit FPA, deg
		gamma0_deg_guess_high : float
			upper bound for the guess of overshoot limit FPA, deg
		gamma_deg_tol : float
			desired accuracy for computation of the overshoot limit, deg
		targetApopasisAltitude_km : float
			target apoapsis altitude , km
		
		Returns
		----------
		underShootLimit : float
			overshoot limit EFPA, deg
		exitflag_us : float
			flag to indicate if a solution could not be found for the 
			undershoot limit

		exitflag_us = 1.0 indicates undershoot limit was found.
		exitflag_us = 0.0 indicates overshoot limit was not found 
		within user specified bounds.
		"""
		
		delta_deg = 0.0   # full lift up bank angle

		temp_var = self.gamma0_deg

		# compute the apoapsis altitude flag for the lower bound 
		# entry flight path angle and the upper bound entry flight path angle
		self.gamma0_deg = gamma0_deg_guess_low
		ans1 = self.hitsTargetApoapsis2(t_sec, dt, delta_deg, targetApopasisAltitude_km)
		self.gamma0_deg = gamma0_deg_guess_high
		ans2 = self.hitsTargetApoapsis2(t_sec, dt, delta_deg, targetApopasisAltitude_km)	
		# if product of above flags is negative, then overshoot limit is 
		# within user specified bounds, proceed to bisection.
		if ans1*ans2 < 0:
			# print("EFPA = "+str(gamma0_deg_guess_low)+ " deg. exceeds \
			# target apoapsis altitude or is not captured.")
			# print("EFPA = "+str(gamma0_deg_guess_high)+" deg. does not \
			# hit target apospasis but is captured.")
			# print("Overshoot limit is within user specified bounds.
			# Beginning bisection search...")
			# print("")
			# bisection algorithm begins here, while abs(ub-lb)>tol continue bisection
			while abs(gamma0_deg_guess_high - gamma0_deg_guess_low) > gamma_deg_tol:
				gamma0_deg_guess_mid = 0.5*(gamma0_deg_guess_low+gamma0_deg_guess_high)
				self.gamma0_deg = gamma0_deg_guess_low
				ans1 = self.hitsTargetApoapsis2(t_sec, dt, delta_deg, targetApopasisAltitude_km)
				self.gamma0_deg = gamma0_deg_guess_mid
				ans2 = self.hitsTargetApoapsis2(t_sec, dt, delta_deg, targetApopasisAltitude_km)
				self.gamma0_deg = gamma0_deg_guess_high
				ans3 = self.hitsTargetApoapsis2(t_sec, dt, delta_deg, targetApopasisAltitude_km)
				# reset upper / lower bounds as appropriate
				if ans1*ans2 < 0:
					# print('ans1*ans2<0')
					gamma0_deg_guess_high = gamma0_deg_guess_mid
				elif ans2*ans3 < 0:
					# print('ans2*ans3<0')
					gamma0_deg_guess_low  = gamma0_deg_guess_mid
				# print("EFPA = "+str(gamma0_deg_guess_low)+ " deg. exceeds target apoapsis\
				# altitude or is not captured.")
				# print("EFPA = "+str(gamma0_deg_guess_high)+" deg. does not hit target \
				# apoapsis but is captured.")
				# print("")
				# set overShootLimit to upper bound
				# set exitflag to 1.0
				underShootLimit = gamma0_deg_guess_high
				exitflag_us = 1.0
		
		# if product of flags is positive, overshootlimit is outside user specified bounds, 
		# print warning message.
		else:
			print("Undershoot limit is outside user specified bounds.")
			underShootLimit    = 0.0
			exitflag_us        = 0.0

		self.gamma0_deg = temp_var

		return underShootLimit, exitflag_us

	def computeTCW(self, t_sec, dt, gamma0_deg_guess_low_os, gamma0_deg_guess_high_os, gamma0_deg_guess_low_us,
					gamma0_deg_guess_high_us, gamma_deg_tol_os, gamma_deg_tol_us, targetApopasisAltitude_km):
		"""
		Computes the theoretical corridor width (TCW) for 
		lift modulation aerocapture.

		TCW = overShootLimit - underShootLimit
		
		Parameters
		----------
		t_sec : float
			propogation time, seconds
		dt : float
			max. time step, seconds
		gamma0_deg_guess_low_os : float
			lower bound for the guess of overshoot limit FPA, deg
		gamma0_deg_guess_high_os : float
			upper bound for the guess of overshoot limit FPA, deg
		gamma0_deg_guess_low_us : float
			lower bound for the guess of undershoot limit FPA, deg
		gamma0_deg_guess_high_us : float
			upper bound for the guess of undershoot limit FPA, deg
		gamma_deg_tol_os : float
			desired accuracy for computation of the overshoot limit, deg
		gamma_deg_tol_us : float
			desired accuracy for computation of the undershoot limit, deg
		targetApopasisAltitude_km : float
			target apoapsis altitude , km
		
		Returns
		----------
		TCW : float
			Theoretical Corridor Width, deg
		"""
		
		# Compute the overshoot and undershoot limit entry flight path angles
		overShootLimit, exitflag_os = self.findOverShootLimit(t_sec,dt,gamma0_deg_guess_low_os,\
										gamma0_deg_guess_high_os,gamma_deg_tol_os, targetApopasisAltitude_km)
		underShootLimit,exitflag_us = self.findUnderShootLimit(t_sec,dt,gamma0_deg_guess_low_us,\
										gamma0_deg_guess_high_us,gamma_deg_tol_us, targetApopasisAltitude_km)

		# Display the computed overshoot and undershoot limit entry 
		# flight path angles
		print("Overshoot Limit  : "+str(overShootLimit) + " deg.")
		print("Undershoot Limit : "+str(underShootLimit) + " deg.")

		# Compute the TCW and print to console
		TCW = overShootLimit - underShootLimit
		print("Corridor Width   : "+str(TCW) + " deg.")

		# Return TCW, FLOAT, SCALAR to program
		return TCW

	def computeTCW2(self, t_sec, dt, gamma0_deg_guess_low_os, gamma0_deg_guess_high_os, gamma0_deg_guess_low_us,\
						gamma0_deg_guess_high_us, gamma_deg_tol_os, gamma_deg_tol_us, targetApopasisAltitude_km):
		"""
		Computes the theoretical corridor width (TCW) for 
		lift modulation aerocapture. Includes effect of planet rotation.

		TCW = overShootLimit - underShootLimit
		
		Parameters
		----------
		t_sec : float
			propogation time, seconds
		dt : float
			max. time step, seconds
		gamma0_deg_guess_low_os : float
			lower bound for the guess of overshoot limit FPA, deg
		gamma0_deg_guess_high_os : float
			upper bound for the guess of overshoot limit FPA, deg
		gamma0_deg_guess_low_us : float
			lower bound for the guess of undershoot limit FPA, deg
		gamma0_deg_guess_high_us : float
			upper bound for the guess of undershoot limit FPA, deg
		gamma_deg_tol_os : float
			desired accuracy for computation of the overshoot limit, deg
		gamma_deg_tol_us : float
			desired accuracy for computation of the undershoot limit, deg
		targetApopasisAltitude_km : float
			target apoapsis altitude , km
		
		Returns
		----------
		TCW : float
			Theoretical Corridor Width, deg
		"""
		
		# Compute the overshoot and undershoot limit entry flight path angles
		overShootLimit, exitflag_os = self.findOverShootLimit2(t_sec,dt,gamma0_deg_guess_low_os,
										gamma0_deg_guess_high_os,gamma_deg_tol_os, targetApopasisAltitude_km)
		underShootLimit,exitflag_us = self.findUnderShootLimit2(t_sec,dt,gamma0_deg_guess_low_us,
										gamma0_deg_guess_high_us,gamma_deg_tol_us, targetApopasisAltitude_km)

		# Display the computed overshoot and undershoot limit entry 
		# flight path angles
		print("Overshoot Limit  : "+str(overShootLimit) + " deg.")
		print("Undershoot Limit : "+str(underShootLimit) + " deg.")

		# Compute the TCW and print to console
		TCW = overShootLimit - underShootLimit
		print("Corridor Width   : "+str(TCW) + " deg.")

		# Return TCW, FLOAT, SCALAR to program
		return TCW

	def setDragModulationVehicleParams(self, beta1, betaRatio):
		"""
		Set the beta1 and betaRatio params for a drag modulation vehicle.

		Parameters
		----------
		beta1 : float
			small value of ballistic coefficient, kg/m2
		betaRatio : float
			ballistic coefficient ratio
		"""

		self.beta1 = beta1
		self.betaRatio = betaRatio

	def findEFPALimitD(self, t_sec, dt, gamma0_deg_guess_low, gamma0_deg_guess_high, gamma_deg_tol,
							targetApopasisAltitude_km):
		"""
		This function computes the limiting EFPA for drag modulation
		aerocapture. Does not include planetary rotation correction.

		A bisection algorithm is used to compute the limit.

		Parameters
		----------
		t_sec : float
			propogation time, seconds
		dt : float
			max. time step, seconds
		gamma0_deg_guess_low : float
			lower bound for the guess of limit FPA, deg
		gamma0_deg_guess_high : float
			upper bound for the guess of limit FPA, deg
		gamma_deg_tol : float
			desired accuracy for computation of the limit, deg
		targetApopasisAltitude_km : float
			target apoapsis altitude , km
		
		Returns
		----------
		EFPALimit : float
			limit EFPA, deg
		exitflag : float
			flag to indicate if a solution could not be found for the 
			limit EFPA

		exitflag = 1.0 indicates over shoot limit was found.
		exitflag = 0.0 indicates overshoot limit was not found 
		within user specified bounds.
		
		"""
		temp_var_1 = self.gamma0_deg
		temp_var_2 = self.LD

		delta_deg = 0.0

		self.LD = 0.0

		self.gamma0_deg = gamma0_deg_guess_low
		ans1 = self.hitsTargetApoapsis(t_sec, dt, delta_deg, targetApopasisAltitude_km)

		self.gamma0_deg = gamma0_deg_guess_high
		ans2 = self.hitsTargetApoapsis(t_sec, dt, delta_deg, targetApopasisAltitude_km)
		
		# if product of above flags is negative, then overshoot limit is within user 
		# specified bounds, proceed to bisection.
		if ans1*ans2 < 0:
			# print("EFPA = "+str(gamma0_deg_guess_low)+ " deg. does not meet target \
			# apoapsis altitude.")
			# print("EFPA = "+str(gamma0_deg_guess_high)+" deg. exceeds target
			# apoapsis altitude.")
			# print("Undershoot limit is within user specified bounds. Beginning
			# bisection search...")
			# print("")
			# bisection algorithm begins here, while abs(ub-lb)>tol continue bisection
			while abs(gamma0_deg_guess_high - gamma0_deg_guess_low) > gamma_deg_tol:
				gamma0_deg_guess_mid = 0.5*(gamma0_deg_guess_low+gamma0_deg_guess_high)

				self.gamma0_deg = gamma0_deg_guess_low
				ans1 = self.hitsTargetApoapsis(t_sec, dt, delta_deg, targetApopasisAltitude_km)

				self.gamma0_deg = gamma0_deg_guess_mid
				ans2 = self.hitsTargetApoapsis(t_sec, dt, delta_deg, targetApopasisAltitude_km)
				
				self.gamma0_deg = gamma0_deg_guess_high				
				ans3 = self.hitsTargetApoapsis(t_sec, dt, delta_deg, targetApopasisAltitude_km)

				if ans1*ans2 < 0:
					gamma0_deg_guess_high = gamma0_deg_guess_mid
				elif ans2*ans3 < 0:
					gamma0_deg_guess_low = gamma0_deg_guess_mid

				# print("EFPA = "+str(gamma0_deg_guess_low)+ " deg. does not meet\
				# target apoapsis altitude.")
				# print("EFPA = "+str(gamma0_deg_guess_high)+" deg. exceeds target\
				# apoapsis altitude.")
				# print("")

				# set overShootLimit to lower bound
				# set exitflag to 1.0
				EFPALimit = gamma0_deg_guess_high
				exitflag = 1.0

		# if product of flags is positive, overshootlimit is outside user \
		# specified bounds, print warning message.
		else:
			print("EFPA limit is outside user specified bounds.")
			EFPALimit = 0.0
			exitflag = 0.0

		self.gamma0_deg = temp_var_1
		self.LD = temp_var_2

		return EFPALimit, exitflag

	def findEFPALimitD2(self, t_sec, dt, gamma0_deg_guess_low, gamma0_deg_guess_high, gamma_deg_tol,
					   targetApopasisAltitude_km):
		"""
		This function computes the limiting EFPA for drag modulation
		aerocapture. Includes planetary rotation correction. Includes planet rotation.

		A bisection algorithm is used to compute the limit.

		Parameters
		----------
		t_sec : float
			propogation time, seconds
		dt : float
			max. time step, seconds
		gamma0_deg_guess_low : float
			lower bound for the guess of limit FPA, deg
		gamma0_deg_guess_high : float
			upper bound for the guess of limit FPA, deg
		gamma_deg_tol : float
			desired accuracy for computation of the limit, deg
		targetApopasisAltitude_km : float
			target apoapsis altitude , km

		Returns
		----------
		EFPALimit : float
			limit EFPA, deg
		exitflag : float
			flag to indicate if a solution could not be found for the
			limit EFPA

		exitflag = 1.0 indicates over shoot limit was found.
		exitflag = 0.0 indicates overshoot limit was not found
		within user specified bounds.

		"""
		temp_var_1 = self.gamma0_deg
		temp_var_2 = self.LD

		delta_deg = 0.0

		self.LD = 0.0

		self.gamma0_deg = gamma0_deg_guess_low
		ans1 = self.hitsTargetApoapsis2(t_sec, dt, delta_deg, targetApopasisAltitude_km)

		self.gamma0_deg = gamma0_deg_guess_high
		ans2 = self.hitsTargetApoapsis2(t_sec, dt, delta_deg, targetApopasisAltitude_km)

		# if product of above flags is negative, then overshoot limit is within user
		# specified bounds, proceed to bisection.
		if ans1 * ans2 < 0:
			# print("EFPA = "+str(gamma0_deg_guess_low)+ " deg. does not meet target \
			# apoapsis altitude.")
			# print("EFPA = "+str(gamma0_deg_guess_high)+" deg. exceeds target
			# apoapsis altitude.")
			# print("Undershoot limit is within user specified bounds. Beginning
			# bisection search...")
			# print("")
			# bisection algorithm begins here, while abs(ub-lb)>tol continue bisection
			while abs(gamma0_deg_guess_high - gamma0_deg_guess_low) > gamma_deg_tol:
				gamma0_deg_guess_mid = 0.5 * (gamma0_deg_guess_low + gamma0_deg_guess_high)

				self.gamma0_deg = gamma0_deg_guess_low
				ans1 = self.hitsTargetApoapsis2(t_sec, dt, delta_deg, targetApopasisAltitude_km)

				self.gamma0_deg = gamma0_deg_guess_mid
				ans2 = self.hitsTargetApoapsis2(t_sec, dt, delta_deg, targetApopasisAltitude_km)

				self.gamma0_deg = gamma0_deg_guess_high
				ans3 = self.hitsTargetApoapsis2(t_sec, dt, delta_deg, targetApopasisAltitude_km)

				if ans1 * ans2 < 0:
					gamma0_deg_guess_high = gamma0_deg_guess_mid
				elif ans2 * ans3 < 0:
					gamma0_deg_guess_low = gamma0_deg_guess_mid

				# print("EFPA = "+str(gamma0_deg_guess_low)+ " deg. does not meet\
				# target apoapsis altitude.")
				# print("EFPA = "+str(gamma0_deg_guess_high)+" deg. exceeds target\
				# apoapsis altitude.")
				# print("")

				# set overShootLimit to lower bound
				# set exitflag to 1.0
				EFPALimit = gamma0_deg_guess_high
				exitflag = 1.0

		# if product of flags is positive, overshootlimit is outside user \
		# specified bounds, print warning message.
		else:
			print("EFPA limit is outside user specified bounds.")
			EFPALimit = 0.0
			exitflag = 0.0

		self.gamma0_deg = temp_var_1
		self.LD = temp_var_2

		return EFPALimit, exitflag

	def findUnderShootLimitD(self, t_sec, dt, gamma0_deg_guess_low, gamma0_deg_guess_high, gamma_deg_tol,
									targetApopasisAltitude_km):
		"""
		This function computes the limiting undershoot 
		EFPA for drag modulation aerocapture. Does not include planet rotation.

		Parameters
		-------------
		t_sec : float
			propogation time, seconds
		dt : float
			max. time step, seconds
		gamma0_deg_guess_low : float
			lower bound for the guess of limit FPA, deg
		gamma0_deg_guess_high : float
			upper bound for the guess of limit FPA, deg
		gamma_deg_tol : float
			desired accuracy for computation of the limit, deg
		targetApopasisAltitude_km : float
			target apoapsis altitude , km
		
		Returns
		----------
		underShootLimitD : float
			undershoot limit EFPA, deg
		exitflagD_us : float
			flag to indicate if a solution could not be found for the 
			undershoot limit EFPA

		exitflagD_us = 1.0 indicates undershoot limit was found.
		exitflagD_us = 0.0 indicates undershoot limit was not found 
		within user specified bounds.
		
		"""

		self.beta = self.beta1*self.betaRatio
		self.CD = self.mass / (self.beta*self.A)
		
		underShootLimitD, exitflagD_us = self.findEFPALimitD(t_sec, dt, gamma0_deg_guess_low,
															gamma0_deg_guess_high, gamma_deg_tol,
															targetApopasisAltitude_km)

		return underShootLimitD, exitflagD_us

	def findUnderShootLimitD2(self, t_sec, dt, gamma0_deg_guess_low, gamma0_deg_guess_high, gamma_deg_tol,
							 targetApopasisAltitude_km):
		"""
		This function computes the limiting undershoot
		EFPA for drag modulation aerocapture. Includes planet rotation.

		Parameters
		-------------
		t_sec : float
			propogation time, seconds
		dt : float
			max. time step, seconds
		gamma0_deg_guess_low : float
			lower bound for the guess of limit FPA, deg
		gamma0_deg_guess_high : float
			upper bound for the guess of limit FPA, deg
		gamma_deg_tol : float
			desired accuracy for computation of the limit, deg
		targetApopasisAltitude_km : float
			target apoapsis altitude , km

		Returns
		----------
		underShootLimitD : float
			undershoot limit EFPA, deg
		exitflagD_us : float
			flag to indicate if a solution could not be found for the
			undershoot limit EFPA

		exitflagD_us = 1.0 indicates undershoot limit was found.
		exitflagD_us = 0.0 indicates undershoot limit was not found
		within user specified bounds.

		"""

		self.beta = self.beta1 * self.betaRatio
		self.CD = self.mass / (self.beta * self.A)

		underShootLimitD, exitflagD_us = self.findEFPALimitD2(t_sec, dt, gamma0_deg_guess_low,
															 gamma0_deg_guess_high, gamma_deg_tol,
															 targetApopasisAltitude_km)

		return underShootLimitD, exitflagD_us

	def findOverShootLimitD(self, t_sec, dt, gamma0_deg_guess_low, gamma0_deg_guess_high, gamma_deg_tol,
								targetApopasisAltitude_km):
		"""
		This function computes the limiting overshoot 
		EFPA for drag modulation aerocapture. Does not include planet rotation.

		Parameters
		------------
		t_sec : float
			propogation time, seconds
		dt : float
			max. time step, seconds
		gamma0_deg_guess_low : float
			lower bound for the guess of limit FPA, deg
		gamma0_deg_guess_high : float
			upper bound for the guess of limit FPA, deg
		gamma_deg_tol : float
			desired accuracy for computation of the limit, deg
		targetApopasisAltitude_km : float
			target apoapsis altitude , km
		
		Returns
		----------
		overShootLimitD : float
			overshoot limit EFPA, deg
		exitflagD_os : float
			flag to indicate if a solution could not be found for the 
			overshoot limit EFPA

		exitflagD_os = 1.0 indicates over shoot limit was found.
		exitflagD_os = 0.0 indicates overshoot limit was not found 
		within user specified bounds.
		
		"""
		self.beta = self.beta1
		self.CD = self.mass / (self.beta*self.A)
		
		overShootLimitD,exitflagD_os = self.findEFPALimitD(t_sec,dt,gamma0_deg_guess_low,
										gamma0_deg_guess_high,gamma_deg_tol, targetApopasisAltitude_km)
		# print('overShootLimit: '+str(overShootLimitD))
		
		return overShootLimitD, exitflagD_os

	def findOverShootLimitD2(self, t_sec, dt, gamma0_deg_guess_low, gamma0_deg_guess_high, gamma_deg_tol,
							targetApopasisAltitude_km):
		"""
		This function computes the limiting overshoot
		EFPA for drag modulation aerocapture. Includes planet rotation.

		Parameters
		------------
		t_sec : float
			propogation time, seconds
		dt : float
			max. time step, seconds
		gamma0_deg_guess_low : float
			lower bound for the guess of limit FPA, deg
		gamma0_deg_guess_high : float
			upper bound for the guess of limit FPA, deg
		gamma_deg_tol : float
			desired accuracy for computation of the limit, deg
		targetApopasisAltitude_km : float
			target apoapsis altitude , km

		Returns
		----------
		overShootLimitD : float
			overshoot limit EFPA, deg
		exitflagD_os : float
			flag to indicate if a solution could not be found for the
			overshoot limit EFPA

		exitflagD_os = 1.0 indicates over shoot limit was found.
		exitflagD_os = 0.0 indicates overshoot limit was not found
		within user specified bounds.

		"""
		self.beta = self.beta1
		self.CD = self.mass / (self.beta * self.A)

		overShootLimitD, exitflagD_os = self.findEFPALimitD2(t_sec, dt, gamma0_deg_guess_low,
															gamma0_deg_guess_high, gamma_deg_tol,
															targetApopasisAltitude_km)
		# print('overShootLimit: '+str(overShootLimitD))

		return overShootLimitD, exitflagD_os

	def computeTCWD(self, t_sec, dt, gamma0_deg_guess_low, gamma0_deg_guess_high,
							gamma_deg_tol, targetApopasisAltitude_km):
		"""
		Computes the theoretical corridor width (TCWD) for 
		drag modulation aerocapture. Does not include planet rotation.

		TCWD = overShootLimit - underShootLimit
		
		Parameters
		------------
		t_sec : float
			propogation time, seconds
		dt : float
			max. time step, seconds
		gamma0_deg_guess_low : float
			lower bound for the EFPA guess, deg
		gamma0_deg_guess_high : float
			upper bound for the EFPA guess, deg
		gamma_deg_tol : float
			EFPA error tolerance
		targetApopasisAltitude_km : float
			target apoapsis altitude , km
		
		Returns
		----------
		TCWD : float
			Theoretical Corridor Width (Drag Modulation), deg
		"""
		
		underShootLimitD,exitflagD_us = self.findUnderShootLimitD(t_sec, dt, gamma0_deg_guess_low,
											gamma0_deg_guess_high, gamma_deg_tol, targetApopasisAltitude_km)
		overshootLimitD, exitflagD_os = self.findOverShootLimitD (t_sec,dt,gamma0_deg_guess_low,\
										gamma0_deg_guess_high, gamma_deg_tol, targetApopasisAltitude_km)
		# print('underShootLimitD: '+str(underShootLimitD))
		# print('overShootLimitD : '+str(overshootLimitD))
		# print('TCWD            : '+str(overshootLimitD-underShootLimitD))

		TCWD = overshootLimitD-underShootLimitD
		
		return TCWD

	def computeTCWD2(self, t_sec, dt, gamma0_deg_guess_low, gamma0_deg_guess_high,
					gamma_deg_tol, targetApopasisAltitude_km):
		"""
		Computes the theoretical corridor width (TCWD) for
		drag modulation aerocapture. Includes planet rotation.

		TCWD = overShootLimit - underShootLimit

		Parameters
		------------
		t_sec : float
			propogation time, seconds
		dt : float
			max. time step, seconds
		gamma0_deg_guess_low : float
			lower bound for the EFPA guess, deg
		gamma0_deg_guess_high : float
			upper bound for the EFPA guess, deg
		gamma_deg_tol : float
			EFPA error tolerance
		targetApopasisAltitude_km : float
			target apoapsis altitude , km

		Returns
		----------
		TCWD : float
			Theoretical Corridor Width (Drag Modulation), deg
		"""

		underShootLimitD, exitflagD_us = self.findUnderShootLimitD2(t_sec, dt, gamma0_deg_guess_low,
																   gamma0_deg_guess_high, gamma_deg_tol,
																   targetApopasisAltitude_km)
		overshootLimitD, exitflagD_os = self.findOverShootLimitD2(t_sec, dt, gamma0_deg_guess_low, \
																 gamma0_deg_guess_high, gamma_deg_tol,
																 targetApopasisAltitude_km)
		# print('underShootLimitD: '+str(underShootLimitD))
		# print('overShootLimitD : '+str(overshootLimitD))
		# print('TCWD            : '+str(overshootLimitD-underShootLimitD))

		TCWD = overshootLimitD - underShootLimitD

		return TCWD

	def createQPlot(self, t_sec, dt, delta_deg):
		"""
		Creates q-plots as described by Cerimele and Gamble, 1985.

		Parameters
		----------
		t_sec : float
			propogation time, seconds
		dt : float
			max. solver time step
		delta_deg : float
			commanded bank angle, degrees

		Returns
		---------
		plt.plot object

		"""

		self.propogateEntry(t_sec, dt, delta_deg)
		
		# Values for -13.64 deg, these are the linear fit 
		# y = ax + b parameters used to compute Ghdot and Gq (see refs.)
		a = -0.16856558809141109
		b = 50777.960704102341
	
		x_arr = np.linspace(100.0E3,298.0E3,101)
		y_arr = a*x_arr + b
		
		fig = plt.figure()
		fig.set_size_inches([3.25,3.25])
		plt.rc('font',family='Times New Roman')
		params = {'mathtext.default': 'regular' }          
		plt.rcParams.update(params)
		
		plt.plot(self.h_kmc*1E3,self.dyn_pres_atm*1.01325E5,'r-',linewidth=3.0)

		plt.xlim(100.0E3, 700.0E3)
		plt.ylim(0.0,12500.0)
		
		# plt.plot(x_arr,y_arr,'k-',linewidth=2.0, linestyle='dashed')
		plt.xlabel("Altitude, m", fontsize=10)
		plt.ylabel("Dynamic pressure, Pa ",fontsize=10)

		plt.xticks(np.array([200.0E3, 400.00E3, 600.0E3]), fontsize=10)
		
		ax=plt.gca()
		ax.tick_params(direction='in')
		ax.yaxis.set_ticks_position('both')
		ax.xaxis.set_ticks_position('both')
		ax.tick_params(axis='x',labelsize=10)
		ax.tick_params(axis='y',labelsize=10)
		ax.tick_params(direction='in')
		
		ax.annotate(r'$\bar{q} = -0.1686h + 50778$',
			xy=(248586, 9103.4) ,
			xytext=(331181, 9103.4),
			arrowprops=dict(arrowstyle="<-"),  va="center", ha="left", fontsize=9)
		'''
		plt.savefig('plots/girijaSaikia2020-dyn-pres-profile.png', \
					bbox_inches='tight')
		plt.savefig('plots/girijaSaikia2020-dyn-pres-profile.pdf', \
					dpi=300,bbox_inches='tight')
		plt.savefig('plots/girijaSaikia2020-dyn-pres-profile.eps', \
					dpi=300,bbox_inches='tight')
		'''
		plt.show()

	def compute_ApoapsisAltitudeKm(self, terminal_r, terminal_v, terminal_g, terminal_theta, terminal_phi, terminal_psi):
		"""
		Compute the apoapsis altitude given conditions at 
		atmospheric exit interface. Note this function includes
		correction to account for rotation of planet.

		Terminal values refer to those at atmospheric exit.

		Parameters
		---------
		terminal_r : float
			radial distance, m
		terminal_v : float
			terminal speed, m/s
		terminal_g : float
			terminal FPA, rad
		terminal_theta : float
			terminal longitude, rad
		terminal_phi : float
			terminal latitude, rad
		terminal_psi : float
			terminal heading angle

		Returns
		--------
		hp_km : float
			apoapsis altitude, km

		"""

		# Compute planet relative speed in Cartesian XYZ coordinates
		v_pr_x = terminal_v*np.sin(terminal_g)*np.cos(terminal_phi)*np.cos(terminal_theta) +\
				terminal_v*np.cos(terminal_g)*np.cos(terminal_psi)*(-1*np.sin(terminal_theta)) +\
				terminal_v*np.cos(terminal_g)*np.sin(terminal_psi)*(-1*np.sin(terminal_phi)*np.cos(terminal_theta))

		v_pr_y = terminal_v*np.sin(terminal_g)*np.cos(terminal_phi)*np.sin(terminal_theta) +\
				terminal_v*np.cos(terminal_g)*np.cos(terminal_psi)*np.cos(terminal_theta) +\
				terminal_v*np.cos(terminal_g)*np.sin(terminal_psi)*(-1*np.sin(terminal_phi)*np.sin(terminal_theta))
		
		v_pr_z = terminal_v*np.sin(terminal_g)*np.sin(terminal_phi) + \
				terminal_v*np.cos(terminal_g)*np.sin(terminal_psi)*np.cos(terminal_phi)

		# Compute inertial speed in Cartesian XYZ coordinates
		v_ie_x = v_pr_x + terminal_r*self.planetObj.OMEGA*np.cos(terminal_phi)*np.sin(terminal_theta)*(-1.0)
		v_ie_y = v_pr_y + terminal_r*self.planetObj.OMEGA*np.cos(terminal_phi)*np.cos(terminal_theta)
		v_ie_z = v_pr_z


		# Compute terminal radial vector
		terminal_r_vec = terminal_r*np.array([np.cos(terminal_phi)*np.cos(terminal_theta),
										np.cos(terminal_phi)*np.sin(terminal_theta),
										np.sin(terminal_phi)])

		# Compute terminal radial unit vector
		terminal_r_hat_vec = terminal_r_vec / np.linalg.norm(terminal_r_vec)

		# Compute inertial velocity vector
		terminal_v_ie_vec = np.array([v_ie_x, v_ie_y, v_ie_z])
		
		# Compute inertial velocity unit vector
		terminal_v_ie_hat_vec = terminal_v_ie_vec / np.linalg.norm(terminal_v_ie_vec)
		
		# Compute inertial flight path angle at exit using
		# terminal inertial radial and velocity vectors
		terminal_fpa_ie_deg = 90.0 - (180/np.pi)*np.arccos(np.dot(terminal_r_hat_vec,terminal_v_ie_hat_vec))
		
		terminal_fpa_ie_rad = terminal_fpa_ie_deg*np.pi/180.0

		# Compute inertial velocity magnitude
		v_ie_mag = np.sqrt(v_ie_x**2 + v_ie_y**2 + v_ie_z**2)
		
		# Compute orbit energy using inertial speed
		terminal_E = self.computeEnergyScalar(terminal_r, v_ie_mag)
		terminal_h = self.computeAngMomScalar(terminal_r, v_ie_mag, terminal_fpa_ie_rad)

		# Compute semi-major axis and eccentricity of the post atmospheric exit orbit
		terminal_a = self.computeSemiMajorAxisScalar(terminal_E)
		terminal_e = self.computeEccScalar(terminal_h, terminal_E)

		# Compute apoapsis radius, apoapsis altitude, apoapsis altitude in KM
		rp = terminal_a*(1.0+terminal_e)
		hp = rp - self.planetObj.RP
		hp_km = hp / 1.0E3

		return hp_km

	def compute_PeriapsisAltitudeKm(self, terminal_r, terminal_v, terminal_g, terminal_theta, terminal_phi, terminal_psi):
		"""
		Compute the periapsis altitude given conditions at 
		atmospheric exit interface. Note this function includes
		correction to account for rotation of planet.

		Terminal values refer to those at atmospheric exit.

		Parameters
		---------
		terminal_r : float
			radial distance, m
		terminal_v : float
			terminal speed, m/s
		terminal_g : float
			terminal FPA, rad
		terminal_theta : float
			terminal longitude, rad
		terminal_phi : float
			terminal latitude, rad
		terminal_psi : float
			terminal heading angle

		Returns
		--------
		hp_km : float
			periapsis altitude, km

		"""
		# Compute planet relative speed in Cartesian XYZ coordinates
		v_pr_x = terminal_v*np.sin(terminal_g)*np.cos(terminal_phi)*np.cos(terminal_theta) +\
				terminal_v*np.cos(terminal_g)*np.cos(terminal_psi)*(-1*np.sin(terminal_theta)) +\
				terminal_v*np.cos(terminal_g)*np.sin(terminal_psi)*(-1*np.sin(terminal_phi)*np.cos(terminal_theta))
		
		v_pr_y = terminal_v*np.sin(terminal_g)*np.cos(terminal_phi)*np.sin(terminal_theta) +\
				terminal_v*np.cos(terminal_g)*np.cos(terminal_psi)*np.cos(terminal_theta) +\
				terminal_v*np.cos(terminal_g)*np.sin(terminal_psi)*(-1*np.sin(terminal_phi)*np.sin(terminal_theta))

		v_pr_z = terminal_v*np.sin(terminal_g)*np.sin(terminal_phi) + \
				 terminal_v*np.cos(terminal_g)*np.sin(terminal_psi)*np.cos(terminal_phi)

		# Compute inertial speed in Cartesian XYZ coordinates
		v_ie_x = v_pr_x + terminal_r*self.planetObj.OMEGA*np.cos(terminal_phi)*np.sin(terminal_theta)*(-1.0)
		v_ie_y = v_pr_y + terminal_r*self.planetObj.OMEGA*np.cos(terminal_phi)*np.cos(terminal_theta)
		v_ie_z = v_pr_z

		# Compute terminal radial vector
		terminal_r_vec = terminal_r*np.array([np.cos(terminal_phi)*np.cos(terminal_theta),
											np.cos(terminal_phi)*np.sin(terminal_theta),
											np.sin(terminal_phi)])

		# Compute terminal radial unit vector
		terminal_r_hat_vec = terminal_r_vec / np.linalg.norm(terminal_r_vec)

		# Compute inertial velocity vector
		terminal_v_ie_vec = np.array([v_ie_x, v_ie_y, v_ie_z])
		
		# Compute inertial velocity unit vector
		terminal_v_ie_hat_vec = terminal_v_ie_vec / np.linalg.norm(terminal_v_ie_vec)
		
		# Compute inertial flight path angle at exit using
		# terminal inertial radial and velocity vectors
		terminal_fpa_ie_deg = 90.0 - (180/np.pi)*np.arccos(np.dot(terminal_r_hat_vec,terminal_v_ie_hat_vec))
		
		terminal_fpa_ie_rad = terminal_fpa_ie_deg*np.pi/180.0

		# Compute inertial velocity magnitude
		v_ie_mag = np.sqrt(v_ie_x**2 + v_ie_y**2 + v_ie_z**2)
		
		# Compute orbit energy using inertial speed
		terminal_E = self.computeEnergyScalar(terminal_r, v_ie_mag)
		terminal_h = self.computeAngMomScalar(terminal_r, v_ie_mag, terminal_fpa_ie_rad)

		# Compute semi-major axis and eccentricity of the post atmospheric exit orbit
		terminal_a = self.computeSemiMajorAxisScalar(terminal_E)
		terminal_e = self.computeEccScalar(terminal_h, terminal_E)

		# Compute apoapsis radius, apoapsis altitude, apoapsis altitude in KM
		rp = terminal_a*(1.0-terminal_e)
		hp = rp - self.planetObj.RP
		hp_km = hp / 1.0E3

		return hp_km

	def createDensityMeasuredFunction(self, h_step_array, \
									  density_mes_array, lowAlt_km, \
									  numPoints_lowAlt):
		"""
		Computes a density function based on measurements made during the 
		descending leg of the aerocapture maneuver.

		Parameters
		----------
		h_step_array : numpy.ndarray
			height array at which density is measured, km
		density_mes_array : numpy.ndarray
			density array corresponding to h_step_array, kg/m3
		lowAlt_km : float
			lower altitude to which density model is to be extrapolated
			based on available measurements, km
		numPoints_lowAlt : int
			number of points to evaluate extrapolation at below the 
			altitude where measurements are available.

		Returns
		----------
		density_mes_int : scipy.interpolate.interpolate.interp1d
			interpolated measured density lookup function 
		minAlt : float       
			minimum altitude at which density measurements were available
		
		"""
		# clean h_step_array and density_mes_array by deleting first 
		# zero element
		# The first entry in these arrays is 0, used for initialization 
		# purpose and has to be removed.
		h_step_array = np.delete(h_step_array, 0)
		density_mes_array = np.delete(density_mes_array, 0)

		# lowAlt_km = 120E3
		# numPoints_lowAlt = 101

		# Compute index of minimum altitude in h_step_array
		minAltIndex = np.argmin(h_step_array)
		minAlt = h_step_array[minAltIndex]*1E3

		# Compute measured density interpolation function using a linear 
		# interpolation between available data points.
		# For values of h outside the data range, use the value at the bounds

		density_mes_int_upper = interp1d(h_step_array*1000.0, density_mes_array, kind='linear',
								  fill_value=(max(density_mes_array), min(density_mes_array)), bounds_error=False)

		scaleHeightminAlt = self.planetObj.scaleHeight(minAlt, density_mes_int_upper)

		h_low_array = np.linspace(minAlt-1000.0, lowAlt_km*1E3, numPoints_lowAlt)

		d_low_array = density_mes_int_upper(minAlt)*np.exp((minAlt-h_low_array)/(scaleHeightminAlt))

		h_array = np.concatenate((h_step_array*1000.0,h_low_array), axis=0)
		d_array = np.concatenate((density_mes_array, d_low_array), axis=0)

		density_mes_int = interp1d(h_array, d_array, kind='linear', fill_value=(max(d_array),min(d_array)),
									bounds_error=False)

		return density_mes_int, minAlt

	def setMaxRollRate(self, maxRollRate):
		"""
		Set the maximum allowed vehicle roll rate (deg/s)
		
		Parameters
		----------
		maxRollRate : float
			maximum roll rate, degrees per second
		"""
		self.maxRollRate = maxRollRate

	def psuedoController(self,DeltaCMD_deg_command, Delta_deg_ini, timestep):
		"""
		Pseudo controller implenetation for maximum roll rate 
		constraint.
		
		Parameters
		----------
		DeltaCMD_deg_command : float
			commanded bank angle from guidance algorithm, deg
		Delta_deg_ini : float      
			current vehicle bank angle, deg
		maxBankRate : float
			maximum allowed roll rate, deg/s
		timestep : float
			guidance cycle timestep
		
		Returns
		----------
		DeltaCMD_deg : float   
			actual bank angle response using pseudocontroller, deg
		Delta_deg_ini : float
			current bank angle, same as actual bank angle 
			(is redundant)
		
		"""

		if np.abs(DeltaCMD_deg_command - Delta_deg_ini) > self.maxRollRate*timestep:
			# if the error between current and target bank angle is greater 
			# than what can be achieved using max. roll
			# rate in one guidance cycle, then a maximum rate turn is 
			# commanded in the correct direction (controlled by np.sign())
			DeltaCMD_deg = Delta_deg_ini + np.sign(DeltaCMD_deg_command - Delta_deg_ini)*self.maxRollRate*timestep
			Delta_deg_ini = DeltaCMD_deg
		
		else:
			# else, the commanded bank angle is acheived in the 
			# guidance cycle by rolling the vehicle at a lower 
			# (< maxBankRate ) roll rate
			# so as to achieve the desired bank angle at the 
			# end of the guidance cycle.
			bankRate = np.abs(DeltaCMD_deg_command - Delta_deg_ini) / timestep
			DeltaCMD_deg = Delta_deg_ini + np.sign(DeltaCMD_deg_command - Delta_deg_ini)*self.maxRollRate*timestep
			Delta_deg_ini = DeltaCMD_deg

		return DeltaCMD_deg, Delta_deg_ini

	def predictApoapsisAltitudeKm_withLiftUp(self, h0_km, theta0_deg, phi0_deg, v0_kms, gamma0_deg, psi0_deg,
											 drange0_km, heatLoad0, t_sec, dt, delta_deg, density_mes_int):
		"""
		Compute apoapsis altitude using full lift up bank 
		command from current vehicle state till atmospheric exit.

		Parameters
		----------
		h0_km : float
			current vehicle altitude, km
		theta0_deg : float
			current vehicle longitude, deg
		phi0_deg : float
			current vehicle latitude, deg
		v0_kms : float
			current vehicle speed, km/s
		gamma0_deg : float
			current FPA, deg
		psi0_deg : float
			current heading angle, deg
		drange0_km : float
			current downrange, km
		heatLoad0 : float
			current heatload, J/cm2
		t_sec : float
			propogation time, seconds
		dt : float
			max. solver timestep
		delta_deg : float
			commanded bank angle, deg
		density_mes_int : scipy.interpolate.interpolate.interp1d
			measured density interpolation function

		"""

		t_minc, h_kmc, v_kmsc, phi_degc, psi_degc, theta_degc, gamma_degc, drange_kmc, exitflag, acc_net_g,\
		dyn_pres_atm, stag_pres_atm, q_stag_total, heatload, acc_drag_g = \
	    self.propogateEntry_util(h0_km, theta0_deg, phi0_deg, v0_kms, gamma0_deg, psi0_deg, drange0_km,
								 heatLoad0, t_sec, dt, delta_deg, density_mes_int)
		
		terminal_apoapsis_km = self.compute_ApoapsisAltitudeKm(self.planetObj.RP+h_kmc[-1]*1E3,
															v_kmsc[-1]*1E3, gamma_degc[-1]*np.pi/180.0,
															theta_degc[-1]*np.pi/180.0, phi_degc[-1]*np.pi/180.0,
															psi_degc[-1]*np.pi/180.0)

		return terminal_apoapsis_km

	def predictApoapsisAltitudeKm_withLiftUp2(self, h0_km, theta0_deg, phi0_deg, v0_kms, gamma0_deg, psi0_deg,
												drange0_km, heatLoad0, t_sec, dt, delta_deg, density_mes_int):
		"""
		Compute apoapsis altitude using full lift up bank
		command from current vehicle state till atmospheric exit.

		Parameters
		----------
		h0_km : float
			current vehicle altitude, km
		theta0_deg : float
			current vehicle longitude, deg
		phi0_deg : float
			current vehicle latitude, deg
		v0_kms : float
			current vehicle speed, km/s
		gamma0_deg : float
			current FPA, deg
		psi0_deg : float
			current heading angle, deg
		drange0_km : float
			current downrange, km
		heatLoad0 : float
			current heatload, J/cm2
		t_sec : float
			propogation time, seconds
		dt : float
			max. solver timestep
		delta_deg : float
			commanded bank angle, deg
		density_mes_int : scipy.interpolate.interpolate.interp1d
			measured density interpolation function

		"""

		t_minc, h_kmc, v_kmsc, phi_degc, psi_degc, theta_degc, gamma_degc, \
		drange_kmc, exitflag, acc_net_g, dyn_pres_atm, stag_pres_atm, q_stag_total, \
		heatload, acc_drag_g = \
		self.propogateEntry_util2(h0_km, theta0_deg, phi0_deg, v0_kms, gamma0_deg, psi0_deg, drange0_km,
									heatLoad0, t_sec, dt, delta_deg, density_mes_int)

		terminal_apoapsis_km = self.compute_ApoapsisAltitudeKm(self.planetObj.RP + h_kmc[-1] * 1E3,
																v_kmsc[-1] * 1E3, gamma_degc[-1] * np.pi / 180.0,
																theta_degc[-1] * np.pi / 180.0,
																phi_degc[-1] * np.pi / 180.0,
																psi_degc[-1] * np.pi / 180.0)

		return terminal_apoapsis_km

	def setTargetOrbitParams(self, target_peri_km, target_apo_km, target_apo_km_tol):
		"""
		Set the target capture orbit parameters.

		Parameters
		----------
		target_peri_km : float
			target periapsis altitude, km
		target_apo_km : float
			target apoapsis altitude, km
		target_apo_km_tol : float
			target apoapsis altitude error tolerance, km
			used by guidance algorithm
		"""
		self.target_peri_km = target_peri_km
		self.target_apo_km = target_apo_km
		self.target_apo_km_tol = target_apo_km_tol

	def compute_periapsis_raise_DV(self, current_peri_km, current_apo_km, target_peri_km):
		"""
		Compute the propulsive DV to raise the orbit periapsis to 
		the target value.

		Parameters
		----------
		current_peri_km : float
			current periapsis altitude, km
		current_apo_km  : float
			current apoapsis altitude, km
		target_peri_km  : float
			target periapsis altitude, km
		
		Returns
		----------
		dV :float
			periapse raise DV, m/s
		
		"""

		# compute the energy of the current orbit, E = -GM / 2a
		E_current = -self.planetObj.GM / ((self.planetObj.RP+current_apo_km*1E3 + current_peri_km*1E3 + self.planetObj.RP))
		# Compute the orbital speed at the apoapsis
		v_apo_current = np.sqrt(2*E_current + 2*self.planetObj.GM/(self.planetObj.RP+current_apo_km*1E3))
		# Compute the energy of the target orbit
		E_target = -self.planetObj.GM / ((self.planetObj.RP+current_apo_km*1E3 + target_peri_km*1E3 + self.planetObj.RP))
		
		# Compute the orbital speed at apoapsis for the target orbit
		v_apo_target = np.sqrt(2*E_target + 2*self.planetObj.GM/(self.planetObj.RP+current_apo_km*1E3))
		
		# Compute the increment in velocity required to raise 
		# the periapsis to target value
		dV = v_apo_target - v_apo_current

		return dV

	def compute_apoapsis_raise_DV(self, peri_km_current, apo_km_current, apo_km_target):
		"""
		Compute the propulsive DV to raise the orbit apoapsis to 
		the target value.

		Parameters
		----------
		peri_km_current : float
			current periapsis altitude, km
		apo_km_current  : float
			current apoapsis altitude, km
		apo_km_target  : float
			target apoapsis altitude, km
		
		Returns
		----------
		dV :float
			apoapsis raise DV, m/s
		
		"""

		# compute the energy of the current orbit, E = -GM / 2a
		E_current  = -self.planetObj.GM / ((self.planetObj.RP+apo_km_current*1E3 + peri_km_current*1E3 + self.planetObj.RP))
		
		# Compute the orbital speed at the periapsis
		v_peri_current =  np.sqrt(2*E_current + 2*self.planetObj.GM/(peri_km_current*1E3+self.planetObj.RP))
		# Compute the energy of the target orbit
		E_target = -self.planetObj.GM / ((apo_km_target*1E3+self.planetObj.RP  + peri_km_current*1E3+self.planetObj.RP))
		# Compute the orbital speed at periapsis for the target orbit
		v_peri_target =  np.sqrt(2*E_target + 2*self.planetObj.GM/(peri_km_current*1E3+self.planetObj.RP))
		# Compute the increment/decrement in velocity required to correct the apoapsis
		dV = v_peri_target - v_peri_current

		return dV

	def setEquilibriumGlideParams(self, Ghdot, Gq, v_switch_kms, lowAlt_km, numPoints_lowAlt, hdot_threshold):
		"""
		Set equilibrium glide phase guidance parameters

		Parameters
		-----------
		Ghdot : float
			Ghdot term
		Gq : float
			Gq term
		v_switch_kms : float
			speed below which eq. glide phase is terminated
		lowAlt_km : float
			lower altitude to which density model is to be extrapolated
			based on available measurements, km
		numPoints_lowAlt : int
			number of points to evaluate extrapolation at below the 
			altitude where measurements are available
		hdot_threshold : float
			threshold altitude rate (m/s) above which density measurement
			is terminated and apoapsis prediction is initiated

		"""
		self.Ghdot = Ghdot
		self.Gq = Gq
		self.v_switch_kms = v_switch_kms
		self.lowAlt_km = lowAlt_km
		self.numPoints_lowAlt = numPoints_lowAlt
		self.hdot_threshold = hdot_threshold

	def propogateEquilibriumGlide(self, timeStep, dt, maxTimeSecs):
		"""
		Implements the equilibrium glide phase of the guidance scheme.

		Parameters
		--------
		timeStep : float
			Guidance cycle time, seconds
		dt : float
			Solver max. time step, seconds
		maxTimeSecs : float
			max. time for propogation, seconds

		"""
		
		counter = 0

		# -------------------------------------------
		g0 = self.planetObj.GM/(self.planetObj.RP**2)

		h_skip_km = self.planetObj.h_skip / 1000.0

		self.t_step_array = np.array([0.0])
		self.delta_deg_array = np.array([0.0])
		self.hdot_array = np.array([0.0])
		self.hddot_array = np.array([0.0])
		self.qref_array = np.array([0.0])
		self.q_array = np.array([0.0])
		self.h_step_array = np.array([0.0])
		self.acc_step_array = np.array([0.0])
		self.acc_drag_array = np.array([0.0])
		self.density_mes_array = np.array([0.0])

		self.propogateEntry(1.0, dt, 0.0)

		t_min = self.t_minc
		h_km = self.h_kmc
		v_kms = self.v_kmsc
		phi_deg = self.phi_degc
		psi_deg = self.psi_degc
		theta_deg = self.theta_degc
		gamma_deg = self.gamma_degc
		drange_km = self.drange_kmc

		acc_net_g = self.acc_net_g
		dyn_pres_atm = self.dyn_pres_atm
		stag_pres_atm = self.stag_pres_atm
		q_stag_total = self.q_stag_total
		heatload = self.heatload
		acc_drag_g = self.acc_drag_g

		# Set the current vehicle speed here.
		self.h_current_km = h_km[-1]
		self.v_current_kms = v_kms[-1]

		customFlag = 0.0

		Delta_deg_ini = 0.0

		# if the currrent speed is greater than the exit phase 
		# switch speed continue iterating in the equilibrium glide 
		# phase mode
		while self.v_current_kms > self.v_switch_kms:
			# Reset the initial conditions to the terminal conditions 
			# of the propgated solution from the previous step.
			
			h0_km = h_km[-1]           # Entry altitude above planet surface
			theta0_deg = theta_deg[-1]      # Entry longitude
			phi0_deg = phi_deg[-1]        # Entry latitude
			v0_kms = v_kms[-1]          # Entry velocity
			psi0_deg = psi_deg[-1]        # Entry heading angle
			gamma0_deg = gamma_deg[-1]      # Entry flight path angle
			drange0_km = drange_km[-1]      # Downrange

			h0 = h0_km*1.0E3             # Entry altitude above planet surface in meters
			theta0 = theta0_deg*np.pi/180.0  # Entry longitude in radians
			phi0 = phi0_deg*np.pi/180.0    # Entry latitude in radians
			v0 = v0_kms*1.000E3          # Entry velocity in meters/sec, relative to planet
			psi0 = psi0_deg*np.pi/180.0    # Entry heading angle in radians
			gamma0 = gamma0_deg*np.pi/180.0  # Entry flight path angle in radians
			# ------------------------------------------------------------------------------------------------

			# ------------------------------------------------------------------------------------------------
			# Initialize iterable variables used in guidance loop, i = at the current time
			# ------------------------------------------------------------------------------------------------
			hi = h0                           # Set the current altitude as the entry altitude
			ri = h0 + self.planetObj.RP       # Set the current radial distance 
			vi = v0                           # Set the current speed (m/s) as the entry speed
			qi = 0.5*self.planetObj.density(hi)*vi**2.0      # Set the current dynamic pressure
			gammai = gamma0                   # Set the current FPA to entry FPA
			hdoti = vi*np.sin(gammai)         # Compute the current altitude rate hdot

			# ------------------------------------------------------------------------------------------------
			# Initialize reference quantities used in the algorithm
			qrefi = (-1.0*self.mass*g0)/(0.75*self.CL*self.A)*(1 - vi**2.0 / (g0*ri))  
			# Set initial reference dyn pres.
			# ------------------------------------------------------------------------------------------------

			# Set the current heatload Ji = terminal heatload 
			# of the previous propogated solution
			Ji = self.heatload[-1]

			# Compute the equilibrium glide bank angle (sigma_EQ.GL)
			cosDeltaEQGL = ((self.mass*g0) / (self.CL*qi*self.A)) * (1.0 - vi**2.0 / (g0*ri))
			
			# Compute the commanded bank angle (sigma CMD)
			cosDeltaCMD = cosDeltaEQGL - self.Ghdot*hdoti/qi + self.Gq*((qi-qrefi)/qi)

			# if the |cosDeltaCMD| > 1.0 then set the bank angle  = 0.0 (full lift up)
			if cosDeltaCMD > 1.0:
				DeltaCMD = 0.0
			# else compute the commanded bank angle using arccos(cosDeltaCMD)
			elif cosDeltaCMD < -1.0:
				DeltaCMD = np.pi
			else:
				DeltaCMD = np.arccos(cosDeltaCMD)

			# Compute the commanded bank angle in degrees	
			DeltaCMD_deg_command = DeltaCMD*180.0/np.pi 

			DeltaCMD_deg, Delta_deg_ini = self.psuedoController(DeltaCMD_deg_command, Delta_deg_ini, timeStep)

			# propogate the vehicle state to 1 second in advance from 
			# the current state using the commanded bank angle deltaCMD
			self.setInitialState(h0_km, theta0_deg, phi0_deg, v0_kms, psi0_deg, gamma0_deg, drange0_km, Ji)
			self.propogateEntry(timeStep, dt, DeltaCMD_deg)

			t_min_c = self.t_minc
			h_km_c = self.h_kmc
			v_kms_c = self.v_kmsc
			phi_deg_c = self.phi_degc
			psi_deg_c = self.psi_degc
			theta_deg_c = self.theta_degc
			gamma_deg_c = self.gamma_degc
			drange_km_c = self.drange_kmc

			acc_net_g_c = self.acc_net_g
			dyn_pres_atm_c = self.dyn_pres_atm
			stag_pres_atm_c = self.stag_pres_atm
			q_stag_total_c = self.q_stag_total
			heatload_c = self.heatload
			acc_drag_g_c = self.acc_drag_g

			# Update the time solution array to account for non-zero start time 
			t_min_c = t_min_c + t_min[-1]

			self.t_step_array = np.append(self.t_step_array, t_min[-1])
			self.delta_deg_array = np.append(self.delta_deg_array, DeltaCMD_deg)
			self.hdot_array = np.append(self.hdot_array, hdoti)
			self.qref_array = np.append(self.qref_array, qrefi)
			self.q_array = np.append(self.q_array, qi)
			self.h_step_array = np.append(self.h_step_array, h0_km)

			self.hdotref_array = np.zeros(len(self.t_step_array))
			self.hddoti = (self.hdot_array[-1] - self.hdot_array[-2]) / (self.t_step_array[-1]*60.0 - self.t_step_array[-2]*60.0)
			self.hddot_array = np.append(self.hddot_array, self.hddoti)
			self.acc_step_array = np.append(self.acc_step_array, self.acc_net_g[-1])
			self.acc_drag_array = np.append(self.acc_drag_array, self.acc_drag_g[-1])

			# Update time and other solution vectors
			t_min = np.concatenate((t_min, t_min_c), axis=0)
			h_km = np.concatenate((h_km , h_km_c ), axis=0)
			v_kms = np.concatenate((v_kms, v_kms_c ), axis=0)
			phi_deg = np.concatenate((phi_deg, phi_deg_c ), axis=0)
			psi_deg = np.concatenate((psi_deg, psi_deg_c ), axis=0)
			theta_deg = np.concatenate((theta_deg, theta_deg_c), axis=0)
			gamma_deg = np.concatenate((gamma_deg, gamma_deg_c), axis=0)
			drange_km = np.concatenate((drange_km, drange_km_c), axis=0)

			# Update entry parameter vectors
			acc_net_g = np.concatenate((acc_net_g,acc_net_g_c), axis=0)
			acc_drag_g = np.concatenate((acc_drag_g,acc_drag_g_c), axis=0)
			dyn_pres_atm = np.concatenate((dyn_pres_atm, dyn_pres_atm_c),axis=0)
			stag_pres_atm = np.concatenate((stag_pres_atm, stag_pres_atm_c), axis=0)
			q_stag_total = np.concatenate((q_stag_total, q_stag_total_c), axis=0)
			heatload = np.concatenate((heatload, heatload_c), axis=0)

			self.acc_step_array = np.append(self.acc_step_array, acc_net_g[-1])
			density_mes = 2*self.mass*acc_drag_g[-1]*self.planetObj.EARTHG / (self.CD*self.A*(0.5*(vi+v_kms[-1]*1E3))**2.0)
			self.density_mes_array = np.append(self.density_mes_array, density_mes)

			if hdoti > self.hdot_threshold and customFlag == 0:
				self.density_mes_int, self.minAlt = self.createDensityMeasuredFunction(self.h_step_array,
												   self.density_mes_array, self.lowAlt_km, self.numPoints_lowAlt)
				customFlag = 1.0
			
			if hdoti > self.hdot_threshold:
				terminal_apoapsis_km = \
				self.predictApoapsisAltitudeKm_withLiftUp(h_km[-1],
					theta_deg[-1], phi_deg[-1], v_kms[-1], gamma_deg[-1], psi_deg[-1], drange_km[-1],
					heatload[-1], maxTimeSecs, dt, 0.0 , self.density_mes_int)
				
			else:
				terminal_apoapsis_km = 0.0

			# print("H (km): "+ str('{:.2f}'.format(h0_km))+" HDOT (m/s): \
			# "+ str('{:.2f}'.format(hdoti))+", DeltaCMD :"+str('{:.2f}'. \
			# format(DeltaCMD_deg_command))+", DELTAACT: "+str('{:.2f}'. \
			# format(DeltaCMD_deg))+", PRED. APO: "+str('{:.2f}'.\
			# format(terminal_apoapsis_km)))
			counter+=1
			# update current speed
			h_current_km  = h_km[-1]
			v_current_kms = v_kms[-1]

			if terminal_apoapsis_km> 0 and terminal_apoapsis_km < self.target_apo_km:
				break

			if abs(terminal_apoapsis_km - self.target_apo_km) < self.target_apo_km_tol:
				break

			if h_current_km > self.planetObj.h_skip/1000 - 2.0:
				break

		
		self.t_min_eg = t_min
		self.h_km_eg = h_km
		self.v_kms_eg = v_kms
		self.theta_deg_eg = theta_deg	
		self.phi_deg_eg = phi_deg
		self.psi_deg_eg = psi_deg
		self.gamma_deg_eg = gamma_deg
		self.drange_km_eg = drange_km
		
		self.acc_net_g_eg = acc_net_g
		self.dyn_pres_atm_eg = dyn_pres_atm
		self.stag_pres_atm_eg = stag_pres_atm
		self.q_stag_total_eg = q_stag_total
		self.heatload_eg = heatload

	def propogateEquilibriumGlide2(self, timeStep, dt, maxTimeSecs):
		"""
		Implements the equilibrium glide phase of the guidance scheme.

		Parameters
		--------
		timeStep : float
			Guidance cycle time, seconds
		dt : float
			Solver max. time step, seconds
		maxTimeSecs : float
			max. time for propogation, seconds

		"""

		counter = 0

		# -------------------------------------------
		g0 = self.planetObj.GM / (self.planetObj.RP ** 2)

		h_skip_km = self.planetObj.h_skip / 1000.0

		self.t_step_array = np.array([0.0])
		self.delta_deg_array = np.array([0.0])
		self.hdot_array = np.array([0.0])
		self.hddot_array = np.array([0.0])
		self.qref_array = np.array([0.0])
		self.q_array = np.array([0.0])
		self.h_step_array = np.array([0.0])
		self.acc_step_array = np.array([0.0])
		self.acc_drag_array = np.array([0.0])
		self.density_mes_array = np.array([0.0])

		self.propogateEntry2(1.0, dt, 0.0)

		t_min = self.t_minc
		h_km = self.h_kmc
		v_kms = self.v_kmsc
		phi_deg = self.phi_degc
		psi_deg = self.psi_degc
		theta_deg = self.theta_degc
		gamma_deg = self.gamma_degc
		drange_km = self.drange_kmc

		acc_net_g = self.acc_net_g
		dyn_pres_atm = self.dyn_pres_atm
		stag_pres_atm = self.stag_pres_atm
		q_stag_total = self.q_stag_total
		heatload = self.heatload
		acc_drag_g = self.acc_drag_g

		# Set the current vehicle speed here.
		self.h_current_km = h_km[-1]
		self.v_current_kms = v_kms[-1]

		customFlag = 0.0

		Delta_deg_ini = 0.0

		# if the currrent speed is greater than the exit phase
		# switch speed continue iterating in the equilibrium glide
		# phase mode
		while self.v_current_kms > self.v_switch_kms:
			# Reset the initial conditions to the terminal conditions
			# of the propgated solution from the previous step.

			h0_km = h_km[-1]  # Entry altitude above planet surface
			theta0_deg = theta_deg[-1]  # Entry longitude
			phi0_deg = phi_deg[-1]  # Entry latitude
			v0_kms = v_kms[-1]  # Entry velocity
			psi0_deg = psi_deg[-1]  # Entry heading angle
			gamma0_deg = gamma_deg[-1]  # Entry flight path angle
			drange0_km = drange_km[-1]  # Downrange

			h0 = h0_km * 1.0E3  # Entry altitude above planet surface in meters
			theta0 = theta0_deg * np.pi / 180.0  # Entry longitude in radians
			phi0 = phi0_deg * np.pi / 180.0  # Entry latitude in radians
			v0 = v0_kms * 1.000E3  # Entry velocity in meters/sec, relative to planet
			psi0 = psi0_deg * np.pi / 180.0  # Entry heading angle in radians
			gamma0 = gamma0_deg * np.pi / 180.0  # Entry flight path angle in radians
			# ------------------------------------------------------------------------------------------------

			# ------------------------------------------------------------------------------------------------
			# Initialize iterable variables used in guidance loop, i = at the current time
			# ------------------------------------------------------------------------------------------------
			hi = h0  # Set the current altitude as the entry altitude
			ri = h0 + self.planetObj.RP  # Set the current radial distance
			vi = v0  # Set the current speed (m/s) as the entry speed
			qi = 0.5 * self.planetObj.density(hi) * vi ** 2.0  # Set the current dynamic pressure
			gammai = gamma0  # Set the current FPA to entry FPA
			hdoti = vi * np.sin(gammai)  # Compute the current altitude rate hdot

			# ------------------------------------------------------------------------------------------------
			# Initialize reference quantities used in the algorithm
			qrefi = (-1.0 * self.mass * g0) / (0.75 * self.CL * self.A) * (1 - vi ** 2.0 / (g0 * ri))
			# Set initial reference dyn pres.
			# ------------------------------------------------------------------------------------------------

			# Set the current heatload Ji = terminal heatload
			# of the previous propogated solution
			Ji = self.heatload[-1]

			# Compute the equilibrium glide bank angle (sigma_EQ.GL)
			cosDeltaEQGL = ((self.mass * g0) / (self.CL * qi * self.A)) * (1.0 - vi ** 2.0 / (g0 * ri))

			# Compute the commanded bank angle (sigma CMD)
			cosDeltaCMD = cosDeltaEQGL - self.Ghdot * hdoti / qi + self.Gq * ((qi - qrefi) / qi)

			# if the |cosDeltaCMD| > 1.0 then set the bank angle  = 0.0 (full lift up)
			if cosDeltaCMD > 1.0:
				DeltaCMD = 0.0
			# else compute the commanded bank angle using arccos(cosDeltaCMD)
			elif cosDeltaCMD < -1.0:
				DeltaCMD = np.pi
			else:
				DeltaCMD = np.arccos(cosDeltaCMD)

			# Compute the commanded bank angle in degrees
			DeltaCMD_deg_command = DeltaCMD * 180.0 / np.pi

			DeltaCMD_deg, Delta_deg_ini = self.psuedoController(DeltaCMD_deg_command, Delta_deg_ini, timeStep)

			# propogate the vehicle state to 1 second in advance from
			# the current state using the commanded bank angle deltaCMD
			self.setInitialState(h0_km, theta0_deg, phi0_deg, v0_kms, psi0_deg, gamma0_deg, drange0_km, Ji)
			self.propogateEntry2(timeStep, dt, DeltaCMD_deg)

			t_min_c = self.t_minc
			h_km_c = self.h_kmc
			v_kms_c = self.v_kmsc
			phi_deg_c = self.phi_degc
			psi_deg_c = self.psi_degc
			theta_deg_c = self.theta_degc
			gamma_deg_c = self.gamma_degc
			drange_km_c = self.drange_kmc

			acc_net_g_c = self.acc_net_g
			dyn_pres_atm_c = self.dyn_pres_atm
			stag_pres_atm_c = self.stag_pres_atm
			q_stag_total_c = self.q_stag_total
			heatload_c = self.heatload
			acc_drag_g_c = self.acc_drag_g

			# Update the time solution array to account for non-zero start time
			t_min_c = t_min_c + t_min[-1]

			self.t_step_array = np.append(self.t_step_array, t_min[-1])
			self.delta_deg_array = np.append(self.delta_deg_array, DeltaCMD_deg)
			self.hdot_array = np.append(self.hdot_array, hdoti)
			self.qref_array = np.append(self.qref_array, qrefi)
			self.q_array = np.append(self.q_array, qi)
			self.h_step_array = np.append(self.h_step_array, h0_km)

			self.hdotref_array = np.zeros(len(self.t_step_array))
			self.hddoti = (self.hdot_array[-1] - self.hdot_array[-2]) / (self.t_step_array[-1] * 60.0 - self.t_step_array[-2] * 60.0)
			self.hddot_array = np.append(self.hddot_array, self.hddoti)
			self.acc_step_array = np.append(self.acc_step_array, self.acc_net_g[-1])
			self.acc_drag_array = np.append(self.acc_drag_array, self.acc_drag_g[-1])

			# Update time and other solution vectors
			t_min = np.concatenate((t_min, t_min_c), axis=0)
			h_km = np.concatenate((h_km, h_km_c), axis=0)
			v_kms = np.concatenate((v_kms, v_kms_c), axis=0)
			phi_deg = np.concatenate((phi_deg, phi_deg_c), axis=0)
			psi_deg = np.concatenate((psi_deg, psi_deg_c), axis=0)
			theta_deg = np.concatenate((theta_deg, theta_deg_c), axis=0)
			gamma_deg = np.concatenate((gamma_deg, gamma_deg_c), axis=0)
			drange_km = np.concatenate((drange_km, drange_km_c), axis=0)

			# Update entry parameter vectors
			acc_net_g = np.concatenate((acc_net_g, acc_net_g_c), axis=0)
			acc_drag_g = np.concatenate((acc_drag_g, acc_drag_g_c), axis=0)
			dyn_pres_atm = np.concatenate((dyn_pres_atm, dyn_pres_atm_c), axis=0)
			stag_pres_atm = np.concatenate((stag_pres_atm, stag_pres_atm_c), axis=0)
			q_stag_total = np.concatenate((q_stag_total, q_stag_total_c), axis=0)
			heatload = np.concatenate((heatload, heatload_c), axis=0)

			self.acc_step_array = np.append(self.acc_step_array, acc_net_g[-1])
			density_mes = 2 * self.mass * acc_drag_g[-1] * self.planetObj.EARTHG / (self.CD * self.A * (0.5 * (vi + v_kms[-1] * 1E3)) ** 2.0)
			self.density_mes_array = np.append(self.density_mes_array, density_mes)

			if hdoti > self.hdot_threshold and customFlag == 0:
				self.density_mes_int, self.minAlt = \
					self.createDensityMeasuredFunction(self.h_step_array, self.density_mes_array,
														self.lowAlt_km, self.numPoints_lowAlt)
				customFlag = 1.0

			if hdoti > self.hdot_threshold:
				terminal_apoapsis_km = \
					self.predictApoapsisAltitudeKm_withLiftUp(h_km[-1],
															  theta_deg[-1], phi_deg[-1], v_kms[-1], gamma_deg[-1],
															  psi_deg[-1], drange_km[-1], heatload[-1], maxTimeSecs, dt,
															  0.0, self.density_mes_int)

			else:
				terminal_apoapsis_km = 0.0

			# print("H (km): "+ str('{:.2f}'.format(h0_km))+" HDOT (m/s): \
			# "+ str('{:.2f}'.format(hdoti))+", DeltaCMD :"+str('{:.2f}'. \
			# format(DeltaCMD_deg_command))+", DELTAACT: "+str('{:.2f}'. \
			# format(DeltaCMD_deg))+", PRED. APO: "+str('{:.2f}'.\
			# format(terminal_apoapsis_km)))

			counter += 1

			# update current speed
			h_current_km = h_km[-1]
			v_current_kms = v_kms[-1]

			if terminal_apoapsis_km > 0 and terminal_apoapsis_km < self.target_apo_km:
				break

			if abs(terminal_apoapsis_km - self.target_apo_km) < self.target_apo_km_tol:
				break

			if h_current_km > self.planetObj.h_skip / 1000 - 2.0:
				break

		self.t_min_eg = t_min
		self.h_km_eg = h_km
		self.v_kms_eg = v_kms
		self.theta_deg_eg = theta_deg
		self.phi_deg_eg = phi_deg
		self.psi_deg_eg = psi_deg
		self.gamma_deg_eg = gamma_deg
		self.drange_km_eg = drange_km

		self.acc_net_g_eg = acc_net_g
		self.dyn_pres_atm_eg = dyn_pres_atm
		self.stag_pres_atm_eg = stag_pres_atm
		self.q_stag_total_eg = q_stag_total
		self.heatload_eg = heatload

	def propogateExitPhase(self, timeStep, dt, maxTimeSecs):
		"""
		Implements the exit phase of the guidance scheme (full lift-up).

		Parameters
		--------
		timeStep : float
			Guidance cycle time, seconds
		dt : float
			Solver max. time step, seconds
		maxTimeSecs : float
			max. time for propogation, seconds

		"""
		self.t_switch = self.t_min_eg[-1]
		self.h_switch = self.h_km_eg[-1]
		self.v_switch = self.v_kms_eg[-1]
		self.p_switch = self.delta_deg_array[-1]

		
		t_min = self.t_min_eg
		h_km = self.h_km_eg
		v_kms = self.v_kms_eg
		theta_deg = self.theta_deg_eg
		phi_deg = self.phi_deg_eg
		psi_deg = self.psi_deg_eg
		gamma_deg = self.gamma_deg_eg
		drange_km = self.drange_km_eg

		acc_net_g = self.acc_net_g_eg
		dyn_pres_atm = self.dyn_pres_atm_eg
		stag_pres_atm = self.stag_pres_atm_eg
		q_stag_total = self.q_stag_total_eg
		heatload = self.heatload_eg
	
		# Set the current altitude to the terminal altitude of
		# the equlibrium glide phase (km).
		h_current_km = h_km[-1]
		t_current_min = t_min[-1]
		
		# Set the skip altitude (km) based on definition in planet.py
		h_skip_km = self.planetObj.h_skip / 1.0E3

		# initialize hdot_ref as terminal hdot of equilibrium glide phase
		hdot_refi = self.hdot_array[-1]
		Ji = self.heatload_eg[-1]

		#print('Exit Phase Guidance Initiated')

		Delta_deg_ini = self.p_switch
		
		while h_current_km < h_skip_km:
			# print(minAlt)
			# Reset the initial conditions to the terminal conditions of 
			# the propgated solution from the previous step.
			# Terminal conditions of the equilibrium glide phase are initial 
			# conditions for the exit phase algorithm.
			# ----------------------------------------------------------------
			h0_km = h_km[-1]           # Entry altitude above planet surface
			theta0_deg = theta_deg[-1]      # Entry longitude
			phi0_deg = phi_deg[-1]        # Entry latitude
			v0_kms = v_kms[-1]          # Entry velocity
			psi0_deg = psi_deg[-1]        # Entry heading angle
			gamma0_deg = gamma_deg[-1]      # Entry flight path angle
			drange0_km = drange_km[-1]      # Downrange

			# ----------------------------------------------------------------------------
			# Convert entry state variables from IO/plot units to calculation (SI) units
			# ----------------------------------------------------------------------------
			h0 = h0_km*1.0E3             # Entry altitude above planet
			theta0 = theta0_deg*np.pi/180.0  # Entry longitude in radians
			phi0 = phi0_deg*np.pi/180.0    # Entry latitude in radians
			v0  = v0_kms*1.000E3          # Entry velocity in meters/se
			psi0 = psi0_deg*np.pi/180.0    # Entry heading angle in radians
			gamma0 = gamma0_deg*np.pi/180.0  # Entry flight path angle in radians
			# -------------------------------------------------------------------------------

			# --------------------------------------------------------------------------
			# Initialize iterable variables used in guidance loop, i = at the current time
			# ----------------------------------------------------------------------------
			hi = h0                           # Set the current altitude as the entry altitude
			ri = h0 + self.planetObj.RP       # Set the current radial distance 	
			vi = v0                           # Set the current speed (m/s) as the entry speed
			qi = 0.5*self.planetObj.density(hi)*vi**2.0      # Set the current dynamic pressure
			gammai = gamma0                   # Set the current FPA to entry FPA
			hdoti = vi*np.sin(gammai)         # Compute the current altitude rate hdot

			DeltaCMD_deg_command = 0.0
			DeltaCMD_deg, Delta_deg_ini = self.psuedoController(DeltaCMD_deg_command, Delta_deg_ini, timeStep)
			# DeltaCMD_deg = 0.0

			# propogate the vehicle state to advance from the current 
			# state using the commanded bank angle deltaCMD
			self.setInitialState(h0_km, theta0_deg, phi0_deg, v0_kms, psi0_deg, gamma0_deg, drange0_km, Ji)
			self.propogateEntry(timeStep, dt, DeltaCMD_deg)

			t_min_c = self.t_minc
			h_km_c = self.h_kmc
			v_kms_c = self.v_kmsc
			phi_deg_c = self.phi_degc
			psi_deg_c = self.psi_degc
			theta_deg_c = self.theta_degc
			gamma_deg_c = self.gamma_degc
			drange_km_c = self.drange_kmc

			acc_net_g_c = self.acc_net_g
			dyn_pres_atm_c = self.dyn_pres_atm
			stag_pres_atm_c = self.stag_pres_atm
			q_stag_total_c = self.q_stag_total
			heatload_c = self.heatload
			acc_drag_g_c = self.acc_drag_g

			# Update the time solution array to account for non-zero start time 
			t_min_c = t_min_c + t_min[-1]

			self.t_step_array = np.append(self.t_step_array, t_min[-1])
			self.delta_deg_array = np.append(self.delta_deg_array, DeltaCMD_deg)
			self.hdot_array = np.append(self.hdot_array, hdoti)
			self.hdotref_array = np.append(self.hdotref_array, hdot_refi)

			self.hddoti = (self.hdot_array[-1] - self.hdot_array[-2]) / \
									   (self.t_step_array[-1]*60.0 - self.t_step_array[-2]*60.0)

			self.hddot_array = np.append(self.hddot_array, self.hddoti)

			# Update time and other solution vectors
			t_min = np.concatenate((t_min, t_min_c), axis=0)
			h_km = np.concatenate((h_km , h_km_c ), axis=0)
			v_kms = np.concatenate((v_kms, v_kms_c ), axis=0)
			phi_deg = np.concatenate((phi_deg, phi_deg_c ), axis=0)
			psi_deg = np.concatenate((psi_deg, psi_deg_c ), axis=0)
			theta_deg = np.concatenate((theta_deg, theta_deg_c), axis=0)
			gamma_deg = np.concatenate((gamma_deg, gamma_deg_c), axis=0)
			drange_km = np.concatenate((drange_km, drange_km_c), axis=0)

			# Update entry parameter vectors
			acc_net_g = np.concatenate((acc_net_g,acc_net_g_c), axis=0)
			dyn_pres_atm = np.concatenate((dyn_pres_atm, dyn_pres_atm_c),axis=0)
			stag_pres_atm = np.concatenate((stag_pres_atm, stag_pres_atm_c), axis=0)
			q_stag_total = np.concatenate((q_stag_total, q_stag_total_c), axis=0)
			heatload = np.concatenate((heatload, heatload_c), axis=0)

			terminal_apoapsis_km = self.compute_ApoapsisAltitudeKm(
								   self.planetObj.RP+h_km[-1]*1E3, v_kms[-1]*1E3,
								   gamma_deg[-1]*np.pi/180.0, theta_deg[-1]*np.pi/180.0,
								   phi_deg[-1]*np.pi/180.0, psi_deg[-1]*np.pi/180.0)


			# print("H (km): "+ str('{:.2f}'.format(h0_km))+", HDOT: "+str('{:.2f}'.format(hdoti)) +", DeltaCMD :"+str('{:.2f}'.format(DeltaCMD_deg_command))+", DELTAACT: "+str('{:.2f}'.format(DeltaCMD_deg))+", PREDICT. APO. ALT. :"+str(terminal_apoapsis_km))

			if hi > self.planetObj.h_skip - 10.0E3:
				break

			h_current_km  = h_km[-1]
			t_current_min = t_min[-1]

		self.t_min_full = t_min
		self.h_km_full = h_km
		self.v_kms_full = v_kms
		self.theta_deg_full = theta_deg	
		self.phi_deg_full = phi_deg
		self.psi_deg_full = psi_deg
		self.gamma_deg_full = gamma_deg
		self.drange_km_full = drange_km
		
		self.acc_net_g_full = acc_net_g
		self.dyn_pres_atm_full = dyn_pres_atm
		self.stag_pres_atm_full = stag_pres_atm
		self.q_stag_total_full = q_stag_total
		self.heatload_full = heatload


		self.terminal_apoapsis  = self.compute_ApoapsisAltitudeKm(
								  self.planetObj.RP+h_km[-1]*1E3, v_kms[-1]*1E3,
								  gamma_deg[-1]*np.pi/180.0, theta_deg[-1]*np.pi/180.0,
								  phi_deg[-1]*np.pi/180.0, psi_deg[-1]*np.pi/180.0)
		self.terminal_periapsis = self.compute_PeriapsisAltitudeKm(
								  self.planetObj.RP+h_km[-1]*1E3, v_kms[-1]*1E3,
								  gamma_deg[-1]*np.pi/180.0, theta_deg[-1]*np.pi/180.0,
								  phi_deg[-1]*np.pi/180.0, psi_deg[-1]*np.pi/180.0)
		self.apoapsis_perc_error= (self.terminal_apoapsis - self.target_apo_km)*100.0 / self.target_apo_km

		self.periapsis_raise_DV = self.compute_periapsis_raise_DV(
								  self.terminal_periapsis, self.terminal_apoapsis, self.target_peri_km)
		self.apoapsis_raise_DV  = self.compute_apoapsis_raise_DV(
								  self.target_peri_km, self.terminal_apoapsis, self.target_apo_km)

		# print("Apoapis Altitude at Exit: "+str(self.terminal_apoapsis)+" km")
		# print("Periapsis Altitude at Exit: "+str(terminal_periapsis)+" km")

	def propogateExitPhase2(self, timeStep, dt, maxTimeSecs):
		"""
		Implements the exit phase of the guidance scheme (full lift-up).

		Parameters
		--------
		timeStep : float
			Guidance cycle time, seconds
		dt : float
			Solver max. time step, seconds
		maxTimeSecs : float
			max. time for propogation, seconds

		"""
		self.t_switch = self.t_min_eg[-1]
		self.h_switch = self.h_km_eg[-1]
		self.v_switch = self.v_kms_eg[-1]
		self.p_switch = self.delta_deg_array[-1]

		t_min = self.t_min_eg
		h_km = self.h_km_eg
		v_kms = self.v_kms_eg
		theta_deg = self.theta_deg_eg
		phi_deg = self.phi_deg_eg
		psi_deg = self.psi_deg_eg
		gamma_deg = self.gamma_deg_eg
		drange_km = self.drange_km_eg

		acc_net_g = self.acc_net_g_eg
		dyn_pres_atm = self.dyn_pres_atm_eg
		stag_pres_atm = self.stag_pres_atm_eg
		q_stag_total = self.q_stag_total_eg
		heatload = self.heatload_eg

		# Set the current altitude to the terminal altitude of
		# the equlibrium glide phase (km).
		h_current_km = h_km[-1]
		t_current_min = t_min[-1]

		# Set the skip altitude (km) based on definition in planet.py
		h_skip_km = self.planetObj.h_skip / 1.0E3

		# initialize hdot_ref as terminal hdot of equilibrium glide phase
		hdot_refi = self.hdot_array[-1]
		Ji = self.heatload_eg[-1]

		# print('Exit Phase Guidance Initiated')

		Delta_deg_ini = self.p_switch

		while h_current_km < h_skip_km:
			# print(minAlt)
			# Reset the initial conditions to the terminal conditions of
			# the propgated solution from the previous step.
			# Terminal conditions of the equilibrium glide phase are initial
			# conditions for the exit phase algorithm.
			# ----------------------------------------------------------------
			h0_km = h_km[-1]  # Entry altitude above planet surface
			theta0_deg = theta_deg[-1]  # Entry longitude
			phi0_deg = phi_deg[-1]  # Entry latitude
			v0_kms = v_kms[-1]  # Entry velocity
			psi0_deg = psi_deg[-1]  # Entry heading angle
			gamma0_deg = gamma_deg[-1]  # Entry flight path angle
			drange0_km = drange_km[-1]  # Downrange

			# ----------------------------------------------------------------------------
			# Convert entry state variables from IO/plot units to calculation (SI) units
			# ----------------------------------------------------------------------------
			h0 = h0_km * 1.0E3  # Entry altitude above planet
			theta0 = theta0_deg * np.pi / 180.0  # Entry longitude in radians
			phi0 = phi0_deg * np.pi / 180.0  # Entry latitude in radians
			v0 = v0_kms * 1.000E3  # Entry velocity in meters/se
			psi0 = psi0_deg * np.pi / 180.0  # Entry heading angle in radians
			gamma0 = gamma0_deg * np.pi / 180.0  # Entry flight path angle in radians
			# -------------------------------------------------------------------------------

			# --------------------------------------------------------------------------
			# Initialize iterable variables used in guidance loop, i = at the current time
			# ----------------------------------------------------------------------------
			hi = h0  # Set the current altitude as the entry altitude
			ri = h0 + self.planetObj.RP  # Set the current radial distance
			vi = v0  # Set the current speed (m/s) as the entry speed
			qi = 0.5 * self.planetObj.density(hi) * vi ** 2.0  # Set the current dynamic pressure
			gammai = gamma0  # Set the current FPA to entry FPA
			hdoti = vi * np.sin(gammai)  # Compute the current altitude rate hdot

			DeltaCMD_deg_command = 0.0
			DeltaCMD_deg, Delta_deg_ini = self.psuedoController(DeltaCMD_deg_command, \
																Delta_deg_ini, timeStep)
			# DeltaCMD_deg = 0.0

			# propogate the vehicle state to advance from the current
			# state using the commanded bank angle deltaCMD
			self.setInitialState(h0_km, theta0_deg, phi0_deg, v0_kms, psi0_deg, gamma0_deg, drange0_km, Ji)
			self.propogateEntry2(timeStep, dt, DeltaCMD_deg)

			t_min_c = self.t_minc
			h_km_c = self.h_kmc
			v_kms_c = self.v_kmsc
			phi_deg_c = self.phi_degc
			psi_deg_c = self.psi_degc
			theta_deg_c = self.theta_degc
			gamma_deg_c = self.gamma_degc
			drange_km_c = self.drange_kmc

			acc_net_g_c = self.acc_net_g
			dyn_pres_atm_c = self.dyn_pres_atm
			stag_pres_atm_c = self.stag_pres_atm
			q_stag_total_c = self.q_stag_total
			heatload_c = self.heatload
			acc_drag_g_c = self.acc_drag_g

			# Update the time solution array to account for non-zero start time
			t_min_c = t_min_c + t_min[-1]

			self.t_step_array = np.append(self.t_step_array, t_min[-1])
			self.delta_deg_array = np.append(self.delta_deg_array, DeltaCMD_deg)
			self.hdot_array = np.append(self.hdot_array, hdoti)
			self.hdotref_array = np.append(self.hdotref_array, hdot_refi)

			self.hddoti = (self.hdot_array[-1] - self.hdot_array[-2]) / \
						  (self.t_step_array[-1] * 60.0 - self.t_step_array[-2] * 60.0)

			self.hddot_array = np.append(self.hddot_array, self.hddoti)

			# Update time and other solution vectors
			t_min = np.concatenate((t_min, t_min_c), axis=0)
			h_km = np.concatenate((h_km, h_km_c), axis=0)
			v_kms = np.concatenate((v_kms, v_kms_c), axis=0)
			phi_deg = np.concatenate((phi_deg, phi_deg_c), axis=0)
			psi_deg = np.concatenate((psi_deg, psi_deg_c), axis=0)
			theta_deg = np.concatenate((theta_deg, theta_deg_c), axis=0)
			gamma_deg = np.concatenate((gamma_deg, gamma_deg_c), axis=0)
			drange_km = np.concatenate((drange_km, drange_km_c), axis=0)

			# Update entry parameter vectors
			acc_net_g = np.concatenate((acc_net_g, acc_net_g_c), axis=0)
			dyn_pres_atm = np.concatenate((dyn_pres_atm, dyn_pres_atm_c), axis=0)
			stag_pres_atm = np.concatenate((stag_pres_atm, stag_pres_atm_c), axis=0)
			q_stag_total = np.concatenate((q_stag_total, q_stag_total_c), axis=0)
			heatload = np.concatenate((heatload, heatload_c), axis=0)

			terminal_apoapsis_km = self.compute_ApoapsisAltitudeKm(
				self.planetObj.RP + h_km[-1] * 1E3, v_kms[-1] * 1E3,
				gamma_deg[-1] * np.pi / 180.0, theta_deg[-1] * np.pi / 180.0,
				phi_deg[-1] * np.pi / 180.0, psi_deg[-1] * np.pi / 180.0)

			# print("H (km): "+ str('{:.2f}'.format(h0_km))+", HDOT: "+str('{:.2f}'.format(hdoti)) +", DeltaCMD :"+str('{:.2f}'.format(DeltaCMD_deg_command))+", DELTAACT: "+str('{:.2f}'.format(DeltaCMD_deg))+", PREDICT. APO. ALT. :"+str(terminal_apoapsis_km))

			if hi > self.planetObj.h_skip - 10.0E3:
				break

			h_current_km = h_km[-1]
			t_current_min = t_min[-1]

		self.t_min_full = t_min
		self.h_km_full = h_km
		self.v_kms_full = v_kms
		self.theta_deg_full = theta_deg
		self.phi_deg_full = phi_deg
		self.psi_deg_full = psi_deg
		self.gamma_deg_full = gamma_deg
		self.drange_km_full = drange_km

		self.acc_net_g_full = acc_net_g
		self.dyn_pres_atm_full = dyn_pres_atm
		self.stag_pres_atm_full = stag_pres_atm
		self.q_stag_total_full = q_stag_total
		self.heatload_full = heatload

		self.terminal_apoapsis = self.compute_ApoapsisAltitudeKm(
			self.planetObj.RP + h_km[-1] * 1E3, v_kms[-1] * 1E3,
			gamma_deg[-1] * np.pi / 180.0, theta_deg[-1] * np.pi / 180.0,
			phi_deg[-1] * np.pi / 180.0, psi_deg[-1] * np.pi / 180.0)
		self.terminal_periapsis = self.compute_PeriapsisAltitudeKm(
			self.planetObj.RP + h_km[-1] * 1E3, v_kms[-1] * 1E3,
			gamma_deg[-1] * np.pi / 180.0, theta_deg[-1] * np.pi / 180.0,
			phi_deg[-1] * np.pi / 180.0, psi_deg[-1] * np.pi / 180.0)
		self.apoapsis_perc_error = (self.terminal_apoapsis - self.target_apo_km) * 100.0 / self.target_apo_km

		self.periapsis_raise_DV = self.compute_periapsis_raise_DV(
			self.terminal_periapsis, self.terminal_apoapsis,
			self.target_peri_km)
		self.apoapsis_raise_DV = self.compute_apoapsis_raise_DV(
			self.target_peri_km, self.terminal_apoapsis,
			self.target_apo_km)

		# print("Apoapis Altitude at Exit: "+str(self.terminal_apoapsis)+" km")
		# print("Periapsis Altitude at Exit: "+str(self.terminal_periapsis)+" km")


	def propogateGuidedEntry(self, timeStep, dt, maxTimeSecs):
		"""
		Implements the full guidance scheme (eq. glide + exit phase)

		Parameters
		--------
		timeStep : float
			Guidance cycle time, seconds
		dt : float
			Solver max. time step, seconds
		maxTimeSecs : float
			max. time for propogation, seconds

		"""
		self.propogateEquilibriumGlide(timeStep, dt, maxTimeSecs)
		self.propogateExitPhase(timeStep, dt, maxTimeSecs)

	def propogateGuidedEntry2(self, timeStep, dt, maxTimeSecs):
		"""
		Implements the full guidance scheme (eq. glide + exit phase)

		Parameters
		--------
		timeStep : float
			Guidance cycle time, seconds
		dt : float
			Solver max. time step, seconds
		maxTimeSecs : float
			max. time for propogation, seconds

		"""
		self.propogateEquilibriumGlide2(timeStep, dt, maxTimeSecs)
		self.propogateExitPhase2(timeStep, dt, maxTimeSecs)


	def setupMonteCarloSimulation(self, NPOS, NMONTE, atmfiles, heightCol, densLowCol, densAvgCol,
								densHighCol, densTotalCol, heightInKmFlag, nominalEFPA,  EFPA_1sigma_value,
								nominalLD, LD_1sigma_value, timeStep, dt, maxTimeSecs, atmSigmaFactor=1):
		"""
		Set the Monte Carlo simulation parameters.

		Parameters
		--------
		NPOS : int
			NPOS value from GRAM model output 
			is the number of data points (altitude) in each atm. profile
		NMONTE : int
			NMONTE is the number of Monte Carlo atm profiles
			from GRAM model output
		atmfiles : str
			location of atmospheric files used in Monte Carlo simulation
		heightCol : int
			column index of height values in atmfiles
		densLowCol : int
			column index of low density (-1 sigma) values in atmfiles
		densAvgCol : int
			column index of average density values in atmfiles
		densHighCol : int
			column index of high density values (+1 sigma) in atmfiles
		densTotalCol : int
			index of perturbed (=avg + pert.) density values
		heightInKmFlag : bool
			set to True if height values in atmfiles are in km
		nominalEFPA : float
			Nominal (target EFPA) value, deg
		EFPA_1sigma_value : float
			1-sigma error for EFPA (from naviation analysis)
		nominalLD : float
			Nominal value of vehicle L/D
		LD_1sigma_value : float
			1-sigma error for L/D (from vehicle aero. design data)
		timeStep : float
			Guidance cycle time step, sec
		dt : float
			max. solver time step
		maxTimeSecs : float
			max. time used for propogation used by guidance scheme

		"""

		self.NPOS = NPOS
		self.NMONTE = NMONTE
		self.atmfiles = atmfiles

		self.heightCol = heightCol
		self.densLowCol = densLowCol
		self.densAvgCol = densAvgCol
		self.densHighCol = densHighCol
		self.densTotalCol = densTotalCol

		self.heightInKmFlag = heightInKmFlag

		self.nominalEFPA = nominalEFPA
		self.EFPA_1sigma_value = EFPA_1sigma_value

		self.nominalLD = nominalLD
		self.LD_1sigma_value = LD_1sigma_value
		self.vehicleCopy = copy.deepcopy(self)

		self.timeStep = timeStep
		self.dt = dt
		self.maxTimeSecs = maxTimeSecs

		self.atmSigmaFactor = atmSigmaFactor


	def runMonteCarlo(self, N, mainFolder):
		"""
		Run a Monte Carlo simulation for lift modulation
		aerocapture.

		Parameters
		--------
		N : int
			Number of trajectories
		mainFolder : str
			path where data is to be stored
		"""

		terminal_apoapsis_arr = np.zeros(N)
		terminal_periapsis_arr = np.zeros(N)
		periapsis_raise_DV_arr = np.zeros(N)
		apoapsis_raise_DV_arr = np.zeros(N)
		acc_net_g_max_arr = np.zeros(N)
		q_stag_max_arr = np.zeros(N)
		heatload_max_arr = np.zeros(N)

		h0_km = self.vehicleCopy.h0_km_ref
		theta0_deg = self.vehicleCopy.theta0_deg_ref
		phi0_deg = self.vehicleCopy.phi0_deg_ref
		v0_kms = self.vehicleCopy.v0_kms_ref
		psi0_deg = self.vehicleCopy.psi0_deg_ref
		drange0_km = self.vehicleCopy.drange0_km_ref
		heatLoad0 = self.vehicleCopy.heatLoad0_ref

		os.makedirs(mainFolder)

		for i in range(N):
			selected_atmfile  = rd.choice(self.atmfiles)
			selected_profile = rd.randint(1, self.NMONTE)
			selected_efpa = np.random.normal(self.nominalEFPA, self.EFPA_1sigma_value)
			selected_atmSigma = np.random.normal(0, self.atmSigmaFactor)
			selected_LD = np.random.normal(self.nominalLD, self.LD_1sigma_value)

			ATM_height, ATM_density_low, ATM_density_avg, ATM_density_high, \
			ATM_density_pert = self.planetObj.loadMonteCarloDensityFile2(selected_atmfile, self.heightCol,
										self.densLowCol,  self.densAvgCol, self.densHighCol, self.densTotalCol,
										self.heightInKmFlag)
			self.planetObj.density_int = self.planetObj.loadAtmosphereModel5(ATM_height,
										 ATM_density_low, ATM_density_avg, ATM_density_high,
										 ATM_density_pert, selected_atmSigma, self.NPOS,
										 selected_profile)

			
			self.setInitialState(h0_km, theta0_deg, phi0_deg, v0_kms, psi0_deg, selected_efpa, drange0_km, heatLoad0)
			self.propogateGuidedEntry(self.timeStep, self.dt, self.maxTimeSecs)

			terminal_apoapsis = self.terminal_apoapsis
			apoapsis_error = self.apoapsis_perc_error
			terminal_periapsis = self.terminal_periapsis

			periapsis_raise_DV = self.periapsis_raise_DV
			apoapsis_raise_DV = self.apoapsis_raise_DV


			terminal_apoapsis_arr[i] = self.terminal_apoapsis
			terminal_periapsis_arr[i] = self.terminal_periapsis

			periapsis_raise_DV_arr[i] = self.periapsis_raise_DV
			apoapsis_raise_DV_arr[i] = self.apoapsis_raise_DV

			acc_net_g_max_arr[i] = max(self.acc_net_g_full)
			q_stag_max_arr[i] = max(self.q_stag_total_full)
			heatload_max_arr[i] = max(self.heatload_full)

			print("BATCH :"+str(mainFolder)+", RUN #: "+str(i+1)+", PROF: "+str(selected_atmfile) +
					", SAMPLE #: "+str(selected_profile)+", EFPA: "+str('{:.2f}'.format(selected_efpa,2)) +
					", SIGMA: "+str('{:.2f}'.format(selected_atmSigma, 2))+", LD: "+str('{:.2f}'.format(selected_LD, 2))+
					", APO : "+str('{:.2f}'.format(terminal_apoapsis, 2)))

			os.makedirs(mainFolder+'/'+'#'+str(i+1))

			np.savetxt(mainFolder+'/'+'#'+str(i+1)+'/'+'atmfile.txt', np.array([selected_atmfile]), fmt='%s')
			np.savetxt(mainFolder+'/'+'#'+str(i+1)+'/'+'profile.txt', np.array([selected_profile]))
			np.savetxt(mainFolder+'/'+'#'+str(i+1)+'/'+'efpa.txt', np.array([selected_efpa]))
			np.savetxt(mainFolder+'/'+'#'+str(i+1)+'/'+'atmSigma.txt', np.array([selected_atmSigma]))
			np.savetxt(mainFolder+'/'+'#'+str(i+1)+'/'+'LD.txt', np.array([selected_LD]))

			np.savetxt(mainFolder+'/'+'#'+str(i+1)+'/'+'terminal_apoapsis.txt', np.array([terminal_apoapsis]))
			np.savetxt(mainFolder+'/'+'#'+str(i+1)+'/'+'apoapsis_error.txt', np.array([apoapsis_error]))
			np.savetxt(mainFolder+'/'+'#'+str(i+1)+'/'+'terminal_periapsis.txt', np.array([terminal_periapsis]))

			np.savetxt(mainFolder+'/'+'#'+str(i+1)+'/'+'periapsis_raise_DV.txt', np.array([periapsis_raise_DV]))
			np.savetxt(mainFolder+'/'+'#'+str(i+1)+'/'+'apoapsis_raise_DV.txt', np.array([apoapsis_raise_DV]))


			np.savetxt(mainFolder+'/'+'terminal_apoapsis_arr.txt', terminal_apoapsis_arr)
			np.savetxt(mainFolder+'/'+'terminal_periapsis_arr.txt', terminal_periapsis_arr)

			np.savetxt(mainFolder+'/'+'periapsis_raise_DV_arr.txt', periapsis_raise_DV_arr)
			np.savetxt(mainFolder+'/'+'apoapsis_raise_DV_arr.txt', apoapsis_raise_DV_arr)

			np.savetxt(mainFolder+'/'+'acc_net_g_max_arr.txt', acc_net_g_max_arr)
			np.savetxt(mainFolder+'/'+'q_stag_max_arr.txt', q_stag_max_arr)
			np.savetxt(mainFolder+'/'+'heatload_max_arr.txt', heatload_max_arr)

	def runMonteCarlo2(self, N, mainFolder):
		"""
		Run a Monte Carlo simulation for lift modulation
		aerocapture.

		Parameters
		--------
		N : int
			Number of trajectories
		mainFolder : str
			path where data is to be stored
		"""

		terminal_apoapsis_arr = np.zeros(N)
		terminal_periapsis_arr = np.zeros(N)
		periapsis_raise_DV_arr = np.zeros(N)
		apoapsis_raise_DV_arr = np.zeros(N)
		acc_net_g_max_arr = np.zeros(N)
		q_stag_max_arr = np.zeros(N)
		heatload_max_arr = np.zeros(N)

		h0_km = self.vehicleCopy.h0_km_ref
		theta0_deg = self.vehicleCopy.theta0_deg_ref
		phi0_deg = self.vehicleCopy.phi0_deg_ref
		v0_kms = self.vehicleCopy.v0_kms_ref
		psi0_deg = self.vehicleCopy.psi0_deg_ref
		drange0_km = self.vehicleCopy.drange0_km_ref
		heatLoad0 = self.vehicleCopy.heatLoad0_ref

		os.makedirs(mainFolder)

		for i in range(N):
			selected_atmfile = rd.choice(self.atmfiles)

			# selected_profile   = 65
			selected_profile = rd.randint(1, self.NMONTE)
			# selected_efpa     = -11.53
			selected_efpa = np.random.normal(self.nominalEFPA, self.EFPA_1sigma_value)
			# selected_atmSigma = +0.0
			selected_atmSigma = np.random.normal(0, self.atmSigmaFactor)
			selected_LD = np.random.normal(self.nominalLD, self.LD_1sigma_value)

			ATM_height, ATM_density_low, ATM_density_avg, ATM_density_high, \
			ATM_density_pert = self.planetObj.loadMonteCarloDensityFile2( \
				selected_atmfile, self.heightCol, \
				self.densLowCol, self.densAvgCol,
				self.densHighCol, self.densTotalCol, self.heightInKmFlag)
			self.planetObj.density_int = self.planetObj.loadAtmosphereModel5(ATM_height, \
																			 ATM_density_low, ATM_density_avg,
																			 ATM_density_high, \
																			 ATM_density_pert, selected_atmSigma,
																			 self.NPOS, \
																			 selected_profile)

			self.setInitialState(h0_km, theta0_deg, phi0_deg, v0_kms, \
								 psi0_deg, selected_efpa, drange0_km, heatLoad0)
			self.propogateGuidedEntry2(self.timeStep, self.dt, self.maxTimeSecs)

			terminal_apoapsis = self.terminal_apoapsis
			apoapsis_error = self.apoapsis_perc_error
			terminal_periapsis = self.terminal_periapsis

			periapsis_raise_DV = self.periapsis_raise_DV
			apoapsis_raise_DV = self.apoapsis_raise_DV

			terminal_apoapsis_arr[i] = self.terminal_apoapsis
			terminal_periapsis_arr[i] = self.terminal_periapsis

			periapsis_raise_DV_arr[i] = self.periapsis_raise_DV
			apoapsis_raise_DV_arr[i] = self.apoapsis_raise_DV

			acc_net_g_max_arr[i] = max(self.acc_net_g_full)
			q_stag_max_arr[i] = max(self.q_stag_total_full)
			heatload_max_arr[i] = max(cumtrapz(self.q_stag_total_full, self.t_min_full*60, initial=0))/1e3

			print("RUN #: " + str(i + 1) + ", SAMPLE #: " + str(selected_profile) + ", EFPA: " + str(
				'{:.2f}'.format(selected_efpa, 2)) + ", SIGMA: " + str(
				'{:.2f}'.format(selected_atmSigma, 2)) + ", LD: " + str(
				'{:.2f}'.format(selected_LD, 2)) + ", APO : " + str('{:.2f}'.format(terminal_apoapsis, 2)))

			os.makedirs(mainFolder + '/' + '#' + str(i + 1))

			np.savetxt(mainFolder + '/' + '#' + str(i + 1) + '/' + 'atmfile.txt', np.array([selected_atmfile]),
					   fmt='%s')
			np.savetxt(mainFolder + '/' + '#' + str(i + 1) + '/' + 'profile.txt', np.array([selected_profile]))
			np.savetxt(mainFolder + '/' + '#' + str(i + 1) + '/' + 'efpa.txt', np.array([selected_efpa]))
			np.savetxt(mainFolder + '/' + '#' + str(i + 1) + '/' + 'atmSigma.txt', np.array([selected_atmSigma]))
			np.savetxt(mainFolder + '/' + '#' + str(i + 1) + '/' + 'LD.txt', np.array([selected_LD]))

			np.savetxt(mainFolder + '/' + '#' + str(i + 1) + '/' + 'terminal_apoapsis.txt',
					   np.array([terminal_apoapsis]))
			np.savetxt(mainFolder + '/' + '#' + str(i + 1) + '/' + 'apoapsis_error.txt', np.array([apoapsis_error]))
			np.savetxt(mainFolder + '/' + '#' + str(i + 1) + '/' + 'terminal_periapsis.txt',
					   np.array([terminal_periapsis]))

			np.savetxt(mainFolder + '/' + '#' + str(i + 1) + '/' + 'periapsis_raise_DV.txt',
					   np.array([periapsis_raise_DV]))
			np.savetxt(mainFolder + '/' + '#' + str(i + 1) + '/' + 'apoapsis_raise_DV.txt',
					   np.array([apoapsis_raise_DV]))

			np.savetxt(mainFolder + '/' + 'terminal_apoapsis_arr.txt', terminal_apoapsis_arr)
			np.savetxt(mainFolder + '/' + 'terminal_periapsis_arr.txt', terminal_periapsis_arr)

			np.savetxt(mainFolder + '/' + 'periapsis_raise_DV_arr.txt', periapsis_raise_DV_arr)
			np.savetxt(mainFolder + '/' + 'apoapsis_raise_DV_arr.txt', apoapsis_raise_DV_arr)

			np.savetxt(mainFolder + '/' + 'acc_net_g_max_arr.txt', acc_net_g_max_arr)
			np.savetxt(mainFolder + '/' + 'q_stag_max_arr.txt', q_stag_max_arr)
			np.savetxt(mainFolder + '/' + 'heatload_max_arr.txt', heatload_max_arr)



	def setDragEntryPhaseParams(self, v_switch_kms,\
								lowAlt_km, numPoints_lowAlt, hdot_threshold):
		"""
		Set entry phase guidance parameters for drag modulation

		Parameters
		-----------
		v_switch_kms : float
			speed below which entry phase is terminated
		lowAlt_km : float
			lower altitude to which density model is to be extrapolated
			based on available measurements, km
		numPoints_lowAlt : int
			number of points to evaluate extrapolation at below the 
			altitude where measurements are available
		hdot_threshold : float
			threshold altitude rate (m/s) above which density measurement
			is terminated and apoapsis prediction is initiated

		"""
		self.v_switch_kms = v_switch_kms
		self.lowAlt_km = lowAlt_km
		self.numPoints_lowAlt = numPoints_lowAlt
		self.hdot_threshold = hdot_threshold



	def propogateEntryPhaseD(self, timeStep, dt, maxTimeSecs):
		"""
		Implements the entry phase of the guided drag modulation
		aerocapture (Single-event discrete drag modulation).

		The entry phase is defined from the atmospheric entry interface
		till drag skirt jettison.

		Parameters
		--------
		timeStep : float
			Guidance cycle time, seconds
		dt : float
			Solver max. time step, seconds
		maxTimeSecs : float
			max. time for propogation, seconds

		"""

		# Set the ballistic coeff = beta1 for entry phase
		# Re-calculate CD based on new beta value
		self.beta    = self.beta1
		self.CD      = self.mass / (self.beta*self.A)
		
		# Initialize solution arrays
		self.t_step_array      = np.array([0.0])
		self.delta_deg_array   = np.array([0.0])
		self.hdot_array        = np.array([0.0])
		self.hddot_array       = np.array([0.0])
		self.qref_array        = np.array([0.0])
		self.q_array           = np.array([0.0])
		self.h_step_array      = np.array([0.0])
		self.acc_step_array    = np.array([0.0])
		self.acc_drag_array    = np.array([0.0])
		self.density_mes_array = np.array([0.0])

		# Propogate for 1 second from EI 
		self.propogateEntry(1.0, dt, 0.0)

		# Store solution at end of 1 sec 
		t_min     = self.t_minc
		h_km      = self.h_kmc
		v_kms     = self.v_kmsc
		phi_deg   = self.phi_degc
		psi_deg   = self.psi_degc
		theta_deg = self.theta_degc
		gamma_deg = self.gamma_degc
		drange_km = self.drange_kmc

		acc_net_g     =  self.acc_net_g
		dyn_pres_atm  =  self.dyn_pres_atm 
		stag_pres_atm =  self.stag_pres_atm
		q_stag_total  =  self.q_stag_total
		heatload      =  self.heatload
		acc_drag_g    =  self.acc_drag_g

		# Set the current vehicle speed here.
		self.h_current_km  = h_km[-1]
		self.v_current_kms = v_kms[-1]

		customFlag = 0.0

		# if the currrent speed is greater than the exit phase 
		# switch speed continue iterating in the entry 
		# phase mode
		while self.v_current_kms > self.v_switch_kms:
			# Reset the initial conditions to the terminal conditions 
			# of the propgated solution from the previous step.
			
			h0_km       = h_km[-1]           # Entry altitude above planet surface 
			theta0_deg  = theta_deg[-1]      # Entry longitude 
			phi0_deg    = phi_deg[-1]        # Entry latitude 
			v0_kms      = v_kms[-1]          # Entry velocity 
			psi0_deg    = psi_deg[-1]        # Entry heading angle 
			gamma0_deg  = gamma_deg[-1]      # Entry flight path angle 
			drange0_km  = drange_km[-1]      # Downrange

			h0      = h0_km*1.0E3             # Entry altitude above planet surface in meters
			theta0  = theta0_deg*np.pi/180.0  # Entry longitude in radians
			phi0    = phi0_deg*np.pi/180.0    # Entry latitude in radians
			v0      = v0_kms*1.000E3          # Entry velocity in meters/sec, relative to planet
			psi0    = psi0_deg*np.pi/180.0    # Entry heading angle in radians
			gamma0  = gamma0_deg*np.pi/180.0  # Entry flight path angle in radians
			# ------------------------------------------------------------------------------------------------

			# ------------------------------------------------------------------------------------------------
			# Initialize iterable variables used in guidance loop, i = at the current time
			# ------------------------------------------------------------------------------------------------
			hi = h0                           # Set the current altitude as the entry altitude
			ri = h0 + self.planetObj.RP       # Set the current radial distance 
			vi = v0                           # Set the current speed (m/s) as the entry speed
			qi = 0.5*self.planetObj.density(hi)*vi**2.0      # Set the current dynamic pressure
			gammai = gamma0                   # Set the current FPA to entry FPA
			hdoti = vi*np.sin(gammai)         # Compute the current altitude rate hdot

			# Set the current heatload Ji = terminal heatload 
			# of the previous propogated solution
			Ji = self.heatload[-1]

			DeltaCMD_deg = 0.0

			# propogate the vehicle state to the next time step. 
			self.setInitialState(h0_km,theta0_deg,phi0_deg,v0_kms,psi0_deg,gamma0_deg,\
								drange0_km,Ji)
			self.propogateEntry(timeStep, dt, DeltaCMD_deg)

			t_min_c     = self.t_minc
			h_km_c      = self.h_kmc
			v_kms_c     = self.v_kmsc
			phi_deg_c   = self.phi_degc
			psi_deg_c   = self.psi_degc
			theta_deg_c = self.theta_degc
			gamma_deg_c = self.gamma_degc
			drange_km_c = self.drange_kmc

			acc_net_g_c     =  self.acc_net_g
			dyn_pres_atm_c  =  self.dyn_pres_atm 
			stag_pres_atm_c =  self.stag_pres_atm
			q_stag_total_c  =  self.q_stag_total
			heatload_c      =  self.heatload
			acc_drag_g_c    =  self.acc_drag_g

			# Update the time solution array to account for non-zero start time 
			t_min_c                 = t_min_c + t_min[-1]

			self.t_step_array             = np.append(self.t_step_array, t_min[-1])
			self.delta_deg_array          = np.append(self.delta_deg_array, DeltaCMD_deg)
			self.hdot_array               = np.append(self.hdot_array, hdoti)
			
			self.q_array                  = np.append(self.q_array, qi)
			self.h_step_array             = np.append(self.h_step_array, h0_km)

			self.hdotref_array            = np.zeros(len(self.t_step_array))
			self.hddoti                   = (self.hdot_array[-1] - self.hdot_array[-2]) / \
											(self.t_step_array[-1]*60.0 - \
											 self.t_step_array[-2]*60.0)
			self.hddot_array              = np.append(self.hddot_array, self.hddoti)
			self.acc_step_array           = np.append(self.acc_step_array, self.acc_net_g[-1])
			self.acc_drag_array           = np.append(self.acc_drag_array, self.acc_drag_g[-1])

			# Update time and other solution vectors
			t_min        = np.concatenate((t_min, t_min_c), axis=0)
			h_km         = np.concatenate((h_km , h_km_c ), axis=0)
			v_kms        = np.concatenate((v_kms, v_kms_c ), axis=0)
			phi_deg      = np.concatenate((phi_deg, phi_deg_c ), axis=0)
			psi_deg      = np.concatenate((psi_deg, psi_deg_c ), axis=0)
			theta_deg    = np.concatenate((theta_deg, theta_deg_c), axis=0)
			gamma_deg    = np.concatenate((gamma_deg, gamma_deg_c), axis=0)
			drange_km    = np.concatenate((drange_km, drange_km_c), axis=0)

			# Update entry parameter vectors
			acc_net_g    = np.concatenate((acc_net_g,acc_net_g_c), axis=0)
			acc_drag_g   = np.concatenate((acc_drag_g,acc_drag_g_c), axis=0)
			dyn_pres_atm = np.concatenate((dyn_pres_atm, dyn_pres_atm_c),axis=0)
			stag_pres_atm= np.concatenate((stag_pres_atm, stag_pres_atm_c), axis=0)
			q_stag_total = np.concatenate((q_stag_total, q_stag_total_c), axis=0)
			heatload     = np.concatenate((heatload, heatload_c), axis=0)

			self.acc_step_array           = np.append(self.acc_step_array, acc_net_g[-1])
			density_mes                   = 2*self.mass*acc_drag_g[-1]*\
											self.planetObj.EARTHG / \
											(self.CD*self.A*(0.5*(vi+v_kms[-1]*1E3))**2.0)
			self.density_mes_array        = np.append(self.density_mes_array, density_mes)

			if hdoti > self.hdot_threshold and customFlag == 0:
				self.density_mes_int, self.minAlt = \
				self.createDensityMeasuredFunction(self.h_step_array, \
					self.density_mes_array, self.lowAlt_km, self.numPoints_lowAlt)
				customFlag = 1.0
			
			if hdoti > self.hdot_threshold:
				terminal_apoapsis_km    = \
				self.predictApoapsisAltitudeKm_afterJettision(h_km[-1], \
					theta_deg[-1], phi_deg[-1], v_kms[-1], gamma_deg[-1], \
					psi_deg[-1], drange_km[-1], heatload[-1], maxTimeSecs, dt, \
					0.0 , self.density_mes_int)
				
			else:
				terminal_apoapsis_km     = 0.0

			#print("H (km): "+ str('{:.2f}'.format(h0_km))+" HDOT (m/s): "+ str('{:.2f}'.format(hdoti))+"  PRED. APO: "+str('{:.2f}'.format(terminal_apoapsis_km)))

			# update current speed
			h_current_km  = h_km[-1]
			v_current_kms = v_kms[-1]

			if terminal_apoapsis_km> 0 and terminal_apoapsis_km < self.target_apo_km:
				break

			if abs(terminal_apoapsis_km - self.target_apo_km) < self.target_apo_km_tol:
				break

			if h_current_km > self.planetObj.h_skip/1000 - 1.0:
				break

		
		self.t_min_en     = t_min
		self.h_km_en      = h_km
		self.v_kms_en     = v_kms
		self.theta_deg_en = theta_deg	
		self.phi_deg_en   = phi_deg
		self.psi_deg_en   = psi_deg
		self.gamma_deg_en = gamma_deg
		self.drange_km_en = drange_km
		
		self.acc_net_g_en    = acc_net_g
		self.dyn_pres_atm_en = dyn_pres_atm
		self.stag_pres_atm_en= stag_pres_atm
		self.q_stag_total_en = q_stag_total
		self.heatload_en     = heatload

	def propogateEntryPhaseD2(self, timeStep, dt, maxTimeSecs):
		"""
		Implements the entry phase of the guided drag modulation
		aerocapture (Single-event discrete drag modulation).

		The entry phase is defined from the atmospheric entry interface
		till drag skirt jettison.

		Parameters
		--------
		timeStep : float
			Guidance cycle time, seconds
		dt : float
			Solver max. time step, seconds
		maxTimeSecs : float
			max. time for propogation, seconds

		"""

		# Set the ballistic coeff = beta1 for entry phase
		# Re-calculate CD based on new beta value
		self.beta = self.beta1
		self.CD = self.mass / (self.beta * self.A)

		# Initialize solution arrays
		self.t_step_array = np.array([0.0])
		self.delta_deg_array = np.array([0.0])
		self.hdot_array = np.array([0.0])
		self.hddot_array = np.array([0.0])
		self.qref_array = np.array([0.0])
		self.q_array = np.array([0.0])
		self.h_step_array = np.array([0.0])
		self.acc_step_array = np.array([0.0])
		self.acc_drag_array = np.array([0.0])
		self.density_mes_array = np.array([0.0])

		# Propogate for 1 second from EI
		self.propogateEntry2(1.0, dt, 0.0)

		# Store solution at end of 1 sec
		t_min = self.t_minc
		h_km = self.h_kmc
		v_kms = self.v_kmsc
		phi_deg = self.phi_degc
		psi_deg = self.psi_degc
		theta_deg = self.theta_degc
		gamma_deg = self.gamma_degc
		drange_km = self.drange_kmc

		acc_net_g = self.acc_net_g
		dyn_pres_atm = self.dyn_pres_atm
		stag_pres_atm = self.stag_pres_atm
		q_stag_total = self.q_stag_total
		heatload = self.heatload
		acc_drag_g = self.acc_drag_g

		# Set the current vehicle speed here.
		self.h_current_km = h_km[-1]
		self.v_current_kms = v_kms[-1]

		customFlag = 0.0

		# if the currrent speed is greater than the exit phase
		# switch speed continue iterating in the entry
		# phase mode
		while self.v_current_kms > self.v_switch_kms:
			# Reset the initial conditions to the terminal conditions
			# of the propgated solution from the previous step.

			h0_km = h_km[-1]  # Entry altitude above planet surface
			theta0_deg = theta_deg[-1]  # Entry longitude
			phi0_deg = phi_deg[-1]  # Entry latitude
			v0_kms = v_kms[-1]  # Entry velocity
			psi0_deg = psi_deg[-1]  # Entry heading angle
			gamma0_deg = gamma_deg[-1]  # Entry flight path angle
			drange0_km = drange_km[-1]  # Downrange

			h0 = h0_km * 1.0E3  # Entry altitude above planet surface in meters
			theta0 = theta0_deg * np.pi / 180.0  # Entry longitude in radians
			phi0 = phi0_deg * np.pi / 180.0  # Entry latitude in radians
			v0 = v0_kms * 1.000E3  # Entry velocity in meters/sec, relative to planet
			psi0 = psi0_deg * np.pi / 180.0  # Entry heading angle in radians
			gamma0 = gamma0_deg * np.pi / 180.0  # Entry flight path angle in radians
			# ------------------------------------------------------------------------------------------------

			# ------------------------------------------------------------------------------------------------
			# Initialize iterable variables used in guidance loop, i = at the current time
			# ------------------------------------------------------------------------------------------------
			hi = h0  # Set the current altitude as the entry altitude
			ri = h0 + self.planetObj.RP  # Set the current radial distance
			vi = v0  # Set the current speed (m/s) as the entry speed
			qi = 0.5 * self.planetObj.density(hi) * vi ** 2.0  # Set the current dynamic pressure
			gammai = gamma0  # Set the current FPA to entry FPA
			hdoti = vi * np.sin(gammai)  # Compute the current altitude rate hdot

			# Set the current heatload Ji = terminal heatload
			# of the previous propogated solution
			Ji = self.heatload[-1]

			DeltaCMD_deg = 0.0

			# propogate the vehicle state to the next time step.
			self.setInitialState(h0_km, theta0_deg, phi0_deg, v0_kms, psi0_deg, gamma0_deg, \
								 drange0_km, Ji)
			self.propogateEntry2(timeStep, dt, DeltaCMD_deg)

			t_min_c = self.t_minc
			h_km_c = self.h_kmc
			v_kms_c = self.v_kmsc
			phi_deg_c = self.phi_degc
			psi_deg_c = self.psi_degc
			theta_deg_c = self.theta_degc
			gamma_deg_c = self.gamma_degc
			drange_km_c = self.drange_kmc

			acc_net_g_c = self.acc_net_g
			dyn_pres_atm_c = self.dyn_pres_atm
			stag_pres_atm_c = self.stag_pres_atm
			q_stag_total_c = self.q_stag_total
			heatload_c = self.heatload
			acc_drag_g_c = self.acc_drag_g

			# Update the time solution array to account for non-zero start time
			t_min_c = t_min_c + t_min[-1]

			self.t_step_array = np.append(self.t_step_array, t_min[-1])
			self.delta_deg_array = np.append(self.delta_deg_array, DeltaCMD_deg)
			self.hdot_array = np.append(self.hdot_array, hdoti)

			self.q_array = np.append(self.q_array, qi)
			self.h_step_array = np.append(self.h_step_array, h0_km)

			self.hdotref_array = np.zeros(len(self.t_step_array))
			self.hddoti = (self.hdot_array[-1] - self.hdot_array[-2]) / \
						  (self.t_step_array[-1] * 60.0 - \
						   self.t_step_array[-2] * 60.0)
			self.hddot_array = np.append(self.hddot_array, self.hddoti)
			self.acc_step_array = np.append(self.acc_step_array, self.acc_net_g[-1])
			self.acc_drag_array = np.append(self.acc_drag_array, self.acc_drag_g[-1])

			# Update time and other solution vectors
			t_min = np.concatenate((t_min, t_min_c), axis=0)
			h_km = np.concatenate((h_km, h_km_c), axis=0)
			v_kms = np.concatenate((v_kms, v_kms_c), axis=0)
			phi_deg = np.concatenate((phi_deg, phi_deg_c), axis=0)
			psi_deg = np.concatenate((psi_deg, psi_deg_c), axis=0)
			theta_deg = np.concatenate((theta_deg, theta_deg_c), axis=0)
			gamma_deg = np.concatenate((gamma_deg, gamma_deg_c), axis=0)
			drange_km = np.concatenate((drange_km, drange_km_c), axis=0)

			# Update entry parameter vectors
			acc_net_g = np.concatenate((acc_net_g, acc_net_g_c), axis=0)
			acc_drag_g = np.concatenate((acc_drag_g, acc_drag_g_c), axis=0)
			dyn_pres_atm = np.concatenate((dyn_pres_atm, dyn_pres_atm_c), axis=0)
			stag_pres_atm = np.concatenate((stag_pres_atm, stag_pres_atm_c), axis=0)
			q_stag_total = np.concatenate((q_stag_total, q_stag_total_c), axis=0)
			heatload = np.concatenate((heatload, heatload_c), axis=0)

			self.acc_step_array = np.append(self.acc_step_array, acc_net_g[-1])
			density_mes = 2 * self.mass * acc_drag_g[-1] * \
						  self.planetObj.EARTHG / \
						  (self.CD * self.A * (0.5 * (vi + v_kms[-1] * 1E3)) ** 2.0)
			self.density_mes_array = np.append(self.density_mes_array, density_mes)

			if hdoti > self.hdot_threshold and customFlag == 0:
				self.density_mes_int, self.minAlt = \
					self.createDensityMeasuredFunction(self.h_step_array, \
													   self.density_mes_array, self.lowAlt_km, self.numPoints_lowAlt)
				customFlag = 1.0

			if hdoti > self.hdot_threshold:
				terminal_apoapsis_km = \
					self.predictApoapsisAltitudeKm_afterJettision2(h_km[-1], \
																  theta_deg[-1], phi_deg[-1], v_kms[-1], gamma_deg[-1], \
																  psi_deg[-1], drange_km[-1], heatload[-1], maxTimeSecs,
																  dt, \
																  0.0, self.density_mes_int)

			else:
				terminal_apoapsis_km = 0.0

			#print("H (km): "+ str('{:.2f}'.format(h0_km))+" HDOT (m/s): "+ str('{:.2f}'.format(hdoti))+"  PRED. APO: "+str('{:.2f}'.format(terminal_apoapsis_km)))

			# update current speed
			h_current_km = h_km[-1]
			v_current_kms = v_kms[-1]

			if terminal_apoapsis_km > 0 and terminal_apoapsis_km < self.target_apo_km:
				break

			if abs(terminal_apoapsis_km - self.target_apo_km) < self.target_apo_km_tol:
				break

			if h_current_km > self.planetObj.h_skip / 1000 - 1.0:
				break

		self.t_min_en = t_min
		self.h_km_en = h_km
		self.v_kms_en = v_kms
		self.theta_deg_en = theta_deg
		self.phi_deg_en = phi_deg
		self.psi_deg_en = psi_deg
		self.gamma_deg_en = gamma_deg
		self.drange_km_en = drange_km

		self.acc_net_g_en = acc_net_g
		self.dyn_pres_atm_en = dyn_pres_atm
		self.stag_pres_atm_en = stag_pres_atm
		self.q_stag_total_en = q_stag_total
		self.heatload_en = heatload

	def predictApoapsisAltitudeKm_afterJettision(self, h0_km, theta0_deg, \
						phi0_deg, v0_kms, gamma0_deg, psi0_deg, drange0_km,\
						heatLoad0, t_sec, dt, delta_deg, density_mes_int ):

		"""
		Compute the apoapsis altitude at exit if the drag skirt is 
		jettisoned at the current vehicle state.

		Parameters
		----------
		h0_km : float
			current vehicle altitude, km
		theta0_deg : float
			current vehicle longitude, deg
		phi0_deg : float
			current vehicle latitude, deg
		v0_kms : float
			current vehicle speed, km/s
		gamma0_deg : float
			current FPA, deg
		psi0_deg : float
			current heading angle, deg
		drange0_km : float
			current downrange, km
		heatLoad0 : float
			current heatload, J/cm2
		t_sec : float
			propogation time, seconds
		dt : float
			max. solver timestep
		delta_deg : float
			commanded bank angle, deg
		density_mes_int : scipy.interpolate.interpolate.interp1d
			measured density interpolation function


		Returns
		----------
		terminal_apoapsis_km : float
			apoapsis altitude achieved if drag skirt is
			jettisoned at the current time, km

		"""

		t_minc, h_kmc, v_kmsc, phi_degc, psi_degc, theta_degc, gamma_degc,\
		drange_kmc, exitflag, acc_net_g, dyn_pres_atm, stag_pres_atm, q_stag_total,\
		heatload, acc_drag_g = \
		self.propogateEntry_utilD(h0_km, theta0_deg, phi0_deg, v0_kms, gamma0_deg,\
		psi0_deg, drange0_km, heatLoad0, t_sec, dt, delta_deg, density_mes_int)
		
		terminal_apoapsis_km = self.compute_ApoapsisAltitudeKm(\
							   self.planetObj.RP+h_kmc[-1]*1E3,\
							   v_kmsc[-1]*1E3, gamma_degc[-1]*np.pi/180.0, \
							   theta_degc[-1]*np.pi/180.0, phi_degc[-1]*np.pi/180.0, \
							   psi_degc[-1]*np.pi/180.0)

		return terminal_apoapsis_km

	def predictApoapsisAltitudeKm_afterJettision2(self, h0_km, theta0_deg, \
												 phi0_deg, v0_kms, gamma0_deg, psi0_deg, drange0_km, \
												 heatLoad0, t_sec, dt, delta_deg, density_mes_int):

		"""
		Compute the apoapsis altitude at exit if the drag skirt is
		jettisoned at the current vehicle state. Uses new solver.

		Parameters
		----------
		h0_km : float
			current vehicle altitude, km
		theta0_deg : float
			current vehicle longitude, deg
		phi0_deg : float
			current vehicle latitude, deg
		v0_kms : float
			current vehicle speed, km/s
		gamma0_deg : float
			current FPA, deg
		psi0_deg : float
			current heading angle, deg
		drange0_km : float
			current downrange, km
		heatLoad0 : float
			current heatload, J/cm2
		t_sec : float
			propogation time, seconds
		dt : float
			max. solver timestep
		delta_deg : float
			commanded bank angle, deg
		density_mes_int : scipy.interpolate.interpolate.interp1d
			measured density interpolation function


		Returns
		----------
		terminal_apoapsis_km : float
			apoapsis altitude achieved if drag skirt is
			jettisoned at the current time, km

		"""

		t_minc, h_kmc, v_kmsc, phi_degc, psi_degc, theta_degc, gamma_degc, \
		drange_kmc, exitflag, acc_net_g, dyn_pres_atm, stag_pres_atm, q_stag_total, \
		heatload, acc_drag_g = \
			self.propogateEntry_utilD2(h0_km, theta0_deg, phi0_deg, v0_kms, gamma0_deg, \
									  psi0_deg, drange0_km, heatLoad0, t_sec, dt, delta_deg, density_mes_int)

		terminal_apoapsis_km = self.compute_ApoapsisAltitudeKm( \
			self.planetObj.RP + h_kmc[-1] * 1E3, \
			v_kmsc[-1] * 1E3, gamma_degc[-1] * np.pi / 180.0, \
			theta_degc[-1] * np.pi / 180.0, phi_degc[-1] * np.pi / 180.0, \
			psi_degc[-1] * np.pi / 180.0)

		return terminal_apoapsis_km



	def propogateEntry_utilD(self, h0_km, theta0_deg, phi0_deg, v0_kms, \
						gamma0_deg, psi0_deg, drange0_km, heatLoad0,\
						t_sec, dt, delta_deg, density_mes_int):
		"""
		Utility propogator routine for prediction of atmospheric exit
		conditions which is then supplied to the apoapis prediction 
		module. Does not include planetary rotation.

		Propogates the vehicle state for using the measured 
		atmospheric profile during the descending leg.
		
		Parameters
		-----------
		h0_km : float
			current altitude, km
		theta0_deg : float
			current longitude, deg
		phi0_deg : float
			current latitude, deg
		v0_kms : float
			current speed, km/s
		gamma0_deg : float
			current FPA, deg
		psi0_deg : float
			current heading angle, deg
		drange0_km : float
			current downrange, km
		heatLoad0 : float
			current heatload, J/cm2
		t_sec : float
			propogation time, seconds
		dt : float
			max. time step, seconds
		delta_deg : float
			bank angle command, deg


		Returns
		----------
		t_minc : numpy.ndarray
			time solution array, min 
		h_kmc : numpy.ndarray
			altitude solution array, km
		v_kmsc : numpy.ndarray
			speed solution array, km/s
		phi_degc : numpy.ndarray
			latitude solution array, deg
		psi_degc : numpy.ndarray
			heading angle solution array, deg
		theta_degc : numpy.ndarray
			longitude solution array, deg
		gamma_degc : numpy.ndarray
			FPA solution array, deg
		drange_kmc : numpy.ndarray
			downrange solution array, km
		exitflag : int
			exitflag
		acc_net_g : numpy.ndarray
			acceleration solution array, Earth g
		dyn_pres_atm : numpy.ndarray
			dynamic pressure solution array, atm
		stag_pres_atm : numpy.ndarray
			stagnation pressure array, atm
		q_stag_total : numpy.ndarray
			stagnation point heat rate array
		heatload : numpy.ndarray
			stagnation point heat load
		acc_drag_g : numpy.ndarray
			acceleration due to drag, Earth g
		
		"""

		"""
		Create a copy of the planet object associated 
		with the vehicle.
		Set the density_int attribute to be the
		density_mes_int which is the measured 
		density function.
		"""
		
		
		planetCopy = copy.deepcopy(self.planetObj)
		planetCopy.density_int = density_mes_int

		# Create a copy of the vehicle object so it does 
		# not affect the existing vehicle state variables
		vehicleCopy = copy.deepcopy(self)
		vehicleCopy.planetObj = planetCopy

		# Set the beta value used by the propogator to be
		# the higher ballistic coefficient value.
		# Update vehicle CD for new ballistic coefficient.
		vehicleCopy.beta = self.beta1*self.betaRatio
		vehicleCopy.CD   = self.mass/(vehicleCopy.beta*self.A)

		# Define entry conditions at entry interface
		# Convert initial state variables from input/plot 
		# units to calculation/SI units

		# Entry altitude above planet surface in meters
		# Entry latitude in radians
		# Entry velocity in meters/sec, relative to planet
		# Entry velocity in meters/sec, relative to planet
		# Entry heading angle in radians
		# Entry flight path angle in radians
		# Entry downrange in m


		h0      = h0_km*1.0E3
		theta0  = theta0_deg*np.pi/180.0  
		phi0    = phi0_deg*np.pi/180.0    
		v0      = v0_kms*1.000E3          
		psi0    = psi0_deg*np.pi/180.0    
		gamma0  = gamma0_deg*np.pi/180.0  
		drange0 = drange0_km*1E3          

		# Define control variables
		# Constant bank angle in radians
		delta   = delta_deg*np.pi/180.0  

		r0      = vehicleCopy.planetObj.computeR(h0)
		
		# Compute non-dimensional entry conditions	
		rbar0,theta0,phi0,vbar0,psi0,gamma0,drangebar0 = \
		vehicleCopy.planetObj.nonDimState(r0,theta0,phi0,v0,psi0,gamma0,drange0)
		
		# Solve for the entry trajectory
		tbar,rbar,theta,phi,vbar,psi,gamma,drangebar   = \
		vehicleCopy.solveTrajectory(rbar0, theta0, phi0, vbar0, psi0, gamma0, \
			drangebar0, t_sec, dt, delta)	
		# Note : solver returns non-dimensional variables
		# Convert to dimensional variables for plotting
		t,r,theta,phi,v,psi,gamma,drange               = \
		vehicleCopy.planetObj.dimensionalize(tbar,rbar,theta,phi,vbar,psi,gamma,\
			drangebar)
		#print(t[-1])
		# dimensional state variables are in SI units
		# convert to more rational units for plotting
		t_min, h_km, v_kms, phi_deg, psi_deg, theta_deg, gamma_deg, drange_km \
		= vehicleCopy.convertToPlotUnits(t,r,v,phi,psi,theta,gamma,drange)
		# classify trajectory
		index,exitflag = vehicleCopy.classifyTrajectory(r)
		# truncate trajectory
		tc,rc,thetac,phic,vc,psic,gammac,drangec\
		       = vehicleCopy.truncateTrajectory(t,r,theta,phi,v,psi,gamma,drange,\
		       	index)
		t_minc, h_kmc, v_kmsc, phi_degc, psi_degc, \
		theta_degc, gamma_degc, drange_kmc = \
		vehicleCopy.truncateTrajectory(t_min, h_km, v_kms, phi_deg, psi_deg, \
		theta_deg, gamma_deg, drange_km, index)
		# compute acceleration loads
		acc_net_g            = vehicleCopy.computeAccelerationLoad(tc,rc,\
			thetac,phic,vc,index,delta)
		# compute drag acceleration 
		acc_drag_g           = vehicleCopy.computeAccelerationDrag(tc,rc,\
			thetac,phic,vc,index,delta)
		# compute dynamic pressure
		dyn_pres_atm         = vehicleCopy.computeDynPres(rc,vc)/(1.01325E5)
	    # compute stagnation pressure
		stag_pres_atm        = vehicleCopy.computeStagPres(rc,vc)/(1.01325E5)

	    # compute stagnation point convective and radiative heating rate
		q_stag_con      = vehicleCopy.qStagConvective(rc,vc)
		q_stag_rad      = vehicleCopy.qStagRadiative (rc,vc)
		# compute total stagnation point heating rate
		q_stag_total    = q_stag_con + q_stag_rad
		# compute stagnation point heating load
		heatload        = cumtrapz(q_stag_total , tc, \
			initial=heatLoad0)

		return t_minc, h_kmc, v_kmsc, phi_degc, psi_degc, theta_degc, \
			   gamma_degc, drange_kmc, exitflag, acc_net_g, dyn_pres_atm, \
			   stag_pres_atm, q_stag_total, heatload, acc_drag_g

	def propogateEntry_utilD2(self, h0_km, theta0_deg, phi0_deg, v0_kms, \
							 gamma0_deg, psi0_deg, drange0_km, heatLoad0, \
							 t_sec, dt, delta_deg, density_mes_int):
		"""
		Utility propogator routine for prediction of atmospheric exit
		conditions which is then supplied to the apoapis prediction
		module. Includes planetary rotation.

		Propogates the vehicle state for using the measured
		atmospheric profile during the descending leg.

		Parameters
		-----------
		h0_km : float
			current altitude, km
		theta0_deg : float
			current longitude, deg
		phi0_deg : float
			current latitude, deg
		v0_kms : float
			current speed, km/s
		gamma0_deg : float
			current FPA, deg
		psi0_deg : float
			current heading angle, deg
		drange0_km : float
			current downrange, km
		heatLoad0 : float
			current heatload, J/cm2
		t_sec : float
			propogation time, seconds
		dt : float
			max. time step, seconds
		delta_deg : float
			bank angle command, deg


		Returns
		----------
		t_minc : numpy.ndarray
			time solution array, min
		h_kmc : numpy.ndarray
			altitude solution array, km
		v_kmsc : numpy.ndarray
			speed solution array, km/s
		phi_degc : numpy.ndarray
			latitude solution array, deg
		psi_degc : numpy.ndarray
			heading angle solution array, deg
		theta_degc : numpy.ndarray
			longitude solution array, deg
		gamma_degc : numpy.ndarray
			FPA solution array, deg
		drange_kmc : numpy.ndarray
			downrange solution array, km
		exitflag : int
			exitflag
		acc_net_g : numpy.ndarray
			acceleration solution array, Earth g
		dyn_pres_atm : numpy.ndarray
			dynamic pressure solution array, atm
		stag_pres_atm : numpy.ndarray
			stagnation pressure array, atm
		q_stag_total : numpy.ndarray
			stagnation point heat rate array
		heatload : numpy.ndarray
			stagnation point heat load
		acc_drag_g : numpy.ndarray
			acceleration due to drag, Earth g

		"""

		"""
		Create a copy of the planet object associated 
		with the vehicle.
		Set the density_int attribute to be the
		density_mes_int which is the measured 
		density function.
		"""

		planetCopy = copy.deepcopy(self.planetObj)
		planetCopy.density_int = density_mes_int

		# Create a copy of the vehicle object so it does
		# not affect the existing vehicle state variables
		vehicleCopy = copy.deepcopy(self)
		vehicleCopy.planetObj = planetCopy

		# Set the beta value used by the propogator to be
		# the higher ballistic coefficient value.
		# Update vehicle CD for new ballistic coefficient.
		vehicleCopy.beta = self.beta1 * self.betaRatio
		vehicleCopy.CD = self.mass / (vehicleCopy.beta * self.A)

		# Define entry conditions at entry interface
		# Convert initial state variables from input/plot
		# units to calculation/SI units

		# Entry altitude above planet surface in meters
		# Entry latitude in radians
		# Entry velocity in meters/sec, relative to planet
		# Entry velocity in meters/sec, relative to planet
		# Entry heading angle in radians
		# Entry flight path angle in radians
		# Entry downrange in m

		h0 = h0_km * 1.0E3
		theta0 = theta0_deg * np.pi / 180.0
		phi0 = phi0_deg * np.pi / 180.0
		v0 = v0_kms * 1.000E3
		psi0 = psi0_deg * np.pi / 180.0
		gamma0 = gamma0_deg * np.pi / 180.0
		drange0 = drange0_km * 1E3

		# Define control variables
		# Constant bank angle in radians
		delta = delta_deg * np.pi / 180.0

		r0 = vehicleCopy.planetObj.computeR(h0)

		# Compute non-dimensional entry conditions
		rbar0, theta0, phi0, vbar0, psi0, gamma0, drangebar0 = \
			vehicleCopy.planetObj.nonDimState(r0, theta0, phi0, v0, psi0, gamma0, drange0)

		# Solve for the entry trajectory
		tbar, rbar, theta, phi, vbar, psi, gamma, drangebar = \
			vehicleCopy.solveTrajectory2(rbar0, theta0, phi0, vbar0, psi0, gamma0, \
										drangebar0, t_sec, dt, delta)
		# Note : solver returns non-dimensional variables
		# Convert to dimensional variables for plotting
		t, r, theta, phi, v, psi, gamma, drange = \
			vehicleCopy.planetObj.dimensionalize(tbar, rbar, theta, phi, vbar, psi, gamma, \
												 drangebar)
		# print(t[-1])
		# dimensional state variables are in SI units
		# convert to more rational units for plotting
		t_min, h_km, v_kms, phi_deg, psi_deg, theta_deg, gamma_deg, drange_km \
			= vehicleCopy.convertToPlotUnits(t, r, v, phi, psi, theta, gamma, drange)
		# classify trajectory
		index, exitflag = vehicleCopy.classifyTrajectory(r)
		# truncate trajectory
		tc, rc, thetac, phic, vc, psic, gammac, drangec \
			= vehicleCopy.truncateTrajectory(t, r, theta, phi, v, psi, gamma, drange, \
											 index)
		t_minc, h_kmc, v_kmsc, phi_degc, psi_degc, \
		theta_degc, gamma_degc, drange_kmc = \
			vehicleCopy.truncateTrajectory(t_min, h_km, v_kms, phi_deg, psi_deg, \
										   theta_deg, gamma_deg, drange_km, index)
		# compute acceleration loads
		acc_net_g = vehicleCopy.computeAccelerationLoad(tc, rc, \
														thetac, phic, vc, index, delta)
		# compute drag acceleration
		acc_drag_g = vehicleCopy.computeAccelerationDrag(tc, rc, \
														 thetac, phic, vc, index, delta)
		# compute dynamic pressure
		dyn_pres_atm = vehicleCopy.computeDynPres(rc, vc) / (1.01325E5)
		# compute stagnation pressure
		stag_pres_atm = vehicleCopy.computeStagPres(rc, vc) / (1.01325E5)

		# compute stagnation point convective and radiative heating rate
		q_stag_con = vehicleCopy.qStagConvective(rc, vc)
		q_stag_rad = vehicleCopy.qStagRadiative(rc, vc)
		# compute total stagnation point heating rate
		q_stag_total = q_stag_con + q_stag_rad
		# compute stagnation point heating load
		heatload = cumtrapz(q_stag_total, tc, \
							initial=heatLoad0)

		return t_minc, h_kmc, v_kmsc, phi_degc, psi_degc, theta_degc, \
			   gamma_degc, drange_kmc, exitflag, acc_net_g, dyn_pres_atm, \
			   stag_pres_atm, q_stag_total, heatload, acc_drag_g

	def propogateExitPhaseD(self, timeStep, dt, maxTimeSecs):
		"""
		Implements the exit phase of the guidance scheme for drag modulation.

		Parameters
		--------
		timeStep : float
			Guidance cycle time, seconds
		dt : float
			Solver max. time step, seconds
		maxTimeSecs : float
			max. time for propogation, seconds

		"""

		self.beta        = self.beta1*self.betaRatio
		self.CD          = self.mass/(self.beta*self.A)


		self.t_switch = self.t_min_en[-1]
		self.h_switch = self.h_km_en[-1]
		self.v_switch = self.v_kms_en[-1]
		self.p_switch = self.delta_deg_array[-1]

		
		t_min     = self.t_min_en
		h_km      = self.h_km_en
		v_kms     = self.v_kms_en
		theta_deg = self.theta_deg_en
		phi_deg   = self.phi_deg_en
		psi_deg   = self.psi_deg_en
		gamma_deg = self.gamma_deg_en
		drange_km = self.drange_km_en

		acc_net_g     = self.acc_net_g_en
		dyn_pres_atm  = self.dyn_pres_atm_en
		stag_pres_atm = self.stag_pres_atm_en
		q_stag_total  = self.q_stag_total_en
		heatload      = self.heatload_en
	
		# Set the current altitude to the terminal altitude of
		# the equlibrium glide phase (km).
		h_current_km  = h_km[-1]
		t_current_min = t_min[-1]
		
		# Set the skip altitude (km) based on definition in planet.py
		h_skip_km    = self.planetObj.h_skip / 1.0E3

		# initialize hdot_ref as terminal hdot of equilibrium glide phase
		hdot_refi    = self.hdot_array[-1]
		Ji           = self.heatload_en[-1]

		#print('Exit Phase Guidance Initiated')
		while h_current_km < h_skip_km:
			# print(minAlt)
			# Reset the initial conditions to the terminal conditions of 
			# the propgated solution from the previous step.
			# Terminal conditions of the equilibrium glide phase are initial 
			# conditions for the exit phase algorithm.
			# ----------------------------------------------------------------
			h0_km       = h_km[-1]           # Entry altitude above planet surface 
			theta0_deg  = theta_deg[-1]      # Entry longitude 
			phi0_deg    = phi_deg[-1]        # Entry latitude 
			v0_kms      = v_kms[-1]          # Entry velocity 
			psi0_deg    = psi_deg[-1]        # Entry heading angle 
			gamma0_deg  = gamma_deg[-1]      # Entry flight path angle 
			drange0_km  = drange_km[-1]      # Downrange

			# ----------------------------------------------------------------------------
			# Convert entry state variables from IO/plot units to calculation (SI) units
			# ----------------------------------------------------------------------------
			h0      = h0_km*1.0E3             # Entry altitude above planet 
			theta0  = theta0_deg*np.pi/180.0  # Entry longitude in radians
			phi0    = phi0_deg*np.pi/180.0    # Entry latitude in radians
			v0      = v0_kms*1.000E3          # Entry velocity in meters/se
			psi0    = psi0_deg*np.pi/180.0    # Entry heading angle in radians
			gamma0  = gamma0_deg*np.pi/180.0  # Entry flight path angle in radians
			# -------------------------------------------------------------------------------

			# --------------------------------------------------------------------------
			# Initialize iterable variables used in guidance loop, i = at the current time
			# ----------------------------------------------------------------------------
			hi = h0                           # Set the current altitude as the entry altitude
			ri = h0 + self.planetObj.RP       # Set the current radial distance 	
			vi = v0                           # Set the current speed (m/s) as the entry speed
			qi = 0.5*self.planetObj.density(hi)*vi**2.0      # Set the current dynamic pressure
			gammai = gamma0                   # Set the current FPA to entry FPA
			hdoti = vi*np.sin(gammai)         # Compute the current altitude rate hdot

			DeltaCMD_deg = 0.0
			

			# propogate the vehicle state to advance from the current 
			# state using the commanded bank angle deltaCMD
			self.setInitialState(h0_km,theta0_deg,phi0_deg,v0_kms,psi0_deg,gamma0_deg,\
								 drange0_km,Ji)
			self.propogateEntry(timeStep, dt, DeltaCMD_deg)

			t_min_c     = self.t_minc
			h_km_c      = self.h_kmc
			v_kms_c     = self.v_kmsc
			phi_deg_c   = self.phi_degc
			psi_deg_c   = self.psi_degc
			theta_deg_c = self.theta_degc
			gamma_deg_c = self.gamma_degc
			drange_km_c = self.drange_kmc

			acc_net_g_c     =  self.acc_net_g
			dyn_pres_atm_c  =  self.dyn_pres_atm 
			stag_pres_atm_c =  self.stag_pres_atm
			q_stag_total_c  =  self.q_stag_total
			heatload_c      =  self.heatload
			acc_drag_g_c    =  self.acc_drag_g

			# Update the time solution array to account for non-zero start time 
			t_min_c      = t_min_c + t_min[-1]

			self.t_step_array        = np.append(self.t_step_array, t_min[-1])
			self.delta_deg_array     = np.append(self.delta_deg_array, DeltaCMD_deg)
			self.hdot_array          = np.append(self.hdot_array, hdoti)
			self.hdotref_array       = np.append(self.hdotref_array, hdot_refi)

			self.hddoti              = (self.hdot_array[-1] - self.hdot_array[-2]) / \
									   (self.t_step_array[-1]*60.0 - self.t_step_array[-2]*60.0)

			self.hddot_array         = np.append(self.hddot_array, self.hddoti)

			# Update time and other solution vectors
			t_min        = np.concatenate((t_min, t_min_c), axis=0)
			h_km         = np.concatenate((h_km , h_km_c ), axis=0)
			v_kms        = np.concatenate((v_kms, v_kms_c ), axis=0)
			phi_deg      = np.concatenate((phi_deg, phi_deg_c ), axis=0)
			psi_deg      = np.concatenate((psi_deg, psi_deg_c ), axis=0)
			theta_deg    = np.concatenate((theta_deg, theta_deg_c), axis=0)
			gamma_deg    = np.concatenate((gamma_deg, gamma_deg_c), axis=0)
			drange_km    = np.concatenate((drange_km, drange_km_c), axis=0)

			# Update entry parameter vectors
			acc_net_g    = np.concatenate((acc_net_g,acc_net_g_c), axis=0)
			dyn_pres_atm = np.concatenate((dyn_pres_atm, dyn_pres_atm_c),axis=0)
			stag_pres_atm= np.concatenate((stag_pres_atm, stag_pres_atm_c), axis=0)
			q_stag_total = np.concatenate((q_stag_total, q_stag_total_c), axis=0)
			heatload     = np.concatenate((heatload, heatload_c), axis=0)

			terminal_apoapsis_km = self.compute_ApoapsisAltitudeKm(\
								   self.planetObj.RP+h_km[-1]*1E3, v_kms[-1]*1E3, \
								   gamma_deg[-1]*np.pi/180.0, theta_deg[-1]*np.pi/180.0, \
								   phi_deg[-1]*np.pi/180.0, psi_deg[-1]*np.pi/180.0)


			#print("H (km): "+ str('{:.2f}'.format(h0_km))+", HDOT: "+str('{:.2f}'.format(hdoti))+", PREDICT. APO. ALT. :"+str(terminal_apoapsis_km))

			if hi > self.planetObj.h_skip - 1.0E3:
				break

			if hi < 20.0E3:
				break

			h_current_km  = h_km[-1]
			t_current_min = t_min[-1]

		self.t_min_full     = t_min
		self.h_km_full      = h_km
		self.v_kms_full     = v_kms
		self.theta_deg_full = theta_deg	
		self.phi_deg_full   = phi_deg
		self.psi_deg_full   = psi_deg
		self.gamma_deg_full = gamma_deg
		self.drange_km_full = drange_km
		
		self.acc_net_g_full    = acc_net_g
		self.dyn_pres_atm_full = dyn_pres_atm
		self.stag_pres_atm_full= stag_pres_atm
		self.q_stag_total_full = q_stag_total
		self.heatload_full     = heatload


		self.terminal_apoapsis  = self.compute_ApoapsisAltitudeKm(
								  self.planetObj.RP+h_km[-1]*1E3, v_kms[-1]*1E3,
								  gamma_deg[-1]*np.pi/180.0, theta_deg[-1]*np.pi/180.0,
								  phi_deg[-1]*np.pi/180.0, psi_deg[-1]*np.pi/180.0)
		self.terminal_periapsis = self.compute_PeriapsisAltitudeKm(
								  self.planetObj.RP+h_km[-1]*1E3, v_kms[-1]*1E3,
								  gamma_deg[-1]*np.pi/180.0, theta_deg[-1]*np.pi/180.0,
								  phi_deg[-1]*np.pi/180.0, psi_deg[-1]*np.pi/180.0)
		self.apoapsis_perc_error= (self.terminal_apoapsis - self.target_apo_km)*100.0 / \
								   self.target_apo_km

		self.periapsis_raise_DV = self.compute_periapsis_raise_DV(
								  self.terminal_periapsis, self.terminal_apoapsis,
								  self.target_peri_km)
		self.apoapsis_raise_DV  = self.compute_apoapsis_raise_DV(
								  self.target_peri_km, self.terminal_apoapsis,
								  self.target_apo_km)

	def propogateExitPhaseD2(self, timeStep, dt, maxTimeSecs):
		"""
		Implements the exit phase of the guidance scheme for drag modulation, with new solver.

		Parameters
		--------
		timeStep : float
			Guidance cycle time, seconds
		dt : float
			Solver max. time step, seconds
		maxTimeSecs : float
			max. time for propogation, seconds

		"""

		self.beta = self.beta1 * self.betaRatio
		self.CD = self.mass / (self.beta * self.A)

		self.t_switch = self.t_min_en[-1]
		self.h_switch = self.h_km_en[-1]
		self.v_switch = self.v_kms_en[-1]
		self.p_switch = self.delta_deg_array[-1]

		t_min = self.t_min_en
		h_km = self.h_km_en
		v_kms = self.v_kms_en
		theta_deg = self.theta_deg_en
		phi_deg = self.phi_deg_en
		psi_deg = self.psi_deg_en
		gamma_deg = self.gamma_deg_en
		drange_km = self.drange_km_en

		acc_net_g = self.acc_net_g_en
		dyn_pres_atm = self.dyn_pres_atm_en
		stag_pres_atm = self.stag_pres_atm_en
		q_stag_total = self.q_stag_total_en
		heatload = self.heatload_en

		# Set the current altitude to the terminal altitude of
		# the equlibrium glide phase (km).
		h_current_km = h_km[-1]
		t_current_min = t_min[-1]

		# Set the skip altitude (km) based on definition in planet.py
		h_skip_km = self.planetObj.h_skip / 1.0E3

		# initialize hdot_ref as terminal hdot of equilibrium glide phase
		hdot_refi = self.hdot_array[-1]
		Ji = self.heatload_en[-1]

		# print('Exit Phase Guidance Initiated')
		while h_current_km < h_skip_km:
			# print(minAlt)
			# Reset the initial conditions to the terminal conditions of
			# the propgated solution from the previous step.
			# Terminal conditions of the equilibrium glide phase are initial
			# conditions for the exit phase algorithm.
			# ----------------------------------------------------------------
			h0_km = h_km[-1]  # Entry altitude above planet surface
			theta0_deg = theta_deg[-1]  # Entry longitude
			phi0_deg = phi_deg[-1]  # Entry latitude
			v0_kms = v_kms[-1]  # Entry velocity
			psi0_deg = psi_deg[-1]  # Entry heading angle
			gamma0_deg = gamma_deg[-1]  # Entry flight path angle
			drange0_km = drange_km[-1]  # Downrange

			# ----------------------------------------------------------------------------
			# Convert entry state variables from IO/plot units to calculation (SI) units
			# ----------------------------------------------------------------------------
			h0 = h0_km * 1.0E3  # Entry altitude above planet
			theta0 = theta0_deg * np.pi / 180.0  # Entry longitude in radians
			phi0 = phi0_deg * np.pi / 180.0  # Entry latitude in radians
			v0 = v0_kms * 1.000E3  # Entry velocity in meters/se
			psi0 = psi0_deg * np.pi / 180.0  # Entry heading angle in radians
			gamma0 = gamma0_deg * np.pi / 180.0  # Entry flight path angle in radians
			# -------------------------------------------------------------------------------

			# --------------------------------------------------------------------------
			# Initialize iterable variables used in guidance loop, i = at the current time
			# ----------------------------------------------------------------------------
			hi = h0  # Set the current altitude as the entry altitude
			ri = h0 + self.planetObj.RP  # Set the current radial distance
			vi = v0  # Set the current speed (m/s) as the entry speed
			qi = 0.5 * self.planetObj.density(hi) * vi ** 2.0  # Set the current dynamic pressure
			gammai = gamma0  # Set the current FPA to entry FPA
			hdoti = vi * np.sin(gammai)  # Compute the current altitude rate hdot

			DeltaCMD_deg = 0.0

			# propogate the vehicle state to advance from the current
			# state using the commanded bank angle deltaCMD
			self.setInitialState(h0_km, theta0_deg, phi0_deg, v0_kms, psi0_deg, gamma0_deg, \
								 drange0_km, Ji)
			self.propogateEntry2(timeStep, dt, DeltaCMD_deg)

			t_min_c = self.t_minc
			h_km_c = self.h_kmc
			v_kms_c = self.v_kmsc
			phi_deg_c = self.phi_degc
			psi_deg_c = self.psi_degc
			theta_deg_c = self.theta_degc
			gamma_deg_c = self.gamma_degc
			drange_km_c = self.drange_kmc

			acc_net_g_c = self.acc_net_g
			dyn_pres_atm_c = self.dyn_pres_atm
			stag_pres_atm_c = self.stag_pres_atm
			q_stag_total_c = self.q_stag_total
			heatload_c = self.heatload
			acc_drag_g_c = self.acc_drag_g

			# Update the time solution array to account for non-zero start time
			t_min_c = t_min_c + t_min[-1]

			self.t_step_array = np.append(self.t_step_array, t_min[-1])
			self.delta_deg_array = np.append(self.delta_deg_array, DeltaCMD_deg)
			self.hdot_array = np.append(self.hdot_array, hdoti)
			self.hdotref_array = np.append(self.hdotref_array, hdot_refi)

			self.hddoti = (self.hdot_array[-1] - self.hdot_array[-2]) / \
						  (self.t_step_array[-1] * 60.0 - self.t_step_array[-2] * 60.0)

			self.hddot_array = np.append(self.hddot_array, self.hddoti)

			# Update time and other solution vectors
			t_min = np.concatenate((t_min, t_min_c), axis=0)
			h_km = np.concatenate((h_km, h_km_c), axis=0)
			v_kms = np.concatenate((v_kms, v_kms_c), axis=0)
			phi_deg = np.concatenate((phi_deg, phi_deg_c), axis=0)
			psi_deg = np.concatenate((psi_deg, psi_deg_c), axis=0)
			theta_deg = np.concatenate((theta_deg, theta_deg_c), axis=0)
			gamma_deg = np.concatenate((gamma_deg, gamma_deg_c), axis=0)
			drange_km = np.concatenate((drange_km, drange_km_c), axis=0)

			# Update entry parameter vectors
			acc_net_g = np.concatenate((acc_net_g, acc_net_g_c), axis=0)
			dyn_pres_atm = np.concatenate((dyn_pres_atm, dyn_pres_atm_c), axis=0)
			stag_pres_atm = np.concatenate((stag_pres_atm, stag_pres_atm_c), axis=0)
			q_stag_total = np.concatenate((q_stag_total, q_stag_total_c), axis=0)
			heatload = np.concatenate((heatload, heatload_c), axis=0)

			terminal_apoapsis_km = self.compute_ApoapsisAltitudeKm( \
				self.planetObj.RP + h_km[-1] * 1E3, v_kms[-1] * 1E3, \
				gamma_deg[-1] * np.pi / 180.0, theta_deg[-1] * np.pi / 180.0, \
				phi_deg[-1] * np.pi / 180.0, psi_deg[-1] * np.pi / 180.0)

			#print("H (km): "+ str('{:.2f}'.format(h0_km))+", HDOT: "+str('{:.2f}'.format(hdoti))+", PREDICT. APO. ALT. :"+str(terminal_apoapsis_km))

			if hi > self.planetObj.h_skip - 1.0E3:
				break

			if hi < 20.0E3:
				break

			h_current_km = h_km[-1]
			t_current_min = t_min[-1]

		self.t_min_full = t_min
		self.h_km_full = h_km
		self.v_kms_full = v_kms
		self.theta_deg_full = theta_deg
		self.phi_deg_full = phi_deg
		self.psi_deg_full = psi_deg
		self.gamma_deg_full = gamma_deg
		self.drange_km_full = drange_km

		self.acc_net_g_full = acc_net_g
		self.dyn_pres_atm_full = dyn_pres_atm
		self.stag_pres_atm_full = stag_pres_atm
		self.q_stag_total_full = q_stag_total
		self.heatload_full = heatload

		self.terminal_apoapsis = self.compute_ApoapsisAltitudeKm(
			self.planetObj.RP + h_km[-1] * 1E3, v_kms[-1] * 1E3,
			gamma_deg[-1] * np.pi / 180.0, theta_deg[-1] * np.pi / 180.0,
			phi_deg[-1] * np.pi / 180.0, psi_deg[-1] * np.pi / 180.0)
		self.terminal_periapsis = self.compute_PeriapsisAltitudeKm(
			self.planetObj.RP + h_km[-1] * 1E3, v_kms[-1] * 1E3,
			gamma_deg[-1] * np.pi / 180.0, theta_deg[-1] * np.pi / 180.0,
			phi_deg[-1] * np.pi / 180.0, psi_deg[-1] * np.pi / 180.0)
		self.apoapsis_perc_error = (self.terminal_apoapsis - self.target_apo_km) * 100.0 / \
								   self.target_apo_km

		self.periapsis_raise_DV = self.compute_periapsis_raise_DV(
			self.terminal_periapsis, self.terminal_apoapsis,
			self.target_peri_km)
		self.apoapsis_raise_DV = self.compute_apoapsis_raise_DV(
			self.target_peri_km, self.terminal_apoapsis,
			self.target_apo_km)

	def propogateGuidedEntryD(self, timeStepEntry, timeStepExit, dt, maxTimeSecs):
		"""
		Implements the full guidance scheme for drag modulation aerocapture
		(entry phase + exit phase)

		Parameters
		--------
		timeStepEntry : float
			Guidance cycle time (entry phase), seconds
		timeStepExit : float
			Guidance cycle time (exit phase), seconds
		dt : float
			Solver max. time step, seconds
		maxTimeSecs : float
			max. time for propogation, seconds

		"""
		self.propogateEntryPhaseD(timeStepEntry, dt, maxTimeSecs)
		self.propogateExitPhaseD(timeStepExit, dt, maxTimeSecs)


	def propogateGuidedEntryD2(self, timeStepEntry, timeStepExit, dt, maxTimeSecs):
		"""
		Implements the full guidance scheme for drag modulation aerocapture
		(entry phase + exit phase). Includes inertial correction.

		Parameters
		--------
		timeStepEntry : float
			Guidance cycle time (entry phase), seconds
		timeStepExit : float
			Guidance cycle time (exit phase), seconds
		dt : float
			Solver max. time step, seconds
		maxTimeSecs : float
			max. time for propogation, seconds

		"""
		self.propogateEntryPhaseD2(timeStepEntry, dt, maxTimeSecs)
		self.propogateExitPhaseD2(timeStepExit, dt, maxTimeSecs)



	def setupMonteCarloSimulationD(self, NPOS, NMONTE, atmfiles,
								  heightCol, densLowCol, densAvgCol,
								  densHighCol, densTotalCol, heightInKmFlag,
								  nominalEFPA,  EFPA_1sigma_value,
								  nominalbeta1, beta1_1sigma_value,
								  timeStepEntry, timeStepExit, dt, maxTimeSecs):
		"""
		Set the Monte Carlo simulation parameters for drag modulation
		aerocapture.

		Parameters
		--------
		NPOS : int
			NPOS value from GRAM model output 
			is the number of data points (altitude) in each atm. profile
		NMONTE : int
			NMONTE is the number of Monte Carlo atm profiles
			from GRAM model output
		atmfiles : str
			location of atmospheric files used in Monte Carlo simulation
		heightCol : int
			column index of height values in atmfiles
		densLowCol : int
			column index of low density (-1 sigma) values in atmfiles
		densAvgCol : int
			column index of average density values in atmfiles
		densHighCol : int
			column index of high density values (+1 sigma) in atmfiles
		densTotalCol : int
			index of perturbed (=avg + pert.) density values
		heightInKmFlag : bool
			set to True if height values in atmfiles are in km
		nominalEFPA : float
			Nominal (target EFPA) value, deg
		EFPA_1sigma_value : float
			1-sigma error for EFPA (from naviation analysis)
		nominalbeta1 : float
			Nominal value of vehicle ballistic coeff.
		beta1_1sigma_value : float
			1-sigma error for beta1 (from vehicle aero. design data)
		timeStepEntry : float
			Guidance cycle time step for entry phase, sec
		timeStepExit : float
			Guidance time step for exit phase, sec
		dt : float
			max. solver time step
		maxTimeSecs : float
			max. time used for propogation used by guidance scheme


		"""

		self.NPOS       = NPOS
		self.NMONTE     = NMONTE
		self.atmfiles   = atmfiles

		self.heightCol   = heightCol
		self.densLowCol  = densLowCol
		self.densAvgCol  = densAvgCol
		self.densHighCol = densHighCol
		self.densTotalCol= densTotalCol

		self.heightInKmFlag = heightInKmFlag

		self.nominalEFPA = nominalEFPA
		self.EFPA_1sigma_value = EFPA_1sigma_value

		self.nominalbeta1 = nominalbeta1
		self.beta1_1sigma_value = beta1_1sigma_value
		self.vehicleCopy = copy.deepcopy(self)

		self.timeStepEntry = timeStepEntry
		self.timeStepExit  = timeStepExit
		
		self.dt = dt
		self.maxTimeSecs = maxTimeSecs


	def setupMonteCarloSimulationD_Earth(self, NPOS, NMONTE, atmfiles,
								  heightCol, densAvgCol,
								  densSD_percCol, densTotalCol, heightInKmFlag,
								  nominalEFPA,  EFPA_1sigma_value,
								  nominalbeta1, beta1_1sigma_value,
								  timeStepEntry, timeStepExit, dt, maxTimeSecs):
		"""
		Set the Monte Carlo simulation parameters for drag modulation
		aerocapture. (Earth aerocapture)

		Parameters
		--------
		NPOS : int
			NPOS value from GRAM model output 
			is the number of data points (altitude) in each atm. profile
		NMONTE : int
			NMONTE is the number of Monte Carlo atm profiles
			from GRAM model output
		atmfiles : str
			location of atmospheric files used in Monte Carlo simulation
		heightCol : int
			column index of height values in atmfiles
		densAvgCol : int
			column index of average density values in atmfiles
		densSD_percCol : int
			column number of mean density one sigma SD
		densTotalCol : int
			index of perturbed (=avg + pert.) density values
		heightInKmFlag : bool
			set to True if height values in atmfiles are in km
		nominalEFPA : float
			Nominal (target EFPA) value, deg
		EFPA_1sigma_value : float
			1-sigma error for EFPA (from naviation analysis)
		nominalbeta1 : float
			Nominal value of vehicle ballistic coeff.
		beta1_1sigma_value : float
			1-sigma error for beta1 (from vehicle aero. design data)
		timeStepEntry : float
			Guidance cycle time step for entry phase, sec
		timeStepExit : float
			Guidance time step for exit phase, sec
		dt : float
			max. solver time step
		maxTimeSecs : float
			max. time used for propogation used by guidance scheme


		"""

		self.NPOS       = NPOS
		self.NMONTE     = NMONTE
		self.atmfiles   = atmfiles

		self.heightCol   = heightCol
		self.densAvgCol  = densAvgCol
		self.densSD_percCol = densSD_percCol
		self.densTotalCol= densTotalCol

		self.heightInKmFlag = heightInKmFlag

		self.nominalEFPA = nominalEFPA
		self.EFPA_1sigma_value = EFPA_1sigma_value

		self.nominalbeta1 = nominalbeta1
		self.beta1_1sigma_value = beta1_1sigma_value
		self.vehicleCopy = copy.deepcopy(self)

		self.timeStepEntry = timeStepEntry
		self.timeStepExit  = timeStepExit
		
		self.dt = dt
		self.maxTimeSecs = maxTimeSecs


	def runMonteCarloD(self, N, mainFolder):
		"""
		Run a Monte Carlo simulation for drag modulation
		aerocapture.

		Parameters
		--------
		N : int
			Number of trajectories
		mainFolder : str
			path where data is to be stored
		"""


		terminal_apoapsis_arr = np.zeros(N)
		terminal_periapsis_arr= np.zeros(N)
		periapsis_raise_DV_arr= np.zeros(N)
		apoapsis_raise_DV_arr = np.zeros(N)
		acc_net_g_max_arr     = np.zeros(N)
		q_stag_max_arr        = np.zeros(N)
		heatload_max_arr      = np.zeros(N)

		h0_km      = self.vehicleCopy.h0_km_ref
		theta0_deg = self.vehicleCopy.theta0_deg_ref
		phi0_deg   = self.vehicleCopy.phi0_deg_ref
		v0_kms     = self.vehicleCopy.v0_kms_ref
		psi0_deg   = self.vehicleCopy.psi0_deg_ref
		drange0_km = self.vehicleCopy.drange0_km_ref
		heatLoad0  = self.vehicleCopy.heatLoad0_ref

		os.makedirs(mainFolder)
	
		
		for i in range(N):
			selected_atmfile  = rd.choice(self.atmfiles)
			
			#selected_profile   = 65
			selected_profile  = rd.randint(1,self.NMONTE)
			#selected_efpa     = -11.53
			selected_efpa     = np.random.normal(self.nominalEFPA, self.EFPA_1sigma_value)
			#selected_atmSigma = +0.0
			selected_atmSigma = np.random.normal(0,1)

			selected_LD       = 0.0

			ATM_height, ATM_density_low, ATM_density_avg, ATM_density_high, \
			ATM_density_pert = self.planetObj.loadMonteCarloDensityFile2(\
				               selected_atmfile, self.heightCol, \
				               self.densLowCol,  self.densAvgCol, 
				               self.densHighCol, self.densTotalCol, self.heightInKmFlag)
			self.planetObj.density_int = self.planetObj.loadAtmosphereModel5(ATM_height, \
										 ATM_density_low, ATM_density_avg, ATM_density_high, \
										 ATM_density_pert, selected_atmSigma, self.NPOS, \
										 selected_profile)

			
			self.setInitialState(h0_km,theta0_deg,phi0_deg,v0_kms,\
						         psi0_deg,selected_efpa,drange0_km,heatLoad0)
			self.propogateGuidedEntryD(self.timeStepEntry, self.timeStepExit,  self.dt, self.maxTimeSecs)

			terminal_apoapsis = self.terminal_apoapsis
			apoapsis_error    = self.apoapsis_perc_error
			terminal_periapsis= self.terminal_periapsis

			periapsis_raise_DV = self.periapsis_raise_DV
			apoapsis_raise_DV =  self.apoapsis_raise_DV


			terminal_apoapsis_arr[i] = self.terminal_apoapsis
			terminal_periapsis_arr[i]= self.terminal_periapsis

			periapsis_raise_DV_arr[i]= self.periapsis_raise_DV
			apoapsis_raise_DV_arr[i] = self.apoapsis_raise_DV

			acc_net_g_max_arr[i]     = max(self.acc_net_g_full)
			q_stag_max_arr[i]        = max(self.q_stag_total_full)
			heatload_max_arr[i]      = max(self.heatload_full)

			print("BATCH :"+str(mainFolder)+", RUN #: "+str(i+1)+", PROF: "+str(selected_atmfile)+", SAMPLE #: "+str(selected_profile)+", EFPA: "+str('{:.2f}'.format(selected_efpa,2))+", SIGMA: "+str('{:.2f}'.format(selected_atmSigma,2))+", APO : "+str('{:.2f}'.format(terminal_apoapsis,2)))


			os.makedirs(mainFolder+'/'+'#'+str(i+1))


			np.savetxt(mainFolder+'/'+'#'+str(i+1)+'/'+'atmfile.txt',np.array([selected_atmfile]), fmt='%s')
			np.savetxt(mainFolder+'/'+'#'+str(i+1)+'/'+'profile.txt',np.array([selected_profile]))
			np.savetxt(mainFolder+'/'+'#'+str(i+1)+'/'+'efpa.txt',np.array([selected_efpa]))
			np.savetxt(mainFolder+'/'+'#'+str(i+1)+'/'+'atmSigma.txt',np.array([selected_atmSigma]))
			np.savetxt(mainFolder+'/'+'#'+str(i+1)+'/'+'LD.txt',np.array([selected_LD]))

			np.savetxt(mainFolder+'/'+'#'+str(i+1)+'/'+'terminal_apoapsis.txt',np.array([terminal_apoapsis]))
			np.savetxt(mainFolder+'/'+'#'+str(i+1)+'/'+'apoapsis_error.txt',np.array([apoapsis_error]))
			np.savetxt(mainFolder+'/'+'#'+str(i+1)+'/'+'terminal_periapsis.txt',np.array([terminal_periapsis]))

			np.savetxt(mainFolder+'/'+'#'+str(i+1)+'/'+'periapsis_raise_DV.txt',np.array([periapsis_raise_DV]))
			np.savetxt(mainFolder+'/'+'#'+str(i+1)+'/'+'apoapsis_raise_DV.txt',np.array([apoapsis_raise_DV]))


			np.savetxt(mainFolder+'/'+'terminal_apoapsis_arr.txt',terminal_apoapsis_arr)
			np.savetxt(mainFolder+'/'+'terminal_periapsis_arr.txt',terminal_periapsis_arr)

			np.savetxt(mainFolder+'/'+'periapsis_raise_DV_arr.txt',periapsis_raise_DV_arr)
			np.savetxt(mainFolder+'/'+'apoapsis_raise_DV_arr.txt', apoapsis_raise_DV_arr)

			np.savetxt(mainFolder+'/'+'acc_net_g_max_arr.txt',acc_net_g_max_arr)
			np.savetxt(mainFolder+'/'+'q_stag_max_arr.txt',q_stag_max_arr)
			np.savetxt(mainFolder+'/'+'heatload_max_arr.txt',heatload_max_arr)

	def runMonteCarloD2(self, N, mainFolder):
		"""
		Run a Monte Carlo simulation for drag modulation
		aerocapture with new solver.

		Parameters
		--------
		N : int
			Number of trajectories
		mainFolder : str
			path where data is to be stored
		"""

		terminal_apoapsis_arr = np.zeros(N)
		terminal_periapsis_arr = np.zeros(N)
		periapsis_raise_DV_arr = np.zeros(N)
		apoapsis_raise_DV_arr = np.zeros(N)
		acc_net_g_max_arr = np.zeros(N)
		q_stag_max_arr = np.zeros(N)
		heatload_max_arr = np.zeros(N)

		h0_km = self.vehicleCopy.h0_km_ref
		theta0_deg = self.vehicleCopy.theta0_deg_ref
		phi0_deg = self.vehicleCopy.phi0_deg_ref
		v0_kms = self.vehicleCopy.v0_kms_ref
		psi0_deg = self.vehicleCopy.psi0_deg_ref
		drange0_km = self.vehicleCopy.drange0_km_ref
		heatLoad0 = self.vehicleCopy.heatLoad0_ref

		os.makedirs(mainFolder)

		for i in range(N):
			selected_atmfile = rd.choice(self.atmfiles)

			# selected_profile   = 65
			selected_profile = rd.randint(1, self.NMONTE)
			# selected_efpa     = -11.53
			selected_efpa = np.random.normal(self.nominalEFPA, self.EFPA_1sigma_value)
			# selected_atmSigma = +0.0
			selected_atmSigma = np.random.normal(0, 1)

			selected_LD = 0.0

			ATM_height, ATM_density_low, ATM_density_avg, ATM_density_high, \
			ATM_density_pert = self.planetObj.loadMonteCarloDensityFile2( \
				selected_atmfile, self.heightCol, \
				self.densLowCol, self.densAvgCol,
				self.densHighCol, self.densTotalCol, self.heightInKmFlag)

			self.planetObj.density_int = self.planetObj.loadAtmosphereModel5(ATM_height, \
																			 ATM_density_low, ATM_density_avg,
																			 ATM_density_high, \
																			 ATM_density_pert, selected_atmSigma,
																			 self.NPOS, \
																			 selected_profile)

			self.setInitialState(h0_km, theta0_deg, phi0_deg, v0_kms, \
								 psi0_deg, selected_efpa, drange0_km, heatLoad0)
			self.propogateGuidedEntryD2(self.timeStepEntry, self.timeStepExit, self.dt, self.maxTimeSecs)

			terminal_apoapsis = self.terminal_apoapsis
			apoapsis_error = self.apoapsis_perc_error
			terminal_periapsis = self.terminal_periapsis

			periapsis_raise_DV = self.periapsis_raise_DV
			apoapsis_raise_DV = self.apoapsis_raise_DV

			terminal_apoapsis_arr[i] = self.terminal_apoapsis
			terminal_periapsis_arr[i] = self.terminal_periapsis

			periapsis_raise_DV_arr[i] = self.periapsis_raise_DV
			apoapsis_raise_DV_arr[i] = self.apoapsis_raise_DV

			acc_net_g_max_arr[i] = max(self.acc_net_g_full)
			q_stag_max_arr[i] = max(self.q_stag_total_full)
			heatload_max_arr[i] = max(cumtrapz(self.q_stag_total_full, self.t_min_full*60, initial=0))/1e3

			print("RUN #: " + str(i + 1) + ", PROF: " + str(
				selected_atmfile) + ", SAMPLE #: " + str(selected_profile) + ", EFPA: " + str(
				'{:.2f}'.format(selected_efpa, 2)) + ", SIGMA: " + str(
				'{:.2f}'.format(selected_atmSigma, 2)) + ", APO : " + str('{:.2f}'.format(terminal_apoapsis, 2)))

			os.makedirs(mainFolder + '/' + '#' + str(i + 1))

			np.savetxt(mainFolder + '/' + '#' + str(i + 1) + '/' + 'atmfile.txt', np.array([selected_atmfile]),
					   fmt='%s')
			np.savetxt(mainFolder + '/' + '#' + str(i + 1) + '/' + 'profile.txt', np.array([selected_profile]))
			np.savetxt(mainFolder + '/' + '#' + str(i + 1) + '/' + 'efpa.txt', np.array([selected_efpa]))
			np.savetxt(mainFolder + '/' + '#' + str(i + 1) + '/' + 'atmSigma.txt', np.array([selected_atmSigma]))
			np.savetxt(mainFolder + '/' + '#' + str(i + 1) + '/' + 'LD.txt', np.array([selected_LD]))

			np.savetxt(mainFolder + '/' + '#' + str(i + 1) + '/' + 'terminal_apoapsis.txt',
					   np.array([terminal_apoapsis]))
			np.savetxt(mainFolder + '/' + '#' + str(i + 1) + '/' + 'apoapsis_error.txt', np.array([apoapsis_error]))
			np.savetxt(mainFolder + '/' + '#' + str(i + 1) + '/' + 'terminal_periapsis.txt',
					   np.array([terminal_periapsis]))

			np.savetxt(mainFolder + '/' + '#' + str(i + 1) + '/' + 'periapsis_raise_DV.txt',
					   np.array([periapsis_raise_DV]))
			np.savetxt(mainFolder + '/' + '#' + str(i + 1) + '/' + 'apoapsis_raise_DV.txt',
					   np.array([apoapsis_raise_DV]))

			np.savetxt(mainFolder + '/' + 'terminal_apoapsis_arr.txt', terminal_apoapsis_arr)
			np.savetxt(mainFolder + '/' + 'terminal_periapsis_arr.txt', terminal_periapsis_arr)

			np.savetxt(mainFolder + '/' + 'periapsis_raise_DV_arr.txt', periapsis_raise_DV_arr)
			np.savetxt(mainFolder + '/' + 'apoapsis_raise_DV_arr.txt', apoapsis_raise_DV_arr)

			np.savetxt(mainFolder + '/' + 'acc_net_g_max_arr.txt', acc_net_g_max_arr)
			np.savetxt(mainFolder + '/' + 'q_stag_max_arr.txt', q_stag_max_arr)
			np.savetxt(mainFolder + '/' + 'heatload_max_arr.txt', heatload_max_arr)


	def runMonteCarloD_Earth(self, N, mainFolder):
		"""
		Run a Monte Carlo simulation for drag modulation
		aerocapture. (Earth application)

		Parameters
		--------
		N : int
			Number of trajectories
		mainFolder : str
			path where data is to be stored
		"""


		terminal_apoapsis_arr = np.zeros(N)
		terminal_periapsis_arr= np.zeros(N)
		periapsis_raise_DV_arr= np.zeros(N)
		apoapsis_raise_DV_arr = np.zeros(N)
		acc_net_g_max_arr     = np.zeros(N)
		q_stag_max_arr        = np.zeros(N)
		heatload_max_arr      = np.zeros(N)

		h0_km      = self.vehicleCopy.h0_km_ref
		theta0_deg = self.vehicleCopy.theta0_deg_ref
		phi0_deg   = self.vehicleCopy.phi0_deg_ref
		v0_kms     = self.vehicleCopy.v0_kms_ref
		psi0_deg   = self.vehicleCopy.psi0_deg_ref
		drange0_km = self.vehicleCopy.drange0_km_ref
		heatLoad0  = self.vehicleCopy.heatLoad0_ref

		os.makedirs(mainFolder)
	
		
		for i in range(N):
			selected_atmfile  = rd.choice(self.atmfiles)
			
			#selected_profile   = 65
			selected_profile  = rd.randint(1,self.NMONTE)
			#selected_efpa     = -11.53
			selected_efpa     = np.random.normal(self.nominalEFPA, self.EFPA_1sigma_value)
			#selected_atmSigma = +0.0
			selected_atmSigma = np.random.normal(0,1)

			selected_LD       = 0.0

			ATM_height, ATM_density_low, ATM_density_avg, ATM_density_high, \
			ATM_density_pert = self.planetObj.loadMonteCarloDensityFile3(\
				               selected_atmfile, self.heightCol, \
				               self.densAvgCol,  self.densSD_percCol,\
				               self.densTotalCol, self.heightInKmFlag)
			
			self.planetObj.density_int = self.planetObj.loadAtmosphereModel5(ATM_height, \
										 ATM_density_low, ATM_density_avg, ATM_density_high, \
										 ATM_density_pert, selected_atmSigma, self.NPOS, \
										 selected_profile)

			
			self.setInitialState(h0_km,theta0_deg,phi0_deg,v0_kms,\
						         psi0_deg,selected_efpa,drange0_km,heatLoad0)
			self.propogateGuidedEntryD(self.timeStepEntry, self.timeStepExit,  self.dt, self.maxTimeSecs)

			terminal_apoapsis = self.terminal_apoapsis
			apoapsis_error    = self.apoapsis_perc_error
			terminal_periapsis= self.terminal_periapsis

			periapsis_raise_DV = self.periapsis_raise_DV
			apoapsis_raise_DV =  self.apoapsis_raise_DV


			terminal_apoapsis_arr[i] = self.terminal_apoapsis
			terminal_periapsis_arr[i]= self.terminal_periapsis

			periapsis_raise_DV_arr[i]= self.periapsis_raise_DV
			apoapsis_raise_DV_arr[i] = self.apoapsis_raise_DV

			acc_net_g_max_arr[i]     = max(self.acc_net_g_full)
			q_stag_max_arr[i]        = max(self.q_stag_total_full)
			heatload_max_arr[i]      = max(self.heatload_full)

			print("BATCH :"+str(mainFolder)+", RUN #: "+str(i+1)+", PROF: "+str(selected_atmfile)+", SAMPLE #: "+str(selected_profile)+", EFPA: "+str('{:.2f}'.format(selected_efpa,2))+", SIGMA: "+str('{:.2f}'.format(selected_atmSigma,2))+", APO : "+str('{:.2f}'.format(terminal_apoapsis,2)))


			os.makedirs(mainFolder+'/'+'#'+str(i+1))


			np.savetxt(mainFolder+'/'+'#'+str(i+1)+'/'+'atmfile.txt',np.array([selected_atmfile]), fmt='%s')
			np.savetxt(mainFolder+'/'+'#'+str(i+1)+'/'+'profile.txt',np.array([selected_profile]))
			np.savetxt(mainFolder+'/'+'#'+str(i+1)+'/'+'efpa.txt',np.array([selected_efpa]))
			np.savetxt(mainFolder+'/'+'#'+str(i+1)+'/'+'atmSigma.txt',np.array([selected_atmSigma]))
			np.savetxt(mainFolder+'/'+'#'+str(i+1)+'/'+'LD.txt',np.array([selected_LD]))

			np.savetxt(mainFolder+'/'+'#'+str(i+1)+'/'+'terminal_apoapsis.txt',np.array([terminal_apoapsis]))
			np.savetxt(mainFolder+'/'+'#'+str(i+1)+'/'+'apoapsis_error.txt',np.array([apoapsis_error]))
			np.savetxt(mainFolder+'/'+'#'+str(i+1)+'/'+'terminal_periapsis.txt',np.array([terminal_periapsis]))

			np.savetxt(mainFolder+'/'+'#'+str(i+1)+'/'+'periapsis_raise_DV.txt',np.array([periapsis_raise_DV]))
			np.savetxt(mainFolder+'/'+'#'+str(i+1)+'/'+'apoapsis_raise_DV.txt',np.array([apoapsis_raise_DV]))


			np.savetxt(mainFolder+'/'+'terminal_apoapsis_arr.txt',terminal_apoapsis_arr)
			np.savetxt(mainFolder+'/'+'terminal_periapsis_arr.txt',terminal_periapsis_arr)

			np.savetxt(mainFolder+'/'+'periapsis_raise_DV_arr.txt',periapsis_raise_DV_arr)
			np.savetxt(mainFolder+'/'+'apoapsis_raise_DV_arr.txt', apoapsis_raise_DV_arr)

			np.savetxt(mainFolder+'/'+'acc_net_g_max_arr.txt',acc_net_g_max_arr)
			np.savetxt(mainFolder+'/'+'q_stag_max_arr.txt',q_stag_max_arr)
			np.savetxt(mainFolder+'/'+'heatload_max_arr.txt',heatload_max_arr)

