#!/usr/bin/env python3
import unittest
import numpy as np

# Python function to fit bcc-iron data
def py_fe_sqw(h, k, l, e, p):
    js = p[0]
    d = p[1]
    om = d + (8*js) * (1 - np.cos(np.pi * h) * np.cos(np.pi * k) * np.cos(np.pi * l))
    q2 = ((1/(2*2.87))**2) * (h**2 + k**2 + l**2)
    ff = 0.0706 * np.exp(-35.008*q2) + 0.3589 * np.exp(-15.358*q2) + 0.5819 * np.exp(-5.561*q2) - 0.0114
    return (ff**2) * (p[4]/np.pi) * (e / (1-np.exp(-11.602*e/p[3]))) * (4 * p[2] * om) / ((e**2 - om**2)**2 + 4*(p[2] * e)**2)
        

class PacePythonTest(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        from pace_neutrons import Matlab
        cls.m = Matlab()
        cls.horace_version = cls.m.horace_version()

    @classmethod
    def tearDownClass(cls):
        with open('success', 'w') as f:
            f.write('success')
    
    @classmethod
    def setup_fe_data(self):
        fe_proj = {'u':[1,1,0], 'v':[-1,1,0], 'type':'rrr'} #TODO: adjust all these directories
        self.w_fe = self.m.cut_sqw('demo/datafiles/fe_cut.sqw', fe_proj,
                                   [-3,0.05,3], [-1.05,-0.95], [-0.05,0.05], [70, 90])
        # Starting parameters for fit
        J = 35;     # Exchange interaction in meV
        D = 0;      # Single-ion anisotropy in meV
        gam = 30;   # Intrinsic linewidth in meV (inversely proportional to excitation lifetime)
        temp = 10;  # Sample measurement temperature in Kelvin
        amp = 300;  # Magnitude of the intensity of the excitation (arbitrary units)
        self.fe_parvec = [J, D, gam, temp, amp]

        # Define linear background function
        self.linear_bg = self.m.eval('@linear_bg')

    @classmethod
    def setup_fe_spinw(self):
        fe = self.m.spinw()
        fe.genlattice('lat_const', [2.87]*3, 'angled', [90, 90, 90], 'spgr', 'I m -3 m')
        fe.addatom('label', 'MFe3', 'r', [0, 0, 0], 'S', 5/2, 'color', 'gold')
        fe.gencoupling()
        fe.addmatrix('label', 'J1', 'value', 1, 'color', 'gray')
        fe.addmatrix('label', 'D', 'value', self.m.diag([0, 0, -1]), 'color', 'green')
        fe.addcoupling('mat', 'J1', 'bond', 1)
        fe.addaniso('D')
        fe.genmagstr('mode', 'direct', 'S', np.array([[0., 0., 1.], [0., 0., 1.]]).T)
        self.sw_fe = fe

    @classmethod
    def setup_euphonic(self):
        from euphonic import ForceConstants
        from euphonic_sqw_models import CoherentCrystal

        fc = ForceConstants.from_castep('demo/datafiles/quartz.castep_bin')
        self.euobj = CoherentCrystal(fc, debye_waller_grid=[6, 6, 6], temperature=100,
                                     negative_e=True, asr=True, chunk=10000, use_c=True)

        self.scalefac = 200
        self.effective_fwhm = 1
        self.intrinsic_fwhm = 0.1

        self.wsc = self.m.cut_sqw('demo/datafiles/quartz_cut.sqw', [-3.02, -2.98], [5, 0.5, 38])


    def test0_CutSqwDnd(self):
        proj = self.m.projaxes([1, 0, 0], [0, 1, 0], 'type', 'rrr')
        w1 = self.m.cut_sqw('demo/datafiles/pcsmo_cut1.sqw', proj,
                            [-1, 0.05, 1], [-1, 0.05, 1], [-10, 10], [10, 20], '-nopix')
        self.assertEqual(np.shape(w1.s), (41, 41))
    
    def test0_CutSqwSqw(self):
        proj = self.m.projaxes([1, 0, 0], [0, 1, 0], 'type', 'rrr')
        w2 = self.m.cut_sqw('demo/datafiles/pcsmo_cut2.sqw', proj,
                            [-1, 0.05, 1], [-0.2, 0.2], [-10, 10], [5, 1, 65])
        
        if self.horace_version.startswith("4"):
            self.assertEqual(w2.main_header.filename, 'pcsmo_cut2.sqw')
        else:
            self.assertEqual(w2.main_header['filename'], 'pcsmo_cut2.sqw')
        self.assertEqual(np.shape(w2.data.s), (41, 61))

        w3 = w2.cut([0.45, 0.55], [5, 1, 65], '-nopix')
        self.assertEqual(np.shape(w3.s), (61, 1))

    def test0_FeSetup(self):
        # Make a cut of the data
        self.setup_fe_data()
        np.testing.assert_allclose(self.w_fe.data.alatt, np.array([[2.87, 2.87, 2.87]]))
        self.assertEqual(np.shape(self.w_fe.data.s), (121, 1))

    def test1_FeSimPython(self):
        # Evaluate the mode on the cut with the starting parameters
        w_cal = self.m.sqw_eval(self.w_fe, py_fe_sqw, self.fe_parvec)
        self.assertEqual(np.shape(w_cal.data.s), np.shape(self.w_fe.data.s))

    def test1_FeFitPython(self):
        kk = self.m.multifit_sqw(self.w_fe)
        kk = kk.set_fun (py_fe_sqw, self.fe_parvec)
        kk = kk.set_free ([1, 0, 1, 0, 1])
        kk = kk.set_bfun (self.linear_bg, [0.1,0])
        kk = kk.set_bfree ([1,0])
        kk = kk.set_options ('list',2)

        # Run and time the fit
        self.m.tic()
        wfit, fitdata = kk.fit('comp')
        t_ana = self.m.toc()
        print(f'Time to run fit: {t_ana}s')
        self.assertEqual(np.shape(wfit['sum'].data.s), np.shape(self.w_fe.data.s))

    def test1_FeSpinWSetup(self):
        self.setup_fe_spinw()
        S_ref = np.array([[0,0], [0,0], [2.5, 2.5]])
        J, D = (self.fe_parvec[0], self.fe_parvec[1])
        mat_ref = np.array([np.eye(3), np.diag([0,0,-1.])])
        np.testing.assert_allclose(self.sw_fe.magstr()['S'], S_ref)
        np.testing.assert_allclose(self.sw_fe.matrix['mat'].T, mat_ref)
        
    def test2_FeSpinW(self):
        # Constant parameters for SpinW model
        # Note that we use the damped harmonic oscillator resolution model ('sho')
        cpars = ['mat', ['J1', 'D(3,3)'], 'hermit', False, 'optmem', 1,
                 'useFast', True, 'resfun', 'sho', 'formfact', True]
        
        kk = self.m.multifit_sqw(self.w_fe)
        kk = kk.set_fun (self.sw_fe.horace_sqw, [self.fe_parvec]+cpars)
        kk = kk.set_free ([1, 0, 1, 0, 1])
        kk = kk.set_bfun (self.linear_bg, [0.1,0])
        kk = kk.set_bfree ([1,0])
        kk = kk.set_options ('list',2)
        
        # Time a single iteration
        self.m.tic()
        wsim = kk.simulate('comp')
        t_spinw_single = self.m.toc()
        print(f'Time to evaluate a single iteration: {t_spinw_single}s')
        self.assertEqual(np.shape(wsim['sum'].data.s), np.shape(self.w_fe.data.s))
 
    def test2_FeBrille(self):
        # Run through it again using Brille
        cpars = ['mat', ['J1', 'D(3,3)'], 'hermit', False, 'optmem', 1,
                 'useFast', False, 'resfun', 'sho', 'formfact', True, 'use_brille', True]

        kk = self.m.multifit_sqw(self.w_fe)
        kk = kk.set_fun (self.sw_fe.horace_sqw, [self.fe_parvec]+cpars)
        kk = kk.set_free ([1, 0, 1, 0, 1])
        kk = kk.set_bfun (self.linear_bg, [0.1,0])
        kk = kk.set_bfree ([1,0])
        kk = kk.set_options ('list',2)

        # Time a single iteration
        self.m.tic()
        wsim = kk.simulate('comp')
        t_spinw_fill = self.m.toc()
        print(f'Time to fill Brille grid: {t_spinw_fill}s')
        self.assertEqual(np.shape(wsim['sum'].data.s), np.shape(self.w_fe.data.s))

    def test2_EuphonicCalc(self):
        self.setup_euphonic()
        # Calculate spectra with simple energy convolution (fixed width Gaussian)
        wsim = self.m.disp2sqw_eval(self.wsc, self.euobj.horace_disp, 
                                    (self.scalefac), self.effective_fwhm)
        self.assertEqual(np.shape(wsim.data.s), np.shape(self.wsc.data.s))

    def test3_EuphonicResolution(self):
        # Calculate spectra with full instrument resolution convolution
        xgeom = [0,0,1]
        ygeom = [0,1,0]
        shape = 'cuboid'
        shape_pars = [0.01,0.05,0.01]
        if self.horace_version.startswith("4"):
            sample = self.m.IX_sample(xgeom= xgeom, ygeom= ygeom, shape=shape, 
                                      ps= shape_pars)
        else:
            sample = self.m.IX_sample(xgeom, ygeom, shape, shape_pars, '-single_crystal', True)
        wsc = self.m.set_sample(self.wsc, sample)
        ei = 40
        freq = 400
        chopper = 'g'
        wsc = self.m.set_instrument(wsc, self.m.merlin_instrument(ei, freq, chopper))
        disp2sqwfun = self.m.eval('@disp2sqw')
        kk = self.m.tobyfit(wsc)
        kk = kk.set_fun(disp2sqwfun, [self.euobj.horace_disp,
                                      [self.scalefac], self.intrinsic_fwhm])
        kk = kk.set_mc_points(5)
        wtoby = kk.simulate()
        self.assertEqual(np.shape(wtoby.data.s), np.shape(self.wsc.data.s))


if __name__ == '__main__':
    unittest.main()
