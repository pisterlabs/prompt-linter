# -*- coding: utf-8 -*-
"""
Microscope control codes

@copyright, Ruizhe Lin and Peter Kner, University of Georgia, 2019
"""

datapath = r'C:/Users/Public/Documents/data'
basepath = r'C:/Users/Public/Documents/python_code/SIM_AO_p36'
import os,sys,time
try:
    sys.path.index(basepath)
except:
    sys.path.append(basepath)

import numpy as N
import tifffile as T
from daq_p35 import daq35 as daq
from pyAndor_p35 import pyAndor_P35_a
import pco
from pol_rotator_p35 import motorized_pol_v2 as mp2
from prior_p35 import zstageP35 as zstage
from prior_p35 import priorxyP35
# import filterWheel_v2 as filterWheel
import Coherent_P33 as Coherent_P33
import qxga_exec_p36 as qx
import attenuator as attu
from Lib64.asdk import DM
from getpass import getuser
from PyQt5.QtCore import QThread, pyqtSignal

FREQ = 10000 # for daq timing signals

TEMPERATURE_SETPOINT = -50
COOLER_MODE = 1 

class scope(QThread):
    
    update = pyqtSignal()
    update2 = pyqtSignal()

    def __init__(self,parent=None):
        super(scope, self).__init__()
        self.delay = 0.02 # delay in loops (StackExt)
        self.handleA = None
        self.handleB = None
        self.QUIT = False
        self.stackparams = {'Date/Time':0,'X':0,'Y':0,'Z1':0,'Z2':0,'Zstep':0,'Exposure(s)':0,'CCD Temperature':0,'Pixel Size(nm)':89,'CCD setting':'','User':''}
        # camera
        self.ccd = pyAndor_P35_a.ccd()
        self.ccd.CoolerON()
        self.ccd.SetCoolerMode(COOLER_MODE)
        self.ccd.SetTemperature(TEMPERATURE_SETPOINT)
        self.data = N.zeros(self.ccd.image_size, dtype=N.uint16)
        # pco camera
        self.cam = pco.Camera()
        self.number_of_images = 4
        self.cam.sdk.set_trigger_mode('external exposure control')
        # piezo
        self.zst = zstage.zstage()
        self.zst.setPositionf(50)
        self.zpos = 0
        # Filter wheel
        # self.fw = filterWheel.filter_wheel()
        # motorized polarizer
        print('Initializing Polarizer')
        self.pol = mp2.A8MRU()
        self.pol.MotorOn()
        self.pol.setSpeed(5000,1)
#        self.pol_array = N.array([0.,0.,0.,60.,60.,60.,30.,30.,30.]) #2D
        self.pol_array = N.array([52.,102.,0.]) #3D
        print(self.pol.getState())
        # priorxy
        self.prior = priorxyP35.prior()
        self.xpos = 0
        self.ypos = 0
        self.prior.limits=[-10000, 10000, -10000, 10000]
        #DM
        self.serialName = 'BAX228'
        self.dm = DM( self.serialName )
        # qxga
        qx.initiate()
        # qx.open_usb_port()
        self.l = 0
        self.k = 0
        # Coherent laser
        try:
            self.ll647 = Coherent_P33.obis('COM6')
            self.ll647.SetDigitalModMode()
            self.ll647.SetLaserOn()
            self.laser647 = True
        except:
            self.ll647 = None
            self.laser647 = False
            print('647nm Laser not on') #raise Exception('647nm Laser not on')
        try:
            self.ll561 = Coherent_P33.obis('COM8')
            self.ll561.SetDigitalModMode()
            self.ll561.SetLaserOn()
            self.laser561 = True
        except:
            self.ll561 = None
            self.laser561 = False
            print('561nm Laser not on') #raise Exception('647nm Laser not on')
        try:
            self.ll405 = Coherent_P33.obis('COM7')
            self.ll405.SetLaserOn()
            self.ll405.SetDigitalModMode()
            self.laser405 = True
        except:
            self.ll405 = None
            self.laser405 = False
            print('405nm Laser not on') #raise Exception('405nm Laser not on')
        #Attenuator
        self.att = attu.attenuate()
        #set save path        
        t=time.localtime()
        self.is_videomode = False
        dpth=str(int(t[0]*1e4+t[1]*1e2+t[2]))+'_'+getuser()
        self.path=os.path.join(datapath,dpth)
        try:
            os.mkdir(self.path)
        except:
            print('Directory already exists')
        # defaults
        self.coord=(200,400,200,400)
        self.Name='default'
        ###
        self.Name = 'default'
        self.coordVal = self.ccd.GetImageCoordinates()
        self.old_Cval = self.ccd.GetImageCoordinates()

    def __del__(self):
        if (self.handleA != None):
            daq.CCDTrig_close(self.handleA,self.handleB)
        x = self.dm.Reset()
        if (x==0):
            print('DM Reset')
        else:
            print('DM cannot reset')
        qx.close()
        self.ccd.ManualShutdown()
        if (self.laser647):
            self.ll647.SetLaserOff()
        if (self.laser561):
            self.ll561.SetLaserOff()
        self.cam.close()    
        
    def get_img(self):
        daq.CCDTrig_run(self.handleA,self.handleB)
        self.ccd.WaitForNewData()
        self.data[:,:] = self.ccd.images
        
    def dont_get_img(self):
        daq.CCDTrig_run(self.handleA,self.handleB)

    def open_Acq(self,exposure=0.1,emgain=200,Blaser=False,Rlaser=False,Ylaser=False,UVlaser=False,LED12=1,FTM=False,conv=False,ccd=True,trig=1):
        # setup camera
        self.ccd.SetTriggerMode(trig) #0:internal, 1:External
        self.ccd.SetShutterMode(2) #auto
        self.ccd.SetReadMode(4) #image
        self.ccd.SetADChannel(0) # 14-bit channel
        self.ccd.SetOutputAmplifier(int(conv))
        self.ccd.SetHSSpeed(0)
        self.ccd.SetEMCCDGain(emgain)
        if FTM: # Frame transfer mode
            self.ccd.SetExposureTime(0)
            self.ccd.SetFrameTransferMode(1)
            self.ccd.SetAcquisitionMode(7) # run until abort
        else:
            self.ccd.SetExposureTime(exposure)
            self.ccd.SetFrameTransferMode(0)
            self.ccd.SetAcquisitionMode(5) # run until abort
        q,w,e = self.ccd.GetAcquisitionTimings()
        # setup triggers
        if not self.handleA == None:
            daq.CCDTrig_close(self.handleA,self.handleB)
        digsig = self.getTiming(FREQ,1e3*exposure,1e3*exposure,1,Blaser,Rlaser,Ylaser,UVlaser,LED=LED12,CCD=ccd)
        b = daq.CCDTrig_open(FREQ,digsig)
        self.handleA = b[0]
        self.handleB = b[1]
        xs = self.ccd.image_size[0]
        ys = self.ccd.image_size[1]
        self.data = N.zeros((xs,ys), dtype=N.uint16)
        self.ccd.SetShutterMode(1)
        self.ccd.Acquire()
        print(q,w,e)
        return (q,w,e)

    def close_Acq(self):        
        self.ccd.AbortAcquisition()
        self.ccd.SetShutterMode(2)
        return True

    def run(self):
        self.is_videomode = True
        if (self.normal_mode==True):
            if (self.norecordimg==False):
                while (self.is_videomode):
                    self.get_img()
                    self.update.emit()
            else:
                 while (self.is_videomode):
                    self.dont_get_img()
                    self.update2.emit()
        else:
            while (self.is_videomode):
                phs = 5
                for m in range(3):
                    self.pol.MoveAbs(self.pol_array[m])
                    time.sleep(0.1)
                    for n in range(phs):
                        qx.selecteorder(15*self.l+m*phs+n)
                        qx.activate()
                        self.get_img()
                        self.update.emit()
                        qx.deactivate()
        self.ccd.AbortAcquisition()
        self.ccd.SetShutterMode(2)

############## External ######################################################
    def SHWFSExt(self):
        self.cam.record(number_of_images=self.number_of_images, mode='ring buffer')
        daq.CCDTrig_run(self.handleA,self.handleB)
        ca = self.cam.rec.get_status()
        img, meta = self.cam.image(-1)
        self.cam.stop()
        return img

    def TimeLapseExt(self, no=200, pol=0, verbose=True):
        pos = self.zst.getPosition()
        xs = self.ccd.image_size[0]
        ys = self.ccd.image_size[1]
        self.data = N.zeros((no,xs,ys), dtype=N.uint16)
        self.ccd.SetShutterMode(1)
        q = self.ccd.Acquire()
        time.sleep(0.01)
        self.zst.setPositionf(pos)
        self.pol.MoveAbs(pol)
        for p in range(no):
            self.zst.setPositionf(pos)
            time.sleep(0.01)
            daq.CCDTrig_run(self.handleA,self.handleB)
            q = self.ccd.WaitForNewData()
            print (p,q)
            self.data[p] = self.ccd.images
            time.sleep(0.01)
        self.ccd.AbortAcquisition()
        self.ccd.SetShutterMode(2)
        if verbose:
            T.imshow(self.data, vmin=self.data.min(), vmax=self.data.max())
        cur_pos = self.prior.getPosition()
        xx = cur_pos[0]
        yy = cur_pos[1]
        self.stackTags(xx,yy,pos,pos,zs=0.,function='Time-Lapse widefield',ps=89)
        return True

    def StackExt(self,start,stop,step=0.2,verbose=True):
        init_loc=self.zst.getPosition()
        no = int((stop-start)/step)+1
        pos = start
        xs = self.ccd.image_size[0]
        ys = self.ccd.image_size[1]
        self.data = N.zeros((no,xs,ys), N.uint16)
        self.ccd.SetShutterMode(1)
        q = self.ccd.Acquire()
        time.sleep(0.2) # was 0.05
        for p in range(no):
            self.zst.setPositionf(pos)
            daq.CCDTrig_run(self.handleA,self.handleB)
            q = self.ccd.WaitForNewData()
            print(p,q)
            self.data[p] = self.ccd.images
            pos += step
            time.sleep(self.delay)
        self.ccd.AbortAcquisition()
        self.ccd.SetShutterMode(2)
        if verbose:
            T.imshow(self.data, vmin=self.data.min(), vmax=self.data.max())
        cur_pos = self.prior.getPosition()
        xx = cur_pos[0]
        yy = cur_pos[1]
        self.stackTags(xx,yy,start,stop,step,function='Z-Stack Widefield',ps=89)
        self.zst.setPositionf(init_loc)
        return True
        
    def Stack_Patterns(self,start,stop,step=0.2, verbose=True):
        init_loc=self.zst.getPosition()
        rots = self.pol_array
        no = int((stop-start)/step)+1
        pos = start
        xs = self.ccd.image_size[0]
        ys = self.ccd.image_size[1]
        psz = 15
        phs = int(psz/3)
        self.data = N.zeros((psz*no,xs,ys), dtype=N.uint16)
        self.ccd.SetShutterMode(1)
        q = self.ccd.Acquire()
        time.sleep(0.05) # was 0.2,  changed 20141114
        for p in range(no):
            self.zst.setPositionf(pos)
            for w in range(3):
                self.pol.MoveAbs(rots[w])
                time.sleep(0.4)
                for m in range(phs):
                    qx.selecteorder(15*self.l+phs*w+m)
                    qx.activate()
                    time.sleep(0.02)
                    daq.CCDTrig_run(self.handleA,self.handleB)
                    q = self.ccd.WaitForNewData()
                    print (p,q)
                    self.data[psz*p + 5*w + m] = self.ccd.images
                    qx.deactivate()
                    time.sleep(0.02)
            pos += step
        self.ccd.AbortAcquisition()
        self.ccd.SetShutterMode(2)
        if verbose:
            T.imshow(self.data, vmin=self.data.min(), vmax=self.data.max())
        cur_pos = self.prior.getPosition()
        self.zst.setPositionf(init_loc)
        xx = cur_pos[0]
        yy = cur_pos[1]
        self.stackTags(xx,yy,start,stop,step,function='Z-Stack 3DSIM patterns',ps=89)
        return True
        
    def Stack_Sectioning(self,start,stop,step=0.2, verbose=True):
        init_loc=self.zst.getPosition()
        no = int((stop-start)/step)+1
        pos = start
        xs = self.ccd.image_size[0]
        ys = self.ccd.image_size[1]
        psz = 3
        self.data = N.zeros((psz*no,xs,ys), dtype=N.uint16)
        self.ccd.SetShutterMode(1)
        q = self.ccd.Acquire()
        self.pol.MoveAbs(0)
        time.sleep(0.3)
        for p in range(no):
            self.zst.setPositionf(pos)
            for m in range(psz):
                qx.selecteorder(30+m)
                qx.activate()
                time.sleep(0.02)
                daq.CCDTrig_run(self.handleA,self.handleB)
                q = self.ccd.WaitForNewData()
                print (p,q)
                self.data[psz*p + m] = self.ccd.images
                qx.deactivate()
                time.sleep(0.02)
            pos += step
        self.ccd.AbortAcquisition()
        self.ccd.SetShutterMode(2)
        if verbose:
            T.imshow(self.data, vmin=self.data.min(), vmax=self.data.max())
        cur_pos = self.prior.getPosition()
        self.zst.setPositionf(init_loc)
        xx = cur_pos[0]
        yy = cur_pos[1]
        self.stackTags(xx,yy,start,stop,step,function='Z-Stack OS patterns',ps=89)
        return True
        
    def Image_Patterns(self, angle=0, no=200, pol=0, verbose=True):
        pos = self.zst.getPosition()
        xs = self.ccd.image_size[0]
        ys = self.ccd.image_size[1]
        psz = 5
        self.data = N.zeros((psz*no,xs,ys), dtype=N.uint16)
        self.ccd.SetShutterMode(1)
        q = self.ccd.Acquire()
        time.sleep(0.01) # was 0.2,  changed 20141114
        self.zst.setPositionf(pos)
        for p in range(no):  
            for m in range(angle*5,psz+angle*5):
                self.pol.MoveAbs(pol)
                qx.selecteorder(15*self.l+m)
                qx.activate()
                daq.CCDTrig_run(self.handleA,self.handleB)
                q = self.ccd.WaitForNewData()
                print (p,q)
                self.data[psz*p+m%5] = self.ccd.images
                qx.deactivate()
                time.sleep(self.delay)
        self.ccd.AbortAcquisition()
        self.ccd.SetShutterMode(2)
        if verbose:
            T.imshow(self.data, vmin=self.data.min(), vmax=self.data.max())
        cur_pos = self.prior.getPosition()
        xx = cur_pos[0]
        yy = cur_pos[1]
        self.stackTags(xx,yy,z1=pos,z2=pos,zs=0.,function='Signle plane SIM patterns',ps=89)
        return True
    
    def singleSnapExt(self,verbose=True):
        pos = self.zst.getPosition()
        xs = self.ccd.image_size[0]
        ys = self.ccd.image_size[1]
        self.data = N.zeros((xs,ys), dtype=N.uint16)
        self.ccd.SetShutterMode(1)
        self.ccd.Acquire()
        time.sleep(0.2) # was 0.05
        daq.CCDTrig_run(self.handleA,self.handleB)
        self.ccd.WaitForNewData()
        self.data[:,:] = self.ccd.images
        self.ccd.AbortAcquisition()
        self.ccd.SetShutterMode(2)
        if verbose:
            T.imshow(self.data, vmin=self.data.min(), vmax=self.data.max())
        cur_pos = self.prior.getPosition()
        xx = cur_pos[0]
        yy = cur_pos[1]
        self.stackTags(xx,yy,z1=pos,z2=pos,zs=0.,function='Single snap',ps=89)
        return True
        
    def setCCD_Conv_ext(self,exposure=0.100,Blaser=False,Rlaser=False,Ylaser=False,UVlaser=False,LED12=1):
        self.is_videMode = False
        # setup camera
        self.ccd.SetTriggerMode(1) #0: internal
        self.ccd.SetShutterMode(2) #0: auto
        self.ccd.SetFrameTransferMode(0)
        self.ccd.SetAcquisitionMode(5) # run until abort
        self.ccd.SetReadMode(4) #image
        self.ccd.SetADChannel(0) # 14-bit channel
        # this sets camera to 1.8us vs speed, and 3MHz readout which doesn't have a lot of pattern
        # to get rid of patterns entirely in conventional mode, go to 1Mhz readout
        self.ccd.SetVSSpeed(3)
        self.ccd.SetHSSpeed(0)
        self.ccd.SetOutputAmplifier(1)
        self.ccd.SetExposureTime(exposure)
        q,w,e = self.ccd.GetAcquisitionTimings()
        # setup triggers
        if not self.handleA == None:
            daq.CCDTrig_close(self.handleA,self.handleB)
        digsig = self.getTiming(FREQ,1e3*exposure,1e3*exposure,1,Blaser,Rlaser,Ylaser,UVlaser,LED=LED12)
        b = daq.CCDTrig_open(FREQ,digsig)
        self.handleA = b[0]
        self.handleB = b[1]
        return (q,w,e)

    def setCCD_EM_ext(self,exposure=0.1,emgain=200,Blaser=False,Rlaser=False,Ylaser=False,UVlaser=False,LED12=1):
        self.is_videMode = False
        # setup camera
        self.ccd.SetTriggerMode(1) #0:internal, 1:External
        self.ccd.SetShutterMode(2) #0: auto
        self.ccd.SetFrameTransferMode(0)
        self.ccd.SetAcquisitionMode(5) # run until abort
        self.ccd.SetReadMode(4) #image
        self.ccd.SetADChannel(0) # 14-bit channel
        self.ccd.SetOutputAmplifier(0)
        self.ccd.SetHSSpeed(0)
        self.ccd.SetEMCCDGain(emgain)
        self.ccd.SetExposureTime(exposure)
        q,w,e = self.ccd.GetAcquisitionTimings()
        # setup triggers
        if not self.handleA.value == 0:
            daq.CCDTrig_close(self.handleA,self.handleB)
        digsig = self.getTiming(FREQ,1e3*exposure,1e3*exposure,1,Blaser,Rlaser,Ylaser,UVlaser,LED=LED12)
        b = daq.CCDTrig_open(FREQ,digsig)
        self.handleA = b[0]
        self.handleB = b[1]
        return (q,w,e)
    
    def setCMOS_ext(self,exposure,Blaser=False,Rlaser=False,Ylaser=False,UVlaser=False,LED12=1,CCD=False,CMOS=True):
        self.cam.set_exposure_time(exposure)
        # setup triggers
        if not self.handleA.value == 0:
            daq.CCDTrig_close(self.handleA,self.handleB)
        digsig = self.getTiming(FREQ,1e3*exposure,1e3*exposure,1,Blaser,Rlaser,Ylaser,UVlaser,LED=LED12,CCD=False,CMOS=True)
        b = daq.CCDTrig_open(FREQ,digsig)
        self.handleA = b[0]
        self.handleB = b[1]
        return True
        
    def getTiming(self,freq,exposure,pulse,delay,Blaser=False,Rlaser=False,Ylaser=False,UVlaser=False,LED=1,CCD=True,CMOS=False):
        ''' frequency is in Hertz, exposure and delay are in ms
            bit 1: CCD Trigger
            bit 2: LED pulse
            bit 3: Laser shutter
            bit 4: Red Laser
            bit 5: Yellow Laser
            bit 6: UV Laser
            bit 7: LED 1/2'''
        # laser shutter is active
        count = int(1.1*(exposure+delay)*0.001*freq)
        td = int(delay*0.001*freq)
        texp = int(exposure*0.001*freq)
        tpulse = int(pulse*0.001*freq)
        trigpulse = int(0.004*freq)
        p = range(count)
        Blaser_arr = N.zeros(count)
        led_arr = N.zeros(count)
        Rlaser_arr = N.zeros(count)
        Ylaser_arr = N.zeros(count)
        UVlaser_arr = N.zeros(count)
        if LED==1:
            LED12_arr = N.zeros(count)
        else:
            LED12_arr = N.ones(count)
        if Blaser:
            Blaser_arr = N.array([(i>td) & (i<(td+texp)) for i in p])
        elif Rlaser:
            Rlaser_arr = N.array([(i>td) & (i<(td+texp)) for i in p])
        elif Ylaser:
            Ylaser_arr = N.array([(i>td) & (i<(td+texp)) for i in p])
        elif UVlaser:
            UVlaser_arr = N.array([(i>td) & (i<(td+texp)) for i in p])
        else:
            led_arr = N.array([(i>td) & (i<(td+tpulse)) for i in p])
        if (CCD):
            ccdtrig = N.array([(i<=(trigpulse)) for i in p])
        else:
            ccdtrig = N.array([(0) for i in p])
        if (CMOS):
            cmostrig = N.array([(i<=(trigpulse)) for i in p])
        else:
            cmostrig = N.array([(0) for i in p])
        out = (ccdtrig + 2*led_arr + 4*Blaser_arr + 8*Rlaser_arr + 16*Ylaser_arr + 32*UVlaser_arr + 64*LED12_arr + 128*cmostrig).astype(N.int)
        return out

####### Internal Trigger ###################################################

    def setCCD_Conv_Int(self,exposure=0.100):
        self.ccd.SetAcquisitionMode(1) # single exposure
        self.ccd.SetOutputAmplifier(1)
        self.ccd.SetTriggerMode(0) #0: internal
        self.ccd.SetShutterMode(2) #0: auto
        self.ccd.SetExposureTime(exposure)
        q = self.ccd.GetAcquisitionTimings()
        return q

    def StackInt(self,start,stop,step=0.2):
        no = int((stop-start)/step)+1
        pos = start
        xs = self.ccd.image_size[0]
        ys = self.ccd.image_size[1]
        self.data = N.zeros((no,xs,ys), dtype=N.uint16)
        self.ccd.SetShutterMode(1)
        for p in range(no):
            self.zst.setPositionf(pos)
            q = self.ccd.Acquire()
            q = self.ccd.WaitForNewData()
            q = self.ccd.AbortAcquisition()
            self.data[p] = self.ccd.images
            pos += step
        self.ccd.AbortAcquisition()
        self.ccd.SetShutterMode(2)
        cur_pos = self.prior.getPosition()
        self.stackTags(cur_pos[0],cur_pos[1],start,stop,step,function='Z-Stack')
        T.imshow(self.data, vmin=self.data.min(), vmax=self.data.max())
        return True

###########################################################################

    def saveTifA(self,slideName='',comments='',Upload=False):
        t=time.localtime()
        x = N.array([1e4,1e2,1])
        t1 = int((t[0:3]*x).sum())
        t2 = int((t[3:6]*x).sum())
        if slideName=='':
            slideName=self.Name
        else:
            self.Name=slideName
        self.stackparams['Comments']=comments
        self.stackparams['Slide Name']=slideName
        fn = "%s-%s_%s_%s" %(t1,t2,slideName,comments)
        fn1 = os.path.join(self.path,fn+'.tif')
        fn2 = os.path.join(self.path,fn+'_ps.txt')
        T.imsave(fn1,self.data)
        self._SaveText(fn2)
        return fn

    def saveTiff(self,comments='',fn=None,Upload=False):
        if fn==None:
            return None
        t = fn.partition('.')
        fn1 = t[0]+'.tif'
        fn2 = t[0]+'_ps.txt'
        T.imsave(fn1,self.data)
        if comments:
            self.stackparams['Comments']=comments
        self._SaveText(fn2)
        return fn

    def _SaveText(self,fn=None):
        if fn==None:
            return False
        s = []
        for parts in self.stackparams:
            s.append('%s : %s \n' % (parts, self.stackparams[parts]))
        s.sort()
        fid = open(fn,'w')
        fid.writelines(s)
        fid.close()
        return True

    def stackTags(self,xx,yy,z1,z2,zs,function='',ps=89):
        '''Date/Time,X,Y,Z1,Z2,Zstep,Exposure(s),CCD Temperature,
        Pixel Size(nm),CCD setting',User
        '''
        self.stackparams.clear()
        self.stackparams['00 function']=function
        self.stackparams['01 Date/Time']=time.asctime()
        self.stackparams['05 X']=xx
        self.stackparams['06 Y']=yy
        self.stackparams['07 Z1']=z1
        self.stackparams['08 Z2']=z2
        self.stackparams['09 Zstep']=zs
        self.stackparams['02 Exposure(s)']=self.ccd.GetExposureTime()
        self.stackparams['03 CCD Temperature']=self.ccd.GetTemperature()
        self.stackparams['10 User']=getuser()
        self.stackparams['11 Coordinates']=self.ccd.GetImageCoordinates()
        self.stackparams['12 Pixel size']=ps
        ccdsett=self.ccd.SettingsString()
        i=13
        for item in ccdsett.splitlines():
            try:
                csi,csv=item.split(':')
                ncsi=str(i)+' '+csi
                self.stackparams[ncsi]=csv
            except:
                csi=0
            i+=1
        return True

    def setCoord(self,xs,xe,ys,ye):
        '''val=(xs,xe,ys,ye)'''
        oxs,oxe,oys,oye = self.coordVal
        self.old_Cval = (oxs,oxe,oys,oye)
        self.coordVal=(xs,xe,ys,ye)
        q=self.ccd.SetImageCoordinates(self.coordVal)
        return q

    def setCoord2(self,xs,ys,nn):
        '''Gets one corner (xs,ys) to make (nn x nn) image size'''
        oxs,oxe,oys,oye = self.coordVal
        self.old_Cval = (oxs,oxe,oys,oye)
        self.coordVal=(xs,xs+nn-1,ys,ys+nn-1)
        q=self.ccd.SetImageCoordinates(self.coordVal)
        return q

    def reSetCoord(self,mode = 0):
        '''mode 0: Full CCD:(1,512,1,512)
           mode 1: return to previous coordinates: old_Cval        
        '''
        if mode==0:
            xs,xe,ys,ye = (1,512,1,512)
        elif mode==1:
            xs,xe,ys,ye = self.old_Cval
        return self.setCoord(xs,xe,ys,ye)
    
