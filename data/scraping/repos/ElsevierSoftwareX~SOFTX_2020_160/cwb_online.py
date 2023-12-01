#!/usr/bin/env python
    
# Copyright (C) 2019 Marco Drago, Serena Vinciguerra
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import commands, os, sys, glob
import cWB_conf

class cwb_create:
  def __init__(self):
    self.add_prod_plugin=False
    self.final_plugin_string=""
    self.add_cuts=""
    #self.web_pages=["run","week","day","mid","hour"]
    #self.name_pages=["the_whole_run","last_week","last_day","last_12_hours","last_hour"]
    self.web_pages=["week","day"]
    self.name_pages=["last_week","last_day"]
    try:
     self.prod=cWB_conf.production_dir    
     self.copy=True
    except:
     self.copy=False

  def make_checks(self):    
    #check that scracth is high enough
    TDSize=12 # time-delay filter size (max 20), from TDSize in $CWB_PARAMETERS_FILE
    SAMP_RATE = 2**14
    
    if cWB_conf.job_offset<=1.5*TDSize/(SAMP_RATE>>(cWB_conf.levelR+cWB_conf.l_high)):
            print "Error - segEdge must be > 1.5x the length for time delay amplitudes!!! Increase job_offset or decrease levelR + l_high"
     	    print "job_offset: ",cWB_conf.job_offset,"levelR ",cWB_conf.levelR,"l_high: ",cWB_conf.l_high	
            exit()
    
    # check existing file plugin 
    if hasattr(cWB_conf, 'prod_plugins'):
      for prod_file in cWB_conf.prod_plugins:
        if not os.path.exists(prod_file):
        	print "Error - Plugin not found: change \"prod_plugins\" %s entry in cWB_conf.py"%(prod_file)
        	sys.exit(1)
    else:
      print "No plugin specified"
    
    # check existing lag
    if hasattr(cWB_conf, 'bkg_laglist'):
      if not os.path.exists(cWB_conf.bkg_laglist):
        print "Error - Bkg lag list %s not found: change \"bkg_laglist\" in cWB_conf.py"%(cWB_conf.bkg_laglist)
        sys.exit(1)
    else:
      print "No Lags specified"

    # check existing superlag
    if hasattr(cWB_conf, 'bkg_superlaglist'):
      if not os.path.exists(cWB_conf.bkg_superlaglist):
        print "Error - Bkg superlag list %s not found: change \"bkg_superlaglist\" in cWB_conf.py"%(cWB_conf.bkg_superlaglist)
        sys.exit(1)
    else:
      print "No Superlags specified"

    
    # check existing file Cut
    if hasattr(cWB_conf, 'Cuts_file'):
      if not os.path.exists(cWB_conf.Cuts_file):
        print "Error - Cut file %s not found: change \"Cuts_file\" in cWB_conf.py"%(cWB_conf.Cuts_file)
        sys.exit(1)
    else:
      print "No Cuts specified"
    
    # check if certificates exist
    if hasattr(cWB_conf, 'gracedb_group'):
      #export X509_USER_CERT=${HOME}/.certificates/waveburst_cwb.ligo.caltech.edu.cert.pem
      try:
        X509_USER_CERT=os.environ['X509_USER_CERT']
        if (len(X509_USER_CERT)==0):
          print "Error - X509_USER_CERT defined void"
          exit()
        else:
          if not os.path.exists(X509_USER_CERT):
            print "Error - X509_USER_CERT file %s not found"%X509_USER_CERT
            exit()
      except:
        print "Error - X509_USER_CERT not defined"
        exit()
      try:
        X509_USER_KEY=os.environ['X509_USER_KEY']
        if (len(X509_USER_KEY)==0):
          print "Error - X509_USER_KEY defined void"
          sys.exit(1)
        else:
          if not os.path.exists(X509_USER_KEY):
            print "Error - X509_USER_KEY file %s not found"%X509_USER_KEY
            sys.exit(1)
      except:
        print "Error - X509_USER_KEY not defined"
        sys.exit(1)
      
#export X509_USER_KEY=${HOME}/.certificates/waveburst.key.pem 

    # check if overwriting directories
    if os.path.isdir(cWB_conf.web_dir):
    	print "Error - web-directory %s already exists, change \"web_dir\" in cWB_conf.py"%(cWB_conf.web_dir)
    	exit()
    if os.path.isdir(cWB_conf.online_dir):
    	print "Error - online-directory %s already exists, change \"online_dir\" in cWB_conf.py"%(cWB_conf.online_dir)
    	exit()
    
  def make_dirs(self):    
    commands.getstatusoutput("mkdir -p %s"%(cWB_conf.run_dir))
    commands.getstatusoutput("mkdir -p %s/%s"%(cWB_conf.run_dir,cWB_conf.summaries_dir))
    commands.getstatusoutput("mkdir -p %s"%(cWB_conf.bkg_run_dir))
    commands.getstatusoutput("mkdir -p %s/%s"%(cWB_conf.bkg_run_dir,cWB_conf.postprod_dir))
    commands.getstatusoutput("mkdir -p %s"%(cWB_conf.web_dir))
    commands.getstatusoutput("mkdir -p %s"%(cWB_conf.config_dir))
    commands.getstatusoutput("mkdir -p %s/%s"%(cWB_conf.run_dir,cWB_conf.jobs_dir))

    if (self.copy):
      commands.getstatusoutput("ln -s %s/%s %s/%s"%(cWB_conf.run_dir.replace(cWB_conf.online_dir,cWB_conf.production_dir),cWB_conf.seg_dir,cWB_conf.run_dir,cWB_conf.seg_dir))
      commands.getstatusoutput("ln -s %s/%s %s/%s"%(cWB_conf.bkg_run_dir.replace(cWB_conf.online_dir,cWB_conf.production_dir),cWB_conf.seg_dir,cWB_conf.bkg_run_dir,cWB_conf.seg_dir))
      commands.getstatusoutput("ln -s %s/%s %s/%s"%(cWB_conf.bkg_run_dir.replace(cWB_conf.online_dir,cWB_conf.production_dir),cWB_conf.jobs_dir,cWB_conf.bkg_run_dir,cWB_conf.jobs_dir))
      for n in range(1,cWB_conf.bkg_njobs):
        commands.getstatusoutput("ln -s %s/%s_%i %s/%s_%i"%(cWB_conf.bkg_run_dir.replace(cWB_conf.online_dir,cWB_conf.production_dir),cWB_conf.jobs_dir,n,cWB_conf.bkg_run_dir,cWB_conf.jobs_dir,n))
    else:
      commands.getstatusoutput("mkdir -p %s/%s"%(cWB_conf.run_dir,cWB_conf.seg_dir))
      commands.getstatusoutput("mkdir -p %s/%s"%(cWB_conf.bkg_run_dir,cWB_conf.seg_dir))
      commands.getstatusoutput("mkdir -p %s/%s"%(cWB_conf.bkg_run_dir,cWB_conf.jobs_dir))
      

    commands.getstatusoutput("mkdir %s/python"%(cWB_conf.online_dir))
    commands.getstatusoutput("cp %s/python/*.py %s/python/."%(os.environ['CWB_ONLINE'],cWB_conf.online_dir))
    commands.getstatusoutput("cp cWB_conf.py %s/python/"%(cWB_conf.online_dir))
    commands.getstatusoutput("cp %s/html/* %s/."%(os.environ['CWB_ONLINE'],cWB_conf.web_dir))

    commands.getstatusoutput("echo > %s/%s/%s"%(cWB_conf.bkg_run_dir,cWB_conf.seg_dir,cWB_conf.considered_segments_file))
    commands.getstatusoutput("echo > %s/%s/%s"%(cWB_conf.bkg_run_dir,cWB_conf.seg_dir,cWB_conf.processed_segments_file))
    commands.getstatusoutput("echo > %s/%s/%s"%(cWB_conf.bkg_run_dir,cWB_conf.seg_dir,cWB_conf.running_segments_file))
    commands.getstatusoutput("echo > %s/%s/%s"%(cWB_conf.bkg_run_dir,cWB_conf.seg_dir,cWB_conf.missing_segments_file))
    commands.getstatusoutput("echo > %s/%s/%s"%(cWB_conf.bkg_run_dir,cWB_conf.seg_dir,cWB_conf.run_segments_file))
    commands.getstatusoutput("echo > %s/%s/%s"%(cWB_conf.bkg_run_dir,cWB_conf.seg_dir,cWB_conf.job_segments_file))
    
    commands.getstatusoutput("ln -s %s/%s %s/."%(cWB_conf.run_dir,cWB_conf.jobs_dir,cWB_conf.web_dir))
    commands.getstatusoutput("ln -s %s/%s %s/."%(cWB_conf.run_dir,cWB_conf.summaries_dir,cWB_conf.web_dir))
    commands.getstatusoutput("ln -s %s/%s %s/."%(cWB_conf.bkg_run_dir,cWB_conf.postprod_dir,cWB_conf.web_dir))
    
  def cuts_file(self):
    try:
      Cuts_lines=open("%s"%(cWB_conf.Cuts_file)).readlines()
      Cuts_file="%s/Cuts.hh"%(cWB_conf.config_dir)#,cWB_conf.Cuts_file.split("/")[len(cWB_conf.Cuts_file.split("/"))-1])
      #com="cp %s %s"%(cWB_conf.Cuts_file,Cuts_file)
      #commands.getstatusoutput(com)
      f=open("%s"%(Cuts_file),"w")
      for line in Cuts_lines:
        print >>f,"%s"%(line.replace("\n",""))
      if (len(cWB_conf.Cuts_list)>1):
        print >>f,"""\nTCut OR_cut      = TCut("OR_cut",(%s).GetTitle());"""%("||".join(cWB_conf.Cuts_list))
      f.close()
      #com="./mkhtml.csh %s"%(Cuts_file)
      com="%s/scripts/cwb_mkhtml.csh %s"%(os.environ['HOME_CWB'],Cuts_file)
      print com
      commands.getstatusoutput(com)
      com="mv %s/Cuts/Cuts.hh.html %s/.;rm -rf %s/Cuts"%(cWB_conf.config_dir,cWB_conf.web_dir,cWB_conf.config_dir)
      commands.getstatusoutput(com)
      self.add_cuts="""#include "%s"
  
  """%(Cuts_file)
    except:
      pass#self.add_cuts=""

  def plugins(self):
    try:
      nplugins=len(cWB_conf.prod_plugins)
      self.add_prod_plugin=True
      if (nplugins==1):
        final_plugin="%s/%s"%(cWB_conf.config_dir,cWB_conf.prod_plugins[0].split("/")[len(cWB_conf.prod_plugins[0].split("/"))-1])
        com="cp %s %s"%(cWB_conf.prod_plugins[0],final_plugin)
      else:
        final_plugin="%s/prod_plugin.C"%(cWB_conf.config_dir)
        com="%s/scripts/cwb_mplugin.csh %s %s"%(os.environ['HOME_CWB'],final_plugin," ".join(cWB_conf.prod_plugins))
      #print commands.getstatusoutput(com)
      com="%s;root -q -b -l %s+"%(com,final_plugin)
      #print "Please compile plugin: %s"%com
      print "Compiling plugin: %s"%com
      commands.getstatusoutput(com)
      self.final_plugin_string="""   plugin = TMacro("%s");        // Macro source
  plugin.SetTitle("%s");"""%(final_plugin,final_plugin.replace(".C","_C.so"))
    except:
      pass#self.add_prod_plugin=False
   
#def phone_par(self): 
    #try:
    #  f=open("%s"%(cWB_conf.phone_par),"w")
    #  print >>f,"A transient candidate from coherent Waveburst of significance less than %g is being passed for follow-up. Please check your email immediately and alert the others."%(cWB_conf.phone_alert)
    #  f.close()
    #except:
    #  print "not sending mail to phone"
    
  def user_parameters(self):    
    f=open("%s"%(cWB_conf.zerolag_par),"w")
    
    ######### from 24/7/2018 in cwb_conf.py
    #  strcpy(analysis,"%s");
    
    #  nIFO = %i;
    #  cfg_search = '%s';
    #  optim=%s;\n"""%(cWB_conf.version_wat,len(cWB_conf.ifos),cWB_conf.search,cWB_conf.optim)
    ############################
    print >>f,"""{"""
    for i in range(len(cWB_conf.ifos)):
        print >>f,"""  strcpy(ifo[%i],"%s");"""%(i,cWB_conf.ifos[i])
    print >>f,"""  strcpy(refIFO,"%s");"""%(cWB_conf.ifos[0])
    print >>f,"""
  //lags
  lagSize = 1;
  lagOff = 0;
  lagMax = 0;

  //jobs
  segLen = %i;
  segMLS = %i;
  segEdge = %i;
  segTHR = 0;"""%(cWB_conf.seg_duration,cWB_conf.seg_duration,cWB_conf.job_offset)
    
    #for line in lines:
    #    print >>f,line
    print >>f,"\n%s"%(cWB_conf.cwb_par)
    for i in range(len(cWB_conf.ifos)):
        print >>f,"""  strcpy(channelNamesRaw[%i],"%s");"""%(i,cWB_conf.channelname[cWB_conf.ifos[i]])
    for i in range(len(cWB_conf.ifos)):
        print >>f,"""  strcpy(frFiles[%i],"input/%s.frames");"""%(i,cWB_conf.ifos[i])
    
    if (self.add_prod_plugin==True):
        print >>f,self.final_plugin_string#"""   plugin = TMacro("%s");        // Macro source
#      plugin.SetTitle("%s");"""%(final_plugin,final_plugin.replace(".C","_C.so"))
    
    print >>f,"""
  nDQF=%i;
  dqfile dqf[%i]={"""%(2*len(cWB_conf.ifos),2*len(cWB_conf.ifos))
    for i in range(len(cWB_conf.ifos)):
        print >>f,"""                     {"%s" ,"input/burst.in",           CWB_CAT1, 0., false, false},"""%(cWB_conf.ifos[i])
    for i in range(len(cWB_conf.ifos)):
        print >>f,"""                     {"%s" ,"input/%s_cat2.in",           CWB_CAT2, 0., false, false},"""%(cWB_conf.ifos[i],cWB_conf.ifos[i])
        #print >>f,"""                     {"%s" ,"input/burst.in",           CWB_CAT2, 0., false, false},"""%(cWB_conf.ifos[i])
    print >>f,"""                   };
  for(int i=0;i<nDQF;i++) DQF[i]=dqf[i];

  strcpy(data_dir,"OUTPUT");

  online = true;
  frRetryTime=0;
  dump = true;
  cedDump = true;
  cedRHO = %f;

}"""%(cWB_conf.th_rho_lum)
    f.close()
    
  def pe_parameters(self):    
    try:
      pe_plugin="%s/%s"%(cWB_conf.config_dir,cWB_conf.pe_plugin.split("/")[len(cWB_conf.pe_plugin.split("/"))-1])
      add_pe_plugin=True
      com="cp %s %s"%(cWB_conf.pe_plugin,pe_plugin)
      commands.getstatusoutput(com)
      com="root -b -l %s+"%(pe_plugin)
      print "Please compile plugin: %s"%com
      #commands.getstatusoutput(com)
    except:
      add_pe_plugin=False
    
    try:
        f=open("%s"%(cWB_conf.pe_par),"w")
    
    ########from 24/07/2018 in cWB_conf.py
    #  strcpy(analysis,"%s");
    
    #  nIFO = %i;
    #  cfg_search = '%s';
    #  optim=%s;\n"""%(cWB_conf.version_wat,len(cWB_conf.ifos),cWB_conf.search,cWB_conf.optim)
    #########################
        print >>f,"""{"""
    
        for i in range(len(cWB_conf.ifos)):
            print >>f,"""  strcpy(ifo[%i],"%s");"""%(i,cWB_conf.ifos[i])
        print >>f,"""  strcpy(refIFO,"%s");"""%(cWB_conf.ifos[0])
        print >>f,"""
  //lags
  lagSize = 1;
  lagOff = 0;
  lagMax = 0;

  //jobs
  segLen = %i;
  segMLS = %i;
  segEdge = %i;
  segTHR = 0;"""%(cWB_conf.seg_duration,cWB_conf.seg_duration,cWB_conf.job_offset)
    
    #for line in lines:
    #    print >>f,line
        print >>f,"\n%s"%(cWB_conf.cwb_par)
        for i in range(len(cWB_conf.ifos)):
            print >>f,"""  strcpy(channelNamesRaw[%i],"%s");"""%(i,cWB_conf.channelname[cWB_conf.ifos[i]])
        for i in range(len(cWB_conf.ifos)):
            print >>f,"""  strcpy(frFiles[%i],"input/%s_scratch.frames");"""%(i,cWB_conf.ifos[i])
    
        if (add_pe_plugin==True):
            print >>f,"""   plugin = TMacro("%s");        // Macro source
  plugin.SetTitle("%s");"""%(pe_plugin,pe_plugin.replace(".C","_C.so"))
    
        print >>f,"""
  nDQF=%i;
  dqfile dqf[%i]={"""%(2*len(cWB_conf.ifos),2*len(cWB_conf.ifos))
        for i in range(len(cWB_conf.ifos)):
            print >>f,"""                      {"%s" ,"input/burst.in",           CWB_CAT1, 0., false, false},"""%(cWB_conf.ifos[i])
            #print >>f,"""                      {"%s" ,"input/burst.in",           CWB_CAT2, 0., false, false},"""%(cWB_conf.ifos[i])
        for i in range(len(cWB_conf.ifos)):
            print >>f,"""                      {"%s" ,"input/%s_cat2.in",           CWB_CAT2, 0., false, false},"""%(cWB_conf.ifos[i],cWB_conf.ifos[i])
        print >>f,"""                   };
  for(int i=0;i<nDQF;i++) DQF[i]=dqf[i];

  strcpy(data_dir,"OUTPUT_PE");
  strcpy(tmp_dir,"tmp_pe");
  cedRHO = 1.000000;

  online = true;
  frRetryTime=0;
  dump = false;

}"""
        f.close()
    except:
        print "no parameter estimation"
    
  def bkg_parameters(self):    
    for l in range(0,3): 
        superlag_string="""
  //super lags
  slagSize   = %i;
  slagMin    = 0;
  slagMax    = %i;
  slagOff    = 0;
"""%(cWB_conf.bkg_njobs+1,cWB_conf.bkg_njobs)
    
        tmpfile=cWB_conf.bkg_par
        lagsize="%i"%cWB_conf.bkg_nlags
        lagoff="1";
        if (l==1):
           tmpfile=tmpfile.replace(".C","_split.C")
           superlag_string=""
        if (l==2):
           tmpfile="user_parameters.C"
           try:
             superlag_string="""%s

  slagFile = new char[1024];
  strcpy(slagFile,"%s");
             """%(superlag_string,cWB_conf.bkg_superlaglist)
           except:
             superlag_string="%s"%(superlag_string)
    
        f=open("%s"%(tmpfile),"w")
    
    ####### from 24/07/2018 in cWB_conf
    #  strcpy(analysis,"%s");
    #
    #  nIFO = %i;
    #  cfg_search = '%s';
    #  optim=%s;\n"""%(cWB_conf.version_wat,len(cWB_conf.ifos),cWB_conf.search,cWB_conf.optim)
    #########################################
        print >>f,"""{"""
        for i in range(len(cWB_conf.ifos)):
            print >>f,"""  strcpy(ifo[%i],"%s");"""%(i,cWB_conf.ifos[i])
        print >>f,"""  strcpy(refIFO,"%s");"""%(cWB_conf.ifos[0])
        print >>f,"""  //lags
  lagSize    = %s;
  lagStep    = 1.;
  lagOff     = %s;
  lagMax     = 0;
  %s
  %s
  //jobs
  segLen = %i;
  segMLS = %i;
  segEdge = %i;
  segTHR = 0;"""%(lagsize,lagoff,self.lag_string,superlag_string,cWB_conf.bkg_job_duration,cWB_conf.bkg_job_minimum,cWB_conf.job_offset)
    
        if (l==1):
            print >>f,"""  
  sprintf(data_label,"%s_%i",data_label,(int)dataShift[1]);
  TString data_Shift=TString(gSystem->Getenv(\"Slag_datashift\"));
  TObjArray*  bitoken   = data_Shift.Tokenize(TString(','));"""
            print >>f,"""
  TObjString* itok[%i];
  TString sitok[%i];"""%(len(cWB_conf.ifos),len(cWB_conf.ifos))
            for i in range(len(cWB_conf.ifos)):
               print >>f,"""
  itok[%i] = (TObjString*)bitoken->At(%i);
  sitok[%i] = itok[%i]->GetString();
  dataShift[%i] = sitok[%i].Atoi();"""%(i,i,i,i,i,i)
     
    
    #for line in lines:
    #    print >>f,line
        print >>f,"\n%s"%(cWB_conf.cwb_par)
        for i in range(len(cWB_conf.ifos)):
            print >>f,"""  strcpy(channelNamesRaw[%i],"%s");"""%(i,cWB_conf.channelname[cWB_conf.ifos[i]])
        for i in range(len(cWB_conf.ifos)):
            print >>f,"""  strcpy(frFiles[%i],"input/%s.frames");"""%(i,cWB_conf.ifos[i])
    
        if (self.add_prod_plugin==True and l<2):
            print >>f,self.final_plugin_string#"""   plugin = TMacro("%s");        // Macro source
#      plugin.SetTitle("%s");"""%(final_plugin,final_plugin.replace(".C","_C.so"))
    
        print >>f,"""
  nDQF=%i;
  dqfile dqf[%i]={"""%(len(cWB_conf.ifos),len(cWB_conf.ifos))
        for i in range(len(cWB_conf.ifos)):
            print >>f,"""                     {"%s" ,"input/%s_burst.in",           CWB_CAT1, dataShift[%i], false, false},"""%(cWB_conf.ifos[i],cWB_conf.ifos[i],i)
            #print >>f,"""                     {"%s" ,"input/%s_burst.in",           CWB_CAT2, 0., false, false},"""%(cWB_conf.ifos[i],cWB_conf.ifos[i])
        if (os.environ['SITE_CLUSTER']=="CASCINA"):
          nodedir_sub="sprintf(nodedir,\"%s/tmp\",gSystem->WorkingDirectory());"
        else:
          nodedir_sub=""
        print >>f,"""                   };
  for(int i=0;i<nDQF;i++) DQF[i]=dqf[i];

  //strcpy(data_dir,"OUTPUT");

  online = true;
  //frRetryTime=0;
  //dump = false;
  %s
}"""%(nodedir_sub)
        f.close()
    
  def user_pparameters(self):    
    f=open("%s"%(cWB_conf.pp_par),"w")
    print >>f,"""#define RUN_LABEL "%s"

//PUT VETO DEFINE HERE
 
{
  %s
  T_cor      = %f;       // cc cut
  T_cut      = 0.0;        // rho high frequency cut

  hours      = 1;         // bin size in hours for rate vs time plot

  pp_irho = %i;
  pp_inetcc =  %i;
  pp_rho_max = 10;
  pp_rho_min = 5;

  pp_batch = true;

  pp_jet_benckmark = -1;
  pp_mem_benckmark = -1;

//PUT VETO FILES HERE

}"""%(cWB_conf.title,self.add_cuts,cWB_conf.th_cc,cWB_conf.id_rho,cWB_conf.id_cc)
    f.close()
    
  def lag_file(self):    
    if hasattr(cWB_conf, 'bkg_laglist'):
      lag_file="%s/laglist.txt"%(cWB_conf.config_dir)
      commands.getstatusoutput("cp %s %s/laglist.txt"%(cWB_conf.bkg_laglist,cWB_conf.config_dir))
      print "cp %s %s/laglist.txt"%(cWB_conf.bkg_laglist,cWB_conf.config_dir)
      self.lag_string="""lagFile = new char[1024];
  strcpy(lagFile,"%s");"""%(lag_file)
    else:
      self.lag_string=""
    
  def superlag_file(self):    
    #create superlag file
    commands.getstatusoutput("mkdir -p config input report/dump")
    commands.getstatusoutput("mv user_parameters.C config/.")
    for i in range(len(cWB_conf.ifos)):
       f=open("input/%s_burst.in"%(cWB_conf.ifos[i]),"w")
       print >>f,"0 %i"%(cWB_conf.bkg_job_duration*(cWB_conf.bkg_njobs+2))
       f.close() 
       f=open("input/%s_cat2.in"%(cWB_conf.ifos[i]),"w")
       print >>f,"0 %i"%(cWB_conf.bkg_job_duration*(cWB_conf.bkg_njobs+2))
       f.close() 
    commands.getstatusoutput("%s/scripts/cwb_dump.csh slag"%(os.environ['HOME_CWB']))
    commands.getstatusoutput("cp report/dump/tmp_ONLINE.slag %s/superlaglist.txt"%(cWB_conf.config_dir))
    
  def create_crontab(self):    
    try:
      dir_for_logfiles="%s"%(cWB_conf.log_dir)
      commands.getstatusoutput("mkdir -p %s"%(dir_for_logfiles))
      commands.getstatusoutput("ln -s %s %s/log"%(dir_for_logfiles,cWB_conf.online_dir))
    except:
      dir_for_logfiles="%s/log"%(cWB_conf.online_dir)
      commands.getstatusoutput("mkdir %s/log"%(cWB_conf.online_dir))

    commands.getstatusoutput("mkdir %s/crontab"%(cWB_conf.online_dir))
    command="""
    * * * * * %s/bin/check_restart.sh %s/python %s/run.log none %s/bin/restart_run.sh >> /tmp/%s_restart_run.log 2>&1
    """%(os.environ['CWB_ONLINE'],cWB_conf.online_dir,dir_for_logfiles,os.environ['CWB_ONLINE'],cWB_conf.user)
    file="%s/crontab/run.crontab"%(cWB_conf.online_dir)
    f=open(file,"w")
    print >>f, command
    f.close()
    
    file="%s/crontab/web.crontab"%(cWB_conf.online_dir)
    ff=open(file,"w")
    command="""* * * * * %s/bin/check_restart.sh %s/python %s/web_pages_%s.log %s %s/bin/restart_web_pages.sh >> /tmp/%s_restart_web_pages.log 2>&1"""%(os.environ['CWB_ONLINE'],cWB_conf.online_dir,dir_for_logfiles,"daily","daily",os.environ['CWB_ONLINE'],cWB_conf.user)
    print >>ff, command
    for w in self.web_pages:
      command="""* * * * * %s/bin/check_restart.sh %s/python %s/web_pages_%s.log %s %s/bin/restart_web_pages.sh >> /tmp/%s_restart_web_pages.log 2>&1"""%(os.environ['CWB_ONLINE'],cWB_conf.online_dir,dir_for_logfiles,w,w,os.environ['CWB_ONLINE'],cWB_conf.user)
      print >>ff, command
    command="""* * * * * %s/bin/check_restart.sh %s/python %s/web_pages_%s.log %s %s/bin/restart_web_pages.sh >> /tmp/%s_restart_web_pages.log 2>&1"""%(os.environ['CWB_ONLINE'],cWB_conf.online_dir,dir_for_logfiles,"check","check",os.environ['CWB_ONLINE'],cWB_conf.user)
    print >>ff, command
    ff.close()
    
    com="cat %s/crontab/run.crontab %s/crontab/web.crontab > %s/crontab/run_and_web.crontab"%(cWB_conf.online_dir,cWB_conf.online_dir,cWB_conf.online_dir)
    commands.getstatusoutput(com)
    
    if (not self.copy):
      command="""
    * * * * * %s/bin/check_restart.sh %s/python %s/run_ts.log none %s/bin/restart_run_ts.sh >> /tmp/%s_restart_run_ts.log 2>&1
    """%(os.environ['CWB_ONLINE'],cWB_conf.online_dir,dir_for_logfiles,os.environ['CWB_ONLINE'],cWB_conf.user)
      file="%s/crontab/bkg.crontab"%(cWB_conf.online_dir)
      f=open(file,"w")
      print >>f, command
      f.close()
    
      com="cat %s/crontab/web.crontab %s/crontab/bkg.crontab > %s/crontab/web_andbkg.crontab"%(cWB_conf.online_dir,cWB_conf.online_dir,cWB_conf.online_dir)
      commands.getstatusoutput(com) 
    
      com="cat %s/crontab/run.crontab %s/crontab/web.crontab %s/crontab/bkg.crontab > %s/crontab/run_and_web_andbkg.crontab"%(cWB_conf.online_dir,cWB_conf.online_dir,cWB_conf.online_dir,cWB_conf.online_dir)
      commands.getstatusoutput(com) 
    
  def create_web(self):    
    file="%s/index.html"%(cWB_conf.web_dir)
    ffindex=open(file,"w")
    command="""
<?xml version="1.0"?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">
<!--                                                                           -->
<!-- Author: CWB team (http://www.virgo.lnl.infn.it/Wiki/index.php/Main_Page)  -->
<!--                                                                           -->
<!--   Date: Sun Jan 13 16:13:38 2013                                          -->
<!--                                                                           -->
<head>
<meta http-equiv="Content-Type" content="text/html; charset=ISO-8859-1" />
<title>CWB Display</title>
<meta name="rating" content="General" />
<meta name="objecttype" content="Report" />
<meta name="keywords" content="software development, GW, Analisys, Virgo, LIGO " />
<meta name="description" content="CWB - An Framework For GW Burst Data Analysis." />
<link rel="shortcut icon" href="/~waveburst/waveburst/logo/cwb_logo_icon_modern.png" type="image/png" />
<link rel="icon" href="/~waveburst/waveburst/logo/cwb_logo_icon_modern.png" type="image/png" />
<link rel="stylesheet" type="text/css" href="ROOT_modern.css" id="CWBstyle" />
<script type="text/javascript" src="ROOT.js"></script>

<script type="text/javascript">
  function toggleVisible(division) {
  if (document.getElementById("div_" + division).style.display == "none") {
    document.getElementById("div_" + division).style.display = "block";
    document.getElementById("input_" + division).checked = true;
  } else {
    document.getElementById("div_" + division).style.display = "none";
    document.getElementById("input_" + division).checked = false;
  }
}
</script>

<script type="text/javascript">
function onSearch() {
var s='http://www.google.com/search?q=%s+site%3A%u+-site%3A%u%2Fsrc%2F+-site%3A%u%2Fexamples%2F';
var ref=String(document.location.href).replace(/https?:\/\//,'').replace(/\/[^\/]*$/,'').replace(/\//g,'%2F');
window.location.href=s.replace(/%u/ig,ref).replace(/%s/ig,escape(document.searchform.t.value));
return false;}
</script>

</head>

<body  onload="javascript:SetValuesFromCookie();"><div id="body_content">
<div id="top"><!-- do not remove this div, it is closed by doxygen! -->
<div id="titlearea">
<table bgcolor="#223E5F" cellspacing="0" cellpadding="0" width="100%" height="120px"  align="center">
  <tr>
    <td>  <img style="height:100px" alt="Logo" src="https://ldas-jobs.ligo.caltech.edu/~waveburst/doc/cwb-lbanner-modern.png" USEMAP="#cwb_lbanner_modern" />  </td>
    <td style="width:45%"> </td>
    <td align="middle" style="color: #FFFFFF" nowrap="nowrap"><font size="6">Coherent WaveBurst</font> &#160;
    <td style="width:55%"> </td>
    <td> <img style="height:100px" alt="Logo" src="https://ldas-jobs.ligo.caltech.edu/~waveburst/doc/ligo_virgo_logo_modern.png" USEMAP="#ligo_virgo_map" /> </td>
  </tr>
</table>
</div>

<map id="cwb_lbanner_modern" name="cwb_lbanner_modern"><area shape="rect" alt="" title="cWB Documentation" coords="0,90,300,0" href="https://ldas-jobs.ligo.caltech.edu/~waveburst/LSC/doc/cWB_documentation/" target="_
blanck" /></map>

<map id="ligo_virgo_map" name="ligo_virgo_map"><area shape="rect" alt="" title="LIGO Homepage" coords="0,45,140,0" href="http://www.ligo.caltech.edu/" target="_blanck" /><area shape="rect" alt="" title="VIRGO Homepage"
coords="0,90,140,45" href="https://www.virgo-gw.eu/" target="_blanck" /></map>

<div id="toplinks">
<div class="descrhead"><div class="descrheadcontent">
<span class="descrtitle">Quick Links:</span>
<a class="descrheadentry" href="http://www.virgo.lnl.infn.it/Wiki/index.php/Main_Page">CWB Wiki</a>
<a class="descrheadentry" href="https://ldas-jobs.ligo.caltech.edu/~waveburst/LSC/doc/cWB_documentation/">CWB Documentation</a>
<a class="descrheadentry" href="https://ldas-jobs.ligo.caltech.edu/~waveburst/doc/cwb/ref">CWB Reference</a>
<a class="descrheadentry" href="https://git.ligo.org/cWB">CWB Repository</a>
<a class="descrheadentry" href="https://git.ligo.org/groups/cWB/-/issues">CWB Issues</a>
<a class="descrheadentry" href="http://root.cern.ch">ROOT Homepage</a>
</div>
</div>
</div>

<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.01//EN" "http://www.w3.org/TR/html4/strict.dtd">
<html><head>
<meta content="text/html;charset=ISO-8859-1" http-equiv="Content-Type">
<title>MAIN_Index</title>
<!-- Include the tabber code -->
<script type="text/javascript" src="tabber.js"></script>
<link rel="stylesheet" href="tabber.css" TYPE="text/css" MEDIA="screen">

<script type="text/javascript">

/* Optional: Temporarily hide the "tabber" class so it does not "flash"   on
 * the page as plain HTML. After tabber runs, the class is changed
   to "tabberlive" and it will appear. */

document.write('<style type="text/css">.tabber{display:none;}<\/style>');

</script>"""

    command+="""
</head>
<title>%s</title>
"""%cWB_conf.title
    
    command+="""
<body>
<html>
<h1 align=center>%s</h1>
<br>
<div class="tabber">
"""%cWB_conf.title
    
    page_length=3500
    
    for n in self.name_pages:
      title=n.replace("_"," ")
      command+="""
<div class="tabbertab">
  <h2>%s</h2>
  <iframe src="%s/%s/%s.html" width="100%%"  height="%ipx" frameborder="0"></iframe>                                                                                                                
</div>"""%(title,cWB_conf.summaries_dir,n,n,page_length)
    
    command+="""
<div class="tabbertab">
  <h2>Calendar</h2>
  <iframe src="main.html" width="100%%"  height="%ipx" frameborder="0"></iframe>
</div>"""%(page_length)
    
    command+="""
<div class="tabbertab">
  <h2>Status</h2>
  <iframe src="%s/check.html" width="100%%"  height="%ipx" frameborder="0"></iframe>
</div>
    
</div>
</html>
    """%(cWB_conf.summaries_dir,page_length)
    print >>ffindex, command
    ffindex.close()
    
if(__name__=="__main__"):    

    cwb=cwb_create() 
    cwb.make_checks()
    cwb.make_dirs()    
    cwb.cuts_file()
    cwb.plugins()
    cwb.lag_file()    
    cwb.user_parameters()    
    cwb.pe_parameters()    
    cwb.bkg_parameters()    
    cwb.user_pparameters()    
    cwb.superlag_file()    
    cwb.create_crontab()
    cwb.create_web()

    print cwb.add_prod_plugin

    print "Analysis dir: %s"%(cWB_conf.online_dir)
