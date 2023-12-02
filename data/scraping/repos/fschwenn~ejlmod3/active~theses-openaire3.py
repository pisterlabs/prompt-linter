# -*- coding: utf-8 -*-
#harvest theses from OpenAIRE via api
#FS: 2023-07-12

import sys
import os
import urllib.request, urllib.error, urllib.parse
from bs4 import BeautifulSoup
import re
import ejlmod3
import time
import datetime

publisher = 'OpenAIRE'
jnlfilename = 'THESES-OpenAIRE-%s' % (ejlmod3.stampofnow())
rpp = 50
skipalreadyharvested = True
alreadyharvestedlogfile = '/afs/desy.de/user/l/library/dok/ejl/backup/THESES-OpenAIRE-%salreadyharvestedviadecicatedcrawler.doki' % (ejlmod3.stampofnow())
#splitting the time erange to be checked to not hit maximal number of records per search
months = 12*15
monthlength = 30/30
absdir = '/afs/desy.de/group/library/publisherdata/abs'
tmpdir = '/afs/desy.de/user/l/library/tmp'
maxrecords = 1000

boring = ['Social Science and Humanities', 'Transport Research', 'Rural Digital Europe',
          'North American Studies', 'Digital Humanities and Cultural Heritage',
          'Neuroinformatics', 'NEANIAS Atmospheric Research Community',
          'Knowmad Institut']
boring += ['BIOMEDICINA I ZDRAVSTVO. Kliničke medicinske znanosti. Neurologija.',
           'BIOMEDICINE AND HEALTHCARE. Clinical Medical Sciences. Neurology.',
           'Coronary Artery Disease', 'Healthcare', 'Heart Failure', 'SARS-CoV-2',
           'M-PSI/02 - PSICOBIOLOGIA E PSICOLOGIA FISIOLOGICA',
           'Settore MED/38 - Pediatria Generale e Specialistica',
           'BIOMEDICINA I ZDRAVSTVO. Kliničke medicinske znanosti. Interna medicina.',
           'BIOMEDICINE AND HEALTHCARE. Clinical Medical Sciences. Internal Medicine.',
           'SDG 3 - Good Health and Well-being', 'Pathology. Clinical medicine',
           'Patologija. Klinička medicina', 'COVID-19', 'Medicina',
           'Medical sciences', 'Surgery. Orthopaedics. Ophthalmology',
           'BIOMEDICINE AND HEALTHCARE. Basic Medical Sciences. Cytology, Histology and Embryology.',
           'Medizinische Fakultät', 'Fakultät für Chemie', 'Settore BIO/09 - Fisiologia',
           'Settore BIO/11 - Biologia Molecolare', '490', 'Chemie', '004',
           'Settore BIO/10 - Biochimica', 'Settore BIO/13 - Biologia Applicata',
           'Settore MED/04 - Patologia Generale', 'Medizinische Fakultät',
           'Medical and Health sciences', 'Lebensqualität', '570', 'Medical sciences Medicine',
           '620', 'Medizin', 'DDC 610 / Medicine &amp; health', '377.5', '610',
           'Quality of Life', 'Settore MED/01 - Statistica Medica', 'VLAG',
           '610 Medical sciences', '610 Medizin', ':Biological sciences [Natural sciences]',
           'Biological sciences', 'Ciências biológicas; Biological sciences; Tese',
           ':Ciências biológicas [Ciências exactas e naturais]', 'Ciências biológicas',
           'climate change', 'Psychology', 'Settore CHIM/03 - Chimica Generale e Inorganica',
           'Settore CHIM/08 - Chimica Farmaceutica', '330',
           '620 Ingenieurwissenschaften, Maschinenbau',
           'DDC 620 / Engineering &amp; allied operations', 'Ingeniería Industrial',
           'Medical sciences Medicine', 'Medizin', '570 Biologie', '540', '550',
           'Settore ING-INF/06 - Bioingegneria Elettronica e Informatica', '610',
           '610 Medizin, Gesundheit', '490']
boring += ['BS (Bachelor of Science)', 'MS (Master of Science)', 'MA (Master of Arts)',
           'EDD (Doctor of Education)', 'DNP (Doctor of Nursing Practice)',
           'MARH (Master of Architectural History)', 'MFA (Master of Fine Arts)',
           'BA (Bachelor of Arts)', 'DNP (Doctor of Nursing Practice)', 'EDD (Doctor of Education)',
           'MA (Master of Arts)', 'MARH (Master of Architectural History)',
           'MFA (Master of Fine Arts)']
neutral = ['Netherlands', 'netherlands']
if skipalreadyharvested:
    alreadyharvested = ejlmod3.getalreadyharvested('THESES')

host = os.uname()[1]
if host == 'l00schwenn':
    bibclassifycommand = "/usr/bin/python3 /afs/desy.de/user/l/library/proc/python3/bibclassify/bibclassify_cli.py  -k /afs/desy.de/user/l/library/akw/HEPont.rdf "
elif host in ['inspire4', 'inspire4.desy.de']:
    bibclassifycommand = "python /afs/desy.de/user/l/library/proc/python3/bibclassify/bibclassify_cli.py  -k /afs/desy.de/user/l/library/akw/HEPont.rdf "


repothdl = re.compile('.*\/handle\/(\d+\/.*)')
repothdl2 = re.compile('.*handle.net\/(\d+\/.*)')
def getrecs(pagen, pagent, page):
    precs = []
    keepit = True
    i = 0
    for results in page.find_all('results'):
        for result in results.children:
            try:
                keepit = True
                rec = {'tc' : 'T', 'jnl' : 'BOOK', 'note' : ['Harvested from OpenAIRE - check carefuly'], 'autaff' : [],
                       'keyw' : []}
                #check whether already in some THESES-OpenAIRE*doki
                for dri in result.find_all('dri:objidentifier'):
                    identifier = dri.text.strip()
                    ejlmod3.printprogress('-', [[pagen, pagent], [i, rpp], [identifier], [len(precs)]])
                    if identifier in bereitsin:
                        print('  %s already in backup' % (identifier))
                        keepit = False
                    elif ejlmod3.checkinterestingDOI(identifier):
                        rec['note'].append('dri:objIdentifier:::' + identifier)
                    else:
                        print('  %s uninteresting' % (identifier))
                        keepit = False
                i += 1
            except:
                continue
            if keepit:
                #title
                for title in result.find_all('title', attrs = {'classid' : 'main title'}):
                    rec['tit'] = title.text.strip()
                #date
                for dateofacceptance in result.find_all('dateofacceptance'):
                    rec['date'] = dateofacceptance.text.strip()
                #author
                for creator in result.find_all('creator'):
                    rec['autaff'].append([creator.text.strip()])
                    for publisher in result.find_all('publisher'):
                        rec['autaff'][-1].append(publisher.text.strip())
                #abstract
                for description in result.find_all('description'):
                    rec['abs'] = description.text.strip()
                #keywords
                for subject in result.find_all('subject', attrs = {'classid' : 'keyword'}):
                    if not subject.text.strip() in rec['keyw']:
                        rec['keyw'].append(subject.text.strip())
                        if rec['keyw'][-1] in boring:
                            keepit = False
                #PID's
                for alternateidentifier in result.find_all('alternateidentifier'):
                    if alternateidentifier.has_key('classid'):
                        if alternateidentifier['classid'] == 'doi':
                            rec['doi'] = alternateidentifier.text.strip()
                            if not ejlmod3.checkinterestingDOI(rec['doi']):
                                print('  %s uninteresting' % (rec['doi']))
                                keepit = False
                        elif alternateidentifier['classid'] == 'urn':
                            rec['urn'] = alternateidentifier.text.strip()
                            if not ejlmod3.checkinterestingDOI(rec['urn']):
                                print('  %s uninteresting' % (rec['urn']))
                                keepit = False
                        elif alternateidentifier['classid'] == 'hdl':
                            rec['hdl'] = alternateidentifier.text.strip()
                        elif alternateidentifier['classid'] == 'isbn':
                            rec['isbn'] = alternateidentifier.text.strip()
                #link
                for webresource in result.find_all('webresource'):
                    for url in webresource.find_all('url'):
                        urltext = url.text.strip()
                        rec['link'] = urltext
                        if rec['autaff'] and 'abs' in rec and rec['keyw'] and 'doi' in rec:
                            pass
                        elif keepit and not re.search('\.pdf$', urltext):
                            print('    check source: %s' % (rec['link']))
                            time.sleep(3)
                            try:
                                sreq = urllib.request.Request(rec['link'], headers=hdr)
                                spage = BeautifulSoup(urllib.request.urlopen(sreq), features="lxml")
                            except:
                                print('         failed :(')
                                spage = False
                            if spage:
                                if not 'doi' in rec and  not re.search('handle.net\/11427\/', rec['link']):
                                    ejlmod3.metatagcheck(rec, spage, ['bepress_citation_doi', 'eprints.doi',
                                                                      'eprints.doi_name', 'DC.Identifier.doi',
                                                                      'citation_doi', 'dc.identifier',
                                                                      'citation_isbn', 'dc.identifier',
                                                                      'dc.Identifier', 'DC.identifier',
                                                                      'DC.Identifier'])
                                if not rec['autaff']:
                                    if 'doi' in rec and rec['doi'][:9] == '10.15099/':
                                        for tr in spage.find_all('tr'):
                                            for th in tr.find_all('th'):
                                                if th.text.strip() == '著者':
                                                    for td in tr.find_all('td'):
                                                        rec['autaff'] = [[ td.text.strip() ]]
                                    else:
                                        ejlmod3.metatagcheck(rec, spage, ['author'])
                                if not 'doi' in rec and re.search('doi.org\/10\.\d+\/', urltext):
                                    rec['doi'] = re.sub('.*doi.org\/', '', urltext)
                                    print('    %s from url=%s' % (rec['doi'], urltext))
                                if not 'abs' in rec:
                                    ejlmod3.metatagcheck(rec, spage, ['abstract', 'citation_abstract',
                                                                      'dcterms.abstract', 'DCTERMS.abstract'])
                                if  not rec['autaff']:
                                    ejlmod3.metatagcheck(rec, spage, ['citation_author'])
                                ejlmod3.metatagcheck(rec, spage, ['citation_language', 'citation_keywords',
                                                                  'citation_pdf_url', 'dc.rights', 'DC.rights',
                                                                  'DC.Rights', 'DC.language'])
                                if not 'hdl' in rec:
                                    if repothdl.search(rec['link']):
                                        hdl = repothdl.sub(r'\1', rec['link'])
                                    elif repothdl2.search(rec['link']):
                                        hdl = repothdl2.sub(r'\1', rec['link'])
                                    else:
                                        hdl = False
                                    if hdl and not re.search('123456789\/', hdl):
                                         print('    check HDL: %s' % (hdl))
                                         #verify
                                         try:
                                             hreq = urllib.request.Request(starturl, headers=hdr)
                                             hdlpage = BeautifulSoup(urllib.request.urlopen(hreq), features="lxml")
                                             for title in hdlpage.find_all('title'):
                                                 if title.text.strip() == 'Not Found':
                                                     rec['note'].append('%s seems not to be a proper HDL' % (hdl))
                                                 else:
                                                     rec['hdl'] = hdl
                                                     rec['note'].append('%s seems to be a proper HDL' % (hdl))
                                         except:
                                             print('    could not check HDL: %s' % (hdl))
                                #University of Virginia
                                if 'doi' in rec and rec['doi'][:9] == '10.18130/':
                                    for div in spage.find_all('div', attrs = {'class' : 'document-row'}):
                                        for span in div.find_all('span', attrs = {'class' : 'document-label'}):
                                            if span.text.strip() == 'Degree:':
                                                for span2 in div.find_all('span', attrs = {'class' : 'document-value'}):
                                                    degree = span2.text.strip()
                                                    if degree in boring:
                                                        keepit = False
                                                        print('   skip %s' % (degree))
                                                    else:
                                                        rec['note'].append('DEG:::' + degree)
                #community
                for com in result.find_all('context', attrs = {'type' : 'community'}):
                    if com.has_attr('label'):
                        community = com['label']
                        if community in boring:
                            print('   skip %s' % (community))
                            keepit = False
                        elif not community in neutral:
                            rec['note'].append('COM:::' + community)
                if keepit:
                    if skipalreadyharvested and 'doi' in rec and rec['doi'] in alreadyharvested:
                        print('   %s already in backup' % (rec['doi']))
                        ouf = open(alreadyharvestedlogfile, 'a')
                        ouf.write('I--dri:objIdentifier:::%s--\n' % (identifier))
                        ouf.close()
                    elif skipalreadyharvested and 'urn' in rec and rec['urn'] in alreadyharvested:
                        print('   %s already in backup' % (rec['urn']))
                        ouf = open(alreadyharvestedlogfile, 'a')
                        ouf.write('I--dri:objIdentifier:::%s--\n' % (identifier))
                        ouf.close()
                    elif skipalreadyharvested and 'hdl' in rec and rec['hdl'] in alreadyharvested:
                        print('   %s already in backup' % (rec['hdl']))
                        ouf = open(alreadyharvestedlogfile, 'a')
                        ouf.write('I--dri:objIdentifier:::%s--\n' % (identifier))
                        ouf.close()
                    else:
                        if not rec['autaff']:
                            rec['autaff'] = [[ 'Doe, John' ]]
                            rec['note'].append('MISSING AUTHOR!')
                        precs.append(rec)
                        ejlmod3.printrecsummary(rec)
                else:
                    ejlmod3.adduninterestingDOI(identifier)
    return precs


#check already harvested
ejldirs = ['/afs/desy.de/user/l/library/dok/ejl/backup/%i' % (ejlmod3.year(backwards=1)),
           '/afs/desy.de/user/l/library/dok/ejl/backup']
redoki = re.compile('THESES.OpenAIRE.*doki$')
reid = re.compile('^I\-\-dri:objIdentifier:::(.*)\-\-$')
nochmal = []
bereitsin = []
for ejldir in ejldirs:
    print(ejldir)
    for datei in os.listdir(ejldir):
        if redoki.search(datei):
            inf = open(os.path.join(ejldir, datei), 'r')
            for line in inf.readlines():
                if len(line) > 1 and line[0] == 'I':
                    if reid.search(line):
                        identifier = reid.sub(r'\1', line.strip())
                        if not identifier in bereitsin:
                            if not identifier in nochmal:
                                bereitsin.append(identifier)
    print('  %6i already from OpenAIRE' % (len(bereitsin)))



hdr = {'User-Agent' : 'Magic Browser'}

restart = re.compile('^Core key')
renum = re.compile('\d')
restop = re.compile('^Field ')
redri = re.compile('dri:objIdentifier:::')
now = datetime.datetime.now()
for month in range(months):
    startdate = now + datetime.timedelta(days=-monthlength*(month+1))
    stopdate = now + datetime.timedelta(days=-monthlength*month)
    startmark = '%4d-%02d-%02d' % (startdate.year, startdate.month, startdate.day)
    stopmark = '%4d-%02d-%02d' % (stopdate.year, stopdate.month, stopdate.day)
    
    starturl = 'https://api.openaire.eu/search/publications?size=' + str(rpp) + '&sortBy=resultdateofacceptance,descending&fromDateAccepted=' + startmark + '&toDateAccepted=' + stopmark + '&instancetype=Doctoral%20thesis'
    ejlmod3.printprogress('=', [[month+1, months, startmark], [1], [starturl]])
    req = urllib.request.Request(starturl, headers=hdr)
    startpage = BeautifulSoup(urllib.request.urlopen(req), features="lxml")
    header = startpage.find_all('header')[0]
    numofrecs = int(header.total.text.strip())
    numofpages = (numofrecs-1) // rpp + 1
    
    recs = getrecs(1, numofpages, startpage)
    for page in range(numofpages-1):
        tocurl = '%s&page=%i' % (starturl, page+2)
        ejlmod3.printprogress('=', [[month+1, months, startmark], [page+2, numofpages], [tocurl], [len(recs), rpp*(page+1), numofrecs]])
        time.sleep(10)
        treq = urllib.request.Request(tocurl, headers=hdr)
        tocpage = BeautifulSoup(urllib.request.urlopen(treq), features="lxml")
        recs += getrecs(page+2, numofpages, tocpage)
        if len(recs) > maxrecords:
            break

    #ejlmod3.writenewXML(recs, publisher, jnlfilename+'m'+str(month), retfilename='retfiles_special')

    #now check whether there are any CORE keywords. I not, forget about it. Just too many theses on OpenAIRE
    frecs = []
    for rec in recs:
        doi1 = False
        if 'doi' in rec:
            doi1 = re.sub('[\(\)\/]', '_', rec['doi'])
        elif 'hdl' in rec:
            doi1 = re.sub('[\(\)\/]', '_', rec['hdl'])
        elif 'urn' in rec:
            doi1 = re.sub('[\(\)\/]', '_', rec['urn'])
        else:
            pseudodoi = False
            if 'isbn' in rec and rec['isbn']:
                pseudodoi = '20.2000/ISBN/' + rec['isbn']
            elif 'isbns' in rec and rec['isbns']:
                pseudodoi = '20.2000/ISBNS'
                for tupel in rec['isbns'][0]:
                    if tupel[0] == 'a':
                        pseudodoi += '/' + tupel[1]
            elif 'link' in rec:
                pseudodoi = '20.2000/LINK/' + re.sub('\W', '', rec['link'][4:])
            elif 'tit' in rec:
                pseudodoi = '30.3000/AUT_TIT'
                if 'autaff' in rec and rec['autaff'] and rec['autaff'][0]:
                    pseudodoi += '/' + re.sub('\W', '', rec['autaff'][0][0])
                if pseudodoi:
                    rec['doi'] = pseudodoi
                    doi1 = re.sub('[\(\)\/]', '_', rec['doi'])
        if doi1:
            absfilename = os.path.join(absdir, doi1)
            bibfilename = os.path.join(tmpdir, doi1+'.hep.bib')
            time.sleep(.3)
            if not os.path.isfile(bibfilename):
                print(' >bibclassify %s' % (doi1))
                try:
                    os.system('%s %s > %s' % (bibclassifycommand, absfilename, bibfilename))
                except:
                    print('FAILURE: %s %s > %s' % (bibclassifycommand, absfilename, bibfilename))
            kws = []
            if os.path.isfile(bibfilename):
                absbib = open(bibfilename, 'r')
                lines = absbib.readlines()
                core = False
                for line in lines:
                    if restart.search(line):
                        core = True
                    elif core and renum.search(line):
                        kws.append('[CORE] ' + line.strip())
                        print('\n[CORE]', line)
                    elif restop.search(line):
                        core = False
                absbib.close()
            if kws:
                frecs.append(rec)
                print(doi1, 'might be interesting', kws)
            else:
                for note in rec['note']:
                    if redri.search(note):
                        dri = redri.sub('', note)
                        print(dri, 'seems to be uninteresting')
                        ejlmod3.adduninterestingDOI(dri)
                        
    if frecs:
        ejlmod3.writenewXML(frecs, publisher, jnlfilename+'M'+str(month))#, retfilename='retfiles_special')
            
