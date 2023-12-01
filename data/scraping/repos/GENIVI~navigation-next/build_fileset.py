import os, stat
import sys
import glob
import subprocess
import time
import shutil
lib_path = os.path.abspath('../../tools')
sys.path.append(lib_path)
import libcom
from tesla import tps
from tesla.tps import lib
from tesla.tps import io
from os import listdir
from os.path import isfile, join

BUILD_DIRECTORY = os.path.join(os.getcwd(), '..', '..', 'supportfiles', 'navfileset')
OUTPUT_DIRECTORY = sys.argv[1]

def chmod(filename):
    if os.path.isfile(filename):
        os.chmod(filename,0777)

def checkPath(p):
    if not os.path.exists(p):
        raise IOError,"Not found:%s" % p
    return p

#get the filesets_branch directory
def getFilesetsBranch():
    path = checkPath(os.path.join(BUILD_DIRECTORY, 'filesets_branch.txt'))
    txtFile = open( path, 'r' )
    branch = ""
    for line in txtFile:
        line = line.rstrip( '\n' )
        if ( line == "" ):
            continue
        else:
            branch = line
    txtFile.close()
    return branch

#get current language
def getCurrentLanguage():
    path = checkPath(os.path.join(BUILD_DIRECTORY, 'current_language.txt'))
    txtFile = open( path, 'r' )
    lang = ""
    for line in txtFile:
        line = line.rstrip( '\n' )
        if ( line == "" ):
            continue
        else:
            lang = line
    txtFile.close()
    return lang

#get current voice style
def getCurrentVoiceStyle():
    path = checkPath(os.path.join(BUILD_DIRECTORY, 'current_voice.txt'))
    txtFile = open( path, 'r' )
    voiceStyle = ""
    for line in txtFile:
        line = line.rstrip( '\n' )
        if ( line == "" ):
            continue
        else:
            voiceStyle = line
    txtFile.close()
    return voiceStyle

def getP4SourceDir():
    """Only parse current path of script to get your P4 root."""

    s = os.getcwd()
    pos = s.find('client')
    if (pos != -1):
        return s[:pos - 1]
    else:
        print 'error finding p4 root dir'
        sys.exit()

#path is here: //depot/client/resources/%BRANCH_NAME%/navigation/...
def getResourceDir():
    p4root = getP4SourceDir()
    filesets_branch = getFilesetsBranch()
    return os.path.join(p4root, 'client', 'resources', filesets_branch, 'navigation')

#generate compiled tpslib file
def generateTplFile(tpslib_path, srcfile, destfile):
    print "Generating tpl file .....\n"
    libcom.generateTpl(tpslib_path, destfile, os.path.join(tpslib_path, srcfile))

    print "Complete tplfile .....\n"

def allFiles(src_dir, extname):
    if (extname == '*'):
        a = os.path.join(src_dir, extname)
        onlyfiles = [ f for f in listdir(src_dir) if isfile(join(src_dir, f)) ]
    else:
        onlyfiles = [ f for f in listdir(src_dir) if (isfile(join(src_dir, f)) and f.endswith("." + extname)) ]

    return onlyfiles

def compileSexp2Tps(sexpfn, outfn, tpslib):
    tl=tps.lib.TemplateLibrary(tpslib)

    try:
        elt=tps.tpselt.fromsexp(open(sexpfn, 'rU').read())
        try:
            open(outfn, 'wb').write(tps.io.pack(tl, elt))
        finally:
            pass
    finally:
        pass

    if not os.path.isfile(outfn):
        sys.exit("Error: creating tps file from sexp file: "+sexpfn+"--> "+outfn)
    return 0

def sexp2tpsFS(sexpfn, tpsfn, tplfn):
    if not os.path.isfile(sexpfn):
        sys.exit("Error: missing sexp file: "+sexpfn)
    newDir=os.path.join(os.path.split(tpsfn)[0])
    if  len(newDir) and not os.path.isdir(newDir) :
        os.makedirs(newDir)
    compileSexp2Tps(sexpfn, tpsfn, tplfn)

#compile resources from guidanceconfig
def compileGuidanceConfigFiles():
    tplDict = os.path.join(OUTPUT_DIRECTORY, 'guidanceconfigdata.tpl')
    resourcePath = getResourceDir()
    dirConfigPath = os.path.join(resourcePath, 'guidanceconfig')
    generateTplFile(dirConfigPath, 'tpslib.txt', tplDict)
    
    for sexpFile in allFiles(dirConfigPath, 'sexp'):
        sexpFilePath = os.path.join(dirConfigPath, sexpFile)
        dstTpsFile = sexpFile.replace(".sexp", ".tps")
        dstTpsFilePath = os.path.join(OUTPUT_DIRECTORY, dstTpsFile)
        print dstTpsFilePath + " is compiling"
        sexp2tpsFS(sexpFilePath, dstTpsFilePath, tplDict)

#compile resources from voices/aac
def compileBasicAudioFiles():
    resourcePath = getResourceDir()
    dirConfigPath = os.path.join(resourcePath, 'voices')
    tplDict = os.path.join(OUTPUT_DIRECTORY, 'basicaudiodata.tpl')
    generateTplFile(dirConfigPath, 'tpslib.txt', tplDict)

    voiceStyle = getCurrentVoiceStyle()
    currentLanguage = getCurrentLanguage()
    #remove '-'
    currentLanguage = currentLanguage.replace("-", "")
    voicesDirName = currentLanguage + '-' + voiceStyle
    voicesPath = os.path.join(resourcePath, 'voices', 'aac', voicesDirName)

    sexpFilePath = os.path.join(voicesPath, "basicaudio.sexp")
    dstTpsFilePath = os.path.join(OUTPUT_DIRECTORY, "basicaudio.tps")
    print dstTpsFilePath + " is compiling"
    sexp2tpsFS(sexpFilePath, dstTpsFilePath, tplDict)

def compileDirections():
    currentLanguage = getCurrentLanguage()
    sexpFileName = 'directions-' + currentLanguage + '.sexp'
    resourcePath = getResourceDir()
    sexpFilePath = os.path.join(resourcePath, 'directions', sexpFileName)
    dstTpsFilePath = os.path.join(OUTPUT_DIRECTORY, "directions.tps")
    print dstTpsFilePath + " is compiling"
    tplDict = os.path.join(OUTPUT_DIRECTORY, 'guidanceconfigdata.tpl')
    sexp2tpsFS(sexpFilePath, dstTpsFilePath, tplDict)

def copy(srcFile,dstFile):
    try:
        shutil.copyfile(srcFile,dstFile)
        os.chmod(dstFile,stat.S_IWRITE | stat.S_IREAD)
    except OSError, (errno,strerror):
        print """Error copying %(path)s, %(error)s """ % {'path' : srcFile, 'error': strerror }

def copyfiles(fspec,dstDir):
    try:
        files = glob.glob(fspec)
        print files
        for fileName in files:
            fileName = fileName.replace('/','\\')
            print fileName + " is copying"
            fname = fileName[fileName.rindex(os.sep)+1:]
            chmod(dstDir + os.sep + fname)
            copy(fileName,dstDir + os.sep + fname)
    except OSError, (errno,strerror):
        print """Error copying files %(path)s, %(error)s """ % {'path' : fspec, 'error': strerror }

def copyBaseAudioFiles():
    resourcePath = getResourceDir()
    voiceStyle = getCurrentVoiceStyle()
    currentLanguage = getCurrentLanguage()
    #remove '-'
    currentLanguage = currentLanguage.replace("-", "")
    voicesDirName = currentLanguage + '-' + voiceStyle
    voicesPath = os.path.join(resourcePath, 'voices', 'aac', voicesDirName)
    outputVoicesPath = os.path.join(OUTPUT_DIRECTORY, 'voices')
    #copy voices once
    if  len(outputVoicesPath) and not os.path.isdir(outputVoicesPath):
        print "base voice files are copying"
        os.makedirs(outputVoicesPath)
        copyfiles(voicesPath + os.sep + '*.aac', outputVoicesPath)

def copyRoutingIcons():
    resourcePath = getResourceDir()
    iconsPath = os.path.join(resourcePath, 'routing_icons')
    outputPath = os.path.join(OUTPUT_DIRECTORY, 'routing_icons')
    #copy routing icons once
    if  len(outputPath) and not os.path.isdir(outputPath):
        print "routing icons are copying"
        os.makedirs(outputPath)
        copyfiles(iconsPath + os.sep + '*.png', outputPath)

def run():
    if  len(OUTPUT_DIRECTORY) and not os.path.isdir(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)
    compileGuidanceConfigFiles()
    compileDirections()
    compileBasicAudioFiles()
    copyBaseAudioFiles()
    copyRoutingIcons()
    copy(os.path.join(BUILD_DIRECTORY, 'current_language.txt'), os.path.join(OUTPUT_DIRECTORY, 'current_language.txt'))
    copy(os.path.join(BUILD_DIRECTORY, 'current_voice.txt'), os.path.join(OUTPUT_DIRECTORY, 'current_voice.txt'))
    print "COMPLETE"

run()
