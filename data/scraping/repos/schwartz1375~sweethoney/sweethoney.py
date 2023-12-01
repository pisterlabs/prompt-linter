#!/usr/bin/env python3

__author__ = 'Matthew Schwartz (@schwartz1375) & Santry (@san4n6)'
__version__ = '3.8'

import argparse
import hashlib
import os
import string
import sys
from datetime import datetime

import magic
import numpy as np
import pefile
import ssdeep
import tlsh
from scipy.stats import chisquare
from termcolor import colored, cprint

import fileUtils
import openAiUtils
import packerUtils

# registry alerts; used for persistence, config data storage, cleanup, and registry management
regalerts = ['RegCreateKeyA', 'RegCreateKeyW', 'RegCreateKeyExA', 'RegCreateKeyExW', 'RegDeleteValueA', 'RegDeleteValueW', 'RegFlushKey',
             'RegCloseKey', 'RegDeleteKeyA', 'RegDeleteKeyW', 'RegDeleteKeyExA', 'RegDeleteKeyExW', 'RegOpenKeyA', 'RegOpenKeyW', 'RegOpenKeyExA',
             'RegOpenKeyExW', 'RegSetValueA', 'RegSetValueW', 'RegSetValueExA', 'RegSetValueExW', 'RtlCreateRegistryKey', 'RtlWriteRegistryValue',]

# Networking and Internet Access; used for exfiltration, and C2
netalerts = ['InternetCloseHandle', 'InternetOpenA', 'InternetOpenW', 'InternetOpenUrlA', 'InternetOpenUrlW', 'InternetConnectA', 'InternetConnectW',
             'HttpOpenRequestA', 'HttpOpenRequestW', 'HttpSendRequestA', 'HttpSendRequestW', 'HttpSendRequestExA', 'HttpSendRequestExW',
             'InternetReadFile', 'InternetReadFileExA', 'InternetReadFileExW', 'InternetWriteFile', 'URLDownloadToFile',
             'FtpPutFileA', 'FtpPutFileW', 'FtpGetFileA', 'FtpGetFileW', 'FtpDeleteFileA', 'FtpDeleteFileW', 'FtpCreateDirectoryA', 'FtpCreateDirectoryW',
             'socket', 'connect', 'bind', 'listen', 'accept', 'send', 'recv', 'sendto', 'recvfrom', 'getaddinfo',
             'gethostbyname', 'gethostbyaddr', 'Gethostname', 'pipe', 'socketpair', 'shmget', 'shmat', 'shmdt', 'semget', 'semop']

# process alerts/Code Injection/Unpacking; process and memory manipulation, and DLL injection
psalerts = ['CreateProcessA', 'CreateProcessW', 'TerminateProcess', 'LoadLibraryA', 'LoadLibraryW', 'LoadLibraryExA', 'LoadLibraryExW',
            'CreateProcess', 'EnumProcesses', 'CreateService', 'ControlService', 'StartService', 'ReadProcessMemory', 'OpenProcess',
            'VirtualFree', 'VirtualFreeEx', 'VirtualAlloc', 'VirtualAllocEx', 'GetProcAddress', 'VirtualProtect', 'VirtualProtectEx', 'LoadLibraryA',
            'GetWindowsThreadProcessId', 'SetWindowsHookEx', 'BroadcastSystemMessage', 'WriteProcessMemory', 'CreateRemoteThread',
            'fork', 'execve', 'clone', 'waitpid', 'kill', 'getpid', 'getppid', 'mprotect', 'mmap', 'munmap', 'dlopen', 'dlsym', 'dlcose']

# malicious general funcitons
miscalerts = ['AdjustTokenPrivileges', 'WinExec', 'ShellExecute', 'FindFirstFile', 'FindNextFile',
              'CreateMutex', 'GetAsyncKeyStat', 'Fopen', 'GetEIP', 'malloc', 'GetTempPathA', 'ShellExecuteA', 'IsWoW64Process', 'LdrLoadDll',
              'MapViewOfFile', 'NetScheduleJobAdd', 'open', 'close', 'read', 'write', 'lseek', 'unlink', 'rename', 'opendir', 'readdir', 'closedir',
              'mkdir', 'rmdir', 'stat', 'fstat', 'chmod', 'chown']

# dropper alerts, cleanup, lateral movement
dropalerts = ['CreateFileA', 'CreateFileW', 'ReadFile', 'ReadFileEx', 'WriteFile', 'WriteFileEx', 'DeleteFileA', 'DeleteFileW',
              'CopyFile', 'CopyFileA', 'CopyFileW', 'CopyFileExA', 'CopyFileExW', 'MoveFileA', 'MoveFileW', 'FindResource', 'LoadResource',
              'SizeOfResource', 'LockResource', 'NtResumeThread', 'NtMapViewOfSection', 'NtCreateSection']

# anti vm/debugging alerts
antialerts = ['GetTickCount', 'CountClipboardFormats', 'GetForeGroundWindow', 'Isdebuggerpresent', 'NtGlobalFlag', 'FindWindow', 'NtClose',
              'CloseHandle', 'OutputDebugString', 'OutputDebugStringA', 'OutputDebugStringW', 'NtQueryInformationProcess', 'GetAdaptersInfo',
              'CheckRemoteDebuggerPresent', 'CreateToolhelp32Snapshot', 'GetModuleHandleA', 'GetModuleHandleW',
              'GetModuleHandleExA', 'GetModuleHandleExW', 'GetModuleFileNameA', 'GetModuleFileNameW']

# keylogger and Data Theft
keyalerts = ['FindWindowA', 'ShowWindow', 'GetAsyncKeyState', 'SetWindowsHookEx', 'RegisterHotKey', 'GetMessage', 'UnhookWindowsHookEx', 'GetClipboardData',
             'GetWindowText']

# System Information and miscellaneous funcations; used for reconnaissance, timestamping, and evasion
sysinfoalerts = ['GetSystemInfo', 'GetSystemTime', 'GetUserNameA', 'GetUserNameW', 'GetComputerNameA', 'GetComputerNameW', 'GetVersion',
                 'GetVersionExA', 'GetVersionExW', 'uname', 'sysctl', 'getuid', 'getgid', 'setuid', 'setgrid', 'geteuid', 'getegid',
                 'getpwnam', 'getpwuid', 'getgrnam', 'getgrid', 'getlogin', 'gethostname', 'gettimeofday', 'localtime', 'strftime']

# crypto stuff
cryptalerts = ['CryptEncrypt', 'CryptAcquireContext',
               'CryptAcquireContext', 'CryptImportPublicKeyInfo', 'CryptoAPI']


def Main(file):
    diec_path=None

    print("Interrogating file: '%s'" % file)
    try:
        pe = pefile.PE(file)
    except pefile.PEFormatError as e:
        cprint('Aw Snap!  PEFormatError: ' + str(e), 'red')
        sys.exit(1)
    except:
        cprint('Something went wrong loading the file with pefile!', 'red')
        sys.exit(1)
    file_size = os.path.getsize(file)
    getFileInfo(pe, file, diec_path)
    getCompileInfo(pe)
    getSecurityFeatures(pe)
    getFileDeclared(pe)
    getFileExports(pe)
    getSectionDetails(pe, file_size)
    getFileStats(pe, file)
    fileUtils.analyzePeFile(file)
    ret = openAiUtils.getOpenAiResults(pe)
    print(ret)


def getSectionDetails(pe, file_size):
    cprint("\n**************************************************", 'blue')
    cprint("Getting section details...", 'blue')
    cprint("**************************************************", 'blue')
    print("%-10s %-12s %-12s %-12s %-35s %-45s %-12s %-12s %-12s " % ("Name", "VirtAddr",
          "VirtSize", "RawSize", "MD5", "SHA-1", "File-Ratio", "Entropy", "Chi2"))
    print("-" * 180)
    for sec in pe.sections:
        raw_size = sec.SizeOfRawData
        ratio = (raw_size / file_size) * 100
        # s = "%-10s %-12s %-12s %-12s %-35s %-45s %-12.2f %-12f %-12.2f" % (''.join([c for c in str(sec.Name, 'utf-8') if c in string.printable]),
        # Ignore invalid bytes
        s = "%-10s %-12s %-12s %-12s %-35s %-45s %-12.2f %-12f %-12.2f" % (''.join([c for c in str(sec.Name, 'utf-8', errors='ignore') if c in string.printable]),
                                                                           hex(sec.VirtualAddress),
                                                                           hex(sec.Misc_VirtualSize),
                                                                           hex(raw_size),
                                                                           sec.get_hash_md5(),
                                                                           sec.get_hash_sha1(),
                                                                           ratio,
                                                                           sec.get_entropy(),
                                                                           getChi2(
                                                                               sec)
                                                                           )
        if raw_size == 0 or \
            (sec.get_entropy() > 0 and sec.get_entropy() < 1) or \
                sec.get_entropy() > 7:
            s += "[SUSPICIOUS]"
        if s.endswith("[SUSPICIOUS]"):
            cprint(s, 'red')
        else:
            print(s)


def getChi2(section):
    section_data = section.get_data()
    # Calculate the observed frequencies of each byte value
    observed_frequencies = np.bincount(bytearray(section_data), minlength=256)

    # The expected frequency for each byte value under the assumption of perfect randomness
    # would be the total number of bytes divided by 256
    total_bytes = len(section_data)
    expected_frequencies = np.full(256, total_bytes / 256)

    # Remove the byte values where the expected frequency is zero
    non_zero_expected = expected_frequencies != 0
    observed_frequencies = observed_frequencies[non_zero_expected]
    expected_frequencies = expected_frequencies[non_zero_expected]

    # If no valid expected frequencies are left after filtering, return -1
    if len(expected_frequencies) == 0:
        return -1

    # Calculate the chi-square statistic and p-value
    chi2_stat, p_value = chisquare(observed_frequencies, expected_frequencies)
    ''' 
	section_name = section.Name.decode('utf-8').rstrip('\x00')
	print(f"Chi-square statistic for section {section_name}: {chi2_stat}")
	print(f"P-value for section {section_name}: {p_value}")
	'''
    return chi2_stat


def getCompileInfo(pe):
    ped = pe.dump_dict()
    cprint("\n**************************************************", 'blue')
    cprint("Compile information:", 'blue')
    cprint("**************************************************", 'blue')
    # Compile time
    comp_time = ped['FILE_HEADER']['TimeDateStamp']['Value']
    comp_time = comp_time.split("[")[-1].strip("]")
    time_stamp, timezone = comp_time.rsplit(" ", 1)
    comp_time = datetime.strptime(time_stamp, "%a %b %d %H:%M:%S %Y")
    print("Compiled on {} {}".format(comp_time, timezone.strip()))


def getFileDeclared(pe):
    cprint("\n**************************************************", 'blue')
    cprint("Functions declared and referenced:", 'blue')
    cprint("**************************************************", 'blue')
    ret, ret1, ret2, ret3, ret4, ret5, ret6, ret7, ret8 = (
        [] for i in range(9))
    try:
        for lib in pe.DIRECTORY_ENTRY_IMPORT:
            print(str(lib.dll, 'utf-8'))
            for imp in lib.imports:
                if imp.name != None:
                    print('\t' + str(imp.name, 'utf-8'))
                if (imp.name != None) and (imp.name != ""):
                    for alert in regalerts:
                        if imp.name.decode('utf-8').startswith(alert):
                            ret.append(imp.name)
                    for alert in netalerts:
                        if imp.name.decode('utf-8').startswith(alert):
                            ret1.append(imp.name)
                    for alert in psalerts:
                        if imp.name.decode('utf-8').startswith(alert):
                            ret2.append(imp.name)
                    for alert in miscalerts:
                        if imp.name.decode('utf-8').startswith(alert):
                            ret3.append(imp.name)
                    for alert in dropalerts:
                        if imp.name.decode('utf-8').startswith(alert):
                            ret4.append(imp.name)
                    for alert in antialerts:
                        if imp.name.decode('utf-8').startswith(alert):
                            ret5.append(imp.name)
                    for alert in keyalerts:
                        if imp.name.decode('utf-8').startswith(alert):
                            ret6.append(imp.name)
                    for alert in sysinfoalerts:
                        if imp.name.decode('utf-8').startswith(alert):
                            ret7.append(imp.name)
                    for alert in cryptalerts:
                        if imp.name.decode('utf-8').startswith(alert):
                            ret8.append(imp.name)
    except AttributeError:
        print("The PE object does not have the DIRECTORY_ENTRY_IMPORT attribute.")
        # handle the error gracefully

    if len(ret) != 0:
        cprint("\n**************************************************", 'blue')
        cprint("Suspicious registry alerts", 'yellow', attrs=['bold'])
        cprint("**************************************************", 'blue')
        for x in ret:
            cprint("\t"+x.decode("utf-8"), 'yellow', attrs=['bold'])

    if len(ret1) != 0:
        cprint("\n**************************************************", 'blue')
        cprint("Suspicious network and or IPC alerts",
               'yellow', attrs=['bold'])
        cprint("**************************************************", 'blue')
        for x in ret1:
            cprint("\t"+x.decode("utf-8"), 'yellow', attrs=['bold'])

    if len(ret2) != 0:
        cprint("\n**************************************************", 'blue')
        cprint("Suspicious process and memory manipulation",
               'yellow', attrs=['bold'])
        cprint("**************************************************", 'blue')
        for x in ret2:
            cprint("\t"+x.decode("utf-8"), 'yellow', attrs=['bold'])

    if len(ret3) != 0:
        cprint("\n**************************************************", 'blue')
        cprint("Suspicious general/miscellaneous alerts",
               'yellow', attrs=['bold'])
        cprint("**************************************************", 'blue')
        for x in ret3:
            cprint("\t"+x.decode("utf-8"), 'yellow', attrs=['bold'])

    if len(ret4) != 0:
        cprint("\n**************************************************", 'blue')
        cprint("Suspicious dropper alerts", 'yellow', attrs=['bold'])
        cprint("**************************************************", 'blue')
        for x in ret4:
            cprint("\t"+x.decode("utf-8"), 'yellow', attrs=['bold'])

    if len(ret5) != 0:
        cprint("\n**************************************************", 'blue')
        cprint("Suspicious anti debugger/vm alerts", 'yellow', attrs=['bold'])
        cprint("**************************************************", 'blue')
        for x in ret5:
            cprint("\t"+x.decode("utf-8"), 'yellow', attrs=['bold'])

    if len(ret6) != 0:
        cprint("\n**************************************************", 'blue')
        cprint("Suspicious keylogger debugger/vm alerts",
               'yellow', attrs=['bold'])
        cprint("**************************************************", 'blue')
        for x in ret6:
            cprint("\t"+x.decode("utf-8"), 'yellow', attrs=['bold'])

    if len(ret7) != 0:
        cprint("\n**************************************************", 'blue')
        cprint("Suspicious System information collection alerts",
               'yellow', attrs=['bold'])
        cprint("**************************************************", 'blue')
        for x in ret7:
            cprint("\t"+x.decode("utf-8"), 'yellow', attrs=['bold'])

    if len(ret8) != 0:
        cprint("\n**************************************************", 'blue')
        cprint("Suspicious CRYPTO alerts", 'yellow', attrs=['bold'])
        cprint("**************************************************", 'blue')
        for x in ret8:
            cprint("\t"+x.decode("utf-8"), 'yellow', attrs=['bold'])


def getFileExports(pe):
    cprint("\n**************************************************", 'blue')
    cprint("Looking for exported sysmbols...", 'blue')
    cprint("**************************************************", 'blue')
    try:
        for exp in pe.DIRECTORY_ENTRY_EXPORT.symbols:
            # print hex(pe.OPTIONAL_HEADER.ImageBase + exp.address), exp.name, exp.ordinal
            print("Name: %s, Ordinal number: %i" %
                  (str(exp.name, 'utf-8'), exp.ordinal))
    except:
        cprint("No exported symbols!", 'magenta')


def getSecurityFeatures(pe):
    cprint("\n**************************************************", 'blue')
    cprint("Getting file security features...", 'blue')
    cprint("**************************************************", 'blue')
    print("Uses NX: ", bool(pe.OPTIONAL_HEADER.DllCharacteristics &
          pefile.DLL_CHARACTERISTICS["IMAGE_DLLCHARACTERISTICS_NX_COMPAT"]))
    print("Uses ASLR: ", bool(pe.OPTIONAL_HEADER.DllCharacteristics &
          pefile.DLL_CHARACTERISTICS["IMAGE_DLLCHARACTERISTICS_DYNAMIC_BASE"]))
    print("Uses High Entropy ASLR: ", bool(pe.OPTIONAL_HEADER.DllCharacteristics &
          pefile.DLL_CHARACTERISTICS["IMAGE_DLLCHARACTERISTICS_HIGH_ENTROPY_VA"]))
    print("Uses SAFESEH: ", bool(pe.OPTIONAL_HEADER.DATA_DIRECTORY[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_LOAD_CONFIG"]
                                                                   ].VirtualAddress != 0 and pe.OPTIONAL_HEADER.DATA_DIRECTORY[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_LOAD_CONFIG"]].Size != 0))
    print("Uses CFG: ", bool(pe.OPTIONAL_HEADER.DllCharacteristics &
          pefile.DLL_CHARACTERISTICS["IMAGE_DLLCHARACTERISTICS_GUARD_CF"]))


def getFileStats(pe, file):
    cprint("\n**************************************************", 'blue')
    cprint("Getting file statics...", 'blue')
    cprint("**************************************************", 'blue')
    raw = pe.write()
    # Get the file size in bytes
    file_size_bytes = os.path.getsize(file)
    # Convert file size to kilobytes (KB)
    file_size_kb = file_size_bytes / 1024
    # Print the file size in both KB and bytes
    print(f"File size: {file_size_kb:.2f} KB ({file_size_bytes} bytes)")
    print('MD5 hash: %s' % hashlib.md5(raw).hexdigest())
    print('SHA-1 hash: %s' % hashlib.sha1(raw).hexdigest())
    print('SHA-256 hash: %s' % hashlib.sha256(raw).hexdigest())
    print('SHA-512 hash: %s' % hashlib.sha512(raw).hexdigest())
    print('Import hash (imphash): %s' % pe.get_imphash())
    with open(file, 'rb') as f:
        file_contents = f.read()
    print('SSDEEP fuzzy hash: %s' % ssdeep.hash(file_contents))
    print('TLSH fuzzy hash: %s' % tlsh.hash(file_contents))


def getFileInfo(pe, file, diec_path):
    cprint("\n**************************************************", 'blue')
    cprint("Getting file information...", 'blue')
    cprint("**************************************************", 'blue')
    print(colored('File name: ', 'green') + '%s' % os.path.basename(file))
    print(colored('Magic Type: ', 'green') + '%s' % magic.from_file(file))

    # Detect-It-Easy (diec)
    if diec_path != None:
        ret = packerUtils.get_packer_info(file, diec_path)
        if ret is not None:
            print(colored("Detect-It-Easy (diec): ", "green") + str(ret))
    else:
        cprint('Path to diec is not set in main!', 'yellow')

    # Check if the file is an executable image
    print(colored('Executable Image: ', 'green') + '%s' % pe.is_exe())

    # Check if the file is a DLL using 2 methods
    is_dll_method_1 = pe.is_dll()
    is_dll_method_2 = pe.FILE_HEADER.IMAGE_FILE_DLL

    # If both methods agree, print the result
    if is_dll_method_1 == is_dll_method_2:
        print(colored('DLL: ', 'green') + '%s' % is_dll_method_1)

    # If the methods disagree, print a warning
    elif is_dll_method_1 != is_dll_method_2:
        print(colored('Warning: DLL check methods returned different results!', 'red'))
        # Print the results
        print(colored('Method 1 - pe.is_dll(): ', 'red') + '%s' % is_dll_method_1)
        print(colored('Method 2 - pe.FILE_HEADER.IMAGE_FILE_DLL: ', 'red') + '%s' % is_dll_method_2)

    # Check other file properties
    print(colored('Driver: ', 'green') + '%s' % pe.is_driver())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A rapid file analysis tool')
    parser.add_argument("file", help="The file to be inspected by the tool")
    args = parser.parse_args()
    Main(args.file)
