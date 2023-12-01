# Use to deal with the data of fix
import json 
import time
import shutil
import OpenAI
import random
import OpenAI_gpt
import re

from Locator        import *
from base58         import b58encode
from utils_GPT      import *
from utils          import *
from glob           import glob
from unidiff        import PatchSet
from n132           import build_from_srcmap
from Diff           import getVulCommit, getDiff
TEMP = 0.75
SAVE_MONEY = True
def get_test_dataset():
    filter1 = "This model's maximum context length"
    fs = glob("./PatchDesc/*")
    res = []
    for fname in fs:
        with open(fname,'r') as f:
            if filter1 not in f.read():
                res.append(int(fname.split("/")[-1][:-4]))
    return res
def _get_reports_id():
    res = glob("./Reports/*")
    return [int(x.split("/")[-1][:-5]) for x in res]

def _get_all_single_mods(DEBUG=True):
    reports = _get_reports_id()
    res = []
    for r in reports:
        if(DEBUG):
            print("[*] Testing localID: " + str(r))
        diff_file = oss_fuzz_get_patch(r)
        if diff_file == False:
            continue
        diff_content = str(diff_file).split("\n")    
        
        mod_cnt = 0
        for z in range(len(diff_content)):
            if(diff_content[z].startswith("@@ ")):
                mod_cnt+=1
                if mod_cnt > 1: # stop when over 2
                    break
        if(DEBUG):
            print("mod: "+str(mod_cnt))
        if mod_cnt==1:
            res.append(r) # localID
            if(DEBUG):
                print("[+] "+ str(r) +" added")
        else:
            if(DEBUG):
                print("[-] "+ str(r) +" skipped")
    print("[!] Done")
    if(DEBUG):
        print(res)
    return res
    
def get_vul_code(diff_file,repo_dir):
    # Get the vul code
    patch = PatchSet.from_filename(diff_file, encoding='utf-8')
    mods = []
    # parse the file
    for _ in range(len(patch)):
        target_set = patch[_]
        if target_set.is_modified_file:
            file_name = target_set.source_file[1:]
            for mod in target_set:
                tmp = (file_name,mod.source_start,mod.source_length)
                mods.append([tmp,target_set])
    
    vul_code = []
    count = 0  # Counter for mod
    for x,y in mods:
        print(repo_dir/x[0][1:])
        with open(repo_dir/x[0][1:],'rb') as f:
            code = f.readlines()[x[1]-1:x[1]+x[2]-1]
        # line info 
        diff_content = str(y).split("\n")
        added_lines   = []
        removed_lines = []
        # pase the diff
        tmp = count
        for z in range(len(diff_content)):
            if(diff_content[z].startswith("@@ ")):
                if tmp !=0:
                    tmp-=1
                else:
                    diff_content = diff_content[z+1:]
                    break
        ct = 0
        while(ct<len(diff_content)):
            if diff_content[ct].startswith("-"):
                removed_lines.append(ct)
            elif diff_content[ct].startswith("+"):
                added_lines.append(ct)
            elif diff_content[ct].startswith("@@ "):
                break
            ct+=1
        # store them
        ori_code = b"".join(code)
        item = [ori_code,removed_lines,added_lines,y.target_file]
        vul_code.append(item)
        count+=1
    return vul_code

def get_bug_info(localId):
    # Get the bug type
    with open("./Reports/"+str(localId)+".json") as f:
        bug_type = json.loads(f.read())['crash_type'].split(" ")[0]
    return (" ".join(bug_type.split("-"))).lower()

def _rev_line_numbers(removed_lines,added_lines):
    all_lines = removed_lines+added_lines
    all_lines.sort()
    reloc_rm = []
    reloc_add = []
    for x in all_lines:
        if x in removed_lines:
            reloc_rm.append(x-all_lines.index(x))
        elif x in added_lines:
            reloc_add.append(x-all_lines.index(x))
    return reloc_rm,reloc_add

def strategy_start_with(bug_info,start_with,mode='Modify'):
    if mode == "Modify":
        instruction = f'Fix the {bug_info} vulnerability on the lines beginning with "{start_with[:0x20].decode()}"'
    elif mode == "Insert":
        instruction = f'Fix the {bug_info} vulnerability by inserting code after the line beginning with "{start_with[:0x20].decode()}"'        
    return instruction

def get_possible_fix(vul_code,bug_info,repo_dir,\
    strategy=strategy_start_with,api=OpenAI.run,n=1,temperature=0):
    [code,removed_lines,added_lines,target_file]  = vul_code
    reloc_rm, reloc_add = \
        _rev_line_numbers(removed_lines,added_lines)
    # TODO: more complex instruction  
    # TODO: More general method to get the modification
    mod = list(set(reloc_rm))
    mod.sort()
    codes = [x.strip() for x in code.strip().split(b"\n")]
    
    if len(mod)>0:#modify
        instruction = strategy(bug_info,codes[mod[0]-1])
    else:#insert
        mod = list(set(reloc_add))
        mod.sort()
        instruction = strategy(bug_info, codes[mod[0]-1],mode="Insert")
    
    # print(instruction)
    res = api(code.decode(),instruction,n=n,temperature=temperature)
    # tmp_fix = res['choices'][0]['text']
    print(res)
    fixed_code = list(set([ x['text'].encode()  for x in res['choices'] if "error" not in x.keys() ]))
    # print(code.decode())
    print(instruction)
    # exit(1)
    # fixed_code.append(tmp_fix)
    # # replace the vul code with the new code
    # with open(repo_dir/target_file[2:],'rb') as f:
    #     raw_code = f.read()
    # # TODO: Better code rather than replace
    # raw_code  = raw_code.replace(code,tmp_fix)
    # with open(repo_dir/target_file[2:],'wb') as f:
    #     f.write(raw_code)
    return fixed_code, code, repo_dir/target_file[2:]
def getDescription(fname,):
    if fname.exists():
        with open(fname,'r') as f:
            patch_desc = f.read()
    
def verify_FIX(localId,repo_dir,pname):
    # TODO: Functional checking
    print(localId)
    # localId, int
    # return value: -1 error, 0 False, 1 True
    def leave(result):
        if CLEAN_TMP and case_dir:
            clean_dir(case_dir)
        if(RM_IMAGES):
            remove_oss_fuzz_img(localId)
        return result
    
    srcmap,issue = getIssueTuple(localId)
    case_dir = tmpDir()
    try:
        case_path = download_reproducer(issue,case_dir,"crash_case")
    except:
        return leave(False)
    if not case_path or not case_path.exists():
        return leave(False)
    
    srcmap =  srcmap[0]

    build_res = \
        build_from_srcmap(srcmap,issue,replace_dep=[pname,repo_dir])
    if not build_res:
        return leave(False)
    not_crash = crash_verfiy(issue,case_path)
    if not_crash == True:
        return leave(True)
    else:
        return leave(False)

def perform_fix(fix,ori_code,target_file):
    with open(target_file,'rb') as f:
        raw_code = f.read()
    raw_code  = raw_code.replace(ori_code,fix)
    with open(target_file,'wb') as f:
        f.write(raw_code)
def get_GPT_fix(localID,vul_code,work_dir):
    print("[*] Getting Fix Description")
    code = vul_code [0]
    localDesc = Path(f"./PatchDesc/{localID}.log")
    if localDesc.exists() and SAVE_MONEY:
        with open(localDesc,'r') as f:
            patch_desc = f.read()
    else:
        patch_desc = oss_fuzz_vul_labeler(localID)
        with open(localDesc,'w') as f:
            f.write(patch_desc)
    
    prompt = f"""
Can you fix the vulnerability in the following code:
```
{code.decode()}
```
    
There is a vulnerability description, a summary of the fix, and a detailed description of the fix for the code:

{patch_desc}

Please only return the code in the response. Do not include explanations in your reply.
"""
    print("[+] Performing GPT Fixing..")
    fixed_code = OpenAI_gpt.run(prompt)
    print("[+] Recieved the result from ChatGPT")
    # extract the code out 
    if fixed_code.startswith("````"):
        fixed_code = re.sub(r'```.*\n', "", fixed_code)
    if fixed_code.endswith('```'):
        fixed_code = re.sub(r'```', "", fixed_code)
    # print(fixed_code)
    # print(type(fixed_code))
    return [fixed_code.encode()], code, work_dir/vul_code[3][2:]

def get_Codex_fix(localID,vul_code,work_dir):
    print("[*] Getting Fix Description")
    code = vul_code [0]
    target_file = vul_code[3] 
    localDesc = Path(f"./PatchDesc/{localID}.log")
    if localDesc.exists() and SAVE_MONEY:
        with open(localDesc,'r') as f:
            patch_desc = f.read()
    else:
        patch_desc = oss_fuzz_vul_labeler(localID)
        with open(localDesc,'w') as f:
            f.write(patch_desc)
    
    prompt = f"""
Can you fix the vulnerability in the given code.
    
There is a vulnerability description, a summary of the fix, and a detailed description of the fix for the code:

{patch_desc}
"""
    print("[+] Performing Codex Fixing..")
    res = OpenAI.run(code.decode(),prompt,n=5,temperature=TEMP)
    print(res)
    fixed_code = list(set([ x['text'].encode()  for x in res['choices'] if "error" not in x.keys() ]))
    return fixed_code, code, work_dir/target_file[2:]

def get_Wizard_fix(localID,vul_code,work_dir):
    print("[*] Getting Wizard Fix Description")
    [code,_,__,target_file]  = vul_code
    localDesc = Path(f"./PatchDesc/{localID}.log")
    if localDesc.exists():
        with open(localDesc,'r') as f:
            patch_desc = f.read()
    else:
        patch_desc = oss_fuzz_vul_labeler(localID)
    prompt = f"""
Can you fix the vulnerability in the following code:
```
{code.decode()}
```
    
There is a vulnerability description, a summary of the fix, and a detailed description of the fix for the code:

{patch_desc}
"""
    # print(prompt)
    res = WizardLM.run(code.decode(),prompt,n=5,temperature=TEMP)
    # print(x['text'].encode() )
    fixed_code = list(set([ x['text'].encode()  for x in res['choices'] if "error" not in x.keys() ]))
    return fixed_code, code, work_dir/target_file[2:]
        

def XxX(localId,module="Wiard",limit=1,chance=1):
    # TODO: CLEAN TMP DIR
    work_dir    = tmpDir()

    # Get meta data
    pname       = getPname(localId)
    diff_file   = getDiff(localId)
    vul_commit  = getVulCommit(localId)
    if vul_commit==False or diff_file==False:
        return False
    # Copy Repo
    repo_dir    = work_dir / pname 
    url         = get_projectInfo(localId,pname)[0]['url']
    shutil.copytree( OSS_DB/b58encode(url).decode()/pname, repo_dir,symlinks=True)
    
    # Check out to vul version
    if check_call(['git','reset','--hard',vul_commit],repo_dir) == False:
        eventLog(f"[-] XxX: Failed to reset {localId}")
        shutil.rmtree(work_dir)
        return False

    # Get code info, maker sure there should be only one case
    vul_code = get_vul_code(diff_file,repo_dir)
    if(len(vul_code)>limit):
        print("[X] More than one modifications. The result could be not precise.")
        return False
    else:
        vul_code = vul_code[0]
    # Perform Fixing
    if(module=="Codex"):
        try:
            fixes, ori_code, target_file = get_Codex_fix(localId,vul_code,repo_dir)
        except:
            return False
    elif(module=="Wizard"):
        try:
            fixes, ori_code, target_file = get_Wizard_fix(localId,vul_code,repo_dir)
        except:
            return False
    elif(module=="GPT"):
        try:
            fixes, ori_code, target_file = get_GPT_fix(localId,vul_code,repo_dir)
        except:
            return False
    else:
        panic("[X] Set your module correctly!")
    for x in fixes:
        print(x)
    # Try to build and verify all possibe fixes
    for fix in fixes[:chance]:
        perform_fix(fix,ori_code,target_file)
        res = verify_FIX(localId,repo_dir,pname)
        if res:
            print("[+] Successful fix: ")
            print(fix)
            break
    shutil.rmtree(work_dir)
    shutil.rmtree(os.path.dirname(diff_file))

    if res:
        print("[+] SUCCESS!")
        return True
    else:
        print("[-] FAIL to FIX.")
        return False

## New Unified Method for Language Model("module") fixes
def LMFix(localId, module="GPT"):
    # TODO: CLEAN TMP DIR
    work_dir = tmpDir()
    # Get necessary data
    pname       = getPname(localId)
    diff_file   = getDiff(id)
    vul_commit  = getVulCommit(id)
    if vul_commit==False or diff_file==False:
        return False
    # Copy Repo
    repo_dir = work_dir / pname 
    proj_json,_ = get_projectInfo(localId,pname)
    url = proj_json['url']
    shutil.copytree( OSS_DB/b58encode(url).decode()/pname, repo_dir,symlinks=True)
    
    # Check out to vul version
    if check_call(['git','reset','--hard',vul_commit],repo_dir) == False:
        eventLog(f"[-] LMFix: Failed to reset {localId}")
        shutil.rmtree(work_dir)
        return False
    # Get code info, there should be only one case
    vul_code = get_vul_code(diff_file,repo_dir)[0]
    if(module=="Codex"):
        try:
            fixes, ori_code, target_file = get_Codex_fix(localId,vul_code,repo_dir)
        except:
            return False
    elif(module=="Wizard"):
        try:
            fixes, ori_code, target_file = get_Wizard_fix(localId,vul_code,repo_dir)
        except:
            return False
    elif(module=="GPT"):
        try:
            fixes, ori_code, target_file = get_GPT_fix(localId,vul_code,repo_dir)
        except:
            return False
    else:
        pass
        bug_info = get_bug_info(localId)
        try:
            fixes, ori_code, target_file = get_possible_fix(vul_code,bug_info,repo_dir,n=5,temperature=TEMP)
        except:
            return False
    
    for fix in fixes:
        perform_fix(fix,ori_code,target_file)
        res = verify_FIX(localId,repo_dir,pname)
        if res:
            print("[+] Successful fix: ")
            print(fix)
            break
    shutil.rmtree(work_dir)
    shutil.rmtree(os.path.dirname(diff_file))

    if res:
        print("[+] SUCCESS!")
        return True
    else:
        print("[-] FAIL to FIX.")
        return False

def patchDesc(localId):
    if Path(f"./PatchDesc/{localId}.log").exists():
        return 
    
    patch_desc = oss_fuzz_vul_labeler(localId)
    with open(f"./PatchDesc/{localId}.log","w") as f:
        f.write(patch_desc)
    print("Sleeping to cool-down...")
    time.sleep(10)
    return

def desc_allPatches():
    ids = _get_all_single_mods()
    print("Start Working")
    for x in ids:
        print(f"[+] Generating the Desc for issue: {x}")
        patchDesc(x)
        
    return 

# Test Area
def TestBenchmark(round, module="GPT"):
    logfile= f"./_fx_local_log/{module}_round_{round}.json"
    logfile = Path(logfile)
    # Init log files
    if not json_file_check(logfile):
        panic(f"Failed to init {logfile}")
    # Get tested issues
    content = json.loads(open(logfile).read())
    tested = [int(x.strip())for x in content.keys()]
    # Do test
    ids = _get_all_single_mods(False)
    while(len(tested) < len(ids)):
        chosen = ids[random.randint(0,len(ids)-1)]
        if chosen in tested:
            continue
        try:
            res=LMFix(chosen, module)
        except:
            res=False
        content[chosen] = res
        with open(logfile,'w') as f:
            json.dumps(content,sort_keys = True,indent=4)
    print("[!] Test ALL Done")
def Benchmark(round, module):
    if module not in ['GPT','Codex','Wizard']:
        print("[X] Select a module from 'GPT','Codex','Wizard'")
        return False
    logfile= f"./Log/BenchMark/{module}_round_{round}.json"
    logfile = Path(logfile)
    # Init log files
    if not json_file_check(logfile):
        panic(f"Failed to init {logfile}")
    # Get tested issues
    content = json.loads(open(logfile).read())
    tested = [int(x.strip())for x in content.keys()]
    # Do test
    ids = _get_all_single_mods(False)
    while(len(tested) < len(ids)):
        chosen = ids[random.randint(0,len(ids)-1)]
        if chosen in tested:
            continue
        try:
            res= XxX(chosen, module)
        except:
            res=False
        content[chosen] = res
        with open(logfile,'w') as f:
            json.dumps(content,sort_keys = True,indent=4)
    print("[!] Test ALL Done")
if __name__ =="__main__":
    # desc_allPatches()
    # Benchmark(1,"GP")
    # ids = _get_all_single_mods(False)
    # print(ids)
    # TestBenchmark(1,"Wizard")
    # print(XxX(5534,"GPT",1))
    # LMFix(5534, "Wizard")
