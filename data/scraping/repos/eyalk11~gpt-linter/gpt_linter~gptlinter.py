
import os
import xml

import logging
import xml.etree.ElementTree as ET

from gpt_linter.common import generate_diff
from gpt_linter.guide import Guidance
from gpt_linter.linter import MyPyLinter

from gpt_linter.logger import Logger
logger=Logger()

DEFAULT_MODEL = "gpt-3.5-turbo-16k"

MYPYARGS = ['--disallow-untyped-defs']

older_path = r"c:\gitproj\Auto-GPT"
DEFAULT_TOKENS =3600
DEFAULT_TOKENS_PER_FIX =400
DEFAULT_TEMP_PER_FIX=0.7
DEFAULT_TEMP = 0.2
import argparse
from typing import List, Dict, Any, Optional, Iterator


class GPTLinter:
    def __init__(self,args: argparse.Namespace):
        #move all the attributes of args to local variables
        self.args = args
        self.file = args.file
        self.original_content = open(self.file, 'rt').read()

        self.debug = args.debug
        self.linter = MyPyLinter()

    def get_new_content(self,err_res: Dict[str, Any]) -> Optional[str]:
        fix_guide= Guidance.guide_for_fixes(self.args)
        fix_res=fix_guide(filename=self.file, file=self.original_content, fixes=err_res['fix'])
        if not 'fixedfile' in fix_res:
            logger.error('no fixed file')
            return None
        fixed = fix_res['fixedfile']
        logger.debug(f'fixed file: {fixed}')
        #bb = json.loads(fix_res["fixedfile"])['pythonfile']
        for k in range(3):

            if k==1:
                try:
                    logger.debug("trying to fix bad formatting")
                    new_content=fixed[fixed.index('<pythonfile>'):]
                    new_content=new_content[:new_content.rindex('</pythonfile>')+len('</pythonfile>')]
                except Exception as e :
                    logger.error(e)
                    logger.error(f"bad formatting manual extraction {k}")
                    continue
            elif k==0:
                try:
                    new_content= fixed[fixed.index('```xml') + 6:]
                    new_content=new_content[:new_content.rindex('```')]
                except:
                    logger.debug("no ```xml found ")
                    if len(fixed)>0.5*len(self.original_content):
                        new_content=fixed
                    else:
                        return None
            elif k==2:
                try:
                    new_content= fixed[fixed.index('```python') + 9:] #it wasn't supposed to be like that
                    new_content=new_content[:new_content.rindex('```')]
                except:
                    logger.debug("no ```python found ")

            try:
                new_content=ET.fromstring(new_content).text #remove the pythonfile element
                return new_content

            except xml.etree.ElementTree.ParseError:
                logger.debug(f"bad formatting {k}")
                logger.debug(new_content)
                if new_content.startswith('<pythonfile>') and new_content.endswith('</pythonfile>') and len(new_content) > 0.8 * len(self.original_content):
                    new_content= new_content[len('<pythonfile>'):(-1)*len('</pythonfile>')] #it is probably too idiot to extract it.
                    return new_content
                if k == 2 and len(new_content) > 0.8 * len(self.original_content):
                    logger.warn("will try anyway")
                    return new_content

        logger.debug("got to the end")
        return None



    def get_issues_string(self,issues: List[Dict[str, Any]]) -> Iterator[str]:
        for issue in issues:
            ln=int(issue['Line Number'])
            lines=self.original_content.split('\n')
            line_range= '\n'.join( lines[ max(ln-1-1,0) :min(ln-1+1,len(lines))])

            issue[f"lines {ln-1} to {ln} in the file"]='\n'+line_range

            st='\n'.join(f"{k}: {v}" for k,v in issue.items())

            logger.debug(st)
            yield st

    def main(self) -> None:
        logger.setup_logger(self.debug)

        if 'OPEN_AI_KEY' not in os.environ:
            logger.error('OPEN_AI_KEY not set')
            return
        Guidance.set_key(self.args)
        logger.info("mypy output:")

        errors = self.linter.get_issues(self.args)
        logger.debug(errors)
        if len(errors)==0:
            logger.info('no errors')
            return

        self.try_to_solve_issues(errors)

    def try_to_solve_issues(self,errors):
        logger.info("trying to solve issues") 
        err_res = self.get_fixes(list(self.get_issues_string(errors)))
        new_content=self.get_new_content(err_res)
        if new_content is None:
            logger.error('cant continue')
            return


        colored_diff,diff= generate_diff(self.original_content, new_content, self.args.file.replace("\\", '/'))

        if self.args.diff_file:
            open(self.args.diff_file, 'wt').write(diff)

        old_errors=errors
        if not self.args.recheck_policy == 'none':
            errors = self.check_new_file(new_content)

         
        print(diff if self.args.no_color else colored_diff)
        update=False


        if (len(errors) == 0 and self.args.auto_update == 'strict') \
            or (self.args.auto_update == 'permissive' and len(old_errors) > len(errors)):
                update=True

        elif not self.args.dont_ask:
            print("do you want to override the file? (y/n)")
            if input() == 'y':
                update=True

        if update:
            open(self.args.file, 'wt').write(new_content)
            self.original_content = new_content
            if self.args.recheck_policy =='recheckandloop' and len(errors) >0:
                self.try_to_solve_issues(errors)

    def check_new_file(self, new_content: str) -> List[Dict[str, Any]]:
        newfile: str = self.args.file.replace('.py', '.fixed.py')  # must be in the same folder sadly.
        open(newfile, 'wt').write(new_content)
        logger.info('output from mypy after applying the fixes:')
        try:
            return self.linter.get_issues(self.args, override_file=newfile)
        finally:
            if not self.args.store_fixed_file:
                try:
                    os.remove(newfile)
                except:
                    logger.error('could not remove file %s' % newfile)


    def get_fixes(self, errors: List[str]) -> Dict[str, Any]:
        err_guide = Guidance.guide_for_errors(self.args)
        err_res: Dict[str, Any] = err_guide(filename=self.file, file=self.original_content, errors=errors)
        if not self.args.dont_print_fixes:
            logger.info('suggested fixes:')
            logger.info('\n'.join(err_res['fix']))
        return err_res


def main() -> None:
    # Create the argument parser
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description='Run mypy on a Python file and use OpenAI GPT to fix the errors. It temporary generates file.fixed.py file to check for errors. You probably want to provide project so that mypy could resolve dependencies.')
    # Add the arguments
    parser.add_argument('file', help='Python file to run mypy on')
    parser.add_argument('mypy_args', nargs=argparse.REMAINDER, help=f'Additional options for mypy after --. By default, uses {MYPYARGS}')
    parser.add_argument('--mypy-path', default='mypy', help='Path to mypy executable (default: "mypy")')
    parser.add_argument('--error_categories', action='store', help='Type of errors to process')
    parser.add_argument('--max_errors', action='store', type=int, default=10, help='Max number of errors to process per cycle')
    parser.add_argument('-p','--proj-path', default='.', help='Path to project')
    parser.add_argument('-d','--diff-file', action='store', help='Store diff in diff file')
    parser.add_argument('-s','--store-fixed-file', action='store_true', default=False, help='Keeps file.fixed.py')

    parser.add_argument('--dont-ask', action='store_true', default=False,
                        help='Dont ask if to apply to changes. Useful for generting diff')
    parser.add_argument('-m','--model', default=DEFAULT_MODEL, help='Openai model to use')
    parser.add_argument('--max_tokens-per-fix', default=DEFAULT_TOKENS_PER_FIX, help='tokens to use for generating each fix')
    parser.add_argument('--temperature-per-fix', default=DEFAULT_TEMP_PER_FIX, help='temperature to use for fixes')
    parser.add_argument('--max_tokens-for-file', default=DEFAULT_TOKENS, help='tokens to use for file')
    parser.add_argument('--temperature-for-file', default=DEFAULT_TEMP, help='temperature to use for generating the file')
    parser.add_argument('-r','--recheck-policy', choices=['recheck','none','recheckandloop'],  default='recheckandloop',
                        help='Recheck the file for issues before suggesting a fix. require to temporarily save file.fixed.py (has to be in the project). recheckandloop will go for another loop if done fixing and there are still errors.')

    parser.add_argument('--debug', action='store_true', default=False, help='debug log level ')
    parser.add_argument('-a','--auto-update', choices=['permissive','no','strict'], default='no', help='auto update file if no errors (if strict). On permissive policy it updates if the number of errors decreased. ')
    parser.add_argument('-D','--dont-print-fixes', action='store_true', default=False, help='dont print fixes')
    #add no colors option  
    parser.add_argument('-N','--no_color', action='store_true', help='dont print color diff')

    # Parse the arguments
    args: argparse.Namespace = parser.parse_args()
    if len(args.mypy_args) ==0:
        args.mypy_args = MYPYARGS

    GPTLinter(args).main()


if __name__ == '__main__':
    main()
