#!/usr/bin/python

# The script is used to move aligned cds file from guidance result into a new file

import os
import argparse
import shutil

#result_file_guidance0 = "/Users/luho/Documents/cds_align_guidance/"
#out_file0 = "/Users/luho/Documents/cds_align_guidance_new/"

def MoveFileForGuidace(result_file_guidance, out_file):
    all_og = os.listdir(result_file_guidance)
    all_og = [x for x in all_og if "_result" in x]

    id_need_check = []
    for result0 in all_og:
        print(result0)
        target_file0 = os.listdir(result_file_guidance + result0)

        target_title = "MSA.MAFFT.Without_low_SP_Col.With_Names"
        if target_title in target_file0:
            old_file = result_file_guidance + result0 + "/" + target_title
            new_file = out_file + result0.replace("_result", "_code.fasta")
            shutil.copy(old_file, new_file)
        else:
            id_need_check.append(result0.replace("_result", ""))
    print("The followed id need check:")
    print(id_need_check)

# for the batch process
# the code file is stored in the document file
def main():
    parser = argparse.ArgumentParser(
            formatter_class = argparse.RawDescriptionHelpFormatter,
            description = 'move the guidance align result into a new file to do th followed analysis!')
    #adding arguments
    parser.add_argument('-r', metavar='input_file', type=str, help='input cds aligned result files')
    parser.add_argument('-n', metavar='output_file', type=str, help='output file to store the cds guidance align result')
    args = parser.parse_args()
    oldfile = args.r
    new_file = args.n
    MoveFileForGuidace(result_file_guidance=oldfile, out_file=new_file)
if __name__ == "__main__":
    main()

# an example
# python /Users/luho/Documents/evolution_analysis/code/code_align/A7_move_align_guidance_into_new_file.py -r /Users/luho/Documents/cds_align_guidance/ -n /Users/luho/Documents/cds_align_guidance_new/