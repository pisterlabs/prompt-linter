#!/usr/bin/python

# Remove file not used in the GUIDANCE alignment
import os
import argparse
#file1 = "/Users/luho/Documents/cds_align_guidance/OG10001_result/"
def RemoveNotUsedFileFrommGuidace(result_file_guidance):
    all_small_file = os.listdir(result_file_guidance)
    target_file = ["MSA.MAFFT.Without_low_SP_Col.With_Names", "MSA.MAFFT.aln.With_Names", "MSA.MAFFT.PROT.aln",
                   "Seqs.Codes", "MSA.MAFFT.Guidance2_col_col.PROT.scr",
                   "Seqs.Orig_DNA.fas.FIXED.MSA.MAFFT.Removed_Col"]
    remove_file = list(set(all_small_file) - set(target_file))
    for i in remove_file:
        whole_file_dir = result_file_guidance + i
        os.remove(whole_file_dir)
    print("Remove all unused files!!")


# for the batch process
# the code file is stored in the document file
def main():
    parser = argparse.ArgumentParser(
            formatter_class = argparse.RawDescriptionHelpFormatter,
            description = 'remove unused file from GUIDANCE alignment!')
    #adding arguments
    parser.add_argument('-n', metavar='input_file', type=str, help='input cds aligned result files')
    args = parser.parse_args()
    cdsfile = args.n
    RemoveNotUsedFileFrommGuidace(result_file_guidance=cdsfile)
if __name__ == "__main__":
    main()

# an example
# python /Users/luho/Documents/evolution_analysis/code/code_align/A4_code_align_all_guidance_remove_files.py -n /Users/luho/Documents/cds_align_guidance/OG10001_result/