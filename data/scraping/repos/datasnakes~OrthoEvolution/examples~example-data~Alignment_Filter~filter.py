import os
from pathlib import Path
from shutil import copy
from guidance2 import Guidance2Commandline
from pal2nal import Pal2NalCommandline
from Bio import SeqIO
from utils import multi_fasta_manipulator
import subprocess


# For one gene at a time
class FilteredAlignment(object):

    def __init__(self, na_fasta, aa_fasta, gene_name=None, home=os.getcwd(), msaProgram='CLUSTALW', na_bootstraps=1,
                 aa_bootstraps=1, na_seqCutoff=0.6, aa_seqCutoff=0.6, na_colCutoff=0, aa_colCutoff=0.88):
        if gene_name is None:
            gene_name = Path(na_fasta).stem

        # Initialize the default command line arguments
        self.G2C_args = dict(outOrder='as_input', dataset=gene_name, msaProgram=msaProgram)
        self.P2N_args = dict(nogap=True, nomismatch=True)

        # Initialize
        self.home = Path(home)
        self.na_guidance_path = self.home / Path('NA_Guidance2')
        self.aa_guidance_path = self.home / Path('AA_Guidance2')
        self.gene = Path(na_fasta).stem

        na_seqFile = str(self.home / Path(na_fasta))  # Guidance NA sequence file
        aa_seqFile = str(self.home / Path(aa_fasta))  # Guidance AA sequence file

        na_fasta = str(self.home / Path(self.gene + '_G2.ffn'))  # Pal2Nal NA sequence file
        na_alignment = str(self.home / Path(self.gene + '_P2N_na.aln'))  # Pal2Nal output file
        # Guidance2 iterations over the nucleic acid sequences
        iteration_flag = True
        iteration = 1
        while iteration_flag is True:
            if iteration > 1:
                na_seqCutoff = 0.7
                na_bootstraps = 1
            iteration_flag, na_seqFile = self.nucleic_acid_guidance(iteration, seqFile=na_seqFile,
                                                                 outDir=str(self.na_guidance_path), bootstraps=na_bootstraps,
                                                                 seqCutoff=na_seqCutoff, colCutoff=na_colCutoff)
            iteration += 1

        # Guidance 2 amino acid alignment filter.  Returns the path as a string
        aa_alignment = self.amino_acid_guidance(seqFile=aa_seqFile, remFile=na_seqFile, outDir=str(self.aa_guidance_path),
                                                bootstraps=aa_bootstraps, seqCutoff=aa_seqCutoff, colCutoff=aa_colCutoff)

        # PAL2NAL nucleic acid alignment
        self.pal2nal_conversion(str(aa_alignment), na_fasta, na_alignment)

    def nucleic_acid_guidance(self, iteration, seqFile, outDir, bootstraps, seqCutoff, colCutoff):
        seqType = 'nuc'

        outDir = Path(outDir) / Path('iter_%s' % iteration)
        Path.mkdir(outDir, parents=True, exist_ok=True)

        G2Cmd = Guidance2Commandline(**self.G2C_args, seqFile=seqFile, seqType=seqType, outDir=str(outDir), bootstraps=bootstraps,
        seqCutoff=seqCutoff, colCutoff=colCutoff)
        print(iteration)
        #print(g2c_args.items())
        print(G2Cmd)
        subprocess.check_call([str(G2Cmd)], stderr=subprocess.STDOUT, shell=True)
        removed_file = str(Path(outDir) / Path('Seqs.Orig.fas.FIXED.Removed_Seq'))
        renamed_file1 = str(copy(removed_file, str(Path(self.gene + '_G2_removed.ffn'))))

        seq_file = str(Path(outDir) / Path('Seqs.Orig.fas.FIXED.Without_low_SP_Seq.With_Names'))
        renamed_file2 = str(copy(seq_file, str(Path(self.gene + '_G2.ffn'))))

        # Copy and rename files
        rem_count = SeqIO.write(SeqIO.parse(removed_file, 'fasta'), renamed_file1, 'fasta')
        if iteration == 1:
            iter_flag = True
            return_file = renamed_file2
        elif iteration > 1:
            if rem_count > 0:
                multi_fasta_manipulator(renamed_file1, removed_file, manipulation='add', added_name='')
                iter_flag = True
                return_file = renamed_file2
            else:
                iter_flag = False
                return_file = renamed_file1
        return iter_flag, return_file

    def amino_acid_guidance(self, seqFile, remFile, outDir, bootstraps, seqCutoff, colCutoff):
        seqType = 'aa'
        filtered_fasta = multi_fasta_manipulator(seqFile, remFile)
        G2Cmd = Guidance2Commandline(**self.G2C_args, seqFile=str(filtered_fasta), seqType=seqType, outDir=outDir, bootstraps=bootstraps,
                                     seqCutoff=seqCutoff, colCutoff=colCutoff)
        print(G2Cmd)
        G2Cmd()

        filtered_alignment = Path(outDir) / Path('%s.CLUSTALW.Without_low_SP_Col.With_Names' % self.G2C_args['dataset'])
        renamed_alignment = copy(str(filtered_alignment), str(Path(self.gene + '_G2_aa.aln')))
        print('Align the filtered amino acid sequences using guidance 2')
        return Path(renamed_alignment)

    def pal2nal_conversion(self, aa_alignment, na_fasta, output_file):
        P2Ncmd = Pal2NalCommandline(**self.P2N_args, pepaln=aa_alignment, nucfasta=na_fasta, output_file=output_file)
        print(P2Ncmd)
        P2Ncmd()
        print('Align the nucleic acids using the amino acid alignment.')
