import os
import subprocess
from pathlib import Path
from shutil import copy

from Bio import SeqIO
from Datasnakes.Orthologs.Align.QualityControl.guidance2 import Guidance2Commandline

from Datasnakes.Orthologs.Align.pal2nal import Pal2NalCommandline
from Datasnakes.Orthologs.Genbank.utils import multi_fasta_manipulator


# TODO-ROB:  Create appropriate class variables for Filtered Tree to inherit
# TODO-ROB:  Create proper class variables to make simpler and for above todo


class FilteredAlignment(object):
    """Filters one alignment per gene at a time."""

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
        na_alignment = str(Path(self.gene + '_P2N_na'))  # Pal2Nal output file

        # Guidance2 iterations over the nucleic acid sequences.
        # Returns the file name that contains the filtered sequences.
        rem_na_seqFile = self.nucleic_acid_guidance(seqFile=na_seqFile,
                                                    outDir=str(self.na_guidance_path),
                                                    bootstraps=na_bootstraps,
                                                    seqCutoff=na_seqCutoff,
                                                    colCutoff=na_colCutoff)

        # Guidance 2 amino acid alignment filter.  Filters the sequences based on the NA_Guidance2 runs.
        # Returns the file name that contains the filtered alignment.
        aa_alignment = self.amino_acid_guidance(seqFile=aa_seqFile,
                                                remFile=rem_na_seqFile,
                                                outDir=str(self.aa_guidance_path),
                                                bootstraps=aa_bootstraps,
                                                seqCutoff=aa_seqCutoff,
                                                colCutoff=aa_colCutoff)
        g2_seqFile = str(self.home / Path(self.gene + '_G2.ffn'))

        # PAL2NAL nucleic acid alignment
        self.pal2nal_conversion(str(aa_alignment), g2_seqFile, na_alignment)

    def nucleic_acid_guidance(self, seqFile, outDir, bootstraps, seqCutoff, colCutoff):
        seqType = 'nuc'
        iteration_flag = True
        iteration = 1
        while iteration_flag is True:
            iterDir = Path(outDir) / Path('iter_%s' % iteration)  # /home/NA_Guidance2/iter_n
            Path.mkdir(iterDir, parents=True, exist_ok=True)

            g2_seqFile = str(self.home / Path(self.gene + '_G2.ffn'))  # Need for all iterations
            g2_rem_file = str(Path(iterDir) / Path('Seqs.Orig.fas.FIXED.Removed_Seq.With_Names'))  # Need for all iterations



            rem_file = str(self.home / Path(self.gene + '_G2_removed.ffn'))   # Need for all iterations

            if iteration == 1:
                # seqFile is the given input
                G2Cmd = Guidance2Commandline(self.G2C_args, seqFile=seqFile,
                                             seqType=seqType, outDir=str(iterDir),
                                             bootstraps=bootstraps,
                                             seqCutoff=seqCutoff, colCutoff=colCutoff)
                print(G2Cmd)
                subprocess.check_call([str(G2Cmd)], stderr=subprocess.STDOUT, shell=True)
                # Copy the Guidance removed seq file and paste it to the home directory
                # Creates the rem_file
                # Files without any removed don't have the file *.With_Names
                if os.path.isfile(g2_rem_file) is False:
                    g2_rem_file = str(Path(iterDir) / Path('Seqs.Orig.fas.FIXED.Removed_Seq'))
                SeqIO.write(SeqIO.parse(g2_rem_file, 'fasta'), rem_file, 'fasta')  # Need for iter_1

                # Filter the input NA fasta file using Guidance output
                # Creates the g2_seqFile
                multi_fasta_manipulator(seqFile, g2_rem_file, g2_seqFile, manipulation='remove')  # Do after copying (iter_1) or adding (iter_n)
                iteration_flag = True
            elif iteration > 1:
                # seqFile changes to g2_seqFile and the cutoffs change
                seqCutoff = 0.7
                colCutoff = 0.1
                G2Cmd = Guidance2Commandline(self.G2C_args, seqFile=g2_seqFile, seqType=seqType, outDir=str(iterDir), bootstraps=bootstraps,
                                             seqCutoff=seqCutoff, colCutoff=colCutoff)
                print(G2Cmd)
                subprocess.check_call([str(G2Cmd)], stderr=subprocess.STDOUT, shell=True)

                # Get the removed sequence count
                rem_count = 0
                # Files without any removed don't have the file *.With_Names
                if os.path.isfile(g2_rem_file) is False:
                    g2_rem_file = str(Path(iterDir) / Path('Seqs.Orig.fas.FIXED.Removed_Seq'))

                for rec in SeqIO.parse(g2_rem_file, 'fasta'):  # Need for all iterations
                    rem_count += 1

                if rem_count > 0:
                    # Add new sequences to the rem_file
                    multi_fasta_manipulator(rem_file, g2_rem_file, rem_file, manipulation='add')
                    # Filter the input NA fasta file using the updated rem_file
                    multi_fasta_manipulator(seqFile, rem_file, g2_seqFile, manipulation='remove')
                    iteration_flag = True
                else:
                    iteration_flag = False
            iteration += 1

        return rem_file

    def amino_acid_guidance(self, seqFile, remFile, outDir, bootstraps, seqCutoff, colCutoff):
        seqType = 'aa'
        g2_seqFile = str(self.home / Path(self.gene + '_G2.faa'))
        multi_fasta_manipulator(seqFile, remFile, g2_seqFile)
        G2Cmd = Guidance2Commandline(self.G2C_args, seqFile=g2_seqFile, seqType=seqType, outDir=outDir, bootstraps=bootstraps,
                                     seqCutoff=seqCutoff, colCutoff=colCutoff)
        print(G2Cmd)
        subprocess.check_call([str(G2Cmd)], stderr=subprocess.STDOUT, shell=True)
        # DO not remove any columns.
        filtered_alignment = Path(outDir) / Path('%s.CLUSTALW.aln.Sorted.With_Names' % self.G2C_args['dataset'])
        renamed_alignment = copy(str(filtered_alignment), str(Path(self.gene + '_G2_aa.aln')))
        renamed_alignment = multi_fasta_manipulator(str(renamed_alignment), str(seqFile), str(renamed_alignment), manipulation='sort')
        print('Align the filtered amino acid sequences using guidance 2')
        return Path(renamed_alignment)

    def pal2nal_conversion(self, aa_alignment, na_fasta, output_file):
        # TODO-ROB:  Add a filter step so if error[0] = Error: #---  ERROR: inconsistency between the following pep and nuc seqs  ---#
        # TODO-ROB:  ....then remove error[2] which is the sequence it can't read
        removed = []
        # Create output directory for PAL2NAL
        outDir = self.home / Path('PAL2NAL')
        Path.mkdir(outDir, exist_ok=True)
        output_file = str(outDir / Path(output_file))

        # Create an alignment for paml input
        P2Ncmd = Pal2NalCommandline(self.P2N_args, pepaln=aa_alignment, nucfasta=na_fasta, output_file=output_file + '.paml.aln',
                                    output='paml')
        print(P2Ncmd)
        pal2nal_flag = True
        while pal2nal_flag is True:
            pal2nal = subprocess.Popen([str(P2Ncmd)], stderr=subprocess.PIPE, stdout=subprocess.PIPE, shell=True, encoding='utf-8')
            error = pal2nal.stderr.readlines()
            out = pal2nal.stdout.readlines()
            pal2nal.wait()
            if 'ERROR: inconsistency between the following pep and nuc seqs' in error[0]:
                print('Caught the pal2nal error!')
                print(error[0])
                for err in error:
                    if '>' in err:
                        removed.append(err.strip('>' '\n'))
                multi_fasta_manipulator(na_fasta, removed, na_fasta)
                multi_fasta_manipulator(aa_alignment, removed, aa_alignment)
            else:
                pal2nal_flag = False

            print('Error: ' + str(error))
            print('Out: ' + str(out))
        # Create an alignment for iqtree input
        P2Ncmd = Pal2NalCommandline(self.P2N_args, pepaln=aa_alignment, nucfasta=na_fasta, output_file=output_file + '.iqtree.aln',
                                    output='fasta')
        print(P2Ncmd)
        pal2nal = subprocess.Popen([str(P2Ncmd)], stderr=subprocess.PIPE, stdout=subprocess.PIPE, shell=True, encoding='utf-8')
        error = pal2nal.stderr.read()
        out = pal2nal.stdout.read()
        pal2nal.wait()

        print('Error: ' + str(error))
        print('Out: ' + str(out))

        print('Align the nucleic acids using the amino acid alignment.')
