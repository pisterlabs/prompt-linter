# Standard Library
import os
import shutil
import subprocess
from pathlib import Path
# BioPython
from Bio import SeqIO
from Bio.Align.Applications import ClustalOmegaCommandline
# OrthoEvol
from OrthoEvol.Tools.logit import LogIt
from OrthoEvol.utilities import FullUtilities
from OrthoEvol.Orthologs.GenBank import GenBank
from OrthoEvol.Orthologs.Align.pal2nal import Pal2NalCommandline
from OrthoEvol.Orthologs.Align.guidance2 import Guidance2Commandline
from OrthoEvol.Orthologs.Align.orthoclustal import ClustalO


class MultipleSequenceAlignment(object):
    """The MultipleSequenceAlignment (MSA) class uses the standard configuration
    along with function dispatching to give the end-user access to multiple
    alignment tools."""

    def __init__(self, project=None, project_path=os.getcwd(), genbank=GenBank, **kwargs):
        """Initialize the MultipleSequenceAlignment class.

        :param project: The project name.
        :param project_path:  The path to the project.
        :param genbank: The composer parameter which is used to configure the
                        GenBank class with the MSA class.
        :param kwargs:  The kwargs are used with the dispatcher as a way to
                        control the alignment pipeline.
        :returns: If the kwargs are utilized with YAML or other
                  configurations, then this class returns an alignment
        dictionary, which can be parsed to run specific alignment algorithms.
        """
        self.dispatcher_options = {"Guidance_config": ["GUIDANCE2", self.guidance2],
                                   "Pal2Nal_config": ["PAL2NAL", self.pal2nal],
                                   "ClustalO_config": ["CLUSTALO", self.clustalo]}
        # Set up loggers
        __log = LogIt()
        __logfile = None
        self.guidancelog = __log.default('guidance2', __logfile)
        self.pal2nallog = __log.default('pal2nal', __logfile)
        self.clustalolog = __log.default('clustalo', __logfile)

        # Initialize Utilities
        self.msa_utils = FullUtilities()

        # stop_codons = ['TAG', 'TAA', 'TGA']

        self.program = None
        self.alignment_dict = {}
        self.project = project
        self.project_path = project_path
        if project_path and project:
            self.project_path = Path(project_path) / Path(project)

        # Configuration of class attributes
        add_self = self.msa_utils.attribute_config(self, composer=genbank, checker=GenBank, project=project, project_path=project_path)
        for var, attr in add_self.__dict__.items():
            setattr(self, var, attr)

        # Determine which alignment to configure
        # And then run that alignment with the configuration.
        for config in self.dispatcher_options.keys():
            if config in kwargs.keys():
                program = self.dispatcher_options[config][0]
                aligner = self.dispatcher_options[config][1]
                aligner_configuration = kwargs[config]
                self.alignment_dict[program] = [aligner, aligner_configuration]

    def guidance2(self, seqFile, msaProgram, seqType, dataset='MSA', seqFilter=None, columnFilter=None, maskFilter=None, **kwargs):
        """Run the GUIDANCE2 command line wrapper from BioPython.

        The Guidance2 algorithm is used to filter sequence alignments in
        different ways.  Here we employ a few of our own strategies on top of
        Guidance2.

        :param seqFile:  The sequence file required by GUIDANCE2.
        :param msaProgram:  The msa program to be used by GUIDANCE2.
                            ("CLUSTALW", "PRANK", "MAFFT", or "MUSCLE")
        :param seqType: The type of sequences to be aligned in GUIDANCE2.
                        ("aa", "nuc", or "codon")
        :param dataset: The name of the dataset, which is used for file
                        naming convention among other things in GUIDANCE2.
        :param seqFilter: The sequence filter parameter is None, "inclusive",
                          or "exclusive".  If inclusive the SeqCutoff
                          decreases for every iteration.  If exclusive the
                          SeqCutoff increases for every iteration, and so the
                          algorithm excludes more genes from the alignment.
                          (An OrthoEvol strategy)
        :param columnFilter: The column filter removes columns from the
                             alignment using GUIDANCE2.
        :param maskFilter: The mask filter uses GUIDANCE2 maskLowScoresResidue
                           script to mask the low scoring residues.
        :param kwargs: The kwargs are used to configure GUIDANCE2 with
                       specific parameters including seqCutoff and colCutoff.
                       It can also be used to set the number of iterations and
                       the increment number, which controls how seqCutoff and
                       colCutoff change for each iteration.
        :return:  Returns Guidance2 files.
        """

        self.guidancelog.info("Guidance2 will be used.")
        # Name and Create the output directory
        self.program = "GUIDANCE2"
        outDir = self.program
        gene = Path(seqFile).stem
        geneDir = self.raw_data / Path(gene)
        self.guidancelog.info(geneDir)
        if seqType == 'nuc':
            g2_seqFile = str(geneDir / Path(gene + '_G2.ffn'))  # Need for all iterations
            rem_file = str(geneDir / Path(gene + '_G2_removed.ffn'))   # Need for all iterations
            g2_alnFile = str(geneDir / Path(gene + '_G2_na.aln'))
            g2_seqcolFilter = str(geneDir / Path(gene + 'G2sfcf_na.aln'))
            g2_colFilter = str(geneDir / Path(gene + '_G2cf_na.aln'))
            g2_maskedFile = str(geneDir / Path(gene + '_G2mf_na.aln'))
        elif seqType == 'aa':
            g2_seqFile = str(geneDir / Path(gene + '_G2.faa'))  # Need for all iterations
            rem_file = str(geneDir / Path(gene + '_G2_removed.faa'))   # Need for all iterations
            g2_alnFile = str(geneDir / Path(gene + '_G2_aa.aln'))
            g2_colFilter = str(geneDir / Path(gene + '_G2cf_aa.aln'))
            g2_maskedFile = str(geneDir / Path(gene + '_G2mf_aa.aln'))

        # Add the Guidance 2 cutoffs to the keyword arguments
        if 'seqCutoff' not in kwargs.keys():
            kwargs['seqCutoff'] = 0.6
        if 'colCutoff' not in kwargs.keys():
            kwargs['colCutoff'] = 0.93

        # Filter Sequences, and then either remove columns, mask residues or do nothing
        if seqFilter is not None:

            # Filter "bad" sequences and iterate over the good ones until the cutoff is reached.
            iterFlag = True
            iteration = 0
            while iterFlag is True:
                set_iter = kwargs['iterations']
                iteration += 1
                # Create paths for output files
                if columnFilter is not None:
                    outDir = self.raw_data / Path(gene) / Path(outDir + '_sf_cf')
                elif maskFilter is not None:
                    outDir = self.raw_data / Path(gene) / Path(outDir + '_sf_mf')
                else:
                    outDir = self.raw_data / Path(gene) / Path(outDir + '_sf')
                Path.mkdir(outDir, parents=True, exist_ok=True)
                iterDir = Path(outDir) / Path('iter_%s' % iteration)
                g2_rem_file = str(iterDir / Path('Seqs.Orig.fas.FIXED.Removed_Seq.With_Names'))  # Need for all iterations
                Path.mkdir(iterDir, parents=True, exist_ok=True)

                # Create files for masking
                if maskFilter is not None:
                    g2_aln2mask = str(iterDir / Path('%s.%s.aln.With_Names' % (dataset, msaProgram)))
                    g2_rprScores = str(iterDir / Path('%s.%s.Guidance2_res_pair_res.scr' % (dataset, msaProgram)))

                if iteration == 1:

                    # seqFile is the given input
                    G2Cmd = Guidance2Commandline(seqFile=seqFile, msaProgram=msaProgram, seqType=seqType,
                                                 outDir=str(iterDir), **kwargs)
                    self.guidancelog.info(G2Cmd)
                    subprocess.check_call([str(G2Cmd)], stderr=subprocess.STDOUT, shell=True)
                    # Copy the Guidance removed seq file and paste it to the home directory
                    # Creates the rem_file
                    # Files without any removed don't have the file *.With_Names
                    if os.path.isfile(g2_rem_file) is False:
                        g2_rem_file = str(iterDir / Path('Seqs.Orig.fas.FIXED.Removed_Seq'))
                    SeqIO.write(SeqIO.parse(g2_rem_file, 'fasta'), rem_file, 'fasta')  # Need for iter_1

                    # Filter the input NA fasta file using Guidance output
                    # Creates the g2_seqFile
                    self.msa_utils.multi_fasta_manipulator(seqFile, g2_rem_file, g2_seqFile, manipulation='remove')  # Do after copying (iter_1) or adding (iter_n)
                    iterFlag = True

                elif set_iter >= iteration > 1:

                    # Depending on the filter strategy increment the seqCutoff
                    if seqFilter == "inclusive":
                        kwargs['seqCutoff'] -= kwargs['increment']
                    elif seqFilter == "exclusive":
                        kwargs['seqCutoff'] += kwargs['increment']
                    # seqFile changes to g2_seqFile and the cutoffs change
                    G2Cmd = Guidance2Commandline(seqFile=g2_seqFile, msaProgram=msaProgram, seqType=seqType,
                                                 outDir=str(iterDir), **kwargs)
                    self.guidancelog.info(G2Cmd)
                    subprocess.check_call([str(G2Cmd)], stderr=subprocess.STDOUT, shell=True)

                    # Get the removed sequence count
                    rem_count = 0
                    # Files without any removed don't have the file *.With_Names
                    if os.path.isfile(g2_rem_file) is False:
                        g2_rem_file = str(Path(iterDir) / Path('Seqs.Orig.fas.FIXED.Removed_Seq'))

                    for rec in SeqIO.parse(g2_rem_file, 'fasta'):  # Need for all iterations
                        rem_count += 1

                    # If sequences are removed, then iterate again on the "good" sequences
                    if rem_count > 0:
                        # Add new sequences to the rem_file
                        self.msa_utils.multi_fasta_manipulator(rem_file, g2_rem_file, rem_file, manipulation='add')
                        # Filter the input fasta file using the updated rem_file
                        self.msa_utils.multi_fasta_manipulator(seqFile, rem_file, g2_seqFile, manipulation='remove')
                        iterFlag = True
                    # If sequences aren't removed, then stop iterating
                    if rem_count < 0 or set_iter == iteration:
                        filtered_alignment = Path(iterDir) / Path('%s.%s.aln.Sorted.With_Names' % (dataset, msaProgram))
                        renamed_alignment = shutil.copy(str(filtered_alignment), g2_alnFile)
                        self.msa_utils.multi_fasta_manipulator(str(renamed_alignment), str(seqFile), str(renamed_alignment), manipulation='sort')
                        iterFlag = False

            if columnFilter is not None:
                col_filt_align = iterDir / Path('%s.%s.Without_low_SP_Col.With_Names' % (dataset, msaProgram))
                shutil.copy(str(col_filt_align), g2_seqcolFilter)

            elif maskFilter is not None:
                G2Cmd = Guidance2Commandline(align=False, seqFile=seqFile, msaProgram=msaProgram, seqType=seqType,
                                             outDir=str(iterDir), maskCutoff=maskFilter, maskFile=g2_aln2mask,
                                             rprScores=g2_rprScores, output=g2_maskedFile, **kwargs)
                self.alignmentlog.info(G2Cmd)
                subprocess.check_call([str(G2Cmd)], stderr=subprocess.STDOUT, shell=True)
                self.msa_utils.multi_fasta_manipulator(g2_maskedFile, str(seqFile), g2_maskedFile, manipulation='sort')

        # Only COLUMN FILTER the bad columns
        elif columnFilter is not None:
            outDir = self.raw_data / Path(gene) / Path(outDir + '_cf')
            Path.mkdir(outDir, parents=True, exist_ok=True)
            G2Cmd = Guidance2Commandline(seqFile=seqFile, msaProgram=msaProgram, seqType=seqType,
                                         outDir=str(outDir), **kwargs)
            self.guidancelog.info(G2Cmd)
            subprocess.check_call([str(G2Cmd)], stderr=subprocess.STDOUT, shell=True)
            col_filt_align = outDir / Path('%s.%s.Without_low_SP_Col.With_Names' % (dataset, msaProgram))
            shutil.copy(str(col_filt_align), g2_colFilter)

        # Only MASK the bad residues
        elif maskFilter is not None:
            outDir = self.raw_data / Path(gene) / Path(outDir + '_sf')
            G2Cmd = Guidance2Commandline(seqFile=seqFile, msaProgram=msaProgram, seqType=seqType,
                                         outDir=str(outDir), maskCutoff=maskFilter, maskFile=kwargs['aln2mask'],
                                         rprScores=kwargs['rprScores'], output=kwargs['maskedFile'], **kwargs)
            self.guidancelog.info(G2Cmd)
            subprocess.check_call([str(G2Cmd)], stderr=subprocess.STDOUT, shell=True)
            self.msa_utils.multi_fasta_manipulator(kwargs['maskedFile'], str(seqFile), kwargs['maskedFile'], manipulation='sort')

    def pal2nal(self, aa_alignment, na_fasta, output_type='paml', nogap=True, nomismatch=True, downstream='paml'):
        """This Pal2Nal method works with the Pal2Nal command line wrapper.

        It uses a protein alignment to generate a codon alignment from the
        corresponding nucleic acid sequences.  This is useful for downstream
        PAML analysis. This function also catches and removes taxa that are
        inconsistent with Pal2Nal's algorithm.

        :param aa_alignment: An amino acid alignment that is used as a guide
                             for a nucleic acid alignment.
        :param na_fasta: The FASTA file that contains matching/ordered
                         sequences corresponding to the aa_alignment.
        :param output_type: The format of the resulting alignment.
                            ("clustal", "paml", "fasta", "codon")
        :param nogap: Removes the gaps and in-frame stop codons from the
                      alignment to work better with PAML.
        :param nomismatch: Removes mismatched codons between protein and DNA
                           sequences.
        :param downstream: Used as a naming convention for a better and more
                           obvious pipeline.
        :return: A codon alignment.
        """

        removed = []
        # Create output directory for PAL2NAL
        outDir = 'PAL2NAL'
        gene = Path(na_fasta).stem
        geneDir = self.raw_data / Path(gene)
        outDir = geneDir / Path(outDir)
        Path.mkdir(outDir, exist_ok=True)

        # Name the output file using the gene and the downstream application
        output_file = str(geneDir / Path(gene + '_P2N.%s.aln' % downstream))

        # Create an alignment
        P2Ncmd = Pal2NalCommandline(pepaln=aa_alignment, nucfasta=na_fasta, output_file=output_file, output=output_type,
                                    nogap=nogap, nomismatch=nomismatch)
        self.pal2nallog.info(P2Ncmd)

        # Use a while loop to catch errors and remove sequences that aren't working with pal2nal
        pal2nal_flag = True
        while pal2nal_flag is True:
            pal2nal = subprocess.Popen([str(P2Ncmd)], stderr=subprocess.PIPE,
                                       stdout=subprocess.PIPE, shell=True,
                                       encoding='utf-8')
            error = pal2nal.stderr.readlines()
            out = pal2nal.stdout.readlines()
            pal2nal.wait()

            # Catch errors
            if 'ERROR: inconsistency between the following pep and nuc seqs' in error[0]:
                self.pal2nallog.warning('Caught the pal2nal error!')
                self.pal2nallog.warning(error[0])
                for err in error:
                    if '>' in err:
                        removed.append(err.strip('>' '\n'))
                self.msa_utils.multi_fasta_manipulator(na_fasta, removed, na_fasta)
                self.msa_utils.multi_fasta_manipulator(aa_alignment, removed, aa_alignment)

            # If no errors then break the while loop
            else:
                if removed is not None:
                    # If any sequences were removed then write them to a file.
                    p2n_remFile = str(geneDir / Path(gene + '_P2N_removed.txt'))
                    with open(p2n_remFile, 'w') as p2n_rem:
                        for name in removed:
                            p2n_rem.write(name)
                pal2nal_flag = False

            self.pal2nallog.info('Error: ' + str(error))
            self.pal2nallog.info('Out: ' + str(out))

    def clustalo(self, infile, outfile, outfmt="fasta"):
        """Align protein/amino acid sequences using Clustal Omega.

        :param infile: Input a multifasta protein file.
        :param outfile: Output an aligned multifasta file.
        :param outfmt:  (Default value = "fasta")
        """
        try:
            clustalo = ClustalO(infile=infile, outfile=outfile, outfmt=outfmt)
            clustalo.runclustalomega()
        except Exception:
            pass
