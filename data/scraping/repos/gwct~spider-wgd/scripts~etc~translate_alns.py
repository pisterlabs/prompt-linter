#############################################################################
# Translate CDS alignments from Guidance to AA alignments for trimal
#
# Authors: Gregg Thomas
#############################################################################

import sys
import os
import seqparse as SEQ

#############################################################################

#infilename = "/n/holylfs05/LABS/informatics/Users/gthomas/spiders/seq/guidance-test-out-cds-2/ORTHOMCL999-cds.MAFFT.Without_low_SP_Col.With_Names";
#outfilename = "/n/holylfs05/LABS/informatics/Users/gthomas/spiders/seq/guidance-test-out-cds-2/test.fa";
#infilename, outfilename = sys.argv[1], sys.argv[2];

indir = sys.argv[1];
indir = "";

for d in os.listdir(indir):
    locus_dir = os.path.join(indir, d);

    infilename = [ os.path.join(locus_dir, f) for f in os.listdir(locus_dir) if f.endswith("-cds.MAFFT.Without_low_SP_Col.With_Names") ];
    outfilename = infilename.replace("-cds.MAFFT.Without_low_SP_Col.With_Names", "-pep.MAFFT.Without_low_SP_Col.With_Names");

    assert os.path.isfile(infilename), " * ERROR: Cannot find file: " + infilename;

    nt_seqs = SEQ.fastaGetDict(infilename);

    with open(outfilename, "w") as outfile:
        for header in nt_seqs:
            codon_seq = SEQ.ntToCodon(nt_seqs[header]);
            aa_seq = SEQ.anotherBioTranslator(codon_seq);

            outfile.write(header + "\n");
            outfile.write(aa_seq + "\n");

