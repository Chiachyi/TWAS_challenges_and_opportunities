# Transcriptome-wide association studies: challenges and opportunities

## Code files

**case_studies.py** generates Manhattan plots for the 3 case study loci (_SORT1_, _IRF2BP2_, _NOD2_) and prints out information about the expression models of each of the genes at those loci.

**global_analysis.py** generates the global analysis plots (Figs. S1 and S4) and reports the number of genes subject to each co-regulation scenario.

**wrong_tissue.py** generates the bar plots in Fig. 6.

To run case_studies.py and global_analysis.py, you will need to obtain the STARNET genotypes, e.g. through dbGAP (https://www.ncbi.nlm.nih.gov/projects/gap/cgi-bin/study.cgi?study_id=phs001203.v1.p1), and place the genotype files (named STARNET.bed, STARNET.bim and STARNET.fam) in data/STARNET/genotypes/ and the expression files (named STARNET.AOR.exp.gz, STARNET.LIV.exp.gz and similar) in data/STARNET/expression/.

## Data files

**data/LDL** and **data/crohns** contain the GWAS summary statistics for LDL and Crohn's.

**data/WEIGHTS** contains the weight files output by Fusion for each gene, in RDat format.  There is one subdirectory for each of the four TWAS (Crohn's/whole blood, LDL/whole blood, Crohn's/liver, LDL/liver).

**data/results** contains the TWAS results.  Again, there is one subdirectory for each of the four TWAS.  **data/results/\*/TWAS_hits.txt** are  human-readable text files with information on the Bonferroni-significant TWAS hits, with 2.5 MB clumping.  **data/results/\*/chr\*.txt** contain the full TWAS summary statistics for each chromosome, in the tab-delimited format output by Fusion.

**data/ensembl** contains a list of Ensembl genes and their locations.
