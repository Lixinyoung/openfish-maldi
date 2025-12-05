from Bio.SeqUtils import MeltingTemp
from Bio.Seq import Seq
from Bio.Seq import complement
import pandas as pd
import numpy as np
import math
from collections import Counter
import pysam as ps

def get_tm(seq, c_seq, Para):
    
    table_dict = {
         '--/AA': (-6, -13),
         'AA/--': (-6, -13),
         'A-/TA': (-6, -13),
         '-A/AT': (-6, -13),
         'TA/A-': (-6, -13),
         'AT/-A': (-6, -13),
         '--/TA': (-6, -13),
         'TA/--': (-6, -13),
         'A-/TT': (-6.5, -13),
         '-A/TT': (-6.5, -13),
         'TT/A-': (-6.5, -13),
         'TT/-A': (-6.5, -13),
         '--/CA': (-6, -13),
         'CA/--': (-6, -13),
         'A-/TC': (-6.5, -13),
         '-A/CT': (-6.5, -13),
         'TC/A-': (-6.5, -13),
         'CT/-A': (-6.5, -13),
         '--/GA': (-6, -13),
         'GA/--': (-6, -13),
         'A-/TG': (-6, -13),
         '-A/GT': (-6, -13),
         'TG/A-': (-6, -13),
         'GT/-A': (-6, -13),
         '--/AT': (-6, -13),
         'AT/--': (-6, -13),
         'T-/AA': (-6, -13),
         '-T/AA': (-6, -13),
         'AA/T-': (-6, -13),
         'AA/-T': (-6, -13),
         '--/TT': (-6, -13),
         'TT/--': (-6, -13),
         'T-/AT': (-6.5, -13),
         '-T/TA': (-6.5, -13),
         'AT/T-': (-6.5, -13),
         'TA/-T': (-6.5, -13),
         '--/CT': (-6, -13),
         'CT/--': (-6, -13),
         'T-/AC': (-6.5, -13),
         '-T/CA': (-6.5, -13),
         'AC/T-': (-6.5, -13),
         'CA/-T': (-6.5, -13),
         '--/GT': (-6, -13),
         'GT/--': (-6, -13),
         'T-/AG': (-6, -13),
         '-T/GA': (-6, -13),
         'AG/T-': (-6, -13),
         'GA/-T': (-6, -13),
         '--/AC': (-6, -13),
         'AC/--': (-6, -13),
         'C-/GA': (-6.5, -14),
         '-C/AG': (-6.5, -14),
         'GA/C-': (-6.5, -14),
         'AG/-C': (-6.5, -14),
         '--/TC': (-6, -13),
         'TC/--': (-6, -13),
         'C-/GT': (-6.7, -12.9),
         '-C/TG': (-6.7, -12.9),
         'GT/C-': (-6.7, -12.9),
         'TG/-C': (-6.7, -12.9),
         '--/CC': (-6, -13),
         'CC/--': (-6, -13),
         'C-/GC': (-6.7, -12.9),
         '-C/CG': (-6.7, -12.9),
         'GC/C-': (-6.7, -12.9),
         'CG/-C': (-6.7, -12.9),
         '--/GC': (-6, -13),
         'GC/--': (-6, -13),
         'C-/GG': (-6.5, -14),
         '-C/GG': (-6.5, -14),
         'GG/C-': (-6.5, -14),
         'GG/-C': (-6.5, -14),
         '--/AG': (-6, -13),
         'AG/--': (-6, -13),
         'G-/CA': (-6.5, -14),
         '-G/AC': (-6.5, -14),
         'CA/G-': (-6.5, -14),
         'AC/-G': (-6.5, -14),
         '--/TG': (-6, -13),
         'TG/--': (-6, -13),
         'G-/CT': (-6.7, -12.9),
         '-G/TC': (-6.7, -12.9),
         'CT/G-': (-6.7, -12.9),
         'TC/-G': (-6.7, -12.9),
         '--/CG': (-6, -13),
         'CG/--': (-6, -13),
         'G-/CC': (-6.7, -12.9),
         '-G/CC': (-6.7, -12.9),
         'CC/G-': (-6.7, -12.9),
         'CC/-G': (-6.7, -12.9),
         '--/GG': (-6, -13),
         'GG/--': (-6, -13),
         'G-/CG': (-6.5, -14),
         '-G/GC': (-6.5, -14),
         'CG/G-': (-6.5, -14),
         'GC/-G': (-6.5, -14),
         'AA/AA': (-6.5, -14),
         'AA/AC': (-6.5, -14),
         'AC/AA': (-6.5, -14),
         'AA/AG': (-6.5, -14),
         'AG/AA': (-6.5, -14),
         'AA/CA': (-6.5, -14),
         'CA/AA': (-6.5, -14),
         'AA/CC': (-6.5, -14),
         'CC/AA': (-6.5, -14),
         'AA/CG': (-6.5, -14),
         'CG/AA': (-6.5, -14),
         'AA/GA': (-6.5, -14),
         'GA/AA': (-6.5, -14),
         'AA/GC': (-6.5, -14),
         'GC/AA': (-6.5, -14),
         'AA/GG': (-6.5, -14),
         'GG/AA': (-6.5, -14),
         'AT/AT': (-6.5, -14),
         'AT/AC': (-6.5, -14),
         'AC/AT': (-6.5, -14),
         'AT/AG': (-6.5, -14),
         'AG/AT': (-6.5, -14),
         'AT/CT': (-6.5, -14),
         'CT/AT': (-6.5, -14),
         'AT/CC': (-6.5, -14),
         'CC/AT': (-6.5, -14),
         'AT/CG': (-6.5, -14),
         'CG/AT': (-6.5, -14),
         'AT/GT': (-6.5, -14),
         'GT/AT': (-6.5, -14),
         'AT/GC': (-6.5, -14),
         'GC/AT': (-6.5, -14),
         'AT/GG': (-6.5, -14),
         'GG/AT': (-6.5, -14),
         'AC/AC': (-6.5, -14),
         'AC/CC': (-6.5, -14),
         'CC/AC': (-6.5, -14),
         'AC/CA': (-6.5, -14),
         'CA/AC': (-6.5, -14),
         'AC/CT': (-6.5, -14),
         'CT/AC': (-6.5, -14),
         'AC/GC': (-6.5, -14),
         'GC/AC': (-6.5, -14),
         'AC/GA': (-6.5, -14),
         'GA/AC': (-6.5, -14),
         'AC/GT': (-6.5, -14),
         'GT/AC': (-6.5, -14),
         'AG/AG': (-6.5, -14),
         'AG/CG': (-6.5, -14),
         'CG/AG': (-6.5, -14),
         'AG/CA': (-6.5, -14),
         'CA/AG': (-6.5, -14),
         'AG/CT': (-6.5, -14),
         'CT/AG': (-6.5, -14),
         'AG/GG': (-6.5, -14),
         'GG/AG': (-6.5, -14),
         'AG/GA': (-6.5, -14),
         'GA/AG': (-6.5, -14),
         'AG/GT': (-6.5, -14),
         'GT/AG': (-6.5, -14),
         'TA/TA': (-6.5, -14),
         'TA/TC': (-6.5, -14),
         'TC/TA': (-6.5, -14),
         'TA/TG': (-6.5, -14),
         'TG/TA': (-6.5, -14),
         'TA/CA': (-6.5, -14),
         'CA/TA': (-6.5, -14),
         'TA/CC': (-6.5, -14),
         'CC/TA': (-6.5, -14),
         'TA/CG': (-6.5, -14),
         'CG/TA': (-6.5, -14),
         'TA/GA': (-6.5, -14),
         'GA/TA': (-6.5, -14),
         'TA/GC': (-6.5, -14),
         'GC/TA': (-6.5, -14),
         'TA/GG': (-6.5, -14),
         'GG/TA': (-6.5, -14),
         'TT/TT': (-6.5, -14),
         'TT/TC': (-6.5, -14),
         'TC/TT': (-6.5, -14),
         'TT/TG': (-6.5, -14),
         'TG/TT': (-6.5, -14),
         'TT/CT': (-6.5, -14),
         'CT/TT': (-6.5, -14),
         'TT/CC': (-6.5, -14),
         'CC/TT': (-6.5, -14),
         'TT/CG': (-6.5, -14),
         'CG/TT': (-6.5, -14),
         'TT/GT': (-6.5, -14),
         'GT/TT': (-6.5, -14),
         'TT/GC': (-6.5, -14),
         'GC/TT': (-6.5, -14),
         'TT/GG': (-6.5, -14),
         'GG/TT': (-6.5, -14),
         'TC/TC': (-6.5, -14),
         'TC/CC': (-6.5, -14),
         'CC/TC': (-6.5, -14),
         'TC/CA': (-6.5, -14),
         'CA/TC': (-6.5, -14),
         'TC/CT': (-6.5, -14),
         'CT/TC': (-6.5, -14),
         'TC/GC': (-6.5, -14),
         'GC/TC': (-6.5, -14),
         'TC/GA': (-6.5, -14),
         'GA/TC': (-6.5, -14),
         'TC/GT': (-6.5, -14),
         'GT/TC': (-6.5, -14),
         'TG/TG': (-6.5, -14),
         'TG/CG': (-6.5, -14),
         'CG/TG': (-6.5, -14),
         'TG/CA': (-6.5, -14),
         'CA/TG': (-6.5, -14),
         'TG/CT': (-6.5, -14),
         'CT/TG': (-6.5, -14),
         'TG/GG': (-6.5, -14),
         'GG/TG': (-6.5, -14),
         'TG/GA': (-6.5, -14),
         'GA/TG': (-6.5, -14),
         'TG/GT': (-6.5, -14),
         'GT/TG': (-6.5, -14),
         'CA/CA': (-6.5, -14),
         'CA/CC': (-6.5, -14),
         'CC/CA': (-6.5, -14),
         'CA/CG': (-6.5, -14),
         'CG/CA': (-6.5, -14),
         'CT/CT': (-6.5, -14),
         'CT/CC': (-6.5, -14),
         'CC/CT': (-6.5, -14),
         'CT/CG': (-6.5, -14),
         'CG/CT': (-6.5, -14),
         'CC/CC': (-6.5, -14),
         'CG/CG': (-6.5, -14),
         'GA/GA': (-6.5, -14),
         'GA/GC': (-6.5, -14),
         'GC/GA': (-6.5, -14),
         'GA/GG': (-6.5, -14),
         'GG/GA': (-6.5, -14),
         'GT/GT': (-6.5, -14),
         'GT/GC': (-6.5, -14),
         'GC/GT': (-6.5, -14),
         'GT/GG': (-6.5, -14),
         'GG/GT': (-6.5, -14),
         'GC/GC': (-6.5, -14),
         'GG/GG': (-6.5, -14),
         'A-/AA': (-6, -14),
         '-A/AA': (-6, -14),
         'AA/A-': (-6, -14),
         'AA/-A': (-6, -14),
         'A-/CA': (-6, -14),
         '-A/AC': (-6, -14),
         'CA/A-': (-6, -14),
         'AC/-A': (-6, -14),
         'A-/GA': (-6, -14),
         '-A/AG': (-6, -14),
         'GA/A-': (-6, -14),
         'AG/-A': (-6, -14),
         'A-/AT': (-6, -14),
         '-A/TA': (-6, -14),
         'AT/A-': (-6, -14),
         'TA/-A': (-6, -14),
         'A-/CT': (-6, -14),
         '-A/TC': (-6, -14),
         'CT/A-': (-6, -14),
         'TC/-A': (-6, -14),
         'A-/GT': (-6, -14),
         '-A/TG': (-6, -14),
         'GT/A-': (-6, -14),
         'TG/-A': (-6, -14),
         'A-/AC': (-6, -14),
         '-A/CA': (-6, -14),
         'AC/A-': (-6, -14),
         'CA/-A': (-6, -14),
         'A-/CC': (-6, -14),
         '-A/CC': (-6, -14),
         'CC/A-': (-6, -14),
         'CC/-A': (-6, -14),
         'A-/GC': (-6, -14),
         '-A/CG': (-6, -14),
         'GC/A-': (-6, -14),
         'CG/-A': (-6, -14),
         'A-/AG': (-6, -14),
         '-A/GA': (-6, -14),
         'AG/A-': (-6, -14),
         'GA/-A': (-6, -14),
         'A-/CG': (-6, -14),
         '-A/GC': (-6, -14),
         'CG/A-': (-6, -14),
         'GC/-A': (-6, -14),
         'A-/GG': (-6, -14),
         '-A/GG': (-6, -14),
         'GG/A-': (-6, -14),
         'GG/-A': (-6, -14),
         'T-/TA': (-6, -14),
         '-T/AT': (-6, -14),
         'TA/T-': (-6, -14),
         'AT/-T': (-6, -14),
         'T-/CA': (-6, -14),
         '-T/AC': (-6, -14),
         'CA/T-': (-6, -14),
         'AC/-T': (-6, -14),
         'T-/GA': (-6, -14),
         '-T/AG': (-6, -14),
         'GA/T-': (-6, -14),
         'AG/-T': (-6, -14),
         'T-/TT': (-6, -14),
         '-T/TT': (-6, -14),
         'TT/T-': (-6, -14),
         'TT/-T': (-6, -14),
         'T-/CT': (-6, -14),
         '-T/TC': (-6, -14),
         'CT/T-': (-6, -14),
         'TC/-T': (-6, -14),
         'T-/GT': (-6, -14),
         '-T/TG': (-6, -14),
         'GT/T-': (-6, -14),
         'TG/-T': (-6, -14),
         'T-/TC': (-6, -14),
         '-T/CT': (-6, -14),
         'TC/T-': (-6, -14),
         'CT/-T': (-6, -14),
         'T-/CC': (-6, -14),
         '-T/CC': (-6, -14),
         'CC/T-': (-6, -14),
         'CC/-T': (-6, -14),
         'T-/GC': (-6, -14),
         '-T/CG': (-6, -14),
         'GC/T-': (-6, -14),
         'CG/-T': (-6, -14),
         'T-/TG': (-6, -14),
         '-T/GT': (-6, -14),
         'TG/T-': (-6, -14),
         'GT/-T': (-6, -14),
         'T-/CG': (-6, -14),
         '-T/GC': (-6, -14),
         'CG/T-': (-6, -14),
         'GC/-T': (-6, -14),
         'T-/GG': (-6, -14),
         '-T/GG': (-6, -14),
         'GG/T-': (-6, -14),
         'GG/-T': (-6, -14),
         'C-/CA': (-6, -14),
         '-C/AC': (-6, -14),
         'CA/C-': (-6, -14),
         'AC/-C': (-6, -14),
         'C-/AA': (-6, -14),
         '-C/AA': (-6, -14),
         'AA/C-': (-6, -14),
         'AA/-C': (-6, -14),
         'C-/TA': (-6, -14),
         '-C/AT': (-6, -14),
         'TA/C-': (-6, -14),
         'AT/-C': (-6, -14),
         'C-/CT': (-6, -14),
         '-C/TC': (-6, -14),
         'CT/C-': (-6, -14),
         'TC/-C': (-6, -14),
         'C-/AT': (-6, -14),
         '-C/TA': (-6, -14),
         'AT/C-': (-6, -14),
         'TA/-C': (-6, -14),
         'C-/TT': (-6, -14),
         '-C/TT': (-6, -14),
         'TT/C-': (-6, -14),
         'TT/-C': (-6, -14),
         'C-/CC': (-6, -14),
         '-C/CC': (-6, -14),
         'CC/C-': (-6, -14),
         'CC/-C': (-6, -14),
         'C-/AC': (-6, -14),
         '-C/CA': (-6, -14),
         'AC/C-': (-6, -14),
         'CA/-C': (-6, -14),
         'C-/TC': (-6, -14),
         '-C/CT': (-6, -14),
         'TC/C-': (-6, -14),
         'CT/-C': (-6, -14),
         'C-/CG': (-6, -14),
         '-C/GC': (-6, -14),
         'CG/C-': (-6, -14),
         'GC/-C': (-6, -14),
         'C-/AG': (-6, -14),
         '-C/GA': (-6, -14),
         'AG/C-': (-6, -14),
         'GA/-C': (-6, -14),
         'C-/TG': (-6, -14),
         '-C/GT': (-6, -14),
         'TG/C-': (-6, -14),
         'GT/-C': (-6, -14),
         'G-/GA': (-6, -14),
         '-G/AG': (-6, -14),
         'GA/G-': (-6, -14),
         'AG/-G': (-6, -14),
         'G-/AA': (-6, -14),
         '-G/AA': (-6, -14),
         'AA/G-': (-6, -14),
         'AA/-G': (-6, -14),
         'G-/TA': (-6, -14),
         '-G/AT': (-6, -14),
         'TA/G-': (-6, -14),
         'AT/-G': (-6, -14),
         'G-/GT': (-6, -14),
         '-G/TG': (-6, -14),
         'GT/G-': (-6, -14),
         'TG/-G': (-6, -14),
         'G-/AT': (-6, -14),
         '-G/TA': (-6, -14),
         'AT/G-': (-6, -14),
         'TA/-G': (-6, -14),
         'G-/TT': (-6, -14),
         '-G/TT': (-6, -14),
         'TT/G-': (-6, -14),
         'TT/-G': (-6, -14),
         'G-/GC': (-6, -14),
         '-G/CG': (-6, -14),
         'GC/G-': (-6, -14),
         'CG/-G': (-6, -14),
         'G-/AC': (-6, -14),
         '-G/CA': (-6, -14),
         'AC/G-': (-6, -14),
         'CA/-G': (-6, -14),
         'G-/TC': (-6, -14),
         '-G/CT': (-6, -14),
         'TC/G-': (-6, -14),
         'CT/-G': (-6, -14),
         'G-/GG': (-6, -14),
         '-G/GG': (-6, -14),
         'GG/G-': (-6, -14),
         'GG/-G': (-6, -14),
         'G-/AG': (-6, -14),
         '-G/GA': (-6, -14),
         'AG/G-': (-6, -14),
         'GA/-G': (-6, -14),
         'G-/TG': (-6, -14),
         '-G/GT': (-6, -14),
         'TG/G-': (-6, -14),
         'GT/-G': (-6, -14)}
    
    NEW_TABLE = MeltingTemp.make_table(
        oldtable=MeltingTemp.R_DNA_NN1,
        values = table_dict)
    
    # c_seq应该为query_seq的forward_comp
    
    tmp = MeltingTemp.Tm_NN(seq, c_seq = c_seq, nn_table=NEW_TABLE, dnac1 = 20, check = False,
                            selfcomp = False, Na = Para.na_conc * 1000, Mg = Para.mg_conc * 1000,
                            tmm_table = MeltingTemp.RNA_DE1)
    
    GC = (seq.count("G") + seq.count("C"))/len(seq) * 100
    
    return MeltingTemp.chem_correction(tmp, fmd=Para.df_Mconc, fmdmethod=2, GC=GC)
    

def basicFilter(probe, Para):
    
    probe = str(probe.seq)
    
    # filter Ns
    if "N" in probe:
        return "FAILED"

    base_count = Counter(probe)
    total_bases = len(probe)
    
    # Filter Complexity
    # Shannon entropy
    frequencies = {base: count / total_bases for base, count in base_count.items()}
    entropy = -sum(freq * math.log2(freq) for freq in frequencies.values() if freq > 0)
    
    if entropy < 1:
        return "FAILED"
    
    # filter base percentage
    GC = (base_count['G'] + base_count['C'])/total_bases * 100
    if GC < Para.gc_thred[0] or GC > Para.gc_thred[1]:
        return "FAILED"
    
    # filter repeats
    REPEATS_LIST = [
        "GGGGG", "CCCCC"
    ]
    
    repeats_count = 0
    for BL in REPEATS_LIST:
        if probe.count(BL) > 0:
            return "FAILED"
    
    # filter tm
    tm = get_tm(probe, complement(probe), Para)
    if tm < Para.sCelsius:
        return "FAILED"
    
            
    return "PASS"


def blastnCheck(row, Para):
    
    """
    该函数用于检查是否比对到了目标基因上
    
    row
    Para
    
    """
    subject_title = row['stitle']
    gene_name = Para.gene_name
    
    if Para.modality == 'mRNA' or Para.modality == 'cDNA':
    
        if gene_name in subject_title:
            return "TARGET"

        # Used to handle specific predicted genes
        elif gene_name == "Etv1" and "Gm5454" in subject_title:
            return "TARGET"

        elif gene_name == "Gapdh" and "(Gm" in subject_title:
            return "TARGET"

        elif gene_name == "Ccl19" and "Gm12407" in subject_title:
            return "TARGET"

        elif gene_name == "Nptxr" and "Npcd" in subject_title:
            return "TARGET"

        elif gene_name == 'Slc6a20a' and "Slc6a20b" in subject_title:
            return "TARGET"
        
        elif gene_name == 'Ccl21a' and '(Ccl21' in subject_title:
            return 'TARGET'

        elif gene_name.startswith('Klra') and gene_name != 'Klra2' and gene_name != 'Klra17':
            if 'Klra' in subject_title and '(Klra2)' not in subject_title and '(Klra17)' not in subject_title:
                return "TARGET"
            else:
                return "OFF-TARGET"
        else:
            return "OFF-TARGET"
    
    else:
        
        qchr_name,qstart,qend = gene_name.split("_")
        qchr = f"chromosome {qchr_name[3:]}"
        qstart = int(qstart)
        qend = int(qend)
        
        if row["sstrand"] == "plus":
            sstart = row["sstart"]
            send = row["send"]
        elif row["sstrand"] == "minus":
            sstart = row["send"]
            send = row["sstart"]
        if qchr in subject_title and qstart <= sstart and qend >= send:
            return "TARGET"
        else:
            return "OFF-TARGET"
    
    
def non_specific_check(probe, alignedProbe, Para):
    
    """
    该函数用于检查非特异性结合的探针是否Tm值满足要求，例如Tm < 杂交温度 - dT
    
    probe:SeqRecord对象
    alignedProbe:全部blastn的结果, dask.dataframe对象
    Para:parameter class
    """
        
    pid = probe.id
    
    try:
        sub_df = alignedProbe.loc[pid,:].copy()
    except:
        if Para.modality == 'Artificial' or Para.modality == 'TE':
            return "PASS"
        else:
            return "FAILED"
        
    if isinstance(sub_df,pd.core.series.Series):
        
        if Para.modality == 'Artificial' or Para.modality == 'TE':
            tm = get_tm(sub_df['sseq'], complement(sub_df['qseq']), Para)
            
            if tm > Para.nCelsius:
                return "FAILED"
            else:
                return "PASS"
        else:
            
            hit_gene = blastnCheck(sub_df, Para)
            if hit_gene == "OFF-TARGET":
                return "FAILED"
            else:
                return "PASS"
                
    else:
        
        # sub_df = sub_df.set_index('qseqid', drop = True)
        
        if Para.modality == 'Artificial' or Para.modality == 'TE':
            pass
        else:
            sub_df["hitgene"] = sub_df.apply(lambda row: blastnCheck(row, Para), axis = 1)
            sub_df = sub_df[sub_df["hitgene"] == "OFF-TARGET"].copy()
        
        for acc, qcc in zip(sub_df["sseq"], sub_df["qseq"]):
            tm = get_tm(acc, complement(qcc), Para)
            if tm > Para.nCelsius:
                return "FAILED"
            
        return "PASS"

def unmapped_bowtie_check(samfilepath):
    
    keep_names = set()
    samfile = ps.AlignmentFile(samfilepath, 'r')
    for r in samfile:
        if not r.is_unmapped:
            keep_names.add(r.query_name)
            
    return list(keep_names)


def specific_count(probe, Para, alignedProbe = None):
    
    """
    该函数用于记录一个target region最多比对到了几个目标转录本
    """
    
    pid = probe.name
    
    # Genome和Artificial的target只有一个
    if Para.modality == 'Genome' or Para.modality == 'Artificial':
        return 1
    
    elif Para.modality == 'mRNA' or Para.modality == 'cDNA':
        
        try:
            sub_df = alignedProbe.loc[pid,:].copy()
        except:
            # 应该不会有这种情况，以防万一
            return 0

        if isinstance(sub_df,pd.core.series.Series):

            return 1
        
        else:
            
            sub_df["hitgene"] = sub_df.apply(lambda row: blastnCheck(row, Para), axis = 1)

            unaligned_count = sum(sub_df["hitgene"] == "OFF-TARGET")

            # 这里需要判断一下是否有一定程度非特异性比对到目标转录本上，即TARGET部分的Tm值是否大于Para.nCelsius
            sub_df = sub_df[sub_df["hitgene"] == "TARGET"].copy()
            sub_df["tm"] = sub_df.apply(lambda row:get_tm(row['sseq'], complement(row['qseq']), Para), axis = 1)
            return sum(sub_df['tm'] > Para.nCelsius)
        
    elif Para.modality == 'TE':
        return 1
    
# def non_specific_mean_Tm(probe, alignedProbe, Para):
    
#     """
#     该函数用于评分计数
#     如果是cDNA，mRNA，返回比对到的特异性转录本数量
#     所有方法都会返回比对到非特异性部分条数
    
#     probe:SeqRecord对象
#     alignedProbe:全部blastn的结果
#     Para:parameter class
    
#     return 1,105,Tm_95th
#     """
        
#     pid = probe.id
    
#     try:
#         sub_df = alignedProbe.loc[pid,:].copy()
#         # Artificial和TE就应该比对不上
#     except:
#         if Para.modality == 'Artificial' or Para.modality == 'TE':
#             return pd.Series({'p1_unaligned':0,
#                               'p1_unTarget_95thTm':0})
            
    
#     if isinstance(sub_df,pd.core.series.Series):
        
#         if Para.modality == 'Artificial' or Para.modality == 'TE':
            
#             return pd.Series({'p1_unaligned':1,
#                               'p1_unTarget_95thTm':get_tm(sub_df["sseq"], Para)})
        
#         else:
#             return pd.Series({'p1_unaligned':0,
#                               'p1_unTarget_95thTm':0})
#     else:   
        
#         if Para.modality == 'Artificial' or Para.modality == 'TE':
#             sub_df["hitgene"] = 'OFF-TARGET'
#         else:
#             sub_df["hitgene"] = sub_df.apply(lambda row: blastnCheck(row, Para), axis = 1)
            
#         unaligned_count = sum(sub_df["hitgene"] == "OFF-TARGET")
        
        
#         if unaligned_count == 0:
#             return pd.Series({'p1_unaligned':0,
#                               'p1_unTarget_95thTm':0})
        
#         else:
#             sub_df = sub_df[sub_df["hitgene"] == "OFF-TARGET"].copy()
#             sub_df["tm"] = sub_df["sseq"].apply(lambda x:get_tm(x, Para))
#             return pd.Series({'p1_unaligned':unaligned_count,
#                               'p1_unTarget_95thTm':np.percentile(sub_df["tm"], 95)})
        
        
def non_specific_pad_check(probe, alignedProbe, Para, isP2 = False):
    
    """
    该函数用于完整探针和已有Pad进行比对后的判断去留
    
    probe:SeqRecord对象
    alignedProbe:全部blastn的结果
    Para:parameter class
    """
        
    pid = probe.id
    

    sub_df = alignedProbe.loc[pid,:].copy()
    
    if isinstance(sub_df,pd.core.series.Series):
        
        return "PASS"
        
    else:
        
        for _, row in sub_df.iterrows():
            
            if isP2 and row['sstart'] >= 75 and row['send'] <= 68:
                
                return 'FAILED'
            
            elif row['sseqid'] != Para.bridge_id and row['sstart'] >= 75 and row['send'] <= 68:
                
                return 'FAILED'
            
        return "PASS"
    
def black_list_check(probe, alignedProbe, Para):
    
    """
    该函数用于去掉和BLACK LIST有重合的探针
    
    probe:SeqRecord对象
    alignedProbe:全部blastn的结果, dask.dataframe对象
    Para:parameter class
    """
        
    pid = probe.id
    
    try:
        sub_df = alignedProbe.loc[pid,:].copy()
    
    except:
        return 'PASS'
    
    if isinstance(sub_df,pd.core.series.Series):
        
        if sub_df['length'] - sub_df['mismatch'] >= 16:
            
            return 'FAILED'
        
        return "PASS"
        
    else:
        
        for _, row in sub_df.iterrows():
            
            if row['length'] - row['mismatch'] >= 16:
                
                return 'FAILED'
            
        return "PASS"
    
    
def bedtools_intersect_filter(bowtie2file, Para):
    
    import subprocess
    
    # 先将sam文件压缩成bam
    
    command = f'samtools view -Sb {bowtie2file} > ./TMP/prbs_candidates_alignment.bam'
    cp = subprocess.run(command, shell = True, check = True, capture_output = True)
    
    # 然后进行intersect
    
    command = f"{Para.bedtools_path} intersect -a ./TMP/prbs_candidates_alignment.bam -b {Para.rmsk_bed} -wa -wb -bed > ./TMP/prbs_candidates_alignment.bed"
    cp = subprocess.run(command, shell = True, check = True, capture_output = True)
    
    # 然后读入结果进行过滤
    TE_level2column = {
                'class': 2,
                'family': 1,
                'subfamily': 0
            }
    
    name_split = Para.gene_name.split("_")
    TE_name = "_".join(name_split[2:])
    TE_level = name_split[1]
    
    Drop = set()
    with open("./TMP/prbs_candidates_alignment.bed", 'r') as handle:
        for line in handle.readlines():
            parsed_line = line.strip().split("\t")
            if TE_name != parsed_line[-3].split("|")[-3].split(":")[TE_level2column[TE_level]]:
                Drop.add(parsed_line[3])
                
    return list(Drop)


import re

def extract_gene(text):
    # 使用正则表达式匹配括号内的内容
    match = re.search(r'\((.*?)\)', text)
    if match:
        return match.group(1)  # 返回第一个捕获组的内容
    else:
        return None  # 如果没有匹配到括号内容，返回None


def non_specific_genes(probe, alignedProbe, Para):
    
    """
    该函数用于评分计数
    如果是cDNA，mRNA，返回比对到的特异性转录本数量
    所有方法都会返回比对到非特异性部分条数
    
    probe:SeqRecord对象
    alignedProbe:全部blastn的结果
    background_list: 一个含有background基因的list，即如果靶标基因的非特异性结合在这个list中，则标记出来
    Para:parameter class
    
    return 1,105,Tm_95th
    """
        
    pid = probe.id
    background_list = Para.background_list
    
    try:
        sub_df = alignedProbe.loc[pid,:].copy()
        # Artificial和TE就应该比对不上
    except:
        if Para.modality == 'Artificial' or Para.modality == 'TE':
            return pd.Series({'p1_unaligned':0,
                              'p1_unTarget_95thTm':0,
                              'p1_unaligned_gene': []})
            
    
    if isinstance(sub_df,pd.core.series.Series):
        
        subject_title = sub_df['stitle']
        sgene = extract_gene(subject_title)
        
        if Para.modality == 'Artificial' or Para.modality == 'TE':
            
            if sgene is not None and sgene in background_list:
                return pd.Series({'p1_unaligned':1,
                                  'p1_unTarget_95thTm':get_tm(sub_df["sseq"], complement(sub_df['qseq']), Para),
                                  'p1_unaligned_gene': [sgene]})
        
        else:
            return pd.Series({'p1_unaligned':0,
                              'p1_unTarget_95thTm':0,
                              'p1_unaligned_gene': []})
    else:   
        
        if Para.modality == 'Artificial' or Para.modality == 'TE':
            sub_df["hitgene"] = 'OFF-TARGET'
        else:
            sub_df["hitgene"] = sub_df.apply(lambda row: blastnCheck(row, Para), axis = 1)
        
        unaligned_count = sum(sub_df["hitgene"] == "OFF-TARGET")
        
        
        if unaligned_count == 0:
            return pd.Series({'p1_unaligned':0,
                              'p1_unTarget_95thTm':0,
                              'p1_unaligned_gene': []})
        
        else:
            sub_df = sub_df[sub_df["hitgene"] == "OFF-TARGET"].copy()
            sgenes_set = set()
            all_tms = []
            for _,row  in sub_df.iterrows():
                sgene = extract_gene(row['stitle'])
                stm = get_tm(row['sseq'], complement(row['qseq']), Para)
                all_tms.append(stm)
                if stm >= Para.sCelsius - 5 and sgene in background_list:
                    sgenes_set.add(sgene)
            
            return pd.Series({'p1_unaligned':unaligned_count,
                              'p1_unTarget_95thTm':np.percentile(all_tms, 100),
                              'p1_unaligned_gene': list(sgenes_set)})