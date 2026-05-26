import importlib
if importlib.util.find_spec('nupack'):
    from nupack import *
    
import pandas as pd
import numpy as np
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio.Seq import reverse_complement
from collections import Counter

def findAllCandidates(seqs: list, prb_strand:str):
    
    """
    p1: 30bp 2bp gap p2: 30bp
    
    target:SeqRecord list
    prb_strand:probe strand relative to target sequence 
        minus: for target mRNA
        plus: for target cDNA
    """
    
    All_probes = []
    ALL_probes_count = Counter()
    
    for target in seqs:
        limit = len(target)
        for i in range(0, limit-62):
            
            start = i
            end = start + 30
            p1 = target.seq[start:end].upper()
            
            target_region = target.seq[start:end+32].upper()
            p2 = target.seq[end+2:end+32].upper()
            
            if prb_strand == 'plus' or prb_strand == 'both':
                
                pp = p2 + p1
                probe = SeqRecord(pp, name = f"{target.name}_{start}_plus", description=str(target_region))
                before_len = len(ALL_probes_count)
                ALL_probes_count[str(target_region)] += 1
                after_len = len(ALL_probes_count)
                if before_len != after_len:
                    All_probes.append(probe)
            
            if prb_strand == 'minus' or prb_strand == 'both':
                
                pp = p1.reverse_complement() + p2.reverse_complement()
                target_region = target_region.reverse_complement()
                probe = SeqRecord(pp, name = f"{target.name}_{start}_minus", description=str(target_region))
                before_len = len(ALL_probes_count)
                ALL_probes_count[str(target_region)] += 1
                after_len = len(ALL_probes_count)
                if before_len != after_len:
                    All_probes.append(probe)
                
 
    prbs1 = []
    prbs2 = []

    # seq: p1/p2
    # id: {target.name}_{start}_plus/minus_p1/p2
    # name: {target.name}_{start}_plus/minus
    # description: target_region
    
    for probe in All_probes:
        p1_rec = probe[0:30]
        p1_rec.id = probe.name + '_p1'
        
        p2_rec = probe[30:60]
        p2_rec.id = probe.name + '_p2'
        
        prbs1.append(p1_rec)
        prbs2.append(p2_rec)
    
    return pd.DataFrame({"Probe1":prbs1, "Probe2":prbs2}), ALL_probes_count


def get_full_probes1(p1, bridge_seq):

    return bridge_seq[0:14] + "ta" + p1

def get_full_probes2(p2, bridge_seq):
    
    return p2 + "ta" + bridge_seq[-14:]

def get_start(probe):

    return int(probe.name.split("_")[-2])


def cal_binding_efficiency(row, Para):

    if importlib.util.find_spec('nupack'):
        config.threads = 12
        workflow = "nupack"
    else:
        print("No nupack installation detected, use primer3-py")
        from primer3 import calc_heterodimer, calc_hairpin
        workflow = "primer3"
    
    if workflow == 'nupack':
        
        P1_strand = Strand(str(row['Full_Probe1'].seq), name='p1')
        P2_strand = Strand(str(row['Full_Probe2'].seq), name='p2')
        
        r_complement = reverse_complement(row['Probe1'].description)
        
        target_region = Strand(str(r_complement), name='target')
        
        my_model = Model(material='dna', celsius=Para.sCelsius - 5, sodium = Para.na_conc, magnesium=Para.mg_conc)
        walker = Complex([P1_strand, P2_strand, target_region])
        my_set = ComplexSet(strands={target_region: 2e-8, P1_strand:2e-8, P2_strand:2e-8}, complexes=SetSpec(max_size=0, include=[walker]))
        
        my_result = complex_analysis(my_set, compute=['pairs'], model=my_model)
        arrays = my_result['(p1+p2+target)'].pairs.to_array()
        np.fill_diagonal(arrays, 0)
        ComplexDNA = arrays[16:46].sum()/2 + arrays[46:76].sum()/2
        
        if Para.modality == 'mRNA':
            
            my_model = Model(material='rna', celsius=Para.sCelsius - 5, sodium = Para.na_conc, magnesium=Para.mg_conc)
            my_result = complex_analysis(my_set, compute=['pairs'], model=my_model)
            arrays = my_result['(p1+p2+target)'].pairs.to_array()
            np.fill_diagonal(arrays, 0)
            SignalPart = (arrays[16:46].sum()/2 + arrays[46:76].sum()/2 + ComplexDNA) / 2
            
        else:
            
            SignalPart = ComplexDNA
        
        my_model = Model(material='dna', celsius=Para.sCelsius - 5, sodium = Para.na_conc, magnesium=Para.mg_conc)
        
        walker = Complex([P1_strand])
        my_set = ComplexSet(strands={P1_strand: 2e-8}, complexes=SetSpec(max_size=0, include=[walker]))
        my_result = complex_analysis(my_set, compute=['pairs'], model=my_model)
        arrays = my_result['(p1)'].pairs.to_array()
        np.fill_diagonal(arrays, 0)
        P1_Pair_Prob = arrays.sum()/2
        
        walker = Complex([P2_strand])
        my_set = ComplexSet(strands={P2_strand: 2e-8}, complexes=SetSpec(max_size=0, include=[walker]))
        my_result = complex_analysis(my_set, compute=['pairs'], model=my_model)
        arrays = my_result['(p2)'].pairs.to_array()
        np.fill_diagonal(arrays, 0)
        P2_Pair_Prob = arrays.sum()/2
        
        walker = Complex([target_region])
        my_set = ComplexSet(strands={target_region: 2e-8}, complexes=SetSpec(max_size=0, include=[walker]))
        my_result = complex_analysis(my_set, compute=['pairs'], model=my_model)
        arrays = my_result['(target)'].pairs.to_array()
        np.fill_diagonal(arrays, 0)
        target_Prob = arrays.sum()/2
        
        BkgPart = target_Prob + P1_Pair_Prob + P2_Pair_Prob

    elif workflow == 'primer3':
        
        P1_strand = str(row['Full_Probe1'].seq)
        P2_strand = str(row['Full_Probe2'].seq)
        
        r_complement = reverse_complement(row['Probe1'].description)
        
        target_region = str(r_complement)

        ComplexDNA1 = calc_heterodimer(P1_strand, target_region, mv_conc=Para.na_conc * 1000, dv_conc=Para.mg_conc * 1000, 
                                       dntp_conc=0., dna_conc=20.0, temp_c=Para.sCelsius - 5, max_loop=30, output_structure=False).dg
        ComplexDNA2 = calc_heterodimer(P2_strand, target_region, mv_conc=Para.na_conc * 1000, dv_conc=Para.mg_conc * 1000, 
                                       dntp_conc=0., dna_conc=20.0, temp_c=Para.sCelsius - 5, max_loop=30, output_structure=False).dg

        SignalPart = ComplexDNA1 + ComplexDNA2
        
        P1_Pair_Prob = calc_hairpin(P1_strand, mv_conc=Para.na_conc * 1000, dv_conc=Para.mg_conc * 1000, temp_c=Para.sCelsius - 5).dg
        P2_Pair_Prob = calc_hairpin(P2_strand, mv_conc=Para.na_conc * 1000, dv_conc=Para.mg_conc * 1000, temp_c=Para.sCelsius - 5).dg
        # target_Prob = calc_hairpin(target_region, mv_conc=Para.na_conc * 1000, dv_conc=Para.mg_conc * 1000, temp_c=Para.sCelsius - 5).dg

        BkgPart = P1_Pair_Prob + P2_Pair_Prob
    
    return SignalPart / (SignalPart + BkgPart)



def get_evalue(row, Para):
    
    part1 = 2.5 / (row['aligned_count'] + 1e-3) # 0 - 2.5
    
    part2 = np.log10((row['p1_unaligned'] + row['p2_unaligned'])/2 + 1) # 0 - 3 probably
    
    part3 = np.log10((row['p1_unTarget_95thTm'] + row['p2_unTarget_95thTm'])/2 + 1) #  0 - 2 probably
    
    part4 = np.log2(1.1 / (row['BindingEfficiency'] + 0.1)) # 0 - 3.5
    
    part5 = 1 / ( - Para.sCelsius + row['probe1_Tm'] - Para.sCelsius + row['probe2_Tm'] + 1) # 0 - 1
    
    return part1 + part2 + part3 + part4 + part5


def generate_order_form(data):
    # This Block is used to generate name + sequence sangong accepted.
    # The name format is "[GENE]_[ProbePosition]_[ProbeNumber][ProbeLength]_[PadNumber]", such as H2-ab1_506_1A30_002.
    final_name = []
    final_primer = []
    gene_check = []
    Final_gene = []
    for i in range(len(data)):
        pos1 = data.loc[i,"name"]
        gene_check.append(pos1)
        number_ = [j for j in gene_check if j == pos1]
        number = len(number_)
        
        bridge_id = data.loc[i,"bridge_id"]
        pos5 = bridge_id.split("_")[-1]
          
        pos2 = str(int(data.loc[i,"position"]))
        pos3 = data.loc[i,"probe A"]
        pos4 = data.loc[i,"probe B"]
        final1 = pos1+"_"+pos2+"_" + str(number) +"A" + "_" + pos5
        final2 = pos1+"_"+pos2+"_" + str(number) +"B" + "_" + pos5

        final_name.append(final1)
        final_name.append(final2)
        final_primer.append(pos3)
        final_primer.append(pos4)
        Final_gene.append(pos1)
        Final_gene.append(pos1)

    print("There are {} sequences in total.".format(str(len(final_name))))
    final_df = pd.DataFrame({"name":final_name, "seq":final_primer,"gene":Final_gene})
    
    return final_df
    

def write_to_fasta(Probes, fastafile):
    
    with open(fastafile, "w") as handle:
        for srec in Probes:
            handle.write(f">{srec.id}\n")
            handle.write(f"{str(srec.seq)}\n")
