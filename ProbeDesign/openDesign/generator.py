import pandas as pd
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio.Seq import reverse_complement
import sys
import warnings
warnings.filterwarnings('ignore')

from .filter import (basicFilter,
                     non_specific_check,
                     specific_count,
                     non_specific_genes,
                     get_tm,
                     non_specific_pad_check,
                     black_list_check)

from .utils import (findAllCandidates,
                    get_full_probes1,
                    get_full_probes2,
                    cal_binding_efficiency,
                    get_start,
                    get_evalue,
                    write_to_fasta)

from .cmdtools import (blastn,
                       blastn_subject)

class Para():
    
    """
    Class for parameter storage
    
    gene_name: target name
    seq: list(SeqRecord)
    bridge_id: Padlock id
    gc_thred: GC% range, default: [30,70]
    sCelsius:hybridization temperature, default: 47
    nCelsiud: expected highest Tm for unspecific binding, default: 37
    df_conc: formamide concentration in %, default: 30
    na_conc: monovalent ion conc. in mM, default: 0.390
    mg_conc: devalent ion conc. in mM, defalut 0.0
    bseqs_path: path to bridge sequences
    
    """
    
    def __init__(self, gene_name = None, seq = None, bridge_id = None, 
                   gc_thred=[30,70],
                   sCelsius = 47, nCelsius = 37, df_conc = 30,
                   na_conc=0.390, mg_conc=0.0,
                   bseqs_path = "./SEQUENCES/Bridge_sequences.csv",
                   UsingPadfile = "./SEQUENCES/UsingPad.fa",
                   TaxonomyID = "10090",
                   species = "Mus musculus",
                   P1evalue = 100,
                   P2evalue = 50,
                   strand = 'minus',
                   BLACK_FA = "./SEQUENCES/BLACK_LIST_FULL.fa",
                   gtype = 'gene',
                   max_threads = 96,
                   max_memory = 256,
                   background_list = None,
                 
                   Entrez_api_key = None,
                   Entrez_email = None,
                   blastn_path = '~/software/ncbi-blast-2.15.0+/bin/blastn',
                   ref_seq_path = "../blastndb/refseq_rna"):
        
        self.gene_name = gene_name
        self.seq = seq
        self.bridge_id = bridge_id
        self.gc_thred = gc_thred
        self.sCelsius = sCelsius
        self.nCelsius = nCelsius
        self.df_conc = df_conc
        self.df_Mconc = df_conc * 0.2222
        self.na_conc = na_conc
        self.mg_conc = mg_conc
        self.bseqs_path = bseqs_path
        self.UsingPadfile = UsingPadfile
        self.TaxonomyID = TaxonomyID
        self.species = species
        self.P2evalue = P2evalue
        self.P1evalue = P1evalue
        self.strand = strand
        self.BLACK_FA = BLACK_FA
        self.gtype = gtype
        self.max_threads = max_threads
        self.max_memory = max_memory
        
        if isinstance(background_list, str):
            BKGdf = pd.read_csv(background_list, index_col=0)
            self.background_list = BKGdf['0'].to_list()
        elif isinstance(background_list, list):
            self.background_list = background_list
        else:
            self.background_list = []
        
        self.Entrez_api_key = Entrez_api_key
        self.Entrez_email = Entrez_email
        self.blastn_path = blastn_path
        self.ref_seq_path = ref_seq_path
        
        
    @property
    def bridge_seq(self):
        
        bseqs = pd.read_csv(self.bseqs_path,index_col=0)
        return bseqs.loc[self.bridge_id,"Seq"]


def gene_generator(Para):
    
    import os
    
    if not os.path.exists("./TMP"):
        os.path.makedirs("./TMP")
    
    from pandarallel import pandarallel
    pandarallel.initialize(progress_bar = False, verbose=0, nb_workers = Para.max_threads)
    
    max_memory_allowed = Para.max_memory
                    
    Probes, probes_counts = findAllCandidates(seqs = Para.seq, prb_strand = Para.strand)
    
    print(f"[INFO]{len(Probes)} candidate probes")
    # Basic filter
    Probes["basicFilter1"] = Probes["Probe1"].parallel_apply(lambda x:basicFilter(x, Para))
    Probes["basicFilter2"] = Probes["Probe2"].parallel_apply(lambda x:basicFilter(x, Para))
    Probes = Probes[(Probes["basicFilter1"] == "PASS") & (Probes["basicFilter2"] == "PASS")].copy()
    
    max_probes_number = 20000
    
    if len(Probes) >= max_probes_number:
        
        Probes.index = Probes['Probe1'].parallel_apply(lambda x: x.description)
        keep_indices = []
        for x,_ in probes_counts.most_common(n = max_probes_number):
            if x in Probes.index:
                keep_indices.append(x)
            if len(keep_indices) >= max_probes_number:
                break
                
        Probes = Probes.loc[keep_indices, :].copy()
        
    print(f"[INFO]{len(Probes)} probes passed basic filter")

    fastafile1 = "./TMP/prbs1_candidates.fasta"
    fastafile2 = "./TMP/prbs2_candidates.fasta"
    alignfile1 = "./TMP/prbs1_candidates_alignment.tsv"
    alignfile2 = "./TMP/prbs2_candidates_alignment.tsv"
    
    Probes = Probes[["Probe1", "Probe2"]].reset_index(drop = True).copy()
    
    # Blastn filter
    write_to_fasta(Probes["Probe1"].to_list(), fastafile1)

    blastn(fastafile = fastafile1, result_path = alignfile1, Para = Para, taxids = Para.TaxonomyID, evalue = Para.P1evalue, strand = Para.strand)
    alignedProbe1 = pd.read_csv(alignfile1, sep = "\t",
                           names=["qseqid","sacc","pident","length","mismatch","qcovs","evalue","qseq","sseq","stitle"],
                               index_col = 0)
    
    
    n_workers = min(Para.max_threads, int(max_memory_allowed // (sys.getsizeof(alignedProbe1) / 1073741824)))
    pandarallel.initialize(progress_bar = False, verbose=0, nb_workers = n_workers)
    Probes["probe1_blastn_check"] = Probes["Probe1"].parallel_apply(lambda x:non_specific_check(x, alignedProbe = alignedProbe1, Para = Para))
    Probes = Probes[Probes["probe1_blastn_check"] == "PASS"].copy()
    print(f"[INFO]{len(Probes)} probes passed p1 blastn filter")
    
    write_to_fasta(Probes["Probe2"].to_list(), fastafile2)
    # _ = SeqIO.write(Probes["Probe2"].to_list(), fastafile2, "fasta-2line")
    
    blastn(fastafile = fastafile2, result_path = alignfile2, Para = Para, taxids = Para.TaxonomyID, evalue = Para.P2evalue, strand = Para.strand)
    alignedProbe2 = pd.read_csv(alignfile2, sep = "\t",
                           names=["qseqid","sacc","pident","length","mismatch","qcovs","evalue","qseq", "sseq","stitle"],
                               index_col = 0)
    
    n_workers = min(Para.max_threads, int(max_memory_allowed // (sys.getsizeof(alignedProbe2) / 1073741824)))
    pandarallel.initialize(progress_bar = False, verbose=0, nb_workers = n_workers)
    Probes["probe2_blastn_check"] = Probes["Probe2"].parallel_apply(lambda x:non_specific_check(x, alignedProbe = alignedProbe2, Para = Para))
    Probes = Probes[Probes["probe2_blastn_check"] == "PASS"].copy()
    print(f"[INFO]{len(Probes)} probes passed p2 blastn filter")
    
    
    Probes = Probes[["Probe1", "Probe2"]].reset_index(drop = True).copy()
    Probes["probe_start"] = Probes["Probe1"].apply(get_start)
    Probes = Probes.reset_index(drop = True).copy()
    
    # whole probe join
    
    Probes["Full_Probe1"] = Probes["Probe1"].parallel_apply(lambda x:get_full_probes1(x, bridge_seq = Para.bridge_seq))
    Probes["Full_Probe2"] = Probes["Probe2"].parallel_apply(lambda x:get_full_probes2(x, bridge_seq = Para.bridge_seq))
    
    # Blastn to BLACK LIST
    if isinstance(Para.BLACK_FA, str):

        write_to_fasta(Probes["Full_Probe1"].to_list(), fastafile1)
        
        blastn_subject(fastafile = fastafile1, subject_file = Para.BLACK_FA, Para = Para, result_path = alignfile1, strand = 'plus')
        
        alignedProbe1 = pd.read_csv(alignfile1, sep = "\t",
                                   names=["qseqid","sseqid","pident","length","mismatch","sstart","send","evalue"],
                                       index_col = 0)
        
        Probes["probe1_blastn_check"] = Probes["Probe1"].parallel_apply(lambda x:black_list_check(x, alignedProbe = alignedProbe1, Para = Para))
        Probes = Probes[Probes["probe1_blastn_check"] == "PASS"].copy()
        print(f"[INFO]{len(Probes)} P1 probes passed black list filter")
        
        write_to_fasta(Probes["Full_Probe2"].to_list(), fastafile2)
        # _ = SeqIO.write(Probes["Full_Probe2"].to_list(), fastafile2, "fasta-2line")
        blastn_subject(fastafile = fastafile2, subject_file = Para.BLACK_FA, Para = Para, result_path = alignfile2, strand = 'plus')
        alignedProbe2 = pd.read_csv(alignfile2, sep = "\t",
                                   names=["qseqid","sseqid","pident","length","mismatch","sstart","send","evalue"],
                                       index_col = 0)
        
        Probes["probe2_blastn_check"] = Probes["Probe2"].parallel_apply(lambda x:black_list_check(x, alignedProbe = alignedProbe2, Para = Para))
        Probes = Probes[Probes["probe2_blastn_check"] == "PASS"].copy()
        print(f"[INFO]{len(Probes)} P2 probes passed black list filter")

    
    # Blast to using pads + tRNA
    
    if Para.bridge_seq == 'CATAGGCGGTTAGATGAGCCCATTGACGAG':
        pass
    else:
        write_to_fasta(Probes["Full_Probe1"].to_list(), fastafile1)

        blastn_subject(fastafile = fastafile1, result_path = alignfile1, Para = Para, subject_file = Para.UsingPadfile)

        alignedProbe1 = pd.read_csv(alignfile1, sep = "\t",
                                   names=["qseqid","sseqid","pident","length","mismatch","sstart","send","evalue"],
                                       index_col = 0)

        Probes["probe1_blastn_check"] = Probes["Probe1"].parallel_apply(lambda x:non_specific_pad_check(x, alignedProbe = alignedProbe1, Para = Para))
        Probes = Probes[Probes["probe1_blastn_check"] == "PASS"].copy()

        print(f"[INFO]{len(Probes)} probes passed P1 system-reaction filter")

        write_to_fasta(Probes["Full_Probe2"].to_list(), fastafile2)

        blastn_subject(fastafile = fastafile2, result_path = alignfile2, Para = Para, subject_file = Para.UsingPadfile)

        alignedProbe2 = pd.read_csv(alignfile2, sep = "\t",
                                   names=["qseqid","sseqid","pident","length","mismatch","sstart","send","evalue"],
                                       index_col = 0)

        Probes["probe2_blastn_check"] = Probes["Probe2"].parallel_apply(lambda x:non_specific_pad_check(x, alignedProbe = alignedProbe2, Para = Para, isP2 = True))
        Probes = Probes[Probes["probe2_blastn_check"] == "PASS"].copy()

        Probes["probe_start"] = Probes["Probe1"].apply(get_start)
        Probes = Probes.reset_index(drop = True).copy()

        print(f"[INFO]{len(Probes)} probes passed P2 system-reaction filter")
    
    # BLASTn evaluation
    
    probe1_list = Probes["Probe1"].to_list()
    
    write_to_fasta(probe1_list, fastafile1)
    write_to_fasta(Probes["Probe2"].to_list(), fastafile2)

    with open("./TMP/region_candidates.fasta", 'w') as handle:
        for p1 in probe1_list:
            handle.write(f">{p1.name}\n")
            handle.write(f"{p1.description}\n")
            
    print(f"[INFO]Preparing Indicator for Efficiency Calculatioin...")
    
    blastn(fastafile = fastafile1, result_path = alignfile1, evalue = 130, Para = Para, taxids = Para.TaxonomyID, strand = 'both')

    blastn(fastafile = fastafile2, result_path = alignfile2, evalue = 130, Para = Para, taxids = Para.TaxonomyID, strand = 'both')

    blastn(fastafile = "./TMP/region_candidates.fasta", 
           result_path = "./TMP/region_candidates_alignment.tsv", evalue = 130, Para = Para, taxids = Para.TaxonomyID, strand = 'both')

    alignedProbe1 = pd.read_csv(alignfile1, sep = "\t",
                           names=["qseqid","sacc","pident","length","mismatch","qcovs","evalue","qseq", "sseq","stitle"],
                           index_col = 0)

    alignedProbe2 = pd.read_csv(alignfile2, sep = "\t",
                           names=["qseqid","sacc","pident","length","mismatch","qcovs","evalue","qseq", "sseq","stitle"],
                           index_col = 0)
    

    alignedProbe3 = pd.read_csv("./TMP/region_candidates_alignment.tsv", sep = "\t",
                           names=["qseqid","sacc","pident","length","mismatch","qcovs","evalue","qseq", "sseq","stitle"],
                           index_col = 0)

    
    n_workers = min(Para.max_threads, int(max_memory_allowed // (sys.getsizeof(alignedProbe1) / 1073741824)))
    pandarallel.initialize(progress_bar = False, verbose=0, nb_workers = n_workers)
    Probes[['p1_unaligned', 'p1_unTarget_95thTm', 'p1_unaligned_gene']] = Probes["Probe1"].parallel_apply(lambda x:non_specific_genes(x, alignedProbe1, Para))
    
    n_workers = min(Para.max_threads, int(max_memory_allowed // (sys.getsizeof(alignedProbe2) / 1073741824)))
    pandarallel.initialize(progress_bar = False, verbose=0, nb_workers = n_workers)
    Probes[['p2_unaligned', 'p2_unTarget_95thTm', 'p2_unaligned_gene']] = Probes["Probe2"].parallel_apply(lambda x:non_specific_genes(x, alignedProbe2, Para))
    
    if Para.modality == 'Artificial':
        n_workers = Para.max_threads
        pandarallel.initialize(progress_bar = False, verbose=0, nb_workers = n_workers)
        Probes['aligned_count'] = Probes["Probe1"].parallel_apply(lambda x:specific_count(x, Para))
    
    elif Para.modality == 'cDNA' or Para.modality == 'mRNA':
        n_workers = min(Para.max_threads, int(max_memory_allowed // (sys.getsizeof(alignedProbe3) / 1073741824)))
        pandarallel.initialize(progress_bar = False, verbose=0, nb_workers = n_workers)
        Probes['aligned_count'] = Probes["Probe1"].parallel_apply(lambda x:specific_count(x, Para, alignedProbe3))

    
    from Bio.Seq import complement

    Probes["probe1_Tm"] = Probes["Probe1"].parallel_apply(lambda x:get_tm(str(x.seq), complement(str(x.seq)), Para))
    Probes["probe2_Tm"] = Probes["Probe2"].parallel_apply(lambda x:get_tm(str(x.seq), complement(str(x.seq)), Para))
    
    # 计算 Binding Efficiency
        
    print(f"[INFO]Calculating Binding Efficiency...")   
    Probes['BindingEfficiency'] = Probes.apply(lambda row:cal_binding_efficiency(row, Para), axis = 1)
    
    Probes['Evaluation'] = Probes.parallel_apply(lambda row: get_evalue(row, Para), axis = 1)
    
    print(f"[INFO]{len(Probes)} pairs of probes are selected to make complete sequence")
    
    result = pd.DataFrame({
        'name': Para.gene_name,
        'probe_name': [x.name for x in Probes["Probe1"]],
        'position':Probes["probe_start"],
        'probe A':[str(x.seq) for x in Probes["Full_Probe1"]],
        'probe B':[str(x.seq) for x in Probes["Full_Probe2"]],
        "Evaluation": Probes['Evaluation'],
        "aligned_count": Probes['aligned_count'],
        "probeA_Tm": Probes["probe1_Tm"],
        "probeB_Tm": Probes["probe2_Tm"],
        "probeA_unTarget_maxTm": Probes['p1_unTarget_95thTm'],
        "probeB_unTarget_maxTm": Probes['p2_unTarget_95thTm'],
        "Off_Genes": Probes['p1_unaligned_gene'],
        'bridge_id':Para.bridge_id,
        'bridge_seq':Para.bridge_seq,
        'probe_strand': [x.name.split("_")[-1] for x in Probes["Probe1"]]})
    
    result = result.sort_values('Evaluation', ascending = True)
    
    return result
    
    
    
    
    
    
    
    
    
    
    
    