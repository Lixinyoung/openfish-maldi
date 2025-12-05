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
                     black_list_check,
                     unmapped_bowtie_check,
                     bedtools_intersect_filter)

from .utils import (findAllCandidates,
                    get_full_probes1,
                    get_full_probes2,
                    cal_binding_efficiency,
                    get_start,
                    get_evalue,
                    select_probe_combinations,
                    _blastn_to_dict,
                    write_to_fasta)

from .cmdtools import (blastn,
                       blastn_subject,
                       blastn_DNA,
                       bowtie2)

class Para():
    
    """
    该类用于储存输入参数
    
    gene_name: 基因名称
    seq:SeqRecord构成的list
    bridge_id:使用的pad编号
    gc_thred:GC% 允许范围，默认为[30,70]
    mfe_thred:允许的完整p1和p2在一定条件下的所有自由能最小值， 默认为-32
    sCelsius:预计杂交使用温度，默认为47
    nCelsiud:期望的非特异性结合最大Tm值， 默认为37
    df_conc:去离子甲酰胺浓度，单位是%，默认为30
    na_conc:单价离子浓度，默认为0.075，单位为mM
    mg_conc:二价离子浓度，默认为0.01，单位为mM
    bseqs_path:储存pad序列的库
    TaxonomyID:10090或者9606
    
    """
    
    def __init__(self, gene_name = None, seq = None, bridge_id = None, 
                   gc_thred=[30,70],
                   sCelsius = 47, nCelsius = 37, df_conc = 30,
                   na_conc=0.075, mg_conc=0.01,
                   bseqs_path = "./BridgeSequence/Bridges_used_for_assemble_20240704.tsv",
                   UsingPadfile = "./BridgeSequence/UsingPad.fa",
                   TaxonomyID = "10090",
                   P1evalue = 100,
                   P2evalue = 50,
                   strand = 'minus',
                   BLACK_FA = "./BridgeSequence/BLACK_LIST_FULL.fa",
                   gtype = 'gene',
                   max_threads = 96,
                   max_memory = 256,
                   background_list = [],
                 
                   Entrez_api_key = "",
                   Entrez_email = "",
                   mouse_fasta_path = "../blastndb/mouse_GRCm39/data/GCF_000001635.27/GCF_000001635.27_GRCm39_genomic.fna",
                   human_fasta_path = "../blastndb/human_GRCh38.p13/data/GCF_000001405.39/GCF_000001405.39_GRCh38.p13_genomic.fna",
                   blastn_path = '~/software/ncbi-blast-2.15.0+/bin/blastn',
                   bowtie2_path = 'bowtie2',
                   bowtie2_mouse_db = '/media/duan/sda2/Reference_DataBase/GRCm39.M32/bowtie2index/GRCm39_VM32',
                   bowtie2_human_db = '/media/duan/sda2/Reference_DataBase/refdata-gex-GRCh38-2020-A/bowtie2index/GRCh38',
                   ref_seq_path = "../blastndb/refseq_rna",
                   mouse_genome_path = "../blastndb/GCF_000001635.27_top_level",
                   human_genome_path = "../blastndb/GCF_000001405.39_top_level",
                   rmsk_path = '/media/duan/sda2/Reference_DataBase/refdata-gex-GRCm39-2024-A/genes/mm39_rmsk.txt.gz',
                   bedtools_path = 'bedtools',
                   rmsk_bed = '/media/duan/sda2/Reference_DataBase/mm39/mm39_rmsk.bed'):
        
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
        self.P2evalue = P2evalue
        self.P1evalue = P1evalue
        self.strand = strand
        self.BLACK_FA = BLACK_FA
        self.gtype = gtype
        self.max_threads = max_threads
        self.max_memory = max_memory
        self.background_list = background_list
        
        self.Entrez_api_key = Entrez_api_key
        self.Entrez_email = Entrez_email
        self.mouse_fasta_path = mouse_fasta_path
        self.human_fasta_path = human_fasta_path
        self.blastn_path = blastn_path
        self.bowtie2_path = bowtie2_path
        self.bowtie2_mouse_db = bowtie2_mouse_db
        self.bowtie2_human_db = bowtie2_human_db
        self.ref_seq_path = ref_seq_path
        self.mouse_genome_path = mouse_genome_path
        self.human_genome_path = human_genome_path
        self.rmsk_path = rmsk_path
        self.bedtools_path = bedtools_path
        self.rmsk_bed = rmsk_bed
        
        
    @property
    def bridge_seq(self):
        
        bseqs = pd.read_csv(self.bseqs_path, sep="\t",index_col=0)
        return bseqs.loc[self.bridge_id,"Seq"]
        
        


def gene_generator(Para):
    
    """
    对于每一条设计出来的探针，将拼接好的探针和正在使用的完整Pad进行比对，去掉任何和Pad末端比对上的序列
    
    另外，新增一个探针的评分系统
    主要包括基础过滤评分(Tm, GC, Repeats), 非特异性比对评分(非特异条数，最大Tm与平均Tm等), Nupack评分，和Pad比对评分
    """
    
    import os
    
    if not os.path.exists("./TMP"):
        os.path.makedirs("./TMP")
    
    from pandarallel import pandarallel
    pandarallel.initialize(progress_bar = False, verbose=0, nb_workers = Para.max_threads)
    
    max_memory_allowed = Para.max_memory
                    
    Probes, probes_counts = findAllCandidates(seqs = Para.seq, prb_strand = Para.strand)
    
    print(f"[INFO]{len(Probes)} candidate probes")
    # Probes.to_csv("check.csv")
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
    
    # For mRNA,cDNA,Artificial_XXX,TE: BLASTn against refseq_rna
    # for Genome: BLASTn against 
    
    # _ = SeqIO.write(Probes["Probe1"].to_list(), fastafile1, "fasta-2line")
    write_to_fasta(Probes["Probe1"].to_list(), fastafile1)
    
    if Para.modality == 'Genome':
        blastn_DNA(fastafile = fastafile1, result_path = alignfile1, Para = Para, evalue = Para.P1evalue, taxids = Para.TaxonomyID)
        
        alignedProbe1 = pd.read_csv(alignfile1, sep = "\t",
                               names=["qseqid","sstart", "send", "qseq", "sseq", "evalue", "stitle", "sstrand"],
                                   index_col = 0)
        
    else:
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
    
    if Para.modality == 'Genome':
        blastn_DNA(fastafile = fastafile2, result_path = alignfile2, Para = Para, evalue = Para.P2evalue, taxids = Para.TaxonomyID)
        alignedProbe2 = pd.read_csv(alignfile2, sep = "\t",
                               names=["qseqid","sstart", "send", "qseq", "sseq", "evalue", "stitle", "sstrand"],
                                   index_col = 0)
    else:
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
    write_to_fasta(Probes["Full_Probe1"].to_list(), fastafile1)
    # _ = SeqIO.write(Probes["Full_Probe1"].to_list(), fastafile1, "fasta-2line")
    
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
        # _ = SeqIO.write(Probes["Full_Probe1"].to_list(), fastafile1, "fasta-2line")

        blastn_subject(fastafile = fastafile1, result_path = alignfile1, Para = Para, subject_file = Para.UsingPadfile)

        alignedProbe1 = pd.read_csv(alignfile1, sep = "\t",
                                   names=["qseqid","sseqid","pident","length","mismatch","sstart","send","evalue"],
                                       index_col = 0)

        Probes["probe1_blastn_check"] = Probes["Probe1"].parallel_apply(lambda x:non_specific_pad_check(x, alignedProbe = alignedProbe1, Para = Para))
        Probes = Probes[Probes["probe1_blastn_check"] == "PASS"].copy()

        print(f"[INFO]{len(Probes)} probes passed P1 system-reaction filter")

        write_to_fasta(Probes["Full_Probe2"].to_list(), fastafile2)
        # _ = SeqIO.write(Probes["Full_Probe2"].to_list(), fastafile2, "fasta-2line")

        blastn_subject(fastafile = fastafile2, result_path = alignfile2, Para = Para, subject_file = Para.UsingPadfile)

        alignedProbe2 = pd.read_csv(alignfile2, sep = "\t",
                                   names=["qseqid","sseqid","pident","length","mismatch","sstart","send","evalue"],
                                       index_col = 0)

        Probes["probe2_blastn_check"] = Probes["Probe2"].parallel_apply(lambda x:non_specific_pad_check(x, alignedProbe = alignedProbe2, Para = Para, isP2 = True))
        Probes = Probes[Probes["probe2_blastn_check"] == "PASS"].copy()

        Probes["probe_start"] = Probes["Probe1"].apply(get_start)
        Probes = Probes.reset_index(drop = True).copy()

        print(f"[INFO]{len(Probes)} probes passed P2 system-reaction filter")
        
    # 评分
    
    # 评分有几个关键点
    # 1.GC含量；2.比对到目标转录本数量；3.比对到非特异转录本数量；
    # 对于 mRNA，cDNA，也使用bowtie进行比对，去除掉比对不上的
    
    # print(f"[INFO]Evaluating...")
       
    bowtie2file1 = "./TMP/prbs1_candidates_alignment.sam"
    bowtie2file2 = "./TMP/prbs2_candidates_alignment.sam"

    # 先运行一遍bowtie2，去掉无法比对到基因组上的probe
    # 算是使用不同比对方法交叉验证
    if Para.modality not in ['TE', 'Artificial']:
        if Para.TaxonomyID in ["9606", "10090"]:
            
            write_to_fasta(Probes["Probe1"].to_list(), fastafile1)
            # _ = SeqIO.write(Probes["Probe1"].to_list(), fastafile1, "fasta-2line")
            bowtie2(fastafile = fastafile1, result_path = bowtie2file1, Para = Para, taxids = Para.TaxonomyID)
            Probes.index = Probes['Probe1'].apply(lambda x: x.id)
            keep_index = unmapped_bowtie_check(bowtie2file1)
            Probes = Probes.loc[keep_index, :].copy()

            write_to_fasta(Probes["Probe2"].to_list(), fastafile2)
            #_ = SeqIO.write(Probes["Probe2"].to_list(), fastafile2, "fasta-2line")
            bowtie2(fastafile = fastafile2, result_path = bowtie2file2, Para = Para, taxids = Para.TaxonomyID)
            Probes.index = Probes['Probe2'].apply(lambda x: x.id)
            keep_index = unmapped_bowtie_check(bowtie2file2)
            Probes = Probes.loc[keep_index, :].copy()
            
        else:
            pass
    
    # BLASTn进行评分
    
    probe1_list = Probes["Probe1"].to_list()
    
    # 这两个文件用于看Off-target
    write_to_fasta(probe1_list, fastafile1)
    write_to_fasta(Probes["Probe2"].to_list(), fastafile2)
    # _ = SeqIO.write(probe1_list, fastafile1, "fasta-2line")
    # _ = SeqIO.write(Probes["Probe2"].to_list(), fastafile2, "fasta-2line")
    
    # 生成第三个文件，完整的target region. 这个文件用于看On-target
    with open("./TMP/region_candidates.fasta", 'w') as handle:
        for p1 in probe1_list:
            handle.write(f">{p1.name}\n")
            handle.write(f"{p1.description}\n")
            
    print(f"[INFO]Preparing Indicator for Efficiency Calculatioin...")
    
    if Para.modality == 'TE':
        
        bowtie2(fastafile = fastafile1, result_path = bowtie2file1, Para = Para, taxids = Para.TaxonomyID)
        Drop = bedtools_intersect_filter(bowtie2file1, Para)
        Probes.index = Probes['Probe1'].apply(lambda x: x.id)
        Probes = Probes.loc[~Probes.index.isin(Drop),:].copy()
        
        write_to_fasta(Probes["Probe2"].to_list(), fastafile2)
        bowtie2(fastafile = fastafile2, result_path = bowtie2file2, Para = Para, taxids = Para.TaxonomyID)
        Drop = bedtools_intersect_filter(bowtie2file2, Para)
        Probes.index = Probes['Probe2'].apply(lambda x: x.id)
        Probes = Probes.loc[~Probes.index.isin(Drop),:].copy()
        
        print(f"[INFO]{len(Probes)} probes passed TE intersection filter")
        

        # TE得考虑比对到其它TE的情况，所以用bowtie2比对一下，然后将结果写成bam用。bedtools进行一下intersect，只保留
        
        # 我可以将所有的TE先写入fasta文件，然后用BLASTn进行比对
        # 我应该在探针生成时保留p1和p2之间的中间序列
        # 对于其它modality也是，不应该将p1的align和p2的align分开考虑，要合并到一起考虑，但是unalign p1可以单独考虑
        
        blastn(fastafile = fastafile1, result_path = alignfile1, evalue = 130, Para = Para, taxids = Para.TaxonomyID, strand = 'both')
        blastn(fastafile = fastafile2, result_path = alignfile2, evalue = 130, Para = Para, taxids = Para.TaxonomyID, strand = 'both')
    
        alignedProbe1 = pd.read_csv(alignfile1, sep = "\t",
                               names=["qseqid","sacc","pident","length","mismatch","qcovs","evalue","qseq", "sseq","stitle"],
                               index_col = 0)

        alignedProbe2 = pd.read_csv(alignfile2, sep = "\t",
                               names=["qseqid","sacc","pident","length","mismatch","qcovs","evalue","qseq", "sseq","stitle"],
                               index_col = 0)
        
    elif Para.modality == 'Genome':
        blastn_DNA(fastafile = fastafile1, result_path = alignfile1, Para = Para, evalue = 130, taxids = Para.TaxonomyID)
        blastn_DNA(fastafile = fastafile2, result_path = alignfile2, Para = Para, evalue = 130, taxids = Para.TaxonomyID)
        alignedProbe1 = pd.read_csv(alignfile1, sep = "\t",
                               names=["qseqid","sstart", "send", "qseq", "sseq", "evalue", "stitle", "sstrand"],
                                   index_col = 0)
        
        alignedProbe2 = pd.read_csv(alignfile2, sep = "\t",
                               names=["qseqid","sstart", "send", "qseq", "sseq", "evalue", "stitle", "sstrand"],
                                   index_col = 0)
        
    else:  
        blastn(fastafile = fastafile1, result_path = alignfile1, evalue = 130, Para = Para, taxids = Para.TaxonomyID, strand = 'both')
        blastn(fastafile = fastafile2, result_path = alignfile2, evalue = 130, Para = Para, taxids = Para.TaxonomyID, strand = 'both')
        # 这个比对结果用于看 aligned 部分
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
    # Probes[["p1_aligned", 'p1_unaligned', 'p1_unTarget_95thTm']] = Probes["Probe1"].parallel_apply(lambda x:non_specific_mean_Tm(x, alignedProbe1, Para))
    
    n_workers = min(Para.max_threads, int(max_memory_allowed // (sys.getsizeof(alignedProbe2) / 1073741824)))
    pandarallel.initialize(progress_bar = False, verbose=0, nb_workers = n_workers)
    Probes[['p2_unaligned', 'p2_unTarget_95thTm', 'p2_unaligned_gene']] = Probes["Probe2"].parallel_apply(lambda x:non_specific_genes(x, alignedProbe2, Para))
    
    if Para.modality == 'Genome' or Para.modality == 'Artificial':
        n_workers = Para.max_threads
        pandarallel.initialize(progress_bar = False, verbose=0, nb_workers = n_workers)
        Probes['aligned_count'] = Probes["Probe1"].parallel_apply(lambda x:specific_count(x, Para))
        Probes['best_comb'] = 'NA'
    
    elif Para.modality == 'cDNA' or Para.modality == 'mRNA':
        n_workers = min(Para.max_threads, int(max_memory_allowed // (sys.getsizeof(alignedProbe3) / 1073741824)))
        pandarallel.initialize(progress_bar = False, verbose=0, nb_workers = n_workers)
        Probes['aligned_count'] = Probes["Probe1"].parallel_apply(lambda x:specific_count(x, Para, alignedProbe3))
        Probes['best_comb'] = 'NA'
    
    elif Para.modality == 'TE':
        
        # 在这里运行贪心算法
        
        # TE要返回的aligned_count是三个探针组合能带来的最大aligned_count数，具体如何返回？
        # 以每一个探针为起点，寻找另外两个探针，构成最大align_count
        # 首先将BLASTn的结果转化成一个dict
        
        probe1_list = Probes["Probe1"].to_list()
        with open("./TMP/region_candidates.fasta", 'w') as handle:
            for p1 in probe1_list:
                handle.write(f">{p1.name}\n")
                handle.write(f"{p1.description}\n")
                
        blastn_subject(fastafile = "./TMP/region_candidates.fasta", 
                       result_path = "./TMP/region_candidates_alignment.tsv", Para = Para, subject_file = "./TMP/forBLASTnTE.fasta",
                       evalue = '0.5', strand = 'both')
        
 
        alignedProbe3 = pd.read_csv("./TMP/region_candidates_alignment.tsv", sep = "\t",
                               names=["qseqid","sseqid","pident","length","mismatch","sstart","send","evalue"],
                                   index_col = 0)
        
        
        print(f"[INFO]Greedy Selecting for Best Combinations...")
        probes_dict = _blastn_to_dict(alignedProbe3)
        greedy_dict = select_probe_combinations(probes_dict)
        
        # 此时greedy_list中每一项是dict，
        # results['start_probe'] = {
        #     'combination': selected_probes,
        #     'covered_objects': len(covered_objects)
        # }
        
        # 最简单的办法是TE的Probe额外增加4列，分别记录p1,p2,p1,p2
        # 或者增加一列记录selected_probes
        
        Probes.index = [x.name for x in Probes['Probe1']]
        Probes = Probes.loc[list(greedy_dict.keys()),:].copy()
        
        Probes['aligned_count'] = Probes['Probe1'].apply(lambda p1: greedy_dict[p1.name]['covered_objects'])
        Probes['best_comb'] = Probes['Probe1'].apply(lambda p1: greedy_dict[p1.name]['combination'])
        

    # Probes[["p2_aligned", 'p2_unaligned', 'p2_unTarget_95thTm']] = Probes["Probe2"].parallel_apply(lambda x:non_specific_mean_Tm(x, alignedProbe2, Para))
    
    # if Para.modality == 'TE':
    #     Probes['p1_aligned'] = list(p1_aligned.values())
    #     Probes['p2_aligned'] = list(p2_aligned.values())
    
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
        'best_comb': Probes['best_comb'],
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
    
    
    
    
    
    
    
    
    
    
    
    