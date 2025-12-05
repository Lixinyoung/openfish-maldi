from .generator import gene_generator
from tqdm import tqdm
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from Bio import Entrez, SeqIO
from Bio.SeqRecord import SeqRecord
import re
from pyfaidx import Fasta


def multigene_generator(OligoInput, Para):
    
    """
    该函数用于对多个基因进行探针生成
    
    genes: 输入dataframe
    
    """
    
    if Para.TaxonomyID == "10090":
        species = "Mus musculus"
    elif Para.TaxonomyID == "9606":
        species = "Homo sapiens"
    elif Para.TaxonomyID == "3847":
        species = "Glycine max"

        
    results = pd.DataFrame(columns = ['name',
                                  'probe_name',
                                  'position', 
                                  'probe A', 
                                  'probe B',
                                  'Evaluation',
                                  "aligned_count",
                                  'probeA_Tm',
                                  'probeB_Tm',
                                  'probeA_unTarget_maxTm',
                                  'probeB_unTarget_maxTm',
                                  "Off_Genes",
                                  'bridge_id', 
                                  'bridge_seq',
                                  'probe_strand'])

    for i in tqdm(range(len(OligoInput)), leave = False):
        
        print(f"[INFO]Generating probes for {OligoInput.loc[i,'target']}")
        
        sequence = OligoInput.loc[i, "seq"]
        target_name = OligoInput.loc[i, "target"]
        
        if target_name.startswith(('TE_class_', 'TE_family_', 'TE_subfamily_')):
            
            Para.modality = 'TE'
            
            # Use /media/duan/sda2/Reference_DataBase/refdata-gex-GRCm39-2024-A/genes/mm39_rmsk.txt.gz
            # Blastn will also be used to exclude unspecific binding.

            TE_level2column = {
                'class': 'repClass',
                'family': 'repFamily',
                'subfamily': 'repName'
            }
            
            name_split = target_name.split("_")
            
            TE_level = name_split[1]
            TE_name = "_".join(name_split[2:])
            
            # _, TE_level, TE_name = target_name.split("_")

            TE_df = pd.read_table(Para.rmsk_path)
            TE_df = TE_df[TE_df[TE_level2column[TE_level]] == TE_name]
            sequences = []
            for idx in TE_df.index:

                chr_name = TE_df.loc[idx, 'genoName']
                start = int(TE_df.loc[idx, 'genoStart'])
                end = int(TE_df.loc[idx, 'genoEnd'])
                strand = TE_df.loc[idx, 'strand']
                
                # 对于TE，生成一个fasta文件用于比对
                
                if re.match(r'^chr(\d+|X|Y|MT)$', chr_name):
                    target = get_genome_seq(chr_name, start, end, Para)
                    target.id = f"{target_name}_{idx}"
                    target.name = f"{target_name}_{idx}"
                    if strand == '+':
                        sequences.append(target)
                    else:
                        sequences.append(target.reverse_complement(id = True, name = True, description=True))

                else:
                    continue
                    
            _ = SeqIO.write(sequences, "./TMP/forBLASTnTE.fasta", "fasta-2line")
        
        elif not isinstance(sequence, str):
            
            # artificial sequences likes Barcodes, exogenous genes
            if target_name.startswith('Artificial_'):
                raise ValueError(f"Please offer sequences for {target_name}.")
            
            elif re.match(r'^chr(\d+|X|Y|MT)_\d+_\d+$', target_name):
                
                Para.modality = 'Genome'
                
                chr_name,start,end = target_name.split("_")
                start = int(start)
                end = int(end)
                target = get_genome_seq(chr_name, start, end, Para)
                sequences = [target]
                
            else:
                if Para.strand == 'minus':
                    Para.modality = 'mRNA'
                else:
                    Para.modality = 'cDNA'
                    
                
                print(f"[INFO]Downloading cDNA for {target_name}")
                sequences = get_seqs(gene_name = target_name, Para = Para, species = species, gtype = Para.gtype)
                        
        
        else:
            
            if target_name.startswith('Artificial_'):
                Para.modality = 'Artificial'
            elif re.match(r'^chr(\d+|X|Y|MT)_\d+_\d+$', target_name):
                Para.modality = 'Genome'
            else:
                if Para.strand == 'minus':
                    Para.modality = 'mRNA'
                else:
                    Para.modality = 'cDNA'
                
            sequences = [SeqRecord(seq = sequence, name = target_name, description=target_name)]
        
        Para.gene_name = target_name
        Para.seq = sequences
        Para.bridge_id = OligoInput.loc[i, "bridge_id"]
        
        # try:
        temp = gene_generator(Para)
        # except:
        #     print(f"[WARNING]Gene {Para.gene_name} has no available probes, check problems.")
        #     continue
            
        results = pd.concat([results, temp], ignore_index=True)
        
    return results

def get_seqs(gene_name: str, Para, species: str = "Mus musculus", gtype: str = 'gene') -> [SeqRecord]:
    
    """
    All transcripts of input gene will be traversed for NM_ or XM_ or NR_
    Sequences for defined feature will be output as a list(Bio.SeqRecord.SeqRecord)
    
    gene_name
    species: Mus musculus or Homo sapiens
    gtype: gene | CDS | exon
    """
    Entrez.api_key = Para.Entrez_api_key
    Entrez.email = Para.Entrez_email
    Entrez.sleep_between_tries = 0.1
    Entrez.max_tries = 60
    
    query = f"{species}[Orgn] AND {gene_name}[Gene] AND (mRNA OR RNA)"
    handle = Entrez.esearch(db="nucleotide", term=query, retmax=30)
    record_full = Entrez.read(handle)
    handle.close()
    
    output_seqs = []
    
    for seq_id in record_full["IdList"]:
        
        handle = Entrez.efetch(db="nucleotide", id=seq_id, rettype="gb", retmode="text")
        record = SeqIO.read(handle, 'genbank')
        
        if f"({gene_name})" in record.description:
            
            # Only Transcripts for mRNA and long non-coding RNA allowed for now
            if record.name.startswith(('NM_', 'XM_', 'NR_')):
                
                # Get all features
                selected_feature = []
                for feature in record.features:
                    if feature.type == gtype:
                        propertity_start = feature.location.start
                        propertity_end = feature.location.end
                        feature_len = propertity_end - propertity_start
                        if feature_len >= 62:
                            selected_feature.append(feature)
                        else:
                            continue
                    else:
                        continue
                    
                # If no defined feature available, output whole seqs
                if len(selected_feature) == 0:
                    print(f"{gtype} is not available for {gene_name}'s transcripts {record.name}.'\nOutput whole sequence.")
                    tmp_seq = SeqRecord(seq = record.seq, id = seq_id, name = record.name, description=f"{gene_name}_whole")
                    output_seqs.append(tmp_seq)
                else:
                    for feature in selected_feature:
                        tmp_seq = SeqRecord(seq = feature.extract(record.seq), id = seq_id, name = record.name, description=f"{gene_name}_{gtype}")
                        output_seqs.append(tmp_seq)
                        
            else:
                continue
                
        else:
            continue
            
    if len(output_seqs) == 0:
        raise ValueError(f"No Available sequences for {gene_name}. May because wrong input or network issue.")
                        
    return output_seqs

def get_genome_seq(chr_name:str, start:int, end:int, Para) -> SeqRecord:
    
    mouse_fasta_path = Para.mouse_fasta_path
    human_fasta_path = Para.human_fasta_path
    
    if Para.TaxonomyID == "10090":
        chr2RefSeqID = {
            "chr1": "NC_000067.7",
            "chr2": "NC_000068.8",
            "chr3": "NC_000069.7",
            "chr4": "NC_000070.7",
            "chr5": "NC_000071.7",
            "chr6": "NC_000072.7",
            "chr7": "NC_000073.7",
            "chr8": "NC_000074.7",
            "chr9": "NC_000075.7",
            "chr10": "NC_000076.7",
            "chr11": "NC_000077.7",
            "chr12": "NC_000078.7",
            "chr13": "NC_000079.7",
            "chr14": "NC_000080.7",
            "chr15": "NC_000081.7",
            "chr16": "NC_000082.7",
            "chr17": "NC_000083.7",
            "chr18": "NC_000084.7",
            "chr19": "NC_000085.7",
            "chrX": "NC_000086.8",
            "chrY": "NC_000087.8",
            "chrMT": "NC_005089.1"
        }

        fasta_ref = Fasta(mouse_fasta_path)

    elif Para.TaxonomyID == "9606":
        chr2RefSeqID = {
            "chr1": "NC_000001.11",
            "chr2": "NC_000002.12",
            "chr3": "NC_000003.12",
            "chr4": "NC_000004.12",
            "chr5": "NC_000005.10",
            "chr6": "NC_000006.12",
            "chr7": "NC_000007.14",
            "chr8": "NC_000008.11",
            "chr9": "NC_000009.12",
            "chr10": "NC_000010.11",
            "chr11": "NC_000011.10",
            "chr12": "NC_000012.12",
            "chr13": "NC_000013.11",
            "chr14": "NC_000014.9",
            "chr15": "NC_000015.10",
            "chr16": "NC_000016.10",
            "chr17": "NC_000017.11",
            "chr18": "NC_000018.10",
            "chr19": "NC_000019.10",
            "chr20": "NC_000020.11",
            "chr21": "NC_000021.9",
            "chr22": "NC_000022.11",
            "chrX": "NC_000023.11",
            "chrY": "NC_000024.10",
            "chrMT": "NC_012920.1"
        }

        fasta_ref = Fasta(human_fasta_path)

    target = fasta_ref[chr2RefSeqID[chr_name]][start:end]
    
    return SeqRecord(seq = target.seq, name = f"{chr_name}_{start}_{end}", description=f"{chr_name}_{start}_{end}")