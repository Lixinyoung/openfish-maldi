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
    This function is used to generate multiple gene probes
    
    genes: input dataframe
    
    """

        
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
        
        
        if not isinstance(sequence, str):
            
            # artificial sequences likes Barcodes, exogenous genes
            if target_name.startswith('Artificial_'):
                raise ValueError(f"Please offer sequences for {target_name}.")
                
            else:
                if Para.strand == 'minus':
                    Para.modality = 'mRNA'
                else:
                    Para.modality = 'cDNA'
                    
                print(f"[INFO]Downloading cDNA for {target_name}")
                sequences = get_seqs(gene_name = target_name, Para = Para, gtype = Para.gtype)
                        
        
        else:
            
            if target_name.startswith('Artificial_'):
                Para.modality = 'Artificial'
            else:
                if Para.strand == 'minus':
                    Para.modality = 'mRNA'
                else:
                    Para.modality = 'cDNA'
                
            sequences = [SeqRecord(seq = sequence, name = target_name, description=target_name)]
        
        Para.gene_name = target_name
        Para.seq = sequences
        Para.bridge_id = OligoInput.loc[i, "bridge_id"]
        
        temp = gene_generator(Para)
   
        results = pd.concat([results, temp], ignore_index=True)
        results.to_csv("TMP/tmp_generated_probes.csv") # This file is for generated probes storage, in case error occurs, don't need to run all target again
        
    return results

def get_seqs(gene_name: str, Para, gtype: str = 'gene') -> [SeqRecord]:
    """
    Retrieve sequences. Retries with increasing retmax if no sequences found.
    """
    Entrez.api_key = Para.Entrez_api_key
    Entrez.email = Para.Entrez_email
    Entrez.sleep_between_tries = 0.1
    Entrez.max_tries = 60

    species = Para.species

    query = f"{species}[Orgn] AND {gene_name}[Gene] AND (mRNA OR RNA)"
    
    for retmax in [30, 60, 90, 100, 500, 1000]:
        output_seqs = []
        handle = Entrez.esearch(db="nucleotide", term=query, retmax=retmax)
        record_full = Entrez.read(handle)
        handle.close()
        
        for seq_id in record_full["IdList"]:
            try:
                handle = Entrez.efetch(db="nucleotide", id=seq_id, rettype="gb", retmode="text")
                record = SeqIO.read(handle, 'genbank')
                handle.close()
            except Exception:
                continue

            if f"({gene_name})" not in record.description or not record.name.startswith(('NM_', 'XM_', 'NR_')):
                continue

            selected_feature = [f for f in record.features 
                                if f.type == gtype and (f.location.end - f.location.start) >= 62]

            if not selected_feature:
                print(f"{gtype} not available for {record.name}. Output whole sequence.")
                output_seqs.append(SeqRecord(seq=record.seq, id=seq_id, name=record.name, description=f"{gene_name}_whole"))
            else:
                for feature in selected_feature:
                    output_seqs.append(SeqRecord(seq=feature.extract(record.seq), id=seq_id, name=record.name, description=f"{gene_name}_{gtype}"))

        if output_seqs:
            return output_seqs
            
    raise ValueError(f"No Available sequences for {gene_name}. May because wrong input or network issue.")