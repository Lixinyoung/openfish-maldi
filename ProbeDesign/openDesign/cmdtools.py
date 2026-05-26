import subprocess


def blastn(fastafile, result_path, Para, evalue = 5, taxids = "10090", strand = 'minus'):
    """
    runs blastn to create alignment results from a fasta file
    """
    call = [
        Para.blastn_path,
        "-query", fastafile,
        "-db", Para.ref_seq_path,
        "-task", "blastn-short",
        "-evalue", str(evalue),
        "-strand", strand,
        "-out", result_path,
        "-outfmt", r'"6 qseqid sacc pident length mismatch qcovs evalue qseq sseq stitle"',
        "-taxids", taxids,
        "-num_threads", str(Para.max_threads),
        '-mt_mode', '2'
        ]
    
    subprocess.run(" ".join(call), shell = True, check = True)
    
    
def blastn_subject(fastafile, subject_file, result_path, Para, strand = 'minus', evalue = '1000'):
    
    call = [Para.blastn_path, 
          "-query",  fastafile,
          "-task", "blastn-short",
          "-strand", strand,
          "-subject", subject_file,
          "-outfmt", r'"6 qseqid sseqid pident length mismatch sstart send evalue"',
          "-evalue", evalue,
          "-max_target_seqs", "10000",
          "-num_threads", "1",
          '-mt_mode', '2',
          "-out", result_path]
    
    subprocess.run(" ".join(call), shell = True, check = True)
    

def blastn_remote(fastafile, result_path, Para, evalue = 5, taxids = "10090", strand = 'minus'):
    """
    runs blastn to create alignment results from a fasta file
    """
    call = [
        Para.blastn_path,
        "-query", fastafile,
        "-db", Para.ref_seq_path,
        "-task", "blastn-short",
        "-evalue", str(evalue),
        "-strand", strand,
        "-out", result_path,
        "-outfmt", r'"6 qseqid sacc pident length mismatch qcovs evalue qseq sseq stitle staxid"',
        "-remote"
        ]
    
    subprocess.run(" ".join(call), shell = True, check = True)