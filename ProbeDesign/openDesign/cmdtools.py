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
        "-num_threads", str(Para.max_threads)
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
          "-out", result_path]
    
    subprocess.run(" ".join(call), shell = True, check = True)
    
    
def blastn_DNA(fastafile:str, result_path:str, Para, evalue = 5, taxids = "10090"):
    """
    runs blastn to create alignment results from a fasta file. FOR DNA
    """
    
    if taxids == "10090":
        db = Para.mouse_genome_path
    elif taxids == "9606":
        db = Para.human_genome_path
    
    call = [
        Para.blastn_path,
        "-query", fastafile,
        "-db", db,
        "-task", "blastn-short",
        # "-evalue", str(evalue),
        "-strand", "both",
        "-out", result_path,
        "-outfmt", r'"6 qseqid sstart send qseq sseq evalue stitle sstrand"',
        "-num_threads", str(Para.max_threads),
        "-max_target_seqs 500"
        ]
    
    subprocess.run(" ".join(call), shell = True, check = True)
    
def bowtie2(fastafile:str, result_path:str, Para, taxids = "10090"):
    
    if taxids == "10090":
        db = Para.bowtie2_mouse_db
    elif taxids == "9606":
        db = Para.bowtie2_human_db
           
    call = [
        Para.bowtie2_path,
        '-x', db,
        '-U', fastafile,
        '-S', result_path,
        '-f',
        '--sensitive',
        '-a',
        '--reorder',
        '-p', str(Para.max_threads)
    ]
    
    cp = subprocess.run(" ".join(call), shell = True, check = True, capture_output=True)