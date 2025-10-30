import pandas as pd
from Bio import SeqIO

# ------------------ Function 1 ------------------
def load_metadata(csv_file):
    return pd.read_csv(csv_file)

# ------------------ Function 2 ------------------
def load_sequences(fasta_file):
    sequences = list(SeqIO.parse(fasta_file, "fasta"))
    sequence_data = [{"ID": record.id, "Protein_Sequence": str(record.seq), "Length": len(record.seq)} for record in sequences]
    return pd.DataFrame(sequence_data)

# ------------------ Function 3 ------------------
def merge_data(metadata_df, sequences_df):
    sequences_df['ID_cleaned'] = sequences_df['ID'].str.split('.').str[0]
    merged_df = pd.merge(metadata_df, sequences_df, how='inner', left_on='Accession', right_on='ID_cleaned')
    return merged_df

# ------------------ NEW FUNCTION: merge nucleotide FASTA ------------------
def load_and_merge_nucleotide_sequences(merged_df, nucleotide_fasta):
    """
    Load nucleotide FASTA sequences and merge with already merged dataframe
    using the 'Nucleotide' column from metadata as key.
    """
    nucleotide_seqs = list(SeqIO.parse(nucleotide_fasta, "fasta"))
    nucleotide_data = [{"Nucleotide_ID": rec.id.split('.')[0], "Nucleotide_Sequence": str(rec.seq)} for rec in nucleotide_seqs]
    nucleotide_df = pd.DataFrame(nucleotide_data)

    if 'Nucleotide' not in merged_df.columns:
        raise KeyError("The merged DataFrame does not have a 'Nucleotide' column for merging.")

    merged_with_nt = pd.merge(merged_df, nucleotide_df, how='left', left_on='Nucleotide', right_on='Nucleotide_ID')
    return merged_with_nt

# ------------------ Function 4 ------------------
def extract_virus_column(df, source_col, target_col):
    df.loc[:, target_col] = df[source_col].str.extract(r'\[(.*?)\]')
    return df

# ------------------ Function 5 ------------------
def remove_unwanted_columns(df, columns_to_remove):
    return df.drop(columns=[col for col in columns_to_remove if col in df.columns], errors='ignore')

# ------------------ Function 6 ------------------
def load_host_mapping_csv(mapping_file):
    return pd.read_csv(mapping_file)

# ------------------ Function 7 ------------------
def map_host_to_agg_with_mapping(df, host_col, target_col, mapping_df):
    mapping_dict = mapping_df.set_index('Host')['Host_agg'].to_dict()

    def map_host(host):
        if pd.isna(host):
            return "unknown host"
        return mapping_dict.get(host, pd.NA)

    df[target_col] = df[host_col].apply(map_host)
    return df

# ------------------ Function 8 ------------------
def add_human_column(df, host_col, human_col):
    df[human_col] = df[host_col].apply(lambda x: 1 if str(x).lower() == "human" else 0)
    return df

# ------------------ Function 9 ------------------
def drop_duplicates_on_sequence(df):
    return df.drop_duplicates(subset=['Protein_Sequence'])

# ------------------------- MAIN SCRIPT -----------------------------
if __name__ == "__main__":
    csv_files = ["C:/Users/vinni/Downloads/respiratory-virus-transmission-model/data/demo_data/metadata.csv"]
    fasta_files = ["C:/Users/vinni/Downloads/respiratory-virus-transmission-model/data/demo_data/protein.fasta"]
    nucleotide_fastas = ["C:/Users/vinni/Downloads/respiratory-virus-transmission-model/data/demo_data/genome.fasta"]  # nucleotide FASTA file
    mapping_file = "C:/Users/vinni/Downloads/respiratory-virus-transmission-model/data/host_mapping.csv"

    if len(csv_files) != len(fasta_files):
        raise ValueError("The number of CSV files must match the number of FASTA files.")

    all_data_frames = []

    for csv_file, fasta_file, nucleotide_fasta in zip(csv_files, fasta_files, nucleotide_fastas):
        metadata_df = load_metadata(csv_file)
        protein_df = load_sequences(fasta_file)
        merged_df = merge_data(metadata_df, protein_df)

        # ---- merge with nucleotide FASTA ----
        merged_with_nt = load_and_merge_nucleotide_sequences(merged_df, nucleotide_fasta)

        df_modified = extract_virus_column(merged_with_nt, source_col="GenBank_Title", target_col="Virus")

        df_col_removed = remove_unwanted_columns(df_modified, [
            "Unnamed: 0", "Organism_Name", "GenBank_RefSeq", "Assembly", "SRA_Accession",
            "Submitters", "Release_Date", "Isolate", "Molecule_type", "Length_x",
            "Genotype", "Segment", "Publications", "Geo_Location", "USA",
            "Tissue_Specimen_Source", "Collection_Date", "BioSample", "BioProject", "GenBank_Title",
            "ID", "Length_y", "ID_cleaned", "Nucleotide_ID"
        ])

        host_mapping_df = load_host_mapping_csv(mapping_file)
        df_host_mapped = map_host_to_agg_with_mapping(df_col_removed, host_col='Host', target_col='Host_agg', mapping_df=host_mapping_df)
        df_human_added = add_human_column(df_host_mapped, host_col='Host_agg', human_col='Human')

        all_data_frames.append(df_human_added)

    all_df_merged = pd.concat(all_data_frames, ignore_index=True)
    final_cleaned_df = drop_duplicates_on_sequence(all_df_merged)

    # ---------------- Save Final Output ----------------
    output_file = "curated_dataset.csv" # can edit file name
    final_cleaned_df.to_csv(output_file, index=False)
    print(f"Data processing completed. Cleaned data with protein and nucleotide sequences saved to {output_file}.")
