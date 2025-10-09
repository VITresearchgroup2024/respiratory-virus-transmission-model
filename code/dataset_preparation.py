# ------------------ Function 1 ------------------
def load_metadata(csv_file):
    return pd.read_csv(csv_file)


# ------------------ Function 2 ------------------
def load_sequences(fasta_file):
    sequences = list(SeqIO.parse(fasta_file, "fasta"))
    sequence_data = [{"ID": record.id, "Sequence": str(record.seq), "Length": len(record.seq)} for record in sequences]
    return pd.DataFrame(sequence_data)


# ------------------ Function 3 ------------------
def merge_data(metadata_df, sequences_df):
    sequences_df['ID_cleaned'] = sequences_df['ID'].str.split('.').str[0]
    merged_df = pd.merge(metadata_df, sequences_df, how='inner', left_on='Accession', right_on='ID_cleaned')
    return merged_df


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
    df[target_col] = df[host_col].map(mapping_dict)
    return df


# ------------------ Function 8 ------------------
def add_human_column(df, host_col, human_col):
    df[human_col] = df[host_col].apply(lambda x: 1 if str(x).lower() == "human" else 0)
    return df


# ------------------ Function 9 ------------------
def drop_duplicates_on_sequence(df):
    return df.drop_duplicates(subset=['Sequence'])


# ------------------ Function 10 ------------------
def load_codon_usage(json_path):
    """Load codon usage table from JSON file."""
    with open(json_path, "r") as f:
        return json.load(f)


# ------------------ Function 11 ------------------
def reverse_translate_protein(df, seq_col, species_col, codon_usage_dict, new_col="Reverse_Translated_DNA"):
    """
    Reverse translate protein sequences into DNA using species-specific codon usage.
    If species not found, fallback to 'default' codon table.
    Handles weighted probabilities or flat codon lists.
    """
    dna_sequences = []

    for _, row in df.iterrows():
        protein_seq = str(row[seq_col]).upper()
        species = str(row[species_col])

        # Handle missing or unknown virus species
        if species not in codon_usage_dict:
            if "default" in codon_usage_dict:
                codon_table = codon_usage_dict["default"]
            else:
                dna_sequences.append("Species_Not_Found")
                continue
        else:
            codon_table = codon_usage_dict[species]

        dna_seq = ""
        for aa in protein_seq:
            if aa in codon_table:
                codon_info = codon_table[aa]

                # Handle weighted or list format
                if isinstance(codon_info, dict):
                    codons = list(codon_info.keys())
                    weights = list(codon_info.values())
                    chosen_codon = random.choices(codons, weights=weights, k=1)[0]
                elif isinstance(codon_info, list):
                    chosen_codon = random.choice(codon_info)
                elif isinstance(codon_info, str):
                    chosen_codon = codon_info
                else:
                    chosen_codon = "NNN"
            else:
                chosen_codon = "NNN"

            dna_seq += chosen_codon

        dna_sequences.append(dna_seq)

    df[new_col] = dna_sequences
    return df


# ------------------------- MAIN SCRIPT -----------------------------
if __name__ == "__main__":
    csv_files = ["./sequences.csv"]
    fasta_files = ["./sequences.fasta"]
    mapping_file = "./host_mapping.csv"
    codon_usage_file = "./codon_usage.json"  # <--- JSON file path

    if len(csv_files) != len(fasta_files):
        raise ValueError("The number of CSV files must match the number of FASTA files.")

    #  Load codon usage JSON once
    codon_usage_dict = load_codon_usage(codon_usage_file)

    all_data_frames = []

    for csv_file, fasta_file in zip(csv_files, fasta_files):
        metadata_df = load_metadata(csv_file)
        sequences_df = load_sequences(fasta_file)
        merged_df = merge_data(metadata_df, sequences_df)
        df_modified = extract_virus_column(merged_df, source_col="GenBank_Title", target_col="Virus")

        df_col_removed = remove_unwanted_columns(df_modified, [
            "Unnamed: 0", "Organism_Name", "GenBank_RefSeq", "Assembly", "Nucleotide", "SRA_Accession",
            "Submitters", "Release_Date", "Isolate", "Molecule_type", "Length_x",
            "Genotype", "Segment", "Publications", "Geo_Location", "USA",
            "Tissue_Specimen_Source", "Collection_Date", "BioSample", "BioProject", "GenBank_Title", "ID", "Length_y",
            "ID_cleaned"
        ])

        host_mapping_df = load_host_mapping_csv(mapping_file)
        df_host_mapped = map_host_to_agg_with_mapping(df_col_removed, host_col='Host', target_col='Host_agg', mapping_df=host_mapping_df)
        df_human_added = add_human_column(df_host_mapped, host_col='Host_agg', human_col='Human')

        all_data_frames.append(df_human_added)

    all_df_merged = pd.concat(all_data_frames, ignore_index=True)
    final_cleaned_df = drop_duplicates_on_sequence(all_df_merged)

    # Reverse translate using codon usage JSON
    final_translated_df = reverse_translate_protein(
        final_cleaned_df,
        seq_col="Sequence",
        species_col="Genus",
        codon_usage_dict=codon_usage_dict
    )

    output_file = "curated_dataset.csv" # edit file name if needed
    final_translated_df.to_csv(output_file, index=False)
    print(f" Data processing completed. Cleaned and reverse-translated data saved to {output_file}.")
