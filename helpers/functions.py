from typing import Optional
from sentence_transformers import SentenceTransformer
import os
import json
import pickle
import re
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz
from concurrent.futures import ThreadPoolExecutor, as_completed
import nirjas
from itertools import combinations

def sample_by_label_limit(df: pd.DataFrame, label_column: str, max_samples_per_label = 5, random_state=None):
    """
    Samples a DataFrame to ensure each unique label appears at most 
    a specified number of times, with control over random sampling.

    Args:
        df: The DataFrame to sample.
        label_column: The name of the column containing the labels.
        max_samples_per_label: The maximum number of samples per label.
        random_state: Seed for the random number generator (for reproducibility).

    Returns:
        A new DataFrame containing the sampled rows.
    """

    sampled_df = pd.DataFrame()
    for label in df[label_column].unique():
        label_df = df[df[label_column] == label]
        sample_size = min(len(label_df), max_samples_per_label) 
        sampled_df = pd.concat([sampled_df, label_df.sample(sample_size, random_state=random_state)])  # Apply random state here
    sampled_df['old_index'] = sampled_df.index
    sampled_df.reset_index(drop=True, inplace=True)
    return sampled_df

def read_json_file(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in '{file_path}' - {e}")

    return None

def create_license_dataset(path_to_license_details_directory, output_path=None):
    """
    Creates a text file containing license information (name, ID, text) from a directory of JSON license details.

    Args:
        path_to_license_details_directory (str): The path to the directory containing the license details JSON files.
        output_path (str, optional): The path to the output text file. If None, the file will be saved in the same parent directory containing the license details

    Raises:
        FileNotFoundError: If the specified license details directory does not exist.

    Returns:
        None: The function writes the license dataset to a file and does not return a value.
    """

    # ! The 'details' folder containing all the license details can be found at:
        # ! https://github.com/spdx/license-list-data/tree/main/json
        # ! Make sure to download the entire 'details' directory

    if not os.path.exists(path_to_license_details_directory):
        raise FileNotFoundError(f"Directory '{path_to_license_details_directory}' not found.")

    license_dataset_file_data = []
    licenses = os.listdir(path_to_license_details_directory)

    # Determine output file path
    if output_path is None:
        parent_dir = os.path.dirname(path_to_license_details_directory)
        license_dataset_file_path = os.path.join(parent_dir, "license_dataset.csv")
    else:
        license_dataset_file_path = output_path

    # Iterate over license files and extract data
    for license in licenses:
        license_path = os.path.join(path_to_license_details_directory, license)
        with open(license_path, 'r') as file:
            license_data = json.load(file)

        license_dataset_file_data.append({
            "licenseName": license_data.get('name', ''),  # Use .get() to avoid KeyError if 'name' is missing
            "licenseId": license_data.get('licenseId', ''),
            "licenseText": license_data.get('licenseText', '')
        })

    # Write the license dataset to the output csv file
    df = pd.DataFrame(columns=['License Name', 'License ID', 'License Text'])
    
    for index, license_data in enumerate(license_dataset_file_data):
        new_row = pd.DataFrame({'License Name': f"{license_data['licenseName']}", 
                       'License ID': f"{license_data['licenseId']}",
                    #    'License Text': f"\n{license_data['licenseText']}"}, index=[index])
                       'License Text': f"\nSPDX-License-Identifier: {license_data['licenseId']}\n\nLicense Name: {license_data['licenseName']}\n\n{license_data['licenseText']}"}, index=[index])
        df = pd.concat([df, new_row], ignore_index=True)
        
    df.to_csv(license_dataset_file_path)

    print(f'License dataset file created successfully at {license_dataset_file_path}')

def get_top_similar_license_lines(
    code_text: str,
    licenses_file_path: str,
    license_embeddings: Optional[np.ndarray] = None,
    model: SentenceTransformer = SentenceTransformer("all-mpnet-base-v2"),
    embeddings_dir: str = "extras/license_information/license_embeddings",
    top_k: int = 5,
    min_similarity: int = 50,
    double_semantic_search: bool = False,
):
    """
    Identifies the top-k most similar lines from a given code text to a set of license texts.
    This function leverages sentence embeddings to compute similarity and can perform an advanced
    double semantic search if specified.

    Args:
        code_text (str): The text content from a code file, including comments.
        licenses_file_path (str): Path to the file containing different license texts.
        license_embeddings (ndarray, optional): Precomputed embeddings of the license texts. If not provided,
                                                they will be computed and saved.
        model (SentenceTransformer, optional): The model used for generating embeddings. Defaults to 'all-mpnet-base-v2'.
        embeddings_dir (str, optional): Directory to store or load embeddings. Defaults to 'extras/license_information/license_embeddings'.
        top_k (int, optional): The number of top similar entries to return. Defaults to 5.
        min_similarity (float, optional): Minimum similarity score (0-100) to consider a match. Defaults to 50.
        double_semantic_search (bool, optional): If True, performs a secondary search for higher accuracy. Defaults to False.

    Returns:
        A list of tuples, each representing a match:
            - Similarity score (float)
            - Matched code line (str)
            - License name (str)
            - License ID (str)
            - Best matching license text line (Optional[str], only present if `double_semantic_search` is True)
            - List of top 5 matching license names with scores (Optional[List[Tuple[str, float]]], only present if `double_semantic_search` is True)
    """

    # Ensure directory for storing embeddings exists
    os.makedirs(embeddings_dir, exist_ok=True)

    # Load license texts from CSV
    licenses = pd.read_csv(licenses_file_path)

    # Load/compute license embeddings
    model_name = model._first_module().auto_model.config._name_or_path.split('/')[1]
    embeddings_file_path = os.path.join(
        embeddings_dir, f"{model.get_sentence_embedding_dimension()}_{model_name.replace('/', '_')}-license-embedding.pkl"
    )

    if license_embeddings is None:
        if os.path.exists(embeddings_file_path):
            # Load pre-computed embeddings if available
            with open(embeddings_file_path, "rb") as fIn:
                stored_data = pickle.load(fIn)
                license_embeddings = stored_data['embeddings']
        else:
            # Compute and store embeddings if not found
            print(f"Pre-embedded licenses not found. Creating embeddings and saving to: {embeddings_file_path}")
            license_embeddings = model.encode(licenses['License Text'])
            with open(embeddings_file_path, "wb") as fOut:
                pickle.dump({'embeddings': license_embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)

    # Clean up special characters from the code text to standardize it for embedding generation
    chars_to_remove = ['—', '…', '•', '§', '«', '»', '„', '・', '−', '*', '>', '<']
    for char_to_remove in chars_to_remove:
        code_text = code_text.replace(char_to_remove, '')

    # Split code into individual lines or chunks
    code_chunks = code_text.split('\n')

    # Generate embeddings for each line/chunk of code
    code_embeddings = model.encode(code_chunks)
    
    # Compute similarity between code embeddings and license embeddings
    similarity_matrix = np.zeros((len(code_embeddings), len(license_embeddings)))
    for i in range(len(code_embeddings)):
        embedding = code_embeddings[i].reshape(1, -1)
        similarity_matrix[i] = cosine_similarity(embedding, license_embeddings)
    
     # Extract and sort results based on similarity scores
    results = []
    for i in range(len(code_embeddings)):  
        max_score_index = np.argmax(similarity_matrix[i])
        max_score = similarity_matrix[i][max_score_index]
        results.append(
            (
                i,
                max_score,
                code_chunks[i],
                licenses.loc[max_score_index, 'License Name'],
                licenses.loc[max_score_index, 'License ID'],
            )
        )
    results.sort(key=lambda x: x[1], reverse=True)

    # Select the top-k results
    top_tuples = results[:top_k]

    if double_semantic_search:
        # Map each license text line/chunk (both for enhanced semantic search) to its correct index in the dataset
        license_index_map = {}
        all_license_texts = []

        # Extract all license texts (lines/chunks) from each license text in the dataset
        for license_index, license_text in enumerate(licenses['License Text']):
            # List of all connect consecutive lines for chunking. (This is also done with file comments)
            current_license_text = []  
            for line in license_text.split('\n'):
                # For each line, append it 
                all_license_texts.append(line)
                license_index_map[len(all_license_texts) - 1] = license_index
                # If this is an empty line, append all consecutive lines already saved in the current_license_text
                # together to the list of license texts
                if len(line) < 4:
                    if current_license_text:
                        all_license_texts.append(''.join(current_license_text))
                        license_index_map[len(all_license_texts) - 1] = license_index
                        current_license_text = []
                else:
                    # If not an empty line, most be connected to lines already in the current_license_text list
                    # Append that line 
                    current_license_text.append(line)
            # In the end, append any left over lines.
            if current_license_text:
                all_license_texts.append(''.join(current_license_text))
                license_index_map[len(all_license_texts) - 1] = license_index

        # Second level semantic search matching the top chunks to their most similar license
        # This is what actually matches a line/chunk to its probable license, the first layer is 
        # mainly used to get the top_k lines themselves but is not good at matching a chunk to its original 
        # license
        similarity_matrix = np.zeros((len(top_tuples), len(all_license_texts)))
        for index, tuple in enumerate(top_tuples):
            code_text = tuple[2]
            
            # Perform similarity matching using fuzzywuzzy which uses Levenshtein distance
            for i in range(len(all_license_texts)):
                similarity_matrix[index][i] = fuzz.ratio(code_text, all_license_texts[i]) 
            
            # Extract the top 5 licenses matches to this chunk
            max_score_index = np.argmax(similarity_matrix[index]) 
            top_5_indices = np.argsort(similarity_matrix[index])[-5:][::-1]
            license_index = license_index_map[max_score_index]
            top_tuples[index] = (
                                    similarity_matrix[index][max_score_index], 
                                    tuple[2], licenses.loc[license_index, 'License Name'],
                                    licenses.loc[license_index, 'License ID'],
                                    all_license_texts[max_score_index],
                                    [
                                        (
                                            licenses.loc[license_index_map[idx], 'License Name'],
                                            similarity_matrix[index][idx]
                                        ) 
                                        for idx in top_5_indices
                                    ]
                                ) 

    # Additional logic for merging potentially related code texts
    # This process attempts to merge code texts identified as similar to enhance similarity detection
    # Mainly used for differentiating between very similar tricky licenses (e.g. ISC, 0BSD, MIT variants, etc.)
    for merge_combo in combinations(top_tuples, 2):
        tuple1, tuple2 = merge_combo
        code_text1, code_text2 = tuple1[1], tuple2[1]

        # Create merged versions of the code texts
        # This tries both possible orders of merging to see if one order yields a higher similarity score
        merged_code_text12 = f"{code_text1}\n{code_text2}"
        merged_code_text21 = f"{code_text2}\n{code_text1}"

        # Analyze each merged code text against all licenses to determine if the merge improves the similarity
        for merged_code_text in [merged_code_text12, merged_code_text21]:
            
            # Compute similarity scores for the merged code text against each license text
            similarity_scores = np.zeros(len(licenses))
            for i, lic_text in enumerate(licenses['License Text']):
                similarity_scores[i] = fuzz.ratio(merged_code_text, lic_text)

            # Identify the maximum similarity score and corresponding license from the scores calculated
            max_index = np.argmax(similarity_scores)
            new_score = similarity_scores[max_index]
            top_5_indices = np.argsort(similarity_scores[index])[-5:][::-1]

            # If the new score meets the minimum similarity threshold, consider this merged code text as a valid potential match
            if new_score >= min_similarity:
                top_tuples.append(
                    (
                        new_score,
                        merged_code_text, 
                        licenses.loc[max_index, 'License Name'],
                        licenses.loc[max_index, 'License ID'],
                        [
                            (
                                licenses.loc[license_index_map[idx], 'License Name'],
                                similarity_scores[idx]
                            )
                            for idx in top_5_indices
                        ],
                    )
                )

    # Filter out results that do not meet the minimum similarity threshold before returning
    filtered_results = [result for result in top_tuples if result[0] >= min_similarity]

    return filtered_results

def extract_comments(df: pd.DataFrame):
        for index, row in df.iterrows():
            try: 
                nirjas_comments = nirjas.extract(os.path.join('extras', row['file path']))
                all_comments = []
                
                with open(os.path.join('extras', row['file path']), "r") as f:
                    all_lines = f.readlines()
                
                for single_line_comment in nirjas_comments['single_line_comment']:
                    all_comments.append(single_line_comment['comment'])
                for cont_single_line_comment in nirjas_comments['cont_single_line_comment']:
                    start = cont_single_line_comment['start_line'] - 1
                    end = cont_single_line_comment['end_line']
                    for line_idx in range(start, end):
                        comment = all_lines[line_idx].strip('\n').strip()
                        all_comments.append(comment)
                for multi_line_comment in nirjas_comments['multi_line_comment']:
                    start = multi_line_comment['start_line'] - 1
                    end = multi_line_comment['end_line']
                    current_comment = []  
                    for line_idx in range(start, end):
                        line = all_lines[line_idx].strip('\n').strip()
                        all_comments.append(line) 
                        if len(line) < 4:
                            if current_comment:
                                all_comments.append(''.join(current_comment))
                                current_comment = []
                        else:
                            current_comment.append(line)
                    if current_comment:
                        all_comments.append(''.join(current_comment))
                comments = "\n".join(all_comments)
            except:
                with open(os.path.join('extras', row['file path']), "r") as f:
                    comments = f.read()
                    # # Group paragraphs together, handling connected lines and preserving indentation
                    # grouped_comments = ""
                    # current_paragraph = []

                    # for line in comments.splitlines():
                    #     if line.strip():  # Non-empty line (part of a paragraph)
                    #         current_paragraph.append(line)
                    #     else:  # Empty line (end of paragraph)
                    #         if current_paragraph:
                    #             grouped_comments += " ".join(current_paragraph) + "\n\n"  # Combine with space if needed
                    #             current_paragraph = []

                    # # Handle last paragraph (if it doesn't end with an empty line)
                    # if current_paragraph:
                    #     grouped_comments += " ".join(current_paragraph) + "\n"  # No extra newline at the end 

                    # # Optional: further clean-up
                    # comments = re.sub(r"\n{3,}", "\n\n", grouped_comments) # Remove excess blank lines
            df.loc[index, 'file_comments'] = comments
        return df

def license_line_found(top_k_lines, relevant_lines):
    top_k_lines = top_k_lines.split('\n')
    relevant_lines = relevant_lines[2:-2].split("', '")

    if relevant_lines[0] == '':
        return 1

    for line in relevant_lines:
        for top_line in top_k_lines:
            similarity_ratio = fuzz.ratio(line, top_line)  
            if similarity_ratio >= 80:
                return 1
    return 0

def asses_coverage(top_k_lines, relevant_lines):
    top_k_lines = top_k_lines.split('\n')
    relevant_lines = relevant_lines[2:-2].split("', '")
    
    coverage = 100

    if relevant_lines[0] == '':
        return coverage

    for line in relevant_lines:
        covered = False
        for top_line in top_k_lines:
            similarity_ratio = fuzz.ratio(line, top_line)  
            if similarity_ratio >= 80:
                covered = True
        if not covered:
            coverage -= (1 / len(relevant_lines)) * 100
    return coverage

def predicted_license_found(license_ids, labels):
    labels = labels.split(' ')
    license_ids = license_ids.split('\n')

    if labels[0] == 'No_license_found':
        return 1

    for label in labels:
        for license_id in license_ids:
            similarity_ratio = fuzz.ratio(label, license_id)  
            if (similarity_ratio >= 35) or label in license_id:
                return 1
    return 0

def predicted_license_covered(license_ids, labels):
    labels = labels.split(' ')
    license_ids = license_ids.split('\n')

    if labels[0] == 'No_license_found':
        return 100

    coverage = 100

    for label in labels:
        covered = False
        for license_id in license_ids:
            similarity_ratio = fuzz.ratio(label, license_id)  
            if similarity_ratio >= 40:
                covered = True
        if not covered:
            coverage -= (1 / len(labels)) * 100
    return coverage