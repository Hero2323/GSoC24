from sentence_transformers import SentenceTransformer
import os
import json
import pickle
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import FastText  # Import the FastText class
from fuzzywuzzy import fuzz
from concurrent.futures import ThreadPoolExecutor, as_completed
import nirjas

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
                       'License Text': f"\nSPDX-License-Identifier: {license_data['licenseId']}\nLicense Name: {license_data['licenseName']}\n{license_data['licenseText']}"}, index=[index])
        df = pd.concat([df, new_row], ignore_index=True)
        
    df.to_csv(license_dataset_file_path)

    print(f'License dataset file created successfully at {license_dataset_file_path}')

def get_top_similar_license_lines_old(
    code_text,
    licenses_file_path,
    license_embeddings,
    model="all-mpnet-base-v2",
    embeddings_dir="extras/license_information/license_embeddings",
    embedding_approach = 'file-embedding',
    top_k=3,
):
    """
    Finds the top-k most similar lines in a code file to the entire license text using either TF-IDF or sentence embeddings.
    """

    os.makedirs(embeddings_dir, exist_ok=True)
    with open(licenses_file_path, 'r', encoding='utf-8') as file:
                licenses = file.read()

    # if model == 'tf-idf':
    #     embeddings_file_path = os.path.join(embeddings_dir, f"tfidf_vectorizer-{embedding_approach}.pkl")
    #     if os.path.exists(embeddings_file_path):
    #         print(f"Loading pre-trained TF-IDF vectorizer from: {embeddings_file_path}")
    #         with open(embeddings_file_path, "rb") as f:
    #             tfidf_data = pickle.load(f)
    #             vectorizer = tfidf_data['vectorizer']
    #             license_vectors = tfidf_data['embeddings']
    #     else:
    #         print(f"Pre-trained TF-IDF vectorizer not found. Training and saving to: {embeddings_file_path}")
    #         vectorizer = TfidfVectorizer()
    #         vectorizer.fit(licenses.split('----------------------------------------'))
    #         if embedding_approach == 'file-embedding':
    #             license_vectors = vectorizer.transform(licenses) #TODO: Fix error in this case. add a acheck
    #             # license_vectors = license_vectors.reshape(1, -1)
    #         elif embedding_approach == 'line-embedding':
    #             license_vectors = vectorizer.transform(licenses.split('\n'))
    #         elif embedding_approach == 'license-embedding':
    #             license_vectors = vectorizer.transform(licenses.split('----------------------------------------'))
    #         else:
    #             # TODO: add error
    #             pass
    #         with open(embeddings_file_path, "wb") as f:
    #             pickle.dump({'vectorizer': vectorizer, 'embeddings': license_vectors}, f, protocol=pickle.HIGHEST_PROTOCOL)

    #     # Encode code chunks (TF-IDF)
    #     code_chunks = code_text.split('\n')
    #     code_vectors = vectorizer.transform(code_chunks)
    # elif model == 'bow':
    #     embeddings_file_path = os.path.join(embeddings_dir, f"bow_vectorizer-{embedding_approach}.pkl")
    #     if os.path.exists(embeddings_file_path):
    #         print(f"Loading pre-trained BoW vectorizer from: {embeddings_file_path}")
    #         with open(embeddings_file_path, "rb") as f:
    #             bow_data = pickle.load(f)
    #             vectorizer = bow_data['vectorizer']
    #             license_vectors = bow_data['embeddings']
    #     else:
    #         print(f"Pre-trained BoW vectorizer not found. Training and saving to: {embeddings_file_path}")
    #         vectorizer = CountVectorizer()  # Use CountVectorizer for BoW

    #         if embedding_approach == 'file-embedding':
    #             # Check for empty licenses
    #             if not licenses.split('----------------------------------------')[0]:
    #                 raise ValueError("No licenses found to embed in license.txt")
    #             license_vectors = vectorizer.fit_transform(licenses)
    #         elif embedding_approach == 'line-embedding':
    #             license_vectors = vectorizer.fit_transform(licenses.split('\n'))
    #         elif embedding_approach == 'license-embedding':
    #             license_vectors = vectorizer.fit_transform(licenses.split('----------------------------------------'))
    #         else:
    #             raise ValueError(f"Invalid embedding approach: {embedding_approach}")

    #         with open(embeddings_file_path, "wb") as f:
    #             pickle.dump({'vectorizer': vectorizer, 'embeddings': license_vectors}, f, protocol=pickle.HIGHEST_PROTOCOL)

    #     # Encode code chunks (BoW)
    #     code_chunks = code_text.split('\n')
    #     code_vectors = vectorizer.transform(code_chunks)
    # elif model == 'fasttext':
    #     embeddings_file_path = os.path.join(embeddings_dir, f"fasttext_model-{embedding_approach}.bin")
    #     if os.path.exists(embeddings_file_path):
    #         print(f"Loading pre-trained FastText model from: {embeddings_file_path}")
    #         fasttext_model = FastText.load(embeddings_file_path)
    #     else:
    #         print(f"Pre-trained FastText model not found. Training and saving to: {embeddings_file_path}")
    #         sentences = [line.strip() for line in licenses.split('\n') if line.strip()]  # Remove empty lines
    #         # Train FastText model with the appropriate sentences
    #         fasttext_model = FastText(sentences, vector_size=100, window=5, min_count=1, workers=4) 
    #         fasttext_model.save(embeddings_file_path)
        
    #     all_words = [word for sentence in sentences for word in sentence.split()]
    #     all_word_embeddings = [fasttext_model.wv.get_vector(word) for word in all_words if word in fasttext_model.wv]
        
    #     # Prepare for the FastText embedding 
    #     if embedding_approach == 'file-embedding':
    #         # Average all word embeddings to get file-level embedding
    #         license_embeddings = np.mean(all_word_embeddings, axis=0).reshape(1, -1)
    #     elif embedding_approach == 'line-embedding':
    #         # Average word embeddings for each line
    #         license_embeddings = []
    #         for sentence in sentences:
    #             words = sentence.split()
    #             line_embedding = np.mean([fasttext_model.wv.get_vector(word) for word in words if word in fasttext_model.wv], axis=0)
    #             license_embeddings.append(line_embedding)

    #     elif embedding_approach == 'license-embedding':
    #         # Average word embeddings within each license section
    #         license_embeddings = []
    #         current_license_embeddings = []
    #         for word in all_words:
    #             current_license_embeddings.append(fasttext_model.wv.get_vector(word))
    #             if word == '----------------------------------------':  # end of license section
    #                 license_embeddings.append(np.mean(current_license_embeddings, axis=0))
    #                 current_license_embeddings = []  # Reset for the next license
    #     else:
    #         pass #TODO: fix
    #     # Encode code chunks (FastText)
    #     code_embeddings = []
    #     code_chunks = code_text.split('\n')
    #     for chunk in code_chunks:
    #         chunk_embeddings = [fasttext_model.wv.get_vector(word) for word in chunk.split()]
    #         if chunk_embeddings:
    #             code_embeddings.append(np.mean(chunk_embeddings, axis=0))
    #         else:
    #             code_embeddings.append(np.zeros(fasttext_model.vector_size))
    # else:
    #     # Use SentenceTransformer for embedding
    #     model = SentenceTransformer(model)
    #     model_name = str(model).split("(")[0].strip()
    #     embeddings_file_path = os.path.join(embeddings_dir, f"{model.get_sentence_embedding_dimension()}_{model_name.replace('/', '_')}-{embedding_approach}.pkl")

    #     if os.path.exists(embeddings_file_path):
    #         print(f"Loading pre-embedded licenses from: {embeddings_file_path}")
    #         with open(embeddings_file_path, "rb") as fIn:
    #             stored_data = pickle.load(fIn)
    #             license_embeddings = stored_data['embeddings']
    #     else:
    #         print(f"Pre-embedded licenses not found. Creating embeddings and saving to: {embeddings_file_path}")
    #         license_embeddings = model.encode(licenses)
    #         if embedding_approach == 'file-embedding':
    #             license_embeddings = model.encode(licenses)
    #             license_embeddings = license_embeddings.reshape(1, -1)
    #         elif embedding_approach == 'line-embedding':
    #             license_embeddings = model.encode(licenses.split('\n'))
    #         elif embedding_approach == 'license-embedding':
    #             license_embeddings = model.encode(licenses.split('----------------------------------------'))
    #         else:
    #             #TODO: add error
    #             pass
    #         with open(embeddings_file_path, "wb") as fOut:
    #             pickle.dump({'embeddings': license_embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)
        
    #     # Encode code chunks (Sentence Transformers)
    #     code_chunks = code_text.split('\n')
    #     code_embeddings = model.encode(code_chunks)
    
    code_chunks = code_text.split('\n')
    code_embeddings = model.encode(code_chunks)
    
    # Calculate cosine similarities between all code chunks and all licenses
    if model == 'tf-idf':
        similarity_matrix = cosine_similarity(code_vectors, license_vectors)
    elif model == 'bow':
        similarity_matrix = cosine_similarity(code_vectors, license_vectors)
    elif model == 'fasttext':
        similarity_matrix = cosine_similarity(code_embeddings, license_embeddings)
    else:
        similarity_matrix = cosine_similarity(code_embeddings, license_embeddings)
    
    if embedding_approach == 'file-embedding' and model != 'bow':
        results = sorted(((index, score, code_chunks[index]) for index, score in enumerate(similarity_matrix)), key=lambda x: x[1], reverse=True)[:top_k]
    elif embedding_approach  == 'file-embedding' and model == 'bow':
        print(similarity_matrix.shape)
        pass
    elif embedding_approach == 'line-embedding':
        results = []
        for i in range(similarity_matrix.shape[0]):
            max_score_index = np.argmax(similarity_matrix[i]) 
            max_score = similarity_matrix[i][max_score_index]  
            results.append((i, max_score, code_chunks[i]))  
        results.sort(key=lambda x: x[1], reverse=True)
    elif embedding_approach == 'license-embedding':
        results = []
        for i in range(similarity_matrix.shape[0]):  
            max_score_index = np.argmax(similarity_matrix[i]) 
            max_score = similarity_matrix[i][max_score_index]  
            match = re.search(r"\bLicense Name: .*\b", licenses.split('----------------------------------------')[max_score_index])
            if match: 
                results.append((i, max_score, code_chunks[i]))
                                # , match.group(0)))
            else:
                results.append((i, max_score, code_chunks[i], 'N/A'))
        results.sort(key=lambda x: x[1], reverse=True)
    else:
        #TODO: add error
        pass

    return results[:top_k]

def get_top_similar_license_lines(
    code_text,
    licenses_file_path,
    license_embeddings = None,
    model = SentenceTransformer("all-mpnet-base-v2"),
    embeddings_dir="extras/license_information/license_embeddings",
    embedding_approach = 'license-embedding',
    top_k=5,
    double_semantic_search = False,
):
    """
    Finds the top-k most similar lines in a code file to the entire license text using either TF-IDF or sentence embeddings.
    
    Args:
        code_text (str): The text content of the code file or comments (if the file extension is supported by nirjas).
        licenses_file_path (str): Path to the file containing license texts.
        embeddings_dir (str, optional): Directory to store/load embeddings. Defaults to "extras/license_information/license_embeddings".
        license_embeddings (multi-dimensional array of floats, optional): Embeddings of the license texts
        model: (SentenceTransformer model, optional): model to be used for embedding, defaults to all-mpnet-base-v2
        threshold (float, optional): Similarity threshold for determining a match. Defaults to 0.8.

    Returns:
        bool: True if license-like text is found, False otherwise.
    """

    os.makedirs(embeddings_dir, exist_ok=True)
    licenses = pd.read_csv(licenses_file_path)

    if model != 'fuzzy':
        model_name = model._first_module().auto_model.config._name_or_path.split('/')[1]
        embeddings_file_path = os.path.join(embeddings_dir, f"{model.get_sentence_embedding_dimension()}_{model_name.replace('/', '_')}-{embedding_approach}.pkl")
        if license_embeddings is None:
            if os.path.exists(embeddings_file_path):
                # print(f"Loading pre-embedded licenses from: {embeddings_file_path}")
                with open(embeddings_file_path, "rb") as fIn:
                    stored_data = pickle.load(fIn)
                    license_embeddings = stored_data['embeddings']
            else:
                print(f"Pre-embedded licenses not found. Creating embeddings and saving to: {embeddings_file_path}")
                if embedding_approach == 'line-embedding':
                    text = '\n'.join(licenses['License Text'])
                    license_embeddings = model.encode(text.split('\n'))
                elif embedding_approach == 'license-embedding':
                    license_embeddings = model.encode(licenses['License Text'])
                else:
                    raise ValueError("embedding approach not recognized. Supported approaches: ['line-embedding', 'license-embedding']")
                
                with open(embeddings_file_path, "wb") as fOut:
                    pickle.dump({'embeddings': license_embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)

        # Encode code chunks (Sentence Transformers)
        code_chunks = code_text.split('\n')
        code_embeddings = model.encode(code_chunks)
        
        similarity_matrix = np.zeros((len(code_embeddings), len(license_embeddings)))

        for i in range(len(code_embeddings)):
            embedding = code_embeddings[i].reshape(1, -1)
            similarity_matrix[i] = cosine_similarity(embedding, license_embeddings)
    
    else:
        similarity_matrix = np.zeros((len(code_embeddings), len(license_embeddings)))

        def calculate_similarity(i, code_chunks, licenses):
            row = np.zeros(len(licenses))
            for j in range(len(licenses)):
                row[j] = fuzz.ratio(code_chunks[i], licenses['License Text'].iloc[j])
            return row
        
        with ThreadPoolExecutor() as executor:
            future_to_row = {executor.submit(calculate_similarity, i, code_chunks, licenses): i for i in range(len(code_embeddings))}

            for future in as_completed(future_to_row):
                i = future_to_row[future]
                try:
                    similarity_matrix[i, :] = future.result()
                except Exception as exc:
                    print(f"Row {i} generated an exception: {exc}")
    
    if embedding_approach == 'line-embedding':
        text = '-----LICENSE_SEPARATOR-----'.join(licenses['License Text'])
        text = re.split(r'\n+', text)
        text = text[1::]
        license_index_map = {}
        current_license_index = 0 
        line_counter = 0
        for line in text:
            if line.strip() == '-----LICENSE_SEPARATOR-----':
                license_index_map[line_counter] = current_license_index
                line_counter += 1
                current_license_index += 1
                continue
            license_index_map[line_counter] = current_license_index
            line_counter += 1
        results = []
        for i in range(similarity_matrix.shape[0]):
            max_score_index = np.argmax(similarity_matrix[i]) 
            max_score = similarity_matrix[i][max_score_index]  

            license_index = license_index_map[max_score_index]

            results.append((i, max_score, code_chunks[i], licenses.loc[license_index, 'License Name'], licenses.loc[license_index, 'License ID']))  
        results.sort(key=lambda x: x[1], reverse=True)
    elif embedding_approach == 'license-embedding':
        results = []
        for i in range(len(code_embeddings)):  
            max_score_index = np.argmax(similarity_matrix[i])
            max_score = similarity_matrix[i][max_score_index]
            results.append((i, max_score, code_chunks[i], licenses.loc[max_score_index, 'License Name'], licenses.loc[max_score_index, 'License ID']))
        results.sort(key=lambda x: x[1], reverse=True)
    else:
        raise ValueError("embedding approach not recognized. Supported approaches: ['line-embedding', 'license-embedding']")

    top_tuples = results[:top_k]

    if double_semantic_search and (embedding_approach != 'line-embedding'):
        text = '\n'.join(licenses['License Text'])
        license_index_map = {}
        line_idx = 0
        for license_index, license in enumerate(licenses['License Text']):
            for line in license.split('\n'):
                license_index_map[line_idx] = license_index
                line_idx += 1  
        similarity_matrix = np.zeros((len(top_tuples), len(text.split('\n'))))
        for index, tuple in enumerate(top_tuples):
            code_text = tuple[2]
            text_lines = text.split('\n')
            for i in range(len(text_lines)):
                similarity_matrix[index][i] = fuzz.ratio(code_text, text_lines[i]) 
            max_score_index = np.argmax(similarity_matrix[index]) 
            license_index = license_index_map[max_score_index]
            top_tuples[index] = (tuple[2], licenses.loc[license_index, 'License Name'], licenses.loc[license_index, 'License ID']) 

    return top_tuples

def extract_comments(df: pd.DataFrame):
        for index, row in df.iterrows():
            try: 
                nirjas_comments = nirjas.extract(os.path.join('extras', row['file path']))
                
                all_comments = []
                
                for key, values in nirjas_comments.items():
                    if "comment" in key:
                        if isinstance(values, list):
                            for entry in values:
                                all_comments.append(entry["comment"])
                        else:
                            all_comments.append(values)

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
            if similarity_ratio >= 40:
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