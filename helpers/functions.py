from sentence_transformers import SentenceTransformer, util
import os
import json
import pickle
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import FastText  # Import the FastText class


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
        license_dataset_file_path = os.path.join(parent_dir, "license_dataset.txt")
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

    # Write the license dataset to the output file
    with open(license_dataset_file_path, 'w', encoding='utf-8') as outfile:
        for license_data in license_dataset_file_data:
            outfile.write(f"License Name: {license_data['licenseName']}\n\n")
            outfile.write(f"License ID: {license_data['licenseId']}\n\n")
            outfile.write(f"{license_data['licenseText']}\n\n")
            outfile.write("-" * 40 + "\n\n")  # Add separator

    print(f'License dataset file created successfully at {license_dataset_file_path}')

def semantic_search(text, query, threshold = 0.5):
    """
    Performs semantic search on a text input using a pre-trained Sentence Transformer model.

    Args:
        text (str or list): The text to search within. Can be a single string or a list of strings.
        query (str): The search query.
        threshold (float, optional): The minimum similarity score for a match. Defaults to 0.5.

    Returns:
        list: A list of tuples containing (matched_text, similarity_score) for matches above the threshold.
            If no matches are found, returns an empty list.
    """

    model = SentenceTransformer('all-mpnet-base-v2')

    # Handle both single string and list of strings input
    if isinstance(text, str):
        text = [text]

    # Embed text and query
    text_embeddings = model.encode(text)
    query_embedding = model.encode(query)

    # Calculate cosine similarities
    similarities = [util.cos_sim(query_embedding, text_embedding)[0][0] for text_embedding in text_embeddings]

    # Filter based on threshold and create result tuples
    results = [(text[i], sim) for i, sim in enumerate(similarities) if sim >= threshold]

    # Sort results by similarity (optional)
    results.sort(key=lambda x: x[1], reverse=True)

    return results

def contains_license_text(code_text, licenses_file_path, model="all-mpnet-base-v2", embeddings_dir="extras/license_information/license_embeddings"):
    """
    Checks if a code file contains text similar to any license in a given file.

    Args:
        code_text (str): The text content of the code file.
        licenses_file_path (str): Path to the file containing license texts.
        embeddings_dir (str, optional): Directory to store/load embeddings. Defaults to "extras/license_information/license_embeddings".
        threshold (float, optional): Similarity threshold for determining a match. Defaults to 0.8.

    Returns:
        bool: True if license-like text is found, False otherwise.
    """
    if model != 'tf-idf':
        model = SentenceTransformer(model)
        # Determine embeddings file path based on model name
        # Create embeddings directory if it doesn't exist
        os.makedirs(embeddings_dir, exist_ok=True)
        model_name = str(model).split('(')[0].strip()  # Get model name from string representation
        embeddings_file_path = os.path.join(embeddings_dir, f"{model.get_sentence_embedding_dimension()}_{model_name.replace('/', '_')}.pkl")
    else:
        # Create embeddings directory if it doesn't exist
        os.makedirs(embeddings_dir, exist_ok=True)
        embeddings_file_path = os.path.join(embeddings_dir, f'{model}.pkl')

    # Load or create embeddings
    if os.path.exists(embeddings_file_path):
        print(f"Loading pre-embedded licenses from: {embeddings_file_path}")
        if model == 'tf-idf':
            tfidf_data = pickle.load(f)
            vectorizer = tfidf_data['vectorizer']
            license_embeddings = stored_data['embeddings']
        else:   
            with open(embeddings_file_path, "rb") as fIn:
                stored_data = pickle.load(fIn)
                license_embeddings = stored_data['embeddings']
    else:
        print(f"Pre-embedded licenses not found. Creating embeddings and saving to: {embeddings_file_path}")
        if model == 'tf-idf':
            dataset_path = os.path.join(licenses_file_path, "license_dataset.txt")
            with open(dataset_path, 'r', encoding='utf-8') as file:
                licenses = file.readlines()
            licenses = licenses.split('----------------------------------------')
            vectorizer = TfidfVectorizer()
            vectorizer.fit(licenses)  # Fit the vectorizer to the licenses
            license_vectors = vectorizer.transform(licenses)
            with open(embeddings_file_path, "wb") as f:
                pickle.dump({'vectorizer': vectorizer, 'embeddings':license_vectors}, f, protocol=pickle.HIGHEST_PROTOCOL)  
        else:
            with open(licenses_file_path, 'r', encoding='utf-8') as file:
                licenses = file.readlines()
            license_embeddings = model.encode(licenses)
            with open(embeddings_file_path, "wb") as fOut:
                pickle.dump({'embeddings': license_embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)

    # Encode code chunks 
    code_chunks = code_text.split('\n')
    code_embeddings = vectorizer.transform(code_chunks) if model == 'tf-idf' else model.encode(code_chunks)
    similarities = []
    # Compare each code chunk directly to the license embedding
    for chunk_embedding in code_embeddings:
        similarity = util.cos_sim(chunk_embedding, license_embeddings)[0][0]  # Cosine similarity
        similarities.append(similarity)
    #     if similarity >= threshold:
    #         return True  # Found license-like text
    return similarities
    # return False  # No license-like text found

def get_top_similar_license_lines(
    code_text,
    licenses_file_path,
    model="all-mpnet-base-v2",
    embeddings_dir="extras/license_information/license_embeddings",
    embedding_approach = 'file-embedding',
    top_k=3
):
    """
    Finds the top-k most similar lines in a code file to the entire license text using either TF-IDF or sentence embeddings.
    """

    os.makedirs(embeddings_dir, exist_ok=True)
    with open(licenses_file_path, 'r', encoding='utf-8') as file:
                licenses = file.read()

    if model == 'tf-idf':
        embeddings_file_path = os.path.join(embeddings_dir, f"tfidf_vectorizer-{embedding_approach}.pkl")
        if os.path.exists(embeddings_file_path):
            print(f"Loading pre-trained TF-IDF vectorizer from: {embeddings_file_path}")
            with open(embeddings_file_path, "rb") as f:
                tfidf_data = pickle.load(f)
                vectorizer = tfidf_data['vectorizer']
                license_vectors = tfidf_data['embeddings']
        else:
            print(f"Pre-trained TF-IDF vectorizer not found. Training and saving to: {embeddings_file_path}")
            vectorizer = TfidfVectorizer()
            vectorizer.fit(licenses.split('----------------------------------------'))
            if embedding_approach == 'file-embedding':
                license_vectors = vectorizer.transform(licenses) #TODO: Fix error in this case. add a acheck
                # license_vectors = license_vectors.reshape(1, -1)
            elif embedding_approach == 'line-embedding':
                license_vectors = vectorizer.transform(licenses.split('\n'))
            elif embedding_approach == 'license-embedding':
                license_vectors = vectorizer.transform(licenses.split('----------------------------------------'))
            else:
                # TODO: add error
                pass
            with open(embeddings_file_path, "wb") as f:
                pickle.dump({'vectorizer': vectorizer, 'embeddings': license_vectors}, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Encode code chunks (TF-IDF)
        code_chunks = code_text.split('\n')
        code_vectors = vectorizer.transform(code_chunks)
    elif model == 'bow':
        embeddings_file_path = os.path.join(embeddings_dir, f"bow_vectorizer-{embedding_approach}.pkl")
        if os.path.exists(embeddings_file_path):
            print(f"Loading pre-trained BoW vectorizer from: {embeddings_file_path}")
            with open(embeddings_file_path, "rb") as f:
                bow_data = pickle.load(f)
                vectorizer = bow_data['vectorizer']
                license_vectors = bow_data['embeddings']
        else:
            print(f"Pre-trained BoW vectorizer not found. Training and saving to: {embeddings_file_path}")
            vectorizer = CountVectorizer()  # Use CountVectorizer for BoW

            if embedding_approach == 'file-embedding':
                # Check for empty licenses
                if not licenses.split('----------------------------------------')[0]:
                    raise ValueError("No licenses found to embed in license.txt")
                license_vectors = vectorizer.fit_transform(licenses)
            elif embedding_approach == 'line-embedding':
                license_vectors = vectorizer.fit_transform(licenses.split('\n'))
            elif embedding_approach == 'license-embedding':
                license_vectors = vectorizer.fit_transform(licenses.split('----------------------------------------'))
            else:
                raise ValueError(f"Invalid embedding approach: {embedding_approach}")

            with open(embeddings_file_path, "wb") as f:
                pickle.dump({'vectorizer': vectorizer, 'embeddings': license_vectors}, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Encode code chunks (BoW)
        code_chunks = code_text.split('\n')
        code_vectors = vectorizer.transform(code_chunks)
    elif model == 'fasttext':
        embeddings_file_path = os.path.join(embeddings_dir, f"fasttext_model-{embedding_approach}.bin")
        if os.path.exists(embeddings_file_path):
            print(f"Loading pre-trained FastText model from: {embeddings_file_path}")
            fasttext_model = FastText.load(embeddings_file_path)
        else:
            print(f"Pre-trained FastText model not found. Training and saving to: {embeddings_file_path}")
            sentences = [line.strip() for line in licenses.split('\n') if line.strip()]  # Remove empty lines
            # Train FastText model with the appropriate sentences
            fasttext_model = FastText(sentences, vector_size=100, window=5, min_count=1, workers=4) 
            fasttext_model.save(embeddings_file_path)
        
        all_words = [word for sentence in sentences for word in sentence.split()]
        all_word_embeddings = [fasttext_model.wv.get_vector(word) for word in all_words if word in fasttext_model.wv]
        
        # Prepare for the FastText embedding 
        if embedding_approach == 'file-embedding':
            # Average all word embeddings to get file-level embedding
            license_embeddings = np.mean(all_word_embeddings, axis=0).reshape(1, -1)
        elif embedding_approach == 'line-embedding':
            # Average word embeddings for each line
            license_embeddings = []
            for sentence in sentences:
                words = sentence.split()
                line_embedding = np.mean([fasttext_model.wv.get_vector(word) for word in words if word in fasttext_model.wv], axis=0)
                license_embeddings.append(line_embedding)

        elif embedding_approach == 'license-embedding':
            # Average word embeddings within each license section
            license_embeddings = []
            current_license_embeddings = []
            for word in all_words:
                current_license_embeddings.append(fasttext_model.wv.get_vector(word))
                if word == '----------------------------------------':  # end of license section
                    license_embeddings.append(np.mean(current_license_embeddings, axis=0))
                    current_license_embeddings = []  # Reset for the next license
        else:
            pass #TODO: fix
        # Encode code chunks (FastText)
        code_embeddings = []
        code_chunks = code_text.split('\n')
        for chunk in code_chunks:
            chunk_embeddings = [fasttext_model.wv.get_vector(word) for word in chunk.split()]
            if chunk_embeddings:
                code_embeddings.append(np.mean(chunk_embeddings, axis=0))
            else:
                code_embeddings.append(np.zeros(fasttext_model.vector_size))
    else:
        # Use SentenceTransformer for embedding
        model = SentenceTransformer(model)
        model_name = str(model).split("(")[0].strip()
        embeddings_file_path = os.path.join(embeddings_dir, f"{model.get_sentence_embedding_dimension()}_{model_name.replace('/', '_')}-{embedding_approach}.pkl")

        if os.path.exists(embeddings_file_path):
            print(f"Loading pre-embedded licenses from: {embeddings_file_path}")
            with open(embeddings_file_path, "rb") as fIn:
                stored_data = pickle.load(fIn)
                license_embeddings = stored_data['embeddings']
        else:
            print(f"Pre-embedded licenses not found. Creating embeddings and saving to: {embeddings_file_path}")
            license_embeddings = model.encode(licenses)
            if embedding_approach == 'file-embedding':
                license_embeddings = model.encode(licenses)
                license_embeddings = license_embeddings.reshape(1, -1)
            elif embedding_approach == 'line-embedding':
                license_embeddings = model.encode(licenses.split('\n'))
            elif embedding_approach == 'license-embedding':
                license_embeddings = model.encode(licenses.split('----------------------------------------'))
            else:
                #TODO: add error
                pass
            with open(embeddings_file_path, "wb") as fOut:
                pickle.dump({'embeddings': license_embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Encode code chunks (Sentence Transformers)
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
                results.append((i, max_score, code_chunks[i], match.group(0)))
            else:
                results.append((i, max_score, code_chunks[i], 'N/A'))
        results.sort(key=lambda x: x[1], reverse=True)
    else:
        #TODO: add error
        pass

    return results[:top_k]
