# -------------------------------------------------
# IMPORTS

# Basic imports
import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import math
import time
import itertools
import re
import os
import requests # type: ignore
import logging

# Tokenization and stopwords
import nltk # type: ignore
from nltk.corpus import stopwords # type: ignore
from nltk.tokenize import word_tokenize # type: ignore

# NER

# Entity linking

# SPARQL queries

# Graphs
import networkx as nx # type: ignore

# Summarization itself
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
from sklearn.metrics.pairwise import cosine_similarity # type: ignore

# Metrics
from bert_score import score as bert_score # type: ignore
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction # type: ignore
from rouge_score import rouge_scorer # type: ignore



# ----------------------------------------------- #
# ------------------ TEXT RANK ------------------ #
# ----------------------------------------------- #

# -------------------------------------------------
# BASIC SETUP

# Supress warnings from Hugging Face Transformers
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
# Suppress transformers library warnings
logging.getLogger("transformers").setLevel(logging.ERROR)

# Function to extract keywords from text
def extract_keywords(sentence):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(sentence, language='english')
    # Filter out stop words and non-alphabetic characters
    keywords = [word for word in words if word.isalpha() and word.lower() not in stop_words]
    return keywords

# Function to search Wikidata for each keyword
def search_wikidata_concepts(keyword):
    url = f"https://www.wikidata.org/w/api.php"
    params = {
        'action': 'wbsearchentities',
        'search': keyword,
        'language': 'en',
        'format': 'json'
    }
    try:
        response = requests.get(url, params=params)
        # Check if rate limiting occurs and add a delay if necessary
        if response.status_code == 429:
            print("Rate limit hit. Waiting for a short period before retrying...")
            time.sleep(1)  # Wait 1 second and then retry
            response = requests.get(url, params=params)
        # If the response is successful, process the results
        if response.status_code == 200:
            results = response.json().get("search", [])
            return [result['id'] for result in results]  # Return Wikidata IDs
        print(f"Error fetching concepts for keyword: {keyword}, Status Code: {response.status_code}")
    except Exception as e:
        print(f"An error occurred: {e}")
    return []

# Function to extract Wikidata concepts for each sentence
def extract_wikidata_concepts(text):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    wikidata_concepts = {}

    for i, sentence in enumerate(sentences):
        keywords = extract_keywords(sentence)
        sentence_concepts = []
        for keyword in keywords:
            concepts = search_wikidata_concepts(keyword)
            sentence_concepts.extend(concepts)
        wikidata_concepts[i] = sentence_concepts
    return wikidata_concepts


# SPAQL query to find relationships between Wikidata entities
def execute_sparql_query(query):
    sparql_endpoint = "https://query.wikidata.org/sparql"
    headers = {'Accept': 'application/sparql-results+json'}
    try:
        response = requests.get(sparql_endpoint, headers=headers, params={'query': query})
        if response.status_code == 200:
            return response.json()  # Returns JSON response with SPARQL results
        else:
            print(f"Error with SPARQL query. Status code: {response.status_code}")
            return None
    except Exception as e:
        print(f"Exception while executing SPARQL query: {e}")
        return None


# Function to fetch the label of a single Wikidata entity
def get_entity_label(entity_id):
    sparql_endpoint = "https://query.wikidata.org/sparql"
    query = f"""SELECT ?label
                WHERE {{ wd:{entity_id} rdfs:label ?label.
                FILTER(LANG(?label) = "en")}}"""
    headers = {'Accept': 'application/sparql-results+json'}
    try:
        response = requests.get(sparql_endpoint, headers=headers, params={'query': query})
        if response.status_code == 200:
            results = response.json().get("results", {}).get("bindings", [])
            if results:
                return results[0]['label']['value']  # Return the label of the entity
            else:
                return "Label not found"
        else:
            print(f"Error fetching label for {entity_id}: {response.status_code}")
            return "Error"
    except Exception as e:
        print(f"Exception while fetching label for {entity_id}: {e}")
        return "Error"


# SPARQL query to find relationships between Wikidata entities, their labels, and the properties used
def query_wikidata_entity_relationships(entity_id):
    # Connection
    sparql_endpoint = "https://query.wikidata.org/sparql"
    # SPARQL query to fetch related entities (P31, P279), their labels (rdfs:label), and the property used
    query = f"""
    SELECT ?relatedEntity ?relatedEntityLabel ?property
    WHERE {{
        wd:{entity_id} ?property ?relatedEntity.
        ?relatedEntity rdfs:label ?relatedEntityLabel.
        FILTER (?property IN (wdt:P31, wdt:P279))  # Only include P31 and P279 properties
        FILTER (LANG(?relatedEntityLabel) = "en")  # Filter to get labels in English
    }}
    """
    
    headers = {'Accept': 'application/sparql-results+json'}
    try:
        response = requests.get(sparql_endpoint, headers=headers, params={'query': query})
        if response.status_code == 200:
            results = response.json().get("results", {}).get("bindings", [])
            if results:
                # Return related entities along with their labels and property used
                return [
                    (result['relatedEntity']['value'].split('/')[-1],
                     result['relatedEntityLabel']['value'],
                     result['property']['value'].split('/')[-1])  # Get the property (P31 or P279)
                    for result in results
                ]
            else:
                print(f"No related entities found for {entity_id}.")
                return []
        else:
            print(f"Error querying SPARQL for {entity_id}: {response.status_code}")
            return []
    except Exception as e:
        print(f"Exception while querying {entity_id}: {e}")
        return []


# Function to fetch the label of a single Wikidata entity
def get_entity_label(entity_id):
    sparql_endpoint = "https://query.wikidata.org/sparql"
    query = f"""
    SELECT ?label WHERE {{
        wd:{entity_id} rdfs:label ?label.
        FILTER(LANG(?label) = "en")  # Filter to get label in English
    }}
    """
    headers = {'Accept': 'application/sparql-results+json'}
    try:
        response = requests.get(sparql_endpoint, headers=headers, params={'query': query})
        if response.status_code == 200:
            results = response.json().get("results", {}).get("bindings", [])
            if results:
                return results[0]['label']['value']  # Return the label of the entity
            else:
                return "Label not found"
        else:
            print(f"Error fetching label for {entity_id}: {response.status_code}")
            return "Error"
    except Exception as e:
        print(f"Exception while fetching label for {entity_id}: {e}")
        return "Error"


# Function to query relationships for multiple Wikidata entities and their labels
def query_multiple_wikidata_entity_relationships(entity_ids):
    results = {}
    for entity_id in entity_ids:
        # Get the main entity's label
        entity_label = get_entity_label(entity_id)
        
        related_entities = query_wikidata_entity_relationships(entity_id)
        if related_entities:
            # Store results if there are related entities
            results[entity_id] = related_entities
            print(f"Entity {entity_id} ({entity_label}) has related entities:")
            for related_entity_id, related_entity_label, property_id in related_entities:
                if property_id == 'P31':
                    print(f"  - {related_entity_id}: {related_entity_label} - instance of ({property_id})")
                elif property_id == 'P279':
                    print(f"  - {related_entity_id}: {related_entity_label} - subclass of ({property_id})")
        else:
            # If no related entities, still print the entity and mention no relations
            print(f"Entity {entity_id} ({entity_label}) has no related entities.")
        # Fixed delay to avoid rate limiting
        time.sleep(1)  # Wait for 1 second before the next request
    return results


# ------------------------------------------------
# SUMMARIZE BY SENTENCE/WORD IMPORTANCE + KG CONCEPTS

# Summarization by "KG weighting" when it is used
def summ_1(text,
           num_sentences,
           percentage_txt,
           counting,
           kind_of_summarization,
           add_wikidata_information,
           wikidata_weight_factor):
    
    start_time = time.time()

    # Split text into sentences
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    total_sentences = len(sentences)

    # Calculate the number of words per sentence
    word_counts_per_sentence = [len(sentence.split()) for sentence in sentences]

    # Define the size of the summary
    if counting == 'num_sentences':
        num_sentences = min(num_sentences, total_sentences)
    elif counting == 'percentage_txt':
        num_sentences = math.ceil(total_sentences * percentage_txt)
        num_sentences = max(1, min(num_sentences, total_sentences))

    # Generate TF-IDF matrix for sentence similarity
    tfidf_vectorizer = TfidfVectorizer()
    tfidf = tfidf_vectorizer.fit_transform(sentences)
    similarity_matrix = cosine_similarity(tfidf)

    # Create a graph from the similarity matrix and apply TextRank
    nx_graph = nx.from_numpy_array(similarity_matrix)
    sentence_scores = nx.pagerank(nx_graph)

    # Initialize Wikidata importance dictionary
    wikidata_importance = {}
    sentence_concepts = {}

    # Only calculate Wikidata scores if requested
    if add_wikidata_information == 'yes':

        # Ensure nltk packages are downloaded
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('all', quiet=True)

        # Extract Wikidata concepts for each sentence
        sentence_concepts = extract_wikidata_concepts(text)

        print(sentence_concepts)

        # Print out the extracted concepts for each sentence
        print("Extracted Wikidata concepts for each sentence:")
        for i, concepts in sentence_concepts.items():
            print(f"Sentence {i}: {concepts}")

        # Collect all concepts across all sentences
        all_concepts = list(itertools.chain.from_iterable(sentence_concepts.values()))

        # Query relationships for all unique concepts
        concept_relationships = query_multiple_wikidata_entity_relationships(set(all_concepts))

        # Print relationships per sentence
        for entity_id, related_entities_list in concept_relationships.items():
            print(f"Entity {entity_id} has related entities:")
            for related_entity_id, related_entity_label in related_entities_list:
                print(f"  - {related_entity_id}: {related_entity_label}")




        # Calculate importance based on relationships found
        for i, concepts in sentence_concepts.items():
            if concepts:
                # Only use the first found relationship for simplicity, can be adjusted
                wikidata_score = sum(len(concept_relationships[concept]) for concept in concepts if concept in concept_relationships)
                wikidata_importance[i] = wikidata_score * wikidata_weight_factor

    # Normalize sentence_scores and wikidata_importance if KG info is included
    if add_wikidata_information == 'yes':
        max_sent_score = max(sentence_scores.values())
        min_sent_score = min(sentence_scores.values())
        norm_sent_scores = {i: (score - min_sent_score) / (max_sent_score - min_sent_score)
                                      for i, score in sentence_scores.items()}
        
        max_wiki_score = max(wikidata_importance.values(), default=1)  # Avoid division by zero
        min_wiki_score = min(wikidata_importance.values(), default=0)
        norm_wiki_scores = {i: (score - min_wiki_score) / (max_wiki_score - min_wiki_score)
                                      for i, score in wikidata_importance.items()}
    else:
        norm_sent_scores = sentence_scores
        norm_wiki_scores = {}

    # Combine sentence scores with Wikidata and word importance
    ranked_sentences = []
    for i in range(total_sentences):
        total_score = norm_sent_scores[i]  # Start with normalized sentence score

        # Add normalized Wikidata importance if included
        if add_wikidata_information == 'yes':
            kg_score = norm_wiki_scores.get(i, 0)
            total_score += kg_score * wikidata_weight_factor  # Amplify KG impact

        # If including word importance (TF-IDF)
        if kind_of_summarization == 'sentences_n_words':
            word_score = tfidf[i].sum()  # Sum TF-IDF scores for words in this sentence
            total_score += word_score * 0.5  # Apply a smaller weight to TF-IDF word importance

        ranked_sentences.append((total_score, sentences[i]))

    # Sort sentences based on combined score
    ranked_sentences = sorted(ranked_sentences, reverse=True)

    # Select top sentences for the summary, ensuring no extra periods
    summary = " ".join([ranked_sentences[i][1].strip() for i in range(num_sentences)])

    final_time = time.time()
    
    return summary, (final_time - start_time), wikidata_importance, word_counts_per_sentence, sentence_concepts


# ------------------------------------------------
# SELECT PARAMETERS

# Text example
text = """Artificial Intelligence (AI) is the field of study that focuses on creating systems capable of performing tasks that typically require human intelligence. These tasks include learning, reasoning, problem solving, perception, and language understanding. AI draws on concepts from several disciplines including computer science, mathematics, linguistics, psychology, and neuroscience to develop algorithms and models that mimic cognitive functions.

AI has evolved significantly since its inception in the mid-20th century. Early AI research aimed at creating machines that could perform narrowly defined tasks, such as playing chess or solving algebraic problems. These early systems were rule-based and relied heavily on pre-programmed knowledge. However, modern AI systems, particularly those using machine learning, are capable of improving their performance over time by learning from data without explicit programming.

One of the most widely used approaches to AI is machine learning (ML), a subset of AI that uses statistical methods to enable machines to improve at tasks through experience. Within ML, deep learning has gained prominence due to its ability to process vast amounts of data and model complex patterns. Deep learning models are particularly effective in fields such as computer vision, natural language processing, and speech recognition.

The application of AI in various industries has brought about transformative changes. In healthcare, AI-powered systems are used to assist in diagnostics, predict patient outcomes, and personalize treatment plans. Financial institutions employ AI to detect fraudulent activities, automate trading, and enhance customer service through chatbots. The automotive industry is exploring AI for autonomous vehicles, where systems can interpret data from sensors to navigate roads safely.

Despite its advantages, AI poses several ethical and societal challenges. Concerns regarding job displacement, privacy, and algorithmic bias are at the forefront of discussions around AI. Additionally, the development of artificial general intelligence (AGI)—machines capable of performing any intellectual task that humans can—raises questions about control, safety, and the potential impact on society.

AI continues to evolve, pushing the boundaries of what machines can achieve. Researchers are working on making AI systems more robust, explainable, and aligned with human values to ensure that AI benefits society as a whole."""

# Define all possible configurations for each parameter
add_kg_info_options = {0: 'no', 1: 'yes'}
wiki_usage_options = {0: 'weight'}
wiki_weight = 2.5
kind_summ_options = {0: 'just_sentences'}
counting_options = {0: 'num_sentences'}

# Values of interest for the results
num_sentences = 3
percentage_txt = 0.2

# Iterate over all possible combinations of options
for add_kg_info, wiki_usage, kind_summ, counting in itertools.product(
    add_kg_info_options.values(),
    wiki_usage_options.values(),
    kind_summ_options.values(),
    counting_options.values()
):
    # Check if we need to add KG info and download resources if required
    if add_kg_info == 'yes':
        t1 = time.time()
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('all', quiet=True)
        t2 = time.time()
    else:
        t1, t2 = 0, 0

    # Generate the summary based on current configuration
    if wiki_usage == 'weight':
        summary, t, c1, c2, data = summ_1(text, num_sentences, percentage_txt, counting, kind_summ, add_kg_info, wiki_weight)
    else:
        pass



from collections import Counter
from itertools import islice

# Create a dictionary to store sorted counts for each sentence
sorted_counts = {}

# Count and sort each list of codes by frequency for each sentence
for key, codes in data.items():
    # Count occurrences of each code
    code_counts = Counter(codes)
    # Sort by frequency (highest first), then alphabetically if frequencies match
    sorted_code_counts = sorted(code_counts.items(), key=lambda x: (-x[1], x[0]))
    # Store sorted result in dictionary
    sorted_counts[key] = sorted_code_counts

# Number of words per sentence:
print("Number of words per sentence")
print(c2)
print("\n")

# Print the sorted counts
for sentence, counts in list(islice(sorted_counts.items(), 3)):
    print(f"Sentence {sentence}: {counts}")
