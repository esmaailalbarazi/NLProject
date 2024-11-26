# Basic imports
import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import math
import time
import itertools
from collections import Counter
import re
import os
import sys
import io
import requests # type: ignore
import logging

# Tokenization and stopwords
import nltk # type: ignore
from nltk.corpus import stopwords, wordnet # type: ignore
from nltk.tokenize import sent_tokenize, word_tokenize # type: ignore
from nltk.util import ngrams # type: ignore

# Entity linking
# SPARQL queries
# Build triples

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
# Set UTF-8 as the default encoding for standard output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# Function to extract keywords from text
def extract_keywords(sentence,
                     n=2):
    # Tokenize the sentence into words
    words = word_tokenize(sentence, language='english')
    # Remove stop words and non-alphabetic characters
    stop_words = set(stopwords.words('english'))
    filtered_words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
    # Generate n-grams (bigrams, trigrams, etc.)
    n_grams = ngrams(filtered_words, n)
    # Join n-grams (e.g. ("artificial", "intelligence") -> "artificial intelligence")
    n_gram_strings = [' '.join(gram) for gram in n_grams]
    # Create a list of valid entities: check if the n-gram appears in the original text
    valid_entities = []
    for n_gram in n_gram_strings:
        # Check if the n-gram appears in the original sentence (case insensitive)
        if n_gram in sentence.lower():
            valid_entities.append(n_gram)
    # Add individual words to the list of valid entities as well
    valid_entities += [word.lower() for word in filtered_words]
    print(valid_entities)
    return valid_entities


# Function to search Wikidata for each keyword
def search_wikidata_concepts(keyword,
                             max_retries=5):
    url = f"https://www.wikidata.org/w/api.php"
    params = {'action': 'wbsearchentities',
              'search': keyword,
              'language': 'en',
              'format': 'json'}
    retries = 0
    while retries < max_retries:
        try:
            response = requests.get(url, params=params)
            if response.status_code == 429:
                wait_time = 2 ** retries  # Exponential backoff: 1s, 2s, 4s, 8s, etc.
                print(f"Rate limit hit. Waiting for {wait_time} seconds before retrying...")
                time.sleep(wait_time)
                retries += 1
                continue
            if response.status_code == 200:
                results = response.json().get("search", [])
                return [result['id'] for result in results]  # Return Wikidata IDs
            print(f"Error fetching concepts for keyword: {keyword}, Status Code: {response.status_code}")
            return []
        except Exception as e:
            print(f"An error occurred: {e}")
            retries += 1
            time.sleep(3 ** retries)
    print(f"Failed to fetch concepts for keyword '{keyword}' after {max_retries} retries.")
    return []


# Function to extract Wikidata main entities for each sentence
def wikidata_main_entities(text):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    wikidata_selected_concepts = {}
    for i, sentence in enumerate(sentences):
        # Extract keywords from the sentence
        keywords = extract_keywords(sentence)
        # Initialize a set to store unique IDs for this sentence
        unique_concept_ids = set()
        for keyword in keywords:
            ids = search_wikidata_concepts(keyword)  # Get IDs for the keyword
            sorted_ids = sorted(ids, key=lambda x: int(x[1:])) if ids else []  # Sort by the number after 'Q'
            
            if sorted_ids:  # Check if sorted_ids is not empty
                first_id = sorted_ids[0]
                print(f"{keyword}: '{first_id}'")  # Print only the first ID
                unique_concept_ids.add(first_id)  # Add only the first ID to the set
        # Convert the set to a list to remove duplicates and store it
        wikidata_selected_concepts[i] = list(unique_concept_ids)
    print('\nWikidata_selected_concepts per sentence:')
    print(wikidata_selected_concepts)
    return wikidata_selected_concepts


# Function to filter main entities from "wikidata_main_entities"
def filter_wikidata_main_entities(concepts):
    # Terms to filter out in descriptions
    unwanted_terms = {"scientific article", "journal"}
    # Default description to filter out
    no_description_text = "No description defined"
    # Initialize an empty dictionary to store filtered concepts
    filtered_concepts = {}
    # Iterate through the concepts and their corresponding entity IDs
    for sentence_idx, entity_ids in concepts.items():
        # Initialize a list to store valid entity IDs for this sentence
        valid_entity_ids = []
        for entity_id in entity_ids:
            # Fetch the description of the entity
            description = get_entity_description(entity_id)
            # Check if the description is missing (None, empty, or default "No description defined")
            if description and description != no_description_text and not any(term in description.lower() for term in unwanted_terms):
                # If valid, add the entity_id to the list
                valid_entity_ids.append(entity_id)
        # Only add the sentence's valid entities if there are any
        if valid_entity_ids:
            filtered_concepts[sentence_idx] = valid_entity_ids
    print('\nWikidata_filtered_concepts and their descriptions:')
    print(filtered_concepts)
    return filtered_concepts


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


# Function to get relationships of P31, P279 and P361 for an entity
def query_wikidata_entity_relationships(entity_id):
    # Terms to filter out
    unwanted_terms = {"scientific", "article", "scholarly", "magazine", "academic major"}
    # Define the SPARQL query
    query = f"""SELECT ?relatedEntity ?relatedEntityLabel ?property
                WHERE {{wd:{entity_id} ?property ?relatedEntity. ?relatedEntity rdfs:label ?relatedEntityLabel.
                FILTER (?property IN (wdt:P31, wdt:P279, wdt:P361))
                FILTER (LANG(?relatedEntityLabel) = "en")}}"""
    # Execute the SPARQL query
    results = execute_sparql_query(query)
    # Check if results were returned
    if results:
        # Process each result
        filtered_results = []
        for result in results["results"]["bindings"]:
            related_entity_label = result['relatedEntityLabel']['value'].lower()
            # Check if any unwanted term is in the label
            if not any(term in related_entity_label for term in unwanted_terms):
                filtered_results.append((
                    result['relatedEntity']['value'].split('/')[-1],  # Extract the entity ID
                    related_entity_label,
                    result['property']['value'].split('/')[-1]  # Extract property (P31, P279 or P361)
                ))
        return filtered_results
    else:
        return []


# Function to fetch the label of a single Wikidata entity
def get_entity_label(entity_id):
    query = f"""SELECT ?label ?description
                WHERE {{wd:{entity_id} rdfs:label ?label; schema:description ?description.
                FILTER(LANG(?label) = "en")
                FILTER(LANG(?description) = "en")}}"""
    results = execute_sparql_query(query)
    if results and results["results"]["bindings"]:
        label = results["results"]["bindings"][0]['label']['value']
        # Check if the label is a number (including integers or floats)
        if label.isdigit() or (label.replace('.', '', 1).isdigit() and label.count('.') == 1):
            return None  # Return None if it's a number
        return label
    return "Label not found"


# Function to fetch the description of a single Wikidata entity
def get_entity_description(entity_id):
    query = f"""SELECT ?description
                WHERE {{wd:{entity_id} schema:description ?description.
                FILTER(LANG(?description) = "en")}}"""
    results = execute_sparql_query(query)    
    if results and results["results"]["bindings"]:
        # Return the description
        return results["results"]["bindings"][0]['description']['value']
    return "Description not found"


# Query all
def query_multiple_wikidata_entity_relationships(entity_ids,
                                                 sentence_idx):
    print(f"\nSentence {sentence_idx}: Main entities + related entities and properties")
    results = {}
    for entity_id in entity_ids:
        entity_label = get_entity_label(entity_id)
        related_entities = query_wikidata_entity_relationships(entity_id)
        if related_entities:
            results[entity_id] = related_entities
            print(f"Entity {entity_id} ({entity_label}) has related entities:")
            for related_entity_id, related_entity_label, property_id in related_entities:
                # Determine the type of relationship
                if property_id == 'P31':
                    relation = "instance of"
                elif property_id == 'P279':
                    relation = "subclass of"
                elif property_id == 'P361':
                    relation = "part of"
                else:
                    relation = "related by unknown property"  # Default fallback
                # Print the related entity and its relation type
                print(f"  - {related_entity_id}: {related_entity_label} - {relation} ({property_id})")
        else:
            print(f"Entity {entity_id} ({entity_label}) has no related entities.")
        time.sleep(3)  # Wait longer to avoid rate limiting across many entities
    print("\nResults:")
    print(results)
    return results


# Function to compile main entities, related entities and their relation
def wikidata_entities_relations(wikidata_filtered_concepts):
    # Initialize dictionary to hold the final related concepts
    wikidata_concepts_with_relations = {}
    # Loop through each sentence and its list of entities
    for sentence_idx, concept_ids in wikidata_filtered_concepts.items():  # Now we pass the filtered concepts
        unique_concept_ids = set(concept_ids)  # Ensure unique IDs for each sentence
        sentence_relationships = query_multiple_wikidata_entity_relationships(unique_concept_ids, sentence_idx)
        # Store relationships for each sentence
        wikidata_concepts_with_relations[sentence_idx] = sentence_relationships
    print("\nFinal Wikidata concepts with relationships:")
    print(wikidata_concepts_with_relations)
    return wikidata_concepts_with_relations


# Function to compile all Wikidata entities (main entities + related entities)
def wikidata_all_entities(data):
    all_entities = []
    # Iterate over each batch
    for batch in data.values():
        # Iterate over each entity in the batch
        for entity, relationships in batch.items():
            # Add the main entity key to the list
            all_entities.append(entity)
            # Add the first element of each tuple in the relationships
            for relationship in relationships:
                first_element = relationship[0]
                all_entities.append(first_element)
    # Sort the entities by their ID number
    sorted_entities = sorted(all_entities, key=lambda x: int(x[1:]))
    print("\n", f"All entities:, {all_entities}")
    print("\n", f"All entities (sorted):, {sorted_entities}")
    return sorted_entities


# Method to query and store in triples
def query_to_generate_triples(concepts):
    print("\nExtracting triples from filtered concepts")
    triples_per_sentence = {}
    all_triples = []
    # Iterate over each sentence and its list of entities
    for sentence_idx, entity_ids in concepts.items():
        print(f"\nSentence {sentence_idx}: Processing entities")
        sentence_triples = []
        for entity_id in entity_ids:
            entity_label = get_entity_label(entity_id)
            print("\nENTITY_LABEL: ", entity_label)

            # Check if entity_label exceeds the word limit
            if len(entity_label.split()) > 10:
                print(f"Skipped entity {entity_id} ({entity_label}) - Label is too long")
                continue  # Skip to the next entity_id

            related_entities = query_wikidata_entity_relationships(entity_id)
            if related_entities:
                for related_entity_id, related_entity_label, property_id in related_entities:                    
                    # Determine the type of relationship
                    if property_id == 'P31':
                        relation = "instance of"
                    elif property_id == 'P279':
                        relation = "subclass of"
                    elif property_id == 'P361':
                        relation = "part of"
                    else:
                        relation = f"unknown relation ({property_id})"
                    # Create the triple and add to both sentence-specific and global lists
                    triple = (entity_label, relation, related_entity_label)
                    sentence_triples.append(triple)
                    all_triples.append(triple)
                    # Print for verification
                    print(f"  - ({entity_label}, {relation}, {related_entity_label})")
            else:
                print(f"Entity {entity_id} ({entity_label}) has no related entities.")
            time.sleep(3)  # Respect API rate limits
        # Add the sentence's triples to the dictionary
        triples_per_sentence[sentence_idx] = sentence_triples
    print("\nGenerated Sentence-to-Triples Dictionary:")
    for sentence_idx, triples in triples_per_sentence.items():
        print(f"Sentence {sentence_idx}: {triples}")
    print("\nGenerated All Triples List:")
    for triple in all_triples:
        print(triple)
    return triples_per_sentence, all_triples


# Function that adds semantic enrichment to the existing triples to form a semantic layer
def semantic_enrichment(triples_per_sentence):
    # Initialize enriched triples structures
    enriched_triples_per_sentence = {}
    all_enriched_triples = []
    # Track all added triples to avoid duplicates
    seen_triples = set()
    # Iterate over sentences and their triples
    for sentence_idx, sentence_triples in triples_per_sentence.items():
        enriched_triples = []  # Store enriched triples for the current sentence
        # Iterate over each triple in the sentence
        for subject, predicate, obj in sentence_triples:
            # Query for additional semantic relationships
            for relation in ["P279", "P361", "P31"]:  # Subclass, part of, instance of
                new_relations = query_wikidata_entity_relationships(subject)
                subject_label = get_entity_label(subject)
                # Filter out entities with labels containing 4 or more words
                if len(subject_label.split()) >= 4:
                    print(f"Skipping {subject_label} - Label contains 4 or more words")
                    continue
                # Check if the subject has any new relations
                for related_entity_id, related_entity_label, property_id in new_relations:
                    # Create a new triple
                    new_triple = (subject, relation, related_entity_label)
                    # Avoid duplicate triples
                    if new_triple not in seen_triples:
                        seen_triples.add(new_triple)  # Mark as seen
                        enriched_triples.append(new_triple)
                        all_enriched_triples.append(new_triple)
                        print(f"Added semantic relationship: {new_triple}")
            # Use WordNet for synonyms or hypernyms
            synsets = wordnet.synsets(obj)
            for synset in synsets:
                synonyms = synset.lemma_names()
                for synonym in synonyms:
                    # Create a new synonym triple
                    new_triple = (subject, "synonym", synonym)
                    # Avoid duplicate synonym triples
                    if new_triple not in seen_triples:
                        seen_triples.add(new_triple)  # Mark as seen
                        enriched_triples.append(new_triple)
                        all_enriched_triples.append(new_triple)
                        print(f"Added synonym relationship: {new_triple}")
            # Delay to respect API limits
            time.sleep(3)
        # Save enriched triples for the current sentence
        enriched_triples_per_sentence[sentence_idx] = enriched_triples
    # Logging the enriched triples
    print("\nEnriched Triples per Sentence:")
    for sentence_idx, triples in enriched_triples_per_sentence.items():
        print(f"Sentence {sentence_idx}: {triples}")
    print("\nAll Enriched Triples:")
    for triple in all_enriched_triples:
        print(triple)
    return enriched_triples_per_sentence, all_enriched_triples


# Function to do each step of TextRank
def textrank_algorithm(text,
                       summary_type,
                       num_sentences,
                       percentage_txt):
    # Initialize stop words
    stop_words = set(stopwords.words("english"))
    
    ### Step 1: Preprocess the text
    original_sentences = sent_tokenize(text)
    clean_sentences = []
    filtered_words = []
    for sentence in original_sentences:
        words = word_tokenize(sentence.lower())
        filtered = [word for word in words if word.isalnum() and word not in stop_words]
        clean_sentence = " ".join(filtered)
        clean_sentences.append(clean_sentence)
        filtered_words.extend(filtered)
    
    ### Step 2: Calculate sentence similarity using TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(clean_sentences)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    ### Step 3: Apply the PageRank algorithm
    def pagerank(similarity_matrix, damping=0.85, max_iter=100, tol=1e-6):
        n = similarity_matrix.shape[0]
        rank = np.ones(n) / n  # Initial rank
        for _ in range(max_iter):
            prev_rank = np.copy(rank)
            for i in range(n):
                rank[i] = (1 - damping) + damping * np.sum(
                    similarity_matrix[i] * prev_rank / np.sum(similarity_matrix, axis=1)
                )
            if np.linalg.norm(rank - prev_rank, ord=1) < tol:
                break
        return rank
    ranks = pagerank(similarity_matrix)
    
    ### Step 4: Extract top-ranked sentences
    ranked_sentences = [(ranks[i], original_sentences[i]) for i in range(len(ranks))]
    ranked_sentences = sorted(ranked_sentences, key=lambda x: x[0], reverse=True)
    
    ### Step 5: Determine number of sentences for summary based on summary_type
    if summary_type == 'num_sentences':
        num_sentences = min(num_sentences, len(original_sentences))  # Ensure we don't exceed the total number of sentences
    elif summary_type == 'percentage_txt':
        num_sentences = math.ceil(len(original_sentences) * percentage_txt)
        num_sentences = max(1, min(num_sentences, len(original_sentences)))

    ### Step 6: Summary itself
    summary = " ".join([ranked_sentences[i][1] for i in range(min(num_sentences, len(ranked_sentences)))])
    return summary


# Function to do each step of TextRank
def textrank_algorithm_with_kg(text,
                               summary_type,
                               num_sentences,
                               percentage_txt,
                               triples_per_sentence,
                               semantics_per_sentence,
                               kg_weight=0.1):
    # Initialize stop words
    stop_words = set(stopwords.words("english"))
    
    ### Step 1: Preprocess the text
    original_sentences = sent_tokenize(text)
    clean_sentences = []
    filtered_words = []
    for sentence in original_sentences:
        words = word_tokenize(sentence.lower())
        filtered = [word for word in words if word.isalnum() and word not in stop_words]
        clean_sentence = " ".join(filtered)
        clean_sentences.append(clean_sentence)
        filtered_words.extend(filtered)
    
    ### Step 2: Calculate sentence similarity using TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(clean_sentences)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    ### Step 3: Apply the PageRank algorithm
    def pagerank(similarity_matrix, damping=0.85, max_iter=100, tol=1e-6):
        n = similarity_matrix.shape[0]
        rank = np.ones(n) / n  # Initial rank
        for _ in range(max_iter):
            prev_rank = np.copy(rank)
            for i in range(n):
                rank[i] = (1 - damping) + damping * np.sum(
                    similarity_matrix[i] * prev_rank / np.sum(similarity_matrix, axis=1)
                )
            if np.linalg.norm(rank - prev_rank, ord=1) < tol:
                break
        return rank
    ranks = pagerank(similarity_matrix)
    
    ### Step 4: Count the number of triples + enriched triples for each sentence
    sentence_triples_count = {
        sentence_idx: len(triples_per_sentence[sentence_idx]) + len(semantics_per_sentence[sentence_idx])
        for sentence_idx in triples_per_sentence
    }
    # Adjust ranks by adding the weighted count of triples
    adjusted_ranks = []
    for i, rank in enumerate(ranks):
        # Adjust each sentence's rank based on the number of triples and constant weight
        adjusted_rank = rank + (sentence_triples_count.get(i, 0) * kg_weight)
        adjusted_ranks.append(adjusted_rank)
        # Print the adjusted rank for each sentence
        print(f"Sentence {i} - Rank: {rank:.4f},"
            f"Rank after adding weighted count of triples: {adjusted_rank:.4f}")
        
    ### Step 5: Extract top-ranked sentences
    ranked_sentences = [(adjusted_ranks[i], original_sentences[i]) for i in range(len(adjusted_ranks))]
    ranked_sentences = sorted(ranked_sentences, key=lambda x: x[0], reverse=True)
    
    ### Step 6: Determine number of sentences for summary based on summary_type
    if summary_type == 'num_sentences':
        num_sentences = min(num_sentences, len(original_sentences))  # Ensure we don't exceed the total number of sentences
    elif summary_type == 'percentage_txt':
        num_sentences = math.ceil(len(original_sentences) * percentage_txt)
        num_sentences = max(1, min(num_sentences, len(original_sentences)))
    
    ### Step 7: Summary itself
    summary = " ".join([ranked_sentences[i][1] for i in range(min(num_sentences, len(ranked_sentences)))] )
    return summary



# ------------------------------------------------
# EVALUATION METRICS
def evaluate_with_rouge(summary, reference_text):
    # Initialize the ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference_text, summary)
    
    # Extract scores for each metric
    rouge_scores = {
        "rouge1": scores['rouge1'].fmeasure,
        "rouge2": scores['rouge2'].fmeasure,
        "rougeL": scores['rougeL'].fmeasure
    }
    return rouge_scores

# SUMMARIZE BY SENTENCE/WORD IMPORTANCE + KG CONCEPTS

# Summarization by "KG weighting" when it is used
def summarization(text,
                  num_sentences,
                  percentage_txt,
                  counting,
                  add_wikidata_information):
    
    start_time = time.time()

    # Split text into sentences
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    total_sentences = len(sentences)

    # Calculate the number of words per sentence
    word_counts_per_sentence = [len(sentence.split()) for sentence in sentences]

    # Initialize Wikidata importance dictionary
    sentence_concepts = {}

    # Only calculate Wikidata scores if requested
    if add_wikidata_information == 'yes':

        # Ensure nltk packages are downloaded
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)

        # Extract Wikidata concepts for each sentence
        sentence_concepts = wikidata_main_entities(text)

        # Filter Wikidata concepts for each sentence (use the filtered version)
        filtered_concepts = filter_wikidata_main_entities(sentence_concepts)

        # Pass the filtered concepts to get relationships
        extract_concepts = wikidata_entities_relations(filtered_concepts)

        # Get all entities and sort them
        sorted_extracted = wikidata_all_entities(extract_concepts)

        # OPCIONAL, É SÓ PARA VER O QUE ESTÁ A PASSAR
        # See all entities ordered by frequency
        entity_counts = Counter(sorted_extracted)
        df = pd.DataFrame(entity_counts.items(), columns=['Entity', 'Frequency'])
        df_sorted = df.sort_values(by='Frequency', ascending=False).reset_index(drop=True)
        print("\n")
        print(df_sorted)

        # Call triples
        initial_triples_per_sentence, initial_triples = query_to_generate_triples(filtered_concepts)
        # Call semantic enrichment with WordNet
        semantics_per_sentence, semantic_triples = semantic_enrichment(initial_triples_per_sentence)
        # Just checking quantities
        print("\nAll tripes: ", len(initial_triples))
        print("Semantic layer: ", len(semantic_triples))
        print("Enriched triples: ", len(initial_triples) + len(semantic_triples))
        print("\n")
        # Call TextRank with KG information
        summary = textrank_algorithm_with_kg(text, counting, num_sentences, percentage_txt, initial_triples_per_sentence, semantics_per_sentence)
    
    else:
        # Call TextRank with no KG information
        summary = textrank_algorithm(text, counting, num_sentences, percentage_txt)

    # ROUGE evaluation
    rouge_scores = evaluate_with_rouge(summary, text)

    final_time = time.time()
    return summary, rouge_scores, (final_time - start_time)


# ------------------------------------------------
# SELECT PARAMETERS

# Text example
text = """Artificial Intelligence (AI) is the field of study that focuses on creating systems capable of performing tasks that typically require human intelligence. These tasks include learning, reasoning, problem solving, perception, and language understanding."""


# Define all possible configurations for each parameter
add_kg_info_options = {0: 'no', 1: 'yes'}
counting_options = {0: 'num_sentences', 1: 'percentage_txt'}

# Values of interest for the results
num_sentences = 1
percentage_txt = 0.5

# Iterate over all possible combinations of options
for add_kg, counting in itertools.product(add_kg_info_options.values(), counting_options.values()):
    # Check if we need to add KG info and download resources if required
    if add_kg == 'yes':
        t1 = time.time()
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('all', quiet=True)
        t2 = time.time()
    else:
        t1, t2 = 0, 0

    # Generate the summary based on current configuration
    summary, rouge_scores, t = summarization(text, num_sentences, percentage_txt, counting, add_kg)

    print(f"{add_kg}, {counting}, {t}s:")
    print(summary, "\n")
    print(f"ROUGE Scores: {rouge_scores}")

    



"""AI draws on concepts from several disciplines including computer science, mathematics, linguistics, psychology, and neuroscience to develop algorithms and models that mimic cognitive functions.

AI has evolved significantly since its inception in the mid-20th century. Early AI research aimed at creating machines that could perform narrowly defined tasks, such as playing chess or solving algebraic problems. These early systems were rule-based and relied heavily on pre-programmed knowledge. However, modern AI systems, particularly those using machine learning, are capable of improving their performance over time by learning from data without explicit programming.

One of the most widely used approaches to AI is machine learning (ML), a subset of AI that uses statistical methods to enable machines to improve at tasks through experience. Within ML, deep learning has gained prominence due to its ability to process vast amounts of data and model complex patterns. Deep learning models are particularly effective in fields such as computer vision, natural language processing, and speech recognition.

The application of AI in various industries has brought about transformative changes. In healthcare, AI-powered systems are used to assist in diagnostics, predict patient outcomes, and personalize treatment plans. Financial institutions employ AI to detect fraudulent activities, automate trading, and enhance customer service through chatbots. The automotive industry is exploring AI for autonomous vehicles, where systems can interpret data from sensors to navigate roads safely.

Despite its advantages, AI poses several ethical and societal challenges. Concerns regarding job displacement, privacy, and algorithmic bias are at the forefront of discussions around AI. Additionally, the development of artificial general intelligence (AGI)—machines capable of performing any intellectual task that humans can—raises questions about control, safety, and the potential impact on society.

AI continues to evolve, pushing the boundaries of what machines can achieve. Researchers are working on making AI systems more robust, explainable, and aligned with human values to ensure that AI benefits society as a whole."""