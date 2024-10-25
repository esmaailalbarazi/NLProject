import requests # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
from sklearn.metrics.pairwise import cosine_similarity # type: ignore
import networkx as nx # type: ignore
import re
import math
import requests # type: ignore
import nltk # type: ignore
from nltk.corpus import stopwords # type: ignore
from nltk.tokenize import word_tokenize # type: ignore
import time


# ----------------------------------------------- #
# ------------------ TEXT RANK ------------------ #
# ----------------------------------------------- #

t1 = time.time()
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('all', quiet=True)
t2 = time.time()

# -------------------------------------------------
# BASIC SETUP

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
        if response.status_code == 200:
            results = response.json().get("search", [])
            return [result['id'] for result in results]  # Return Wikidata IDs
        else:
            print(f"Error fetching concepts for keyword: {keyword}, Status Code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
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

    # Only calculate Wikidata scores if requested
    if add_wikidata_information == 'yes':

        # Initialize Wikidata importance dictionary
        wikidata_importance = {}
        
        # Ensure nltk packages are downloaded
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('all', quiet=True)

        # Extract Wikidata concepts for each sentence
        sentence_concepts = extract_wikidata_concepts(text)
        for i, concepts in sentence_concepts.items():
            wikidata_score = len(concepts) * wikidata_weight_factor
            wikidata_importance[i] = wikidata_score
        
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

    return summary, (final_time - start_time)


# Summarization by "KG filtering" when it is used
def summ_2(text,
           num_sentences,
           percentage_txt,
           counting,
           kind_of_summarization,
           add_wikidata_information):
    
    start_time = time.time()

    # Split text into sentences
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    total_sentences = len(sentences)

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

    # Initialize frequent KG concepts if KG information is included
    frequent_concepts = set()
    if add_wikidata_information == 'yes':
                    
        # Ensure nltk packages are downloaded
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('all', quiet=True)

        # Extract Wikidata concepts for each sentence
        sentence_concepts = extract_wikidata_concepts(text)

        # Count occurrences of each concept across sentences
        concept_frequency = {}
        for concepts in sentence_concepts.values():
            for concept in concepts:
                concept_frequency[concept] = concept_frequency.get(concept, 0) + 1

        # Filter for the most frequent concepts based on a threshold
        threshold = max(1, int(len(sentence_concepts) * 0.1))  # Top 10% frequent
        frequent_concepts = {concept for concept, freq in concept_frequency.items() if freq >= threshold}

    # Select sentences that contain frequent KG concepts, if KG data is included
    ranked_sentences = []
    for i, sentence in enumerate(sentences):
        # Only include sentences that contain frequent KG concepts if KG data is used
        if add_wikidata_information == 'yes' and frequent_concepts:
            sentence_concepts = set(extract_wikidata_concepts(sentence).get(i, []))
            if not frequent_concepts & sentence_concepts:
                continue  # Skip sentences without frequent KG concepts

        # Start with TextRank score
        total_score = sentence_scores[i]

        # If including word importance with TF-IDF scores
        if kind_of_summarization == 'sentences_n_words':
            word_score = tfidf[i].sum()  # Sum TF-IDF scores for words in this sentence
            total_score += word_score * 0.5  # Apply a smaller weight to TF-IDF word importance

        ranked_sentences.append((total_score, sentence))

    # Sort sentences based on combined score
    ranked_sentences = sorted(ranked_sentences, reverse=True)

    # Select top sentences for the summary based on the filtered set
    summary = " ".join([ranked_sentences[i][1].strip() for i in range(min(num_sentences, len(ranked_sentences)))])

    final_time = time.time()

    return summary, (final_time - start_time)


# ------------------------------------------------
# SELECT PARAMETERS

# Text example
text = """Artificial Intelligence (AI) is the field of study that focuses on creating systems capable of performing tasks that typically require human intelligence. These tasks include learning, reasoning, problem-solving, perception, and language understanding. AI draws on concepts from several disciplines including computer science, mathematics, linguistics, psychology, and neuroscience to develop algorithms and models that mimic cognitive functions.

AI has evolved significantly since its inception in the mid-20th century. Early AI research aimed at creating machines that could perform narrowly defined tasks, such as playing chess or solving algebraic problems. These early systems were rule-based and relied heavily on pre-programmed knowledge. However, modern AI systems, particularly those using machine learning, are capable of improving their performance over time by learning from data without explicit programming.

One of the most widely used approaches to AI is machine learning (ML), a subset of AI that uses statistical methods to enable machines to improve at tasks through experience. Within ML, deep learning has gained prominence due to its ability to process vast amounts of data and model complex patterns. Deep learning models are particularly effective in fields such as computer vision, natural language processing, and speech recognition.

The application of AI in various industries has brought about transformative changes. In healthcare, AI-powered systems are used to assist in diagnostics, predict patient outcomes, and personalize treatment plans. Financial institutions employ AI to detect fraudulent activities, automate trading, and enhance customer service through chatbots. The automotive industry is exploring AI for autonomous vehicles, where systems can interpret data from sensors to navigate roads safely.

Despite its advantages, AI poses several ethical and societal challenges. Concerns regarding job displacement, privacy, and algorithmic bias are at the forefront of discussions around AI. Additionally, the development of artificial general intelligence (AGI)—machines capable of performing any intellectual task that humans can—raises questions about control, safety, and the potential impact on society.

AI continues to evolve, pushing the boundaries of what machines can achieve. Researchers are working on making AI systems more robust, explainable, and aligned with human values to ensure that AI benefits society as a whole."""

# Kind of summary based on sentence/word importance
kind_of_summ = {0: 'just_sentences', 1: 'sentences_n_words'}[1]

# Add or not information from BabelNet
add_kg_info = {0: 'no', 1: 'yes'}[1]
# BabelNet importance
wiki_weight = 2.5
# Weight high or filter actively concepts?
wiki_usage = {0: 'weight', 1: 'filter'}[1]

# Results based one what?
counting = {0: 'num_sentences', 1: 'percentage_txt'}[0]
# Values of interest for the results
num_sentences = 3
percentage_txt = 0.2


# ------------------------------------------------
# GENERATE SUMMARY ITSELF

# For each approach, generate the summary
if wiki_usage == 'weight':
    summary = summ_1(text, num_sentences, percentage_txt, counting, kind_of_summ, add_kg_info, wiki_weight)
else:
    summary = summ_2(text, num_sentences, percentage_txt, counting, kind_of_summ, add_kg_info)

# Print the summary
print(summary)




counting_options = ['num_sentences', 'percentage_txt']
num_sentences = 3
percentage_txt = 0.2

kind_summ = ['only_sentences', 'sentences_n_words']

add_kg = ['no', 'yes']

wiki_usage = ['weight', 'filter']
weight_factor = 2

# Loop through all parameter combinations
for counting in counting_options:
    for add in add_kg:
        for kind in kind_summ:
            for usage in wiki_usage:
                # Choose summarization method based on user preference
                if wiki_usage == 'weight':
                    summary, t = summ_1(text, num_sentences, percentage_txt, counting, kind, add, weight_factor)
                else:
                    summary, t = summ_2(text, num_sentences, percentage_txt, counting, kind, add)
                
                print(f"SUMMARY SCHEMA: {counting}, SENTENCES/WORDS: {kind}, ADD KG INFO: {add}, KG USAGE APPROACH: {usage}")

                print("Summary:")
                print(summary)

                print("Spent time:")
                if add == 'yes':
                   print(round(t + t2 - t1, 2))
                else:
                    print(round(t, 2))
                print("-" * 80)
