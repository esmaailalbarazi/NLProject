import requests # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
from sklearn.metrics.pairwise import cosine_similarity # type: ignore
import networkx as nx # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import math
import re
import os
import time
import itertools
import requests # type: ignore
import nltk # type: ignore
import logging
from nltk.corpus import stopwords # type: ignore
from nltk.tokenize import word_tokenize # type: ignore
from sumy.parsers.plaintext import PlaintextParser # type: ignore
from sumy.nlp.tokenizers import Tokenizer # type: ignore
from sumy.summarizers.text_rank import TextRankSummarizer # type: ignore
# Metrics
from sklearn.metrics.pairwise import cosine_similarity # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
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


# Function to replace hyphens and em dashes with a space
def replace_dashes_with_space(text):
    return re.sub(r'[-—]', ' ', text)

# Function to split text into sentences
def split_text_into_sentences(text):
    sentence_pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!|…)\s+'
    sentences = re.split(sentence_pattern, text)
    return sentences

# Function to "clean" text with words that interest
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
    response = requests.get(url, params=params)
    if response.status_code == 200:
        results = response.json().get("search", [])
        return [result['id'] for result in results]  # Return Wikidata IDs
    else:
        print(f"Error fetching concepts for keyword: {keyword}, Status Code: {response.status_code}")
    return []

# Function to extract Wikidata concepts for each sentence
def extract_wikidata_concepts(text):
    sentences = split_text_into_sentences(text)
    wikidata_concepts = {}
    # Extraction itself
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
    sentences = split_text_into_sentences(text)
    total_sentences = len(sentences)
    word_counts_per_sentence = [len(sentence.split()) for sentence in sentences]

    # Define the size of the summary
    if counting == 'num_sentences':
        num_sentences = min(num_sentences, total_sentences)
    elif counting == 'percentage_txt':
        num_sentences = math.ceil(total_sentences * percentage_txt)
        num_sentences = max(1, min(num_sentences, total_sentences))

    # Pre-processing: new text without hyphens and dashes
    new_text = replace_dashes_with_space(text)

    # Parse the text with Sumy
    parser = PlaintextParser.from_string(new_text, Tokenizer("english"))
    summarizer = TextRankSummarizer()

    # Generate the summary using TextRank
    text_rank_summary = summarizer(parser.document, num_sentences)

    # Initialize Wikidata importance dictionary
    wikidata_importance = {}
    summary_text = ""

    # Only calculate Wikidata scores if requested
    if add_wikidata_information == 'yes':
        sentence_concepts = extract_wikidata_concepts(new_text)
        for i, concepts in sentence_concepts.items():
            wikidata_score = len(concepts) * wikidata_weight_factor
            wikidata_importance[i] = wikidata_score

    # Handling summarization based on type
    if kind_of_summarization == 'sentences_n_words':
        # Generate TF-IDF matrix for sentences
        tfidf_vectorizer = TfidfVectorizer()
        
        # Extract the text from each Sentence object
        sentence_texts = [str(sentence) for sentence in parser.document.sentences]
        tfidf = tfidf_vectorizer.fit_transform(sentence_texts)

        # Calculate word importance by summing TF-IDF scores for each sentence
        word_importance = np.array(tfidf.sum(axis=1)).flatten()

        # Combine TextRank scores and word importance
        combined_scores = []
        for i, sentence in enumerate(text_rank_summary):
            # Assuming the sentence is a string, we don't have sentence.score
            total_score = (word_importance[i] * 0.5)  # Only using word importance
            combined_scores.append((total_score, str(sentence)))

        # Sort sentences based on combined score
        combined_scores = sorted(combined_scores, reverse=True)

        # Select top sentences for the summary
        summary_text = " ".join([combined_scores[i][1] for i in range(num_sentences)])

    else:
        # Just using TextRank for summarization
        summary_text = " ".join([str(sentence) for sentence in text_rank_summary[:num_sentences]])

    # Handle case where summary might still be empty
    if not summary_text.strip():
        summary_text = "No valid summary could be generated."

    final_time = time.time()

    return summary_text, (final_time - start_time), wikidata_importance, word_counts_per_sentence


# Summarization by "KG filtering" when it is used
def summ_2(text,
           num_sentences,
           percentage_txt,
           counting,
           kind_of_summarization,
           add_wikidata_information):
    
    start_time = time.time()

    # Split text into sentences for initial processing
    sentences = split_text_into_sentences(text)
    total_sentences = len(sentences)

    # Define the size of the summary
    if counting == 'num_sentences':
        num_sentences = min(num_sentences, total_sentences)
    elif counting == 'percentage_txt':
        num_sentences = math.ceil(total_sentences * percentage_txt)
        num_sentences = max(1, min(num_sentences, total_sentences))

    # Pre-processing: new text without hyphens and dashes
    new_text = replace_dashes_with_space(text)

    # Parse the text with Sumy and apply TextRank
    parser = PlaintextParser.from_string(new_text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    text_rank_summary = summarizer(parser.document, num_sentences)

    # Initialize frequent KG concepts if KG information is included
    frequent_concepts = set()
    if add_wikidata_information == 'yes':
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)

        # Extract Wikidata concepts for each sentence
        sentence_concepts = extract_wikidata_concepts(new_text)

        # Count occurrences of each concept across sentences
        concept_frequency = {}
        for concepts in sentence_concepts.values():
            for concept in concepts:
                concept_frequency[concept] = concept_frequency.get(concept, 0) + 1

        # Filter for the most frequent concepts based on a threshold
        threshold = max(1, int(len(sentence_concepts) * 0.1))  # Top 10% frequent
        frequent_concepts = {concept for concept, freq in concept_frequency.items() if freq >= threshold}

    # Extract sentences with frequent KG concepts if KG data is used
    ranked_sentences = []
    for i, sentence in enumerate(text_rank_summary):
        if add_wikidata_information == 'yes' and frequent_concepts:
            sentence_concepts = set(extract_wikidata_concepts(str(sentence)).get(i, []))
            if not frequent_concepts & sentence_concepts:
                continue  # Skip sentences without frequent KG concepts

        # Calculate the initial score from TextRank
        total_score = 1.0  # Assuming this is a placeholder score

        # Add word importance if specified
        if kind_of_summarization == 'sentences_n_words':
            # Generate TF-IDF for each sentence
            tfidf_vectorizer = TfidfVectorizer()
            tfidf = tfidf_vectorizer.fit_transform([str(sentence) for sentence in text_rank_summary])
            
            # Calculate word importance score for the sentence
            word_importance = np.array(tfidf.sum(axis=1)).flatten()[i]
            total_score += word_importance * 0.5  # Apply weight to word importance

        ranked_sentences.append((total_score, str(sentence)))

    # Sort sentences based on combined score
    ranked_sentences = sorted(ranked_sentences, reverse=True)

    # Select top sentences for the summary based on the filtered set
    summary = " ".join([ranked_sentences[i][1].strip() for i in range(min(num_sentences, len(ranked_sentences)))])

    # Handle case where summary might still be empty
    if not summary.strip():
        summary = "No valid summary could be generated."

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


# Define all possible configurations for each parameter
add_kg_info_options = {0: 'no', 1: 'yes'}
wiki_usage_options = {0: 'weight', 1: 'filter'}
wiki_weight = 2.5
kind_summ_options = {0: 'just_sentences', 1: 'sentences_n_words'}
counting_options = {0: 'num_sentences', 1: 'percentage_txt'}

# Values of interest for the results
num_sentences = 3
percentage_txt = 0.2

# Function to compute ROUGE scores
def compute_rouge(reference, generated):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, generated)
    return scores

# Function to compute BLEU score
def compute_bleu(reference, generated):
    reference_tokens = [reference.split()]
    generated_tokens = generated.split()
    smooth = SmoothingFunction().method1
    score = sentence_bleu(reference_tokens, generated_tokens, smoothing_function=smooth)
    return score

# Function to compute BERTScore
def compute_bert_score(reference, generated):
    P, R, F1 = bert_score([generated], [reference], lang="en")
    return {"Precision": P.mean().item(), "Recall": R.mean().item(), "F1": F1.mean().item()}

# Function to compute cosine similarity with TF-IDF
def compute_tfidf_cosine(reference, generated):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([reference, generated])
    cos_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return cos_similarity[0][0]

# DataFrame to hold all evaluation results for each scenario
all_results = pd.DataFrame()

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
        summary, t, c1, c2 = summ_1(text, num_sentences, percentage_txt, counting, kind_summ, add_kg_info, wiki_weight)
    else:
        summary, t = summ_2(text, num_sentences, percentage_txt, counting, kind_summ, add_kg_info)

    print(f"{add_kg_info}, {wiki_usage}, {kind_summ}, {counting}")
    print(summary)
    print('-----------------------------------')

    # Run evaluation
    rouge_scores = compute_rouge(text, summary)
    bleu_score = compute_bleu(text, summary)
    bert_scores = compute_bert_score(text, summary)
    cosine_sim = compute_tfidf_cosine(text, summary)

    # Time calculation
    spent_time = round(t + (t2 - t1) if add_kg_info == 'yes' else t, 2)

    # Collecting the results into a dictionary for this specific configuration
    config_results = {
        "add_kg_info": add_kg_info,
        "wiki_usage": wiki_usage,
        "kind_summ": kind_summ,
        "counting": counting,
        "ROUGE-1": rouge_scores['rouge1'].fmeasure,
        "ROUGE-2": rouge_scores['rouge2'].fmeasure,
        "ROUGE-L": rouge_scores['rougeL'].fmeasure,
        "BLEU": bleu_score,
        "BERT Precision": bert_scores['Precision'],
        "BERT Recall": bert_scores['Recall'],
        "BERT F1": bert_scores['F1'],
        "Cosine Similarity": cosine_sim,
        "Spent Time": spent_time
    }

    # Add this run's results as a new row in the DataFrame
    all_results = pd.concat([all_results, pd.DataFrame([config_results])], ignore_index=True)

# Transpose for the final format where each run's results are a column
results_df = all_results.T

# Save the results to an Excel file
results_df.to_excel("summarization_textrank_results.xlsx")

# Show all rows and columns in pandas DataFrame for full visibility
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Print final transposed DataFrame
print(results_df)


# ------------------------------------------------
# EXTRA: EXPLORATORY ANALYSIS

# Calculate corr between KG weight per sentence and the number of words in each
new_c1 = np.array(list(c1.values())) / wiki_weight
correlation = np.corrcoef(list(new_c1), c2)
print(correlation)

# Cria o scatter plot
plt.scatter(new_c1, c2, color='blue', alpha=0.7)
plt.xlabel("KG given weight")
plt.ylabel("Number of words")
plt.title("Relation 'Size - KG importance'")
plt.tight_layout()
plt.show()
