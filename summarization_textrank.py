from transformers import pipeline
from transformers import utils

# Initialize the summarization pipeline
summarizer = pipeline("summarization")

# Sample text to summarize
text = """
Natural Language Processing (NLP) is a subfield of artificial intelligence (AI) that focuses on the interaction 
between computers and humans through natural language. The ultimate goal of NLP is to enable computers to 
understand, interpret, and generate human language in a valuable way. NLP combines computational linguistics 
with machine learning and deep learning to facilitate the analysis of large amounts of natural language data. 
Applications of NLP include chatbots, language translation, sentiment analysis, and speech recognition. 
As technology advances, NLP continues to evolve, leading to more sophisticated algorithms and better 
understanding of human language.
"""

# Generate the summary
summary = summarizer(text, max_length=50, min_length=25, do_sample=False)

# Print the summary
print("Summary:")
print(summary[0]['summary_text'])