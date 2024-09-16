import pandas as pd
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
import nltk
from nltk.corpus import cmudict
import re
import os
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor


nltk.download('punkt')
nltk.download('cmudict')

# Load the CMU Pronouncing Dictionary
d = cmudict.dict()

# Load stop words from the StopWords folder
stop_words = set()
stop_words_files = [f for f in os.listdir('StopWords') if f.endswith('.txt')]
for file in stop_words_files:
    with open(os.path.join('StopWords', file), 'r', encoding='ISO-8859-1') as f:
        for line in f:
            stop_words.add(line.strip().lower())

# Load positive and negative word lists from the MasterDictionary folder
positive_words = set()
negative_words = set()

with open('MasterDictionary/positive-words.txt', 'r', encoding='ISO-8859-1') as f:
    for line in f:
        if line.strip() and not line.startswith(';'):
            positive_words.add(line.strip().lower())

with open('MasterDictionary/negative-words.txt', 'r', encoding='ISO-8859-1') as f:
    for line in f:
        if line.strip() and not line.startswith(';'):
            negative_words.add(line.strip().lower())

# Function to count syllables in a word
def syllable_count(word):
    try:
        return [len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]][0]
    except KeyError:
        return 1  # If the word is not found in the dictionary, assume 1 syllable

# Function to count personal pronouns
def count_personal_pronouns(text):
    pronouns = re.findall(r'\b(I|we|my|ours|us)\b', text, re.IGNORECASE)
    pronouns = [pronoun for pronoun in pronouns if pronoun != 'US']
    return len(pronouns)

# Function to clean text using stop words
def clean_text(text):
    tokens = nltk.word_tokenize(text)
    cleaned_tokens = [word for word in tokens if word.lower() not in stop_words and word.isalpha()]
    return " ".join(cleaned_tokens)

# Function to compute text analysis metrics
def text_analysis(text):
    blob = TextBlob(text)
    
    # Positive and Negative Score
    positive_score = sum(1 for word in blob.words if word.lower() in positive_words)
    negative_score = sum(1 for word in blob.words if word.lower() in negative_words)
    
    # Polarity and Subjectivity
    polarity = (positive_score - negative_score) / ((positive_score + negative_score) + 0.000001)
    subjectivity = (positive_score + negative_score) / (len(blob.words) + 0.000001)
    
    # Sentence and word counts
    sentences = blob.sentences
    words = blob.words
    word_count = len(words)
    sentence_count = len(sentences)
    avg_sentence_length = word_count / sentence_count if sentence_count != 0 else 0
    
    # Syllables per word and complex word count
    syllables_per_word = sum(syllable_count(word) for word in words) / word_count if word_count != 0 else 0
    complex_word_count = sum(1 for word in words if syllable_count(word) > 2)
    percentage_of_complex_words = complex_word_count / word_count if word_count != 0 else 0
    
    # FOG Index
    fog_index = 0.4 * (avg_sentence_length + percentage_of_complex_words)
    
    # Average number of words per sentence
    avg_number_of_words_per_sentence = avg_sentence_length
    
    # Average word length
    avg_word_length = sum(len(word) for word in words) / word_count if word_count != 0 else 0
    
    return {
        'POSITIVE SCORE': positive_score,
        'NEGATIVE SCORE': negative_score,
        'POLARITY SCORE': polarity,
        'SUBJECTIVITY SCORE': subjectivity,
        'AVG SENTENCE LENGTH': avg_sentence_length,
        'PERCENTAGE OF COMPLEX WORDS': percentage_of_complex_words,
        'FOG INDEX': fog_index,
        'AVG NUMBER OF WORDS PER SENTENCE': avg_number_of_words_per_sentence,
        'COMPLEX WORD COUNT': complex_word_count,
        'WORD COUNT': word_count,
        'SYLLABLE PER WORD': syllables_per_word,
        'AVG WORD LENGTH': avg_word_length
    }

async def fetch(session, url):
    try:
        async with session.get(url) as response:
            if response.status != 200:
                print(f"Page not found: {url}")
                return None, None
            text = await response.text()
            return url, text
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return None, None

def extract_article(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Extract the title
    title_tag = soup.find('h1')
    title = title_tag.get_text().strip() if title_tag else None
    
    # Extract the main article text
    article_div_classes = [['td-post-content', 'tagdiv-type'], ['tdb-block-inner', 'td-fix-index']]
    article = ''
    
    for div_class in article_div_classes:
        article_div = soup.find('div', class_=div_class)
        if article_div:
            # Remove the pre tag with class 'wp-block-preformatted'
            for pre in article_div.find_all('pre', class_='wp-block-preformatted'):
                pre.decompose()
            article = article_div.get_text(separator=' ').strip()
            if article:
                break
    
    return title, article

# Function to extract article text
def extract_article(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Extract the title
    title_tag = soup.find('h1')
    title = title_tag.get_text().strip() if title_tag else None
    
    # Extract the main article text
    article_div_classes = [['td-post-content', 'tagdiv-type'], ['tdb-block-inner', 'td-fix-index']]
    article = ''
    
    for div_class in article_div_classes:
        article_div = soup.find('div', class_=div_class)
        if article_div:
            # Remove the pre tag with class 'wp-block-preformatted'
            for pre in article_div.find_all('pre', class_='wp-block-preformatted'):
                pre.decompose()
            article = article_div.get_text(separator=' ').strip()
            if article:
                break
    
    return title, article

# Read the input file
input_df = pd.read_excel('Input.xlsx')
urls = input_df['URL'].tolist()

async def main(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, url) for url in urls]
        html_contents = await asyncio.gather(*tasks)
        
        results = []
        for url, html_content in html_contents:
            if html_content:
                title, article = extract_article(html_content)
                if not title or not article:
                    continue
                text = title + "\n\n\n\n" + article
                
                # Save the extracted text to a file
                url_id = input_df[input_df['URL'] == url]['URL_ID'].values[0]
                with open(f"{url_id}.txt", 'w', encoding='utf-8') as file:
                    file.write(text)
                
                personal_pronouns = count_personal_pronouns(text)
                cleaned_text = clean_text(text)
                analysis_results = text_analysis(cleaned_text)
                analysis_results['PERSONAL PRONOUNS'] = personal_pronouns
                analysis_results['URL'] = url
                analysis_results['URL_ID'] = url_id  # Add URL_ID to results
                results.append(analysis_results)
        
        return results

# Run the asynchronous main function
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
results = loop.run_until_complete(main(urls))

# Prepare the output DataFrame
output_columns = [
    'URL_ID', 'URL', 'POSITIVE SCORE', 'NEGATIVE SCORE', 'POLARITY SCORE', 
    'SUBJECTIVITY SCORE', 'AVG SENTENCE LENGTH', 'PERCENTAGE OF COMPLEX WORDS', 
    'FOG INDEX', 'AVG NUMBER OF WORDS PER SENTENCE', 'COMPLEX WORD COUNT', 
    'WORD COUNT', 'SYLLABLE PER WORD', 'PERSONAL PRONOUNS', 'AVG WORD LENGTH'
]
output_df = pd.DataFrame(results, columns=output_columns)

# Adjust column width
with pd.ExcelWriter('Output Data Structure.xlsx', engine='xlsxwriter') as writer:
    output_df.to_excel(writer, sheet_name='Sheet1', index=False)
    worksheet = writer.sheets['Sheet1']
    for i, col in enumerate(output_df.columns):
        max_length = max(output_df[col].astype(str).map(len).max(), len(col)) + 2
        worksheet.set_column(i, i, max_length)

print("Processing complete. File 'Output Data Structure.xlsx' has been saved.")
