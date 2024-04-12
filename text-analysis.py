from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import os


# Function to extract text from HTML file
def extract_text_from_html(file_path):
    with open(file_path, 'r') as file:
        html_doc = file.read()
    soup = BeautifulSoup(html_doc, 'html.parser')
    strings = [text.strip() for text in soup.stripped_strings]
    return ' '.join(strings)

# Function to create TF-IDF matrix for HTML files
def create_tfidf_matrices(html_files):
    tfidf_matrices = []
    for file_path in html_files:
        document = extract_text_from_html(file_path)
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([document])
        tfidf_matrices.append((file_path, vectorizer, tfidf_matrix))
    return tfidf_matrices

# Function to compute cosine similarity between two vectors
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2.T)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    similarity = dot_product / (norm_vec1 * norm_vec2)
    return similarity

# Function to search for keyword in TF-IDF matrices
def search_keyword(keyword, tfidf_matrices):
    results = []
    for file_path, vectorizer, tfidf_matrix in tfidf_matrices:
        feature_names = vectorizer.get_feature_names_out()
        keyword_index = np.where(feature_names == keyword)[0]
        if keyword_index.size != 0:
            keyword_index = keyword_index[0]
            tfidf_vector = tfidf_matrix.toarray()[0]
            keyword_vector = np.zeros_like(tfidf_vector)
            keyword_vector[keyword_index] = 1
            similarity = cosine_similarity(tfidf_vector, keyword_vector)
            results.append((file_path, similarity))
    return results

# Main function
def main():
    # Get list of HTML files
    html_files = [f for f in os.listdir('example_crawl') if f.endswith('.html')]
    html_files = [os.path.join('example_crawl', f) for f in html_files]
    
    
    # Create TF-IDF matrices for HTML files
    tfidf_matrices = create_tfidf_matrices(html_files)
    
    # Input keyword from user
    keyword = input("Enter keyword to search: ")
    
    # Search for keyword in HTML files
    results = search_keyword(keyword, tfidf_matrices)
    
    # Print results
    if results:
        print("HTML files containing the keyword, sorted by relevance:")
        results.sort(key=lambda x: x[1], reverse=True)
        for result in results:
            print(f"{result[0]} (Similarity: {result[1]})")
    else:
        print("No HTML files contain the keyword.")

if __name__ == "__main__":
    main()
