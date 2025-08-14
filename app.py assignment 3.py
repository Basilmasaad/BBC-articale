import streamlit as st
import joblib
import pandas as pd
import re
import string
import nltk
import ssl
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Set NLTK data path to a writable directory
nltk_data_path = os.path.join("/tmp", "nltk_data")
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

# Download necessary NLTK data (if not already downloaded)
try:
    _create_unverified_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_context

@st.cache_resource
def download_nltk_data():
    """Download NLTK data packages with caching and version compatibility"""
    nltk_packages = [
        ('punkt_tab', 'punkt'),  # Try punkt_tab first, fallback to punkt
        ('stopwords', None),
        ('wordnet', None),
        ('omw-1.4', None)  # Additional wordnet data
    ]
    
    success = True
    for primary, fallback in nltk_packages:
        try:
            nltk.download(primary, download_dir=nltk_data_path, quiet=True)
        except Exception as e1:
            if fallback:
                try:
                    nltk.download(fallback, download_dir=nltk_data_path, quiet=True)
                except Exception as e2:
                    st.warning(f"Could not download {primary} or {fallback}: {e2}")
                    success = False
            else:
                try:
                    nltk.download(primary, download_dir=nltk_data_path, quiet=True)
                except:
                    st.warning(f"Could not download {primary}")
                    success = False
    
    return success

# Download NLTK data
if not download_nltk_data():
    st.warning("Some NLTK data may be missing, but trying to continue...")

# Initialize NLTK components with error handling
@st.cache_resource
def init_nltk_components():
    """Initialize NLTK components with error handling"""
    try:
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        return stop_words, lemmatizer, True
    except Exception as e:
        st.error(f"Error initializing NLTK components: {e}")
        return set(), None, False

stop_words, lemmatizer, nltk_ready = init_nltk_components()
def clean_text(text):
    """Clean and preprocess text with better error handling"""
    try:
        if not nltk_ready:
            # Fallback: basic cleaning without NLTK
            text = text.lower()
            text = text.translate(str.maketrans('', '', string.punctuation))
            words = text.split()
            basic_stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
            cleaned_words = [word for word in words if word not in basic_stopwords and len(word) > 1]
            return ' '.join(cleaned_words)
        
        # Regular NLTK processing
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Try tokenization with error handling
        try:
            tokens = nltk.word_tokenize(text)
        except Exception:
            # Fallback to simple split if tokenizer fails
            tokens = text.split()
        
        # Remove stopwords and lemmatize
        if lemmatizer:
            cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 1]
        else:
            cleaned_tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
        
        return ' '.join(cleaned_tokens)
        
    except Exception as e:
        st.error(f"Error cleaning text: {e}")
        # Return basic cleaned version as fallback
        text = text.lower()
        text = ''.join(c for c in text if c.isalnum() or c.isspace())
        return text@st.cache_resource
def load_models():
    """Load the trained model and vectorizer with caching"""
    try:
        # Check if files exist
        if not os.path.exists('saved_models/best_baseline_model.pkl'):
            st.error("Model file 'best_baseline_model.pkl' not found.")
            return None, None
        
        if not os.path.exists('saved_models/tfidf_vectorizer.pkl'):
            st.error("Vectorizer file 'tfidf_vectorizer.pkl' not found.")
            return None, None
        
        # Load models
        baseline_model = joblib.load('saved_models/best_baseline_model.pkl')
        tfidf_vectorizer = joblib.load('saved_models/tfidf_vectorizer.pkl')
        
        return baseline_model, tfidf_vectorizer
        
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

# Load models
baseline_model, tfidf_vectorizer = load_models()

if baseline_model is None or tfidf_vectorizer is None:
    st.error("Failed to load models. Please check the error messages above.")
    st.stop()

# Define the categories (make sure this matches the order used during training)
categories = ['business', 'entertainment', 'politics', 'sport', 'tech']

st.title("BBC News Article Classifier")
st.write("Enter a news article to classify its category.")

# Show NLTK status
if not nltk_ready:
    st.warning("⚠️ Using basic text processing (NLTK components not fully loaded)")
else:
    st.success("✅ All text processing components loaded successfully")

# Text input area
article_text = st.text_area("Enter Article Text", height=300, 
                           placeholder="Paste your news article text here...")

if st.button("Classify", type="primary"):
    if article_text.strip():
        try:
            # Clean the input text
            cleaned_article = clean_text(article_text)
            
            if not cleaned_article.strip():
                st.warning("The text appears to be empty after cleaning. Please try with different content.")
                st.stop()

            # Vectorize the cleaned text
            article_tfidf = tfidf_vectorizer.transform([cleaned_article])

            # Get prediction probabilities
            probabilities = baseline_model.predict_proba(article_tfidf)[0]
# Get the predicted category index and confidence
            predicted_category_index = baseline_model.predict(article_tfidf)[0]
            confidence = probabilities[predicted_category_index]

            # Get the predicted category name
            predicted_category = categories[predicted_category_index]

            # Display results
            st.success("Classification completed!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎯 Prediction")
                st.metric("Category", predicted_category.capitalize())
                st.metric("Confidence", f"{confidence:.1%}")
            
            with col2:
                st.subheader("📊 All Probabilities")
                prob_df = pd.DataFrame({
                    'Category': [cat.capitalize() for cat in categories], 
                    'Probability': [f"{prob:.1%}" for prob in probabilities]
                })
                prob_df = prob_df.sort_values(by='Probability', ascending=False)
                st.dataframe(prob_df, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"An error occurred during classification: {e}")
            st.error("Please try again with different text.")

    else:
        st.warning("⚠️ Please enter some text to classify.")@st.cache_resource
def load_models():
    """Load the trained model and vectorizer with caching"""
    try:
        # Check if files exist
        if not os.path.exists('saved_models/best_baseline_model.pkl'):
            st.error("Model file 'best_baseline_model.pkl' not found.")
            return None, None
        
        if not os.path.exists('saved_models/tfidf_vectorizer.pkl'):
            st.error("Vectorizer file 'tfidf_vectorizer.pkl' not found.")
            return None, None
        
        # Load models
        baseline_model = joblib.load('saved_models/best_baseline_model.pkl')
        tfidf_vectorizer = joblib.load('saved_models/tfidf_vectorizer.pkl')
        
        return baseline_model, tfidf_vectorizer
        
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

# Load models
baseline_model, tfidf_vectorizer = load_models()

if baseline_model is None or tfidf_vectorizer is None:
    st.error("Failed to load models. Please check the error messages above.")
    st.stop()

# Define the categories (make sure this matches the order used during training)
categories = ['business', 'entertainment', 'politics', 'sport', 'tech']

st.title("BBC News Article Classifier")
st.write("Enter a news article to classify its category.")

# Show NLTK status
if not nltk_ready:
    st.warning("⚠️ Using basic text processing (NLTK components not fully loaded)")
else:
    st.success("✅ All text processing components loaded successfully")

# Text input area
article_text = st.text_area("Enter Article Text", height=300, 
                           placeholder="Paste your news article text here...")

if st.button("Classify", type="primary"):
    if article_text.strip():
        try:
            # Clean the input text
            cleaned_article = clean_text(article_text)
            
            if not cleaned_article.strip():
                st.warning("The text appears to be empty after cleaning. Please try with different content.")
                st.stop()

            # Vectorize the cleaned text
            article_tfidf = tfidf_vectorizer.transform([cleaned_article])

            # Get prediction probabilities
            probabilities = baseline_model.predict_proba(article_tfidf)[0]

            # Get the predicted category index and confidence
            predicted_category_index = baseline_model.predict(article_tfidf)[0]
            confidence = probabilities[predicted_category_index]

            # Get the predicted category name
            predicted_category = categories[predicted_category_index]
# Display results
            st.success("Classification completed!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎯 Prediction")
                st.metric("Category", predicted_category.capitalize())
                st.metric("Confidence", f"{confidence:.1%}")
            
            with col2:
                st.subheader("📊 All Probabilities")
                prob_df = pd.DataFrame({
                    'Category': [cat.capitalize() for cat in categories], 
                    'Probability': [f"{prob:.1%}" for prob in probabilities]
                })
                prob_df = prob_df.sort_values(by='Probability', ascending=False)
                st.dataframe(prob_df, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"An error occurred during classification: {e}")
            st.error("Please try again with different text.")

    else:
        st.warning("⚠️ Please enter some text to classify.")@st.cache_resource
def load_models():
    """Load the trained model and vectorizer with caching"""
    try:
        # Check if files exist
        if not os.path.exists('saved_models/best_baseline_model.pkl'):
            st.error("Model file 'best_baseline_model.pkl' not found.")
            return None, None
        
        if not os.path.exists('saved_models/tfidf_vectorizer.pkl'):
            st.error("Vectorizer file 'tfidf_vectorizer.pkl' not found.")
            return None, None
        
        # Load models
        baseline_model = joblib.load('saved_models/best_baseline_model.pkl')
        tfidf_vectorizer = joblib.load('saved_models/tfidf_vectorizer.pkl')
        
        return baseline_model, tfidf_vectorizer
        
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

# Load models
baseline_model, tfidf_vectorizer = load_models()

if baseline_model is None or tfidf_vectorizer is None:
    st.error("Failed to load models. Please check the error messages above.")
    st.stop()

# Define the categories (make sure this matches the order used during training)
categories = ['business', 'entertainment', 'politics', 'sport', 'tech']

st.title("BBC News Article Classifier")
st.write("Enter a news article to classify its category.")

# Show NLTK status
if not nltk_ready:
    st.warning("⚠️ Using basic text processing (NLTK components not fully loaded)")
else:
    st.success("✅ All text processing components loaded successfully")

# Text input area
article_text = st.text_area("Enter Article Text", height=300, 
                           placeholder="Paste your news article text here...")

if st.button("Classify", type="primary"):
    if article_text.strip():
        try:
            # Clean the input text
            cleaned_article = clean_text(article_text)
            
            if not cleaned_article.strip():
                st.warning("The text appears to be empty after cleaning. Please try with different content.")
                st.stop()

            # Vectorize the cleaned text
            article_tfidf = tfidf_vectorizer.transform([cleaned_article])

            # Get prediction probabilities
            probabilities = baseline_model.predict_proba(article_tfidf)[0]

            # Get the predicted category index and confidence
            predicted_category_index = baseline_model.predict(article_tfidf)[0]
            confidence = probabilities[predicted_category_index]

            # Get the predicted category name
            predicted_category = categories[predicted_category_index]

# Display results
            st.success("Classification completed!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎯 Prediction")
                st.metric("Category", predicted_category.capitalize())
                st.metric("Confidence", f"{confidence:.1%}")
            
            with col2:
                st.subheader("📊 All Probabilities")
                prob_df = pd.DataFrame({
                    'Category': [cat.capitalize() for cat in categories], 
                    'Probability': [f"{prob:.1%}" for prob in probabilities]
                })
                prob_df = prob_df.sort_values(by='Probability', ascending=False)
                st.dataframe(prob_df, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"An error occurred during classification: {e}")
            st.error("Please try again with different text.")

    else:
        st.warning("⚠️ Please enter some text to classify.")# Add some example text
with st.expander("📝 Try with example text"):
    example_text = """
    Apple Inc. reported strong quarterly earnings today, beating analyst expectations. 
    The tech giant's revenue increased by 15% compared to the same period last year, 
    driven by robust iPhone sales and growing services revenue. The company's stock 
    price rose 5% in after-hours trading following the announcement.
    """
    st.text_area("Example:", example_text, height=100, disabled=True)
    if st.button("Use this example"):
        st.session_state.article_text = example_text
        st.experimental_rerun()