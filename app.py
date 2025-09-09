import streamlit as st
import joblib
import re
import nltk
import ssl

# Fix SSL certificate issues for NLTK downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# --- NLTK Data Download ---
@st.cache_data
def download_nltk_data():
    """Download required NLTK data and return stopwords set."""
    try:
        # Download required NLTK data
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        
        # Import after download
        from nltk.corpus import stopwords
        return set(stopwords.words('english'))
    except Exception as e:
        st.error(f"Error downloading NLTK data: {e}")
        # Return a basic set of common English stopwords as fallback
        return {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 
                'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 
                'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'am', 
                'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 
                'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'shall'}

# Download NLTK data and get stopwords
stop_words = download_nltk_data()

# --- Load Model and Vectorizer ---
@st.cache_resource
def load_model_and_vectorizer():
    """Loads the saved SVM model and TF-IDF vectorizer."""
    try:
        model = joblib.load('tuned_svm_linear_model.pkl')
        vectorizer = joblib.load('vectorizer.pkl')
        return model, vectorizer
    except FileNotFoundError:
        st.error("Model or vectorizer file not found. Please make sure 'tuned_svm_linear_model.pkl' and 'vectorizer.pkl' are in the same directory.")
        return None, None

model, tfidf_vectorizer = load_model_and_vectorizer()

# --- Text Preprocessing Function ---
def preprocess_text(text):
    """Cleans and preprocesses the input text."""
    # 1. Convert to lowercase
    text = text.lower()
    # 2. Remove punctuation and extra spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 3. Tokenization with better fallback
    try:
        from nltk.tokenize import word_tokenize
        tokens = word_tokenize(text)
    except Exception as e:
        # Fallback to simple split if NLTK tokenization fails
        st.warning("Using simple tokenization due to NLTK issue")
        tokens = text.split()
    
    # 4. Remove stopwords (handle case where stop_words might be None)
    if stop_words:
        tokens = [token for token in tokens if token not in stop_words]
    else:
        # Basic stopword removal if stop_words failed to load
        basic_stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        tokens = [token for token in tokens if token not in basic_stopwords]
    
    # 5. Join tokens back to string
    return ' '.join(tokens)

# --- Custom CSS for Styling ---
st.markdown("""
<style>
/* Main page background */
.main {
    background-color: #FFFFFF; /* White */
}
/* Title color */
h1 {
    color: #800000; /* Maroon */
}
/* Button style */
.stButton>button {
    background-color: #000080; /* Navy Blue */
    color: white;
    border-radius: 5px;
    border: none;
    padding: 10px 20px;
}
.stButton>button:hover {
    background-color: #800000; /* Maroon on hover */
    color: white;
}
/* Text area style */
.stTextArea textarea {
    border: 2px solid #000080; /* Navy Blue border */
    border-radius: 5px;
}
/* Success box for prediction */
.st-success {
    background-color: rgba(0, 0, 128, 0.1); /* Light Navy Blue */
    border: 1px solid #000080; /* Navy Blue */
    border-radius: 5px;
    color: #800000; /* Maroon text */
}
</style>
""", unsafe_allow_html=True)

# --- Streamlit App Interface ---
st.title('📰 BBC News Article Classifier')

st.markdown("""
Enter the text of a news article below, and the model will predict its category:
**Business, Entertainment, Politics, Sport, or Tech.**
""")

# User input text area
user_input = st.text_area("Enter article text here:", height=250)

# Prediction button
if st.button('Classify Article'):
    if model and tfidf_vectorizer:
        if user_input.strip():
            # 1. Preprocess the input
            processed_input = preprocess_text(user_input)

            # 2. Vectorize the processed text
            vectorized_input = tfidf_vectorizer.transform([processed_input])

            # 3. Make a prediction
            prediction = model.predict(vectorized_input)
            prediction_proba = model.predict_proba(vectorized_input)
            confidence = max(prediction_proba[0]) * 100

            # 4. Map prediction to category name with icons and colors
            category_mapping = {
                0: ('Business', '💼', '#1f77b4'),
                1: ('Entertainment', '🎬', '#ff7f0e'), 
                2: ('Politics', '🏛️', '#2ca02c'),
                3: ('Sport', '⚽', '#d62728'),
                4: ('Tech', '💻', '#9467bd')
            }
            
            predicted_category, icon, color = category_mapping.get(prediction[0], ('Unknown', '❓', '#gray'))

            # 5. Display the result with enhanced styling
            st.markdown(f"""
            <div style="background-color: {color}20; border-left: 5px solid {color}; padding: 15px; border-radius: 5px; margin: 10px 0;">
                <h3 style="color: {color}; margin: 0;">{icon} Predicted Category: {predicted_category}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Show confidence score only if available
            if show_confidence:
                st.info(f"🎯 **Confidence:** {confidence:.1f}%")
            else:
                st.info("🎯 **Model:** SVM Linear Classifier")
        else:
            st.warning("Please enter some text to classify.")
    else:
        st.error("Model or vectorizer could not be loaded. Please check your files.")
