# 📰 BBC News Classification (Streamlit App)

A simple, clean, pastel-themed Streamlit app that classifies text into BBC news categories: **business**, **entertainment**, **politics**, **sport**, **tech**.

## Demo Features
- Single text input → prediction
- Probability table + bar chart
- Pastel UI via custom CSS
- Loads your existing `model.pkl` and `vectorizer`

## Project Structure
```
news-classifier/
├── app.py
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml
├── model.pkl
├── vectorizer
└── NLP_Text_Classification.ipynb
```

> Place `model.pkl` and `vectorizer` in the repository root (next to `app.py`).

## Run Locally
```bash
# 1. Create & activate a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push this folder to a public GitHub repo.
2. Go to Streamlit Community Cloud → New app → select your repo/branch.
3. Set `app.py` as the entrypoint.
4. Add the following **secrets / advanced settings** if needed:
   - **Python version**: 3.10+
5. Click **Deploy**.

## Notes
- This app expects a scikit-learn style pipeline: a **vectorizer** (e.g., `TfidfVectorizer`) and a **classifier** with `.predict()` (and ideally `.predict_proba()`).
- If your model has different class labels or order, update the `CATEGORIES` list in `app.py` accordingly.

## License
MIT
