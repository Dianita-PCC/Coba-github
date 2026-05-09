# MBG Streamlit App (No sklearn, No SMOTE)

## Isi
- `app.py` — aplikasi Streamlit untuk:
  - upload dataset `datasetmbg.csv` (separator default `|`)
  - preprocessing (slang/stopwords/elongation + optional stemming Sastrawi)
  - auto-label IndoRoBERTa
  - download `dataset_sentimen_berlabel.csv`
  - train & evaluate TF-IDF + Multinomial Naive Bayes (tanpa scikit-learn)
  - tampil metrik Accuracy/Precision/Recall/F1 + Confusion Matrix

## Cara jalanin
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Format dataset
Disarankan punya kolom `text` (atau `clean_text`).
Kalau file kamu pakai pemisah `|`, pilih separator `|` di sidebar (default sudah `|`).
