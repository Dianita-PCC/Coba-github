import re
import io
import numpy as np
import pandas as pd
import streamlit as st

from transformers import pipeline
from scipy.sparse import csr_matrix

# =========================
# 0) Streamlit setup
# =========================
st.set_page_config(page_title="Analisis Sentimen MBG)", layout="wide")
st.title("Analisis Sentimen MBG")

# =========================
# 1) Preprocessing (same approach as notebook)
# =========================
STOPWORDS_ID = {
    "yang","dan","di","ke","dari","untuk","pada","dengan","atau","itu","ini","aja","sih","nih",
    "gue","gw","aku","kamu","lu","loe","dia","mereka","kita","kami","anda","kak","bang",
    "nya","lah","deh","dong","kok","ya","yah","yahh","lahh","pun","juga","lagi","udah","sudah",
    "nggak","gak","ga","tak","tdk","tidak","bukan",
    "jadi","bisa","biar","supaya","agar","banget","bgt","sangat","amat",
    "tuh","tu","mah","kan","kayak","kaya","gini","gitu","sama","doang","cuma"
}

SLANG_MAP = {
    "gk":"gak","ga":"gak","gak":"gak","ngga":"gak","nggak":"gak","tdk":"tidak",
    "bgt":"banget","bngt":"banget","bgtt":"banget",
    "yg":"yang","dgn":"dengan","utk":"untuk","dr":"dari","tp":"tapi","krn":"karena",
    "emg":"memang","aja":"saja","sm":"sama","km":"kamu","sy":"saya",
    "udh":"sudah","udah":"sudah","blm":"belum","skrg":"sekarang",
    "org":"orang","anak2":"anak anak","krg":"kurang","bkn":"bukan"
}

RE_URL = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
RE_MENTION = re.compile(r"@\w+")
RE_HASHTAG = re.compile(r"#(\w+)")
RE_HTML = re.compile(r"&amp;|&lt;|&gt;|&quot;|&#39;")
RE_NONALNUM = re.compile(r"[^0-9a-zA-Z\s]")
RE_MULTI_SPACE = re.compile(r"\s+")
RE_REPEAT_CHARS = re.compile(r"(.)\1{2,}")     # huruf berulang 3+ kali
RE_LAUGH = re.compile(r"\b(wk+|wkwk+|haha+|hehe+|hihi+|xixix+)\b", re.IGNORECASE)

def normalize_slang(token: str) -> str:
    return SLANG_MAP.get(token, token)

def reduce_elongation(word: str) -> str:
    # contoh: "baguuuuus" -> "baguus" (mengurangi jadi max 2)
    return RE_REPEAT_CHARS.sub(r"\1\1", word)

@st.cache_resource
def load_stemmer():
    try:
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
        return StemmerFactory().create_stemmer()
    except Exception:
        return None

def basic_clean_competition(text: str, use_stemming: bool, stemmer) -> str:
    if text is None:
        return ""

    # lowercase
    text = str(text).lower()

    # unescape html sederhana
    text = RE_HTML.sub(" ", text)

    # hapus url, mention
    text = RE_URL.sub(" ", text)
    text = RE_MENTION.sub(" ", text)

    # hashtag: keep token-nya, buang '#'
    text = RE_HASHTAG.sub(r" \1 ", text)

    # ubah laughter jadi token khusus (biar konsisten)
    text = RE_LAUGH.sub(" LAUGH ", text)

    # normalisasi tanda baca berlebihan: !!!! -> !
    text = re.sub(r"([!?.,])\1{1,}", r"\1", text)

    # buang karakter aneh (sisain spasi)
    text = RE_NONALNUM.sub(" ", text)

    # normalisasi angka: bisa dibuang atau jadi token NUM
    text = re.sub(r"\b\d+\b", " NUM ", text)

    # rapihin spasi dulu
    text = RE_MULTI_SPACE.sub(" ", text).strip()

    # tokenisasi whitespace
    tokens = text.split()

    cleaned = []
    for tok in tokens:
        tok = normalize_slang(tok)
        tok = reduce_elongation(tok)

        # buang token terlalu pendek (kecuali "mbg" dll)
        if len(tok) <= 2 and tok not in {"mbg"}:
            continue

        # stopwords
        if tok in STOPWORDS_ID:
            continue

        cleaned.append(tok)

    text = " ".join(cleaned).strip()

    # stemming (opsional)
    if use_stemming and stemmer is not None and text:
        text = stemmer.stem(text)

    return text.strip()

# =========================
# 2) Model loader
# =========================
@st.cache_resource
def load_sentiment_pipeline(model_name: str):
    return pipeline("sentiment-analysis", model=model_name)

# =========================
# 3) TF-IDF + NB (no sklearn)
# =========================
def train_test_split_np(X, y, test_size=0.2, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    n_test = int(len(X) * test_size)
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def tokenize(text: str):
    return text.split()

def build_vocab(texts, min_df=2):
    df_counts = {}
    for t in texts:
        seen = set(tokenize(t))
        for tok in seen:
            df_counts[tok] = df_counts.get(tok, 0) + 1
    filtered_tokens = [tok for tok, cnt in df_counts.items() if cnt >= min_df]
    vocab = {tok: i for i, tok in enumerate(filtered_tokens)}
    return vocab, df_counts

def tfidf_matrix(texts, vocab, df_counts, n_docs_train):
    rows = []
    cols = []
    data = []

    for i, t in enumerate(texts):
        toks = tokenize(t)
        if not toks:
            continue
        tf = {}
        for tok in toks:
            j = vocab.get(tok)
            if j is None:
                continue
            tf[j] = tf.get(j, 0) + 1
        if not tf:
            continue
        for j, cnt in tf.items():
            rows.append(i)
            cols.append(j)
            data.append(cnt)

    if len(data) == 0:
        return csr_matrix((len(texts), len(vocab)), dtype=np.float64)

    tf_mat = csr_matrix(
        (np.array(data, dtype=np.float64),
         (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32))),
        shape=(len(texts), len(vocab))
    )

    # log-TF
    tf_mat.data = 1.0 + np.log(tf_mat.data)

    # idf
    idf = np.ones(len(vocab), dtype=np.float64)
    for tok, j in vocab.items():
        dfc = df_counts.get(tok, 0)
        idf[j] = np.log((n_docs_train + 1.0) / (dfc + 1.0)) + 1.0

    tf_mat = tf_mat.multiply(idf)

    # normalize rows
    row_norm = np.sqrt(tf_mat.multiply(tf_mat).sum(axis=1)).A1
    row_norm[row_norm == 0] = 1.0
    tf_mat = tf_mat.multiply(1.0 / row_norm[:, None])
    return tf_mat.tocsr()

class MultinomialNB_NoSklearn:
    def __init__(self, alpha=1.0):
        self.alpha = float(alpha)
        self.class_log_prior_ = None
        self.feature_log_prob_ = None
        self.classes_ = None

    def fit(self, X: csr_matrix, y: np.ndarray):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        class_count = np.zeros(n_classes, dtype=np.float64)
        feature_count = np.zeros((n_classes, n_features), dtype=np.float64)

        for ci, c in enumerate(self.classes_):
            X = X.tocsr()  # <-- pastikan CSR
            idx = np.where(y == c)[0]   # index integer
            Xc = X[idx]                #
            class_count[ci] = len(idx)
            feature_count[ci] = np.asarray(Xc.sum(axis=0)).ravel()

        self.class_log_prior_ = np.log(class_count / class_count.sum())

        smoothed_fc = feature_count + self.alpha
        smoothed_cc = smoothed_fc.sum(axis=1, keepdims=True)
        self.feature_log_prob_ = np.log(smoothed_fc / smoothed_cc)
        return self

    def predict_log_proba(self, X: csr_matrix):
        jll = X @ self.feature_log_prob_.T
        jll = jll + self.class_log_prior_
        amax = np.max(jll, axis=1, keepdims=True)
        lse = amax + np.log(np.sum(np.exp(jll - amax), axis=1, keepdims=True))
        return jll - lse

    def predict(self, X: csr_matrix):
        logp = self.predict_log_proba(X)
        idx = np.argmax(logp, axis=1)
        return self.classes_[idx]

def confusion_matrix_binary(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_true==1) & (y_pred==1)).sum())
    tn = int(((y_true==0) & (y_pred==0)).sum())
    fp = int(((y_true==0) & (y_pred==1)).sum())
    fn = int(((y_true==1) & (y_pred==0)).sum())
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}

def metrics_from_cm(cm):
    tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]
    total = tp + tn + fp + fn
    accuracy  = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1}

# =========================
# 4) Sidebar controls
# =========================
with st.sidebar:
    st.header("Pengaturan")
    sep = st.selectbox("Separator dataset", options=["|", ",", ";", "\t"], index=0)
    model_name = st.text_input(
        "Model IndoRoBERTa (HF)",
        value="w11wo/indonesian-roberta-base-sentiment-classifier"
    )
    use_stemming = st.checkbox("Stemming Sastrawi (opsional)", value=True)
    drop_neutral = st.checkbox("Kalau muncul neutral → buang barisnya", value=True)
    neutral_map = st.selectbox("Kalau tidak dibuang, neutral jadi:", options=["negative", "positive"], index=0)

    st.divider()
    st.subheader("Preview realtime saat labeling")
    print_every = st.slider("Replace preview setiap N teks", 0, 50, 10)
    alt_show = st.slider("Jumlah contoh ditampilkan", 2, 20, 6)

    st.divider()
    st.subheader("Training NB")
    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
    min_df = st.slider("min_df (TF-IDF vocab)", 1, 10, 2, 1)
    alpha = st.slider("alpha (Laplace smoothing)", 0.1, 3.0, 1.0, 0.1)
    seed = st.number_input("Random seed", value=42, step=1)

# =========================
# 5) Upload dataset
# =========================
uploaded = st.file_uploader("Upload dataset (contoh: datasetmbg.csv)", type=["csv", "txt"])
if uploaded is None:
    st.info("Upload file dulu untuk mulai.")
    st.stop()

raw = uploaded.read().decode("utf-8", errors="ignore")
df0 = pd.read_csv(io.StringIO(raw), sep=sep)

st.subheader("Preview data (raw)")
st.dataframe(df0.head(10), use_container_width=True)

# pilih kolom teks
default_col = "text" if "text" in df0.columns else ("clean_text" if "clean_text" in df0.columns else df0.columns[0])
text_col = st.selectbox("Pilih kolom teks", options=list(df0.columns), index=list(df0.columns).index(default_col))

colA, colB = st.columns([1, 1])

# =========================
# 6) Run preprocessing + labeling
# =========================
with colA:
    st.subheader("1) Preprocess + Auto-label")
    run_label = st.button("Jalankan Labeling", type="primary")

with colB:
    st.subheader("2) Train NB (setelah labeling)")
    run_train = st.button("Train & Evaluate NB")

# session state
if "df_labeled" not in st.session_state:
    st.session_state.df_labeled = None
if "nb_model" not in st.session_state:
    st.session_state.nb_model = None
if "nb_vocab" not in st.session_state:
    st.session_state.nb_vocab = None
if "nb_dfcounts" not in st.session_state:
    st.session_state.nb_dfcounts = None
if "nb_Xtrain_text" not in st.session_state:
    st.session_state.nb_Xtrain_text = None

if run_label:
    sent_pipe = load_sentiment_pipeline(model_name)
    stemmer = load_stemmer() if use_stemming else None

    df = df0.copy()
    df["raw_text"] = df[text_col].astype(str)

    # --- Preprocessing progress
    st.write("Preprocessing...")
    p_clean = st.progress(0.0, text="Preprocessing (cleaning)...")
    cleaned = []
    n = len(df)
    for i, t in enumerate(df["raw_text"].tolist(), start=1):
        cleaned.append(basic_clean_competition(t, use_stemming, stemmer))
        if i % max(1, n // 200) == 0 or i == n:
            p_clean.progress(i / n, text=f"Preprocessing (cleaning)... {i}/{n}")

    df["clean_text"] = cleaned
    df = df[df["clean_text"].str.len() > 0].copy()
    st.success(f"Preprocessing selesai. Baris valid: {len(df)} / {len(df0)}")

    # --- Labeling progress + replace preview
    st.write("Auto-labeling...")
    p_lab = st.progress(0.0, text="Analisis Sentimen...")
    preview_box = st.empty()

    results = []
    buf_pos, buf_neg = [], []
    texts = df["clean_text"].tolist()
    total = len(texts)

    for i, t in enumerate(texts, start=1):
        out = sent_pipe(t, truncation=True)[0]
        results.append(out)

        lbl = str(out["label"]).lower().strip()
        sc = float(out.get("score", np.nan))

        if lbl == "positive":
            buf_pos.append((sc, t))
        elif lbl == "negative":
            buf_neg.append((sc, t))

        if i % max(1, total // 200) == 0 or i == total:
            p_lab.progress(i / total, text=f"Analisis Sentimen... {i}/{total}")

        # replace preview setiap N
        if print_every > 0 and i % print_every == 0:
            take = max(1, alt_show // 2)
            pos_take = buf_pos[-take:]
            neg_take = buf_neg[-take:]

            lines = [f"Sample (replace) @ {i}/{total} — selang-seling POS/NEG"]
            shown, p, n_ = 0, 0, 0
            while shown < alt_show and (p < len(pos_take) or n_ < len(neg_take)):
                if p < len(pos_take) and shown < alt_show:
                    scp, tp = pos_take[p]
                    lines.append(f"[POS] ({scp:.3f}) {tp[:200]}")
                    p += 1; shown += 1
                if n_ < len(neg_take) and shown < alt_show:
                    scn, tn = neg_take[n_]
                    lines.append(f"[NEG] ({scn:.3f}) {tn[:200]}")
                    n_ += 1; shown += 1

            preview_box.text("\n".join(lines))

    df["sentiment"] = [str(r["label"]).lower().strip() for r in results]
    df["score"] = [float(r.get("score", np.nan)) for r in results]

    # handle neutral
    if drop_neutral:
        df = df[df["sentiment"].isin(["positive", "negative"])].copy()
    else:
        df["sentiment"] = df["sentiment"].replace({"neutral": neutral_map})
        df = df[df["sentiment"].isin(["positive", "negative"])].copy()

    label_map = {"negative": 0, "positive": 1}
    df["label"] = df["sentiment"].map(label_map).astype(int)

    st.session_state.df_labeled = df

    st.success("Labeling selesai!")
    st.subheader("Distribusi label")
    st.dataframe(df["sentiment"].value_counts().rename("count").to_frame(), use_container_width=True)

    st.subheader("Preview hasil")
    st.dataframe(df[["raw_text", "clean_text", "sentiment", "label", "score"]].head(30), use_container_width=True)

    # download labeled
    out_df = df[["clean_text", "sentiment", "label", "score"]].copy()
    csv_bytes = out_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download dataset_sentimen_berlabel.csv",
        data=csv_bytes,
        file_name="dataset_sentimen_berlabel.csv",
        mime="text/csv"
    )

# =========================
# 7) Train & Evaluate NB
# =========================
if run_train:
    df = st.session_state.df_labeled
    if df is None or len(df) < 10:
        st.error("Belum ada dataset berlabel. Jalankan Labeling dulu.")
        st.stop()

    st.write("Training NB...")
    X = df["clean_text"].to_numpy()
    y = df["label"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split_np(X, y, test_size=float(test_size), seed=int(seed))

    vocab, df_counts = build_vocab(X_train, min_df=int(min_df))
    if len(vocab) == 0:
        st.error("Vocab kosong (min_df terlalu tinggi atau teks terlalu sedikit). Turunkan min_df.")
        st.stop()

    Xtr = tfidf_matrix(X_train, vocab, df_counts, n_docs_train=len(X_train))
    Xte = tfidf_matrix(X_test, vocab, df_counts, n_docs_train=len(X_train))

    nb = MultinomialNB_NoSklearn(alpha=float(alpha))
    nb.fit(Xtr, y_train)
    pred = nb.predict(Xte)

    cm = confusion_matrix_binary(y_test, pred)
    m = metrics_from_cm(cm)

    metrics_table = pd.DataFrame([{
        "Accuracy": round(m["accuracy"], 4),
        "Precision": round(m["precision"], 4),
        "Recall": round(m["recall"], 4),
        "F1-Score": round(m["f1_score"], 4),
        "TP": cm["tp"], "TN": cm["tn"], "FP": cm["fp"], "FN": cm["fn"]
    }])

    st.session_state.nb_model = nb
    st.session_state.nb_vocab = vocab
    st.session_state.nb_dfcounts = df_counts
    st.session_state.nb_Xtrain_text = X_train

    st.subheader("Hasil Evaluasi (NB)")
    st.dataframe(metrics_table, use_container_width=True)

    st.subheader("Confusion Matrix (binary)")
    st.json(cm)

# =========================
# 8) Predict manual (NB) after trained
# =========================
st.divider()
st.subheader("Prediksi manual pakai NB (setelah Train)")

user_text = st.text_area("Tulis komentar:", value="Program MBG ini membantu banget, anak jadi lebih fokus belajar.")
if st.button("Prediksi dengan NB"):
    nb = st.session_state.nb_model
    vocab = st.session_state.nb_vocab
    df_counts = st.session_state.nb_dfcounts
    X_train_text = st.session_state.nb_Xtrain_text

    if nb is None or vocab is None or df_counts is None or X_train_text is None:
        st.error("Model NB belum ditrain. Klik Train & Evaluate NB dulu.")
        st.stop()

    stemmer = load_stemmer() if use_stemming else None
    clean = basic_clean_competition(user_text, use_stemming, stemmer)

    Xq = tfidf_matrix(np.array([clean]), vocab, df_counts, n_docs_train=len(X_train_text))
    p = int(nb.predict(Xq)[0])
    st.write("Clean text:", clean)
    st.success("Prediksi: " + ("positive" if p == 1 else "negative"))
