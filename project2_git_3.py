import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import re
from underthesea import word_tokenize, pos_tag
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix, hstack
from datetime import datetime
from text_resources import load_teen_dict, load_stopwords
import plotly.express as px
import textwrap

# ==========================================================
# 1. CACHED LOADERS
# ==========================================================

@st.cache_resource
def get_resources():
    teen_dict = load_teen_dict()
    stop_words = load_stopwords()
    return teen_dict, stop_words

teen_dict, stop_words = get_resources()

def load_models():

    vectorizer = joblib.load("tfidf_vectorizer.pkl")

    with open('tfidf_matrix.pkl', 'rb') as f:
        tfidf_matrix = pickle.load(f)

    with open("kmeans.pkl", "rb") as f:
        kmeans = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    ohe = joblib.load("onehot_encoder.pkl")

    imputer = joblib.load("imputer.pkl")

    with open("pca.pkl", "rb") as f:
        pca = pickle.load(f)

    return vectorizer, tfidf_matrix, kmeans, scaler, ohe, imputer, pca


@st.cache_data
def compute_clusters(df_cluster):
    # models are accessed from global scope:
    global scaler, kmeans, pca

    num_cols = ['age', 'mileage_km', 'min_price', 'max_price', 'log_price']

    X_scaled = scaler.transform(df_cluster[num_cols])
    df_cluster['cluster_label'] = kmeans.predict(X_scaled)

    pca_points = pca.transform(X_scaled)
    df_cluster['x'] = pca_points[:, 0]
    df_cluster['y'] = pca_points[:, 1]

    return df_cluster, num_cols

def load_raw_data():
    data = pd.read_excel('data_motobikes.xlsx').rename(columns={
        'Tiêu đề': 'title',
        'Địa chỉ': 'address',
        'Mô tả chi tiết': 'description',
        'Giá': 'price',
        'Khoảng giá min': 'min_price',
        'Khoảng giá max': 'max_price',
        'Thương hiệu': 'brand',
        'Dòng xe': 'model',
        'Năm đăng ký': 'registration_year',
        'Số Km đã đi': 'mileage_km',
        'Tình trạng': 'condition',
        'Loại xe': 'bike_type',
        'Dung tích xe': 'engine_capacity',
        'Xuất xứ': 'origin',
        'Chính sách bảo hành': 'warranty_policy',
        'Trọng lượng': 'weight'
    })
    return data

def clean_text(text): # tạo hàm xử lý text với text là chuỗi các từ

    text = str(text).lower()
    text = text.replace('\n', ' ')
    text = re.sub(r'[^a-zA-ZÀ-ỹ0-9\s]', '', text)
    text = re.sub(r'\b\w\b', '', text)

    # Teen-code normalization
    words = text.split()
    words = [teen_dict.get(w, w) for w in words]
    text = ' '.join(words)

    # Tokenize & POS filter
    tokenized = word_tokenize(text)
    pos_tagged_text = pos_tag(" ".join(tokenized))
    filtered_words = [word for word, tag in pos_tagged_text if tag != 'T']

    # Stopword removal
    clean_words = [word for word in filtered_words if word not in stop_words]

    # Return string (not list), same as df['content_clean_cosine']
    return " ".join(clean_words)

def clean_df_for_recommender(df):
    ### For numeric part of vector

    # clean price
    df['price'] = (
    df['price']
    .astype(str)
    .str.replace('[^0-9]', '', regex=True)   # chỉ giữ lại chữ số
    .replace('', np.nan)
    .astype(float)
)
    def parse_minmax_price(s):
        if pd.isna(s):
            return np.nan
        s = str(s).lower().replace("tr", "").replace(" ", "")
        try:
            return float(s) * 1_000_000
        except:
            return np.nan

    df['min_price'] = df['min_price'].apply(parse_minmax_price)
    df['max_price'] = df['max_price'].apply(parse_minmax_price)

    # Xác định num/ non-num cols để fill NA
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns

    # Fill NA (num -> median, non-num -> mode)
    # 1. Numeric imputation
    num_imputer = joblib.load('imputer.pkl')
    df[num_cols] = num_imputer.fit_transform(df[num_cols])

    # 2. Categorical imputation
    cat_imputer = SimpleImputer(strategy="most_frequent")
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

    # Thay thế các giá trị không rõ trong cột 'engine_capacity'
    df['engine_capacity'] = df['engine_capacity'].replace(
        ['Không biết rõ', 'Đang cập nhật', 'Nhật Bản'],
        'Unknown'
    )

    # Thay thế các giá trị không rõ trong cột 'origin', giữ nguyên nhóm "Bảo hành hãng" để xử lý text
    df['origin'] = df['origin'].replace(
        ['Đang cập nhật', 'Nước khác'],
        'Nước khác'
    )

    # Chuẩn hóa registration_year
    df['registration_year'] = (
        df['registration_year']
        .astype(str)
        .str.lower()
        .str.replace('trước năm', '1980', regex=False)
        .str.extract('(\d{4})')[0]
    )
    # Chuyển sang numeric, những giá trị không chuyển được sẽ thành NA
    df['registration_year'] = pd.to_numeric(df['registration_year'], errors='coerce')

    # Fill NA ban đầu
    df['registration_year'] = df['registration_year'].fillna(df['registration_year'].median())

    # Gắn giá trị bất hợp lệ thành NA
    df.loc[
        (df['registration_year'] < 1980) | (df['registration_year'] > 2025),
        'registration_year'
    ] = np.nan

    # Fill NA sau khi loại bất hợp lệ
    df['registration_year'] = df['registration_year'].fillna(df['registration_year'].median())

    # Thêm biến age
    current_year = datetime.now().year
    df['age'] = current_year - df['registration_year']

    # gom nhóm brand hiếm và tạo cột 'segment'
    brand_counts = df['brand'].value_counts()
    rare_brands = brand_counts[brand_counts < 50].index
    df['brand_grouped'] = df['brand'].replace(rare_brands, 'Hãng khác')

    def group_model(x):
        counts = x.value_counts()
        rare_models = counts[counts < 100].index
        return x.replace(rare_models, 'Dòng khác')

    df['model_grouped'] = df.groupby('brand_grouped')['model'].transform(group_model)
    df['segment'] = df['brand_grouped'] + '_' + df['model_grouped']

    # One hot encoding 'bike_type', 'engine_capacity'
    encoded = ohe.transform(df[['bike_type', 'engine_capacity']])
    encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(['bike_type', 'engine_capacity']))
    # merge back to original dataframe
    df = pd.concat([df, encoded_df], axis=1)

    # numeric features
    num_features = ['price','mileage_km','min_price','max_price','age', 'registration_year']
    # log normalize numeric features
    normalized_features = []
    for col in num_features:
        new_col = col + "_log"
        df[new_col] = np.log1p(df[col])
        normalized_features.append(new_col)

    # tạo feature brand_meanprice
    brand_mean_log = df.groupby('brand')['price_log'].mean().rename('brand_meanprice')
    df = df.merge(brand_mean_log, on='brand', how='left')
    normalized_features.append('brand_meanprice')

    # features to turn to a vector: 
    onehot_features = ohe.get_feature_names_out(['bike_type', 'engine_capacity']).tolist()
    num_features = onehot_features + normalized_features

    # Xử lý NaN (nếu có) để tạo dense vector cho việc tính toán cosine similarity lúc sau
    X_num = df[num_features].copy()

    # 1️⃣ Impute missing values
    # imputer = SimpleImputer(strategy="median")
    X_num_imputed = imputer.fit_transform(X_num)

    # 2️⃣ Scaling for num features
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X_num_imputed)

    ### For text part of vector
    # Ở đây đã load tfidf_matrix nên không xử lý phần text nữa

    ### Tạo vector đầu vào bằng cách kết hợp vector TF-IDF và array num col (X_num_scaled)
    # from scipy.sparse import csr_matrix, hstack
    # Chuyển array X_num_scaled thành matrix dạng sparse (ko store các giá trị 0)
    X_num_sparse = csr_matrix(X_num_scaled)

    # Ghép ma trận TF-IDF và ma trận X_num_sparse theo chiều ngang
    X_final = hstack([tfidf_matrix, X_num_sparse])

    return df, X_final

def clean_df_for_clustering(df_cluster):
    cols_drop = ['title', 'address', 'description', 'Href']
    df_cluster = df_cluster.drop(columns=[c for c in cols_drop if c in df_cluster.columns], errors='ignore')
    df_cluster = df_cluster.drop(columns=['warranty_policy', 'weight', 'condition'], errors='ignore')
    df_cluster = df_cluster.dropna()

    # Clean price
    df_cluster['price'] = (
        df_cluster['price'].astype(str)
        .str.replace('[^0-9]', '', regex=True)
        .replace('', np.nan).astype(float)
    )

    # Minimal cleaning df price for display
    if 'price' in df_cluster.columns:
        df_cluster['price'] = df_cluster['price'].astype(str).str.replace('[^0-9]', '', regex=True)
        df_cluster.loc[df_cluster['price'] == '', 'price'] = np.nan
        df_cluster['price'] = pd.to_numeric(df_cluster['price'], errors='coerce')

    # ensure registration_year numeric
    if 'registration_year' in df_cluster.columns:
        df_cluster['registration_year'] = (
            df_cluster['registration_year'].astype(str)
            .str.lower()
            .str.replace('trước năm', '1980', regex=False)
            .str.extract(r'(\d{4})')[0]
        )
        df_cluster['registration_year'] = pd.to_numeric(df_cluster['registration_year'], errors='coerce')
        df_cluster.loc[(df_cluster['registration_year'] < 1980) | (df_cluster['registration_year'] > 2025), 'registration_year'] = np.nan
    
    def parse_price(s):
        if pd.isna(s): return np.nan
        s = str(s).lower().replace("tr", "").replace(" ", "")
        try: return float(s) * 1_000_000
        except: return np.nan

    df_cluster['min_price'] = df_cluster['min_price'].apply(parse_price)
    df_cluster['max_price'] = df_cluster['max_price'].apply(parse_price)

    df_cluster = df_cluster[~(df_cluster['price'] == 0)]

    # Remove invalid engine_capacity
    df_cluster = df_cluster[~df_cluster['engine_capacity'].astype(str).str.contains("Nhật Bản", na=False)]

    # Clean origin
    df_cluster = df_cluster[~df_cluster['origin'].astype(str).str.contains('Bảo hành hãng', case=False, na=False)]
    df_cluster['origin'] = df_cluster['origin'].replace(['Đang cập nhật', 'Nước khác'], 'Nước khác')

    # Registration year
    df_cluster['registration_year'] = (
        df_cluster['registration_year'].astype(str)
        .str.lower()
        .str.replace('trước năm', '1980')
        .str.extract('(\d{4})')[0]
    ).astype(float)

    df_cluster.loc[(df_cluster['registration_year'] < 1980) | (df_cluster['registration_year'] > 2025),
            'registration_year'] = np.nan

    df_cluster["age"] = 2025 - df_cluster["registration_year"]

    # Log transforms
    numeric_cols = ['age', 'mileage_km', 'min_price', 'max_price', 'price']
    for c in numeric_cols:
        df_cluster[f"log_{c}"] = np.log1p(df_cluster[c])

    df_cluster = df_cluster.dropna(subset=numeric_cols)

    return df_cluster



# ==========================================================
# LOAD EVERYTHING (CACHED)
# ==========================================================
# 1) Load models
vectorizer, tfidf_matrix, kmeans, scaler, ohe, imputer, pca = load_models()

# 2) Load raw data
df_raw = load_raw_data()

# 3) Prepare recommender dataset
df_clean, X_final = clean_df_for_recommender(df_raw.copy())

# 4) Prepare clustering dataset
df_cluster = clean_df_for_clustering(df_raw.copy())
df_cluster, num_cols = compute_clusters(df_cluster)


# ==========================================================
# FUNCTIONS
# ==========================================================
def preprocess_user_input(price, min_price, max_price, mileage_km, registration_year):
    age = 2025 - registration_year
    log_price = np.log1p(price)
    X = np.array([[age, mileage_km, min_price, max_price, log_price]])
    return scaler.transform(X)

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def get_top_n_similar_by_content(df, X_final, title, top_n=5):
    """
    Given a bike title, return top N most similar bikes based on
    combined TF-IDF + numeric features vector.

    Params:
        df (DataFrame): cleaned dataframe returned from clean_df_for_recommender
        X_final (sparse matrix): combined feature matrix
        title (str): the selected bike title
        top_n (int): number of similar bikes to return

    Returns:
        df_recommend (DataFrame): rows of top-N similar bikes
        scores (list): similarity scores
    """

    # 1️⃣ Find the index of the selected bike
    matches = df.index[df['title'] == title]

    if len(matches) == 0:
        return None, []   # title not found

    idx = matches[0]

    # 2️⃣ Compute cosine similarity for this single item
    sims = cosine_similarity(X_final[idx], X_final).flatten()

    # 3️⃣ Sort by similarity (descending), ignore itself
    ranked_indices = np.argsort(sims)[::-1]

    # Remove itself
    ranked_indices = ranked_indices[ranked_indices != idx]

    # 4️⃣ Take top-N
    top_indices = ranked_indices[:top_n]
    top_scores = sims[top_indices]

    # 5️⃣ Return matching rows + scores
    df_recommend = df.iloc[top_indices].copy()
    df_recommend['similarity_score'] = top_scores

    return df_recommend, top_scores.tolist()

# helper: safe format number
def fmt_vnd(x):
    try:
        return f"{int(x):,} VNĐ"
    except:
        return '-'


# ==========================================================
# STREAMLIT PAGES
# ==========================================================
st.set_page_config(
    page_title="Hệ thống gợi ý xe máy tương tự và phân cụm xe máy",
    page_icon="🏍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.markdown("""
## Hệ thống gợi ý xe máy tương tự và phân cụm xe máy
""")

st.sidebar.markdown("""
### Thành viên nhóm 6
1. Vũ Thị Ngọc Anh
2. Nguyễn Phạm Quỳnh Anh
""")

st.sidebar.markdown("### Menu")   
menu = ["Giới thiệu", "Bài toán nghiệp vụ", "Đánh giá mô hình và Báo cáo",
        "Gợi ý mẫu xe tương tự", "Phân cụm phân khúc xe máy"]
page = st.sidebar.selectbox("", menu)  


# ==========================================================
# STYLES
# ==========================================================

BASE_CSS = """
<style>
:root{
  --accent-1: #ffde37;       /* Your yellow */
  --accent-2: #e5c620;       /* Slightly darker yellow for gradients */
  --muted: #4a4a4a;
  --card-bg: #fff7c2;        /* Soft light yellow background */
  --glass: rgba(255,255,255,0.55);
}

/* Background */
html, body {
  background: linear-gradient(180deg, #fff5a0 0%, #ffef73 100%);
  color: #000000 !important;
}

/* Header / hero section */
.header-hero {
  background: linear-gradient(90deg, var(--accent-1), var(--accent-2));
  padding: 22px;
  border-radius: 12px;
  color: #000000;
  font-weight: 600;
  margin-bottom: 18px;
  box-shadow: 0 6px 24px rgba(0,0,0,0.12);
}

/* Small muted text */
.small-muted {
  color: var(--muted);
  font-size: 13px;
}

/* Cards */
.card {
  background: var(--card-bg);
  padding: 14px;
  border-radius: 12px;
  box-shadow: 0 6px 18px rgba(0,0,0,0.10);
  color: #000000;
}

/* Base typography */
h1, h2, h3, h4, h5, h6, p, span, div {
  color: #000000 !important;
}

/* Bike title / subtitles */
.bike-title{
  font-size:18px;
  font-weight:700;
  margin-bottom:4px;
}

.bike-sub{
  font-size:13px;
  color:var(--muted);
  margin-bottom:6px;
}

/* Cluster cards */
.cluster-card{
  padding:18px;
  border-radius:12px;
  color:#000000;
  margin-bottom:12px;
  font-weight:600;
}

/* Cluster variants using your yellow palette */
.cluster-0{
  background:linear-gradient(135deg, #ffeb7a, #ffde37);
}
.cluster-1{
  background:linear-gradient(135deg, #ffe45c, #e5c620);
}
.cluster-2{
  background:linear-gradient(135deg, #fff1a1, #ffde37);
}
</style>
"""

st.markdown(BASE_CSS, unsafe_allow_html=True)


# ==========================================================
# PAGE CONTENT
# ==========================================================

if page == 'Giới thiệu':
    # st.title("Hệ thống gợi ý xe máy tương tự và phân cụm xe máy")
    st.markdown("""
        <h1 style='font-size:48px; font-weight:800; margin-bottom:8px;'>
            Hệ thống gợi ý xe máy tương tự và phân cụm xe máy
        </h1>
        <div style='width:90px; height:6px; background:#FF9A00; border-radius:3px; margin-bottom:24px;'></div>
    """, unsafe_allow_html=True)    
    st.image("xe_may_cu2.jpg")
    st.subheader("[Trang chủ Chợ Tốt](https://www.chotot.com/)")

        # Function for light yellow pad header
    def yellow_pad_header(text):
        st.markdown(f"""
            <div style="
                background: #FFF4C2;
                border-left: 6px solid #FFDE37;
                padding: 12px 18px;
                border-radius: 6px;
                font-size: 24px;
                font-weight: bold;
                color: #333;
                margin: 15px 0 10px 0;
            ">
                {text}
            </div>
        """, unsafe_allow_html=True)
    
    yellow_pad_header('Giới thiệu dự án')
    st.markdown('''Đây là dự án xây dựng hệ thống hỗ trợ **gợi ý mẫu xe máy tương tự** 
và **phân khúc xe máy bằng phương pháp phân cụm** trên nền tảng *Chợ Tốt* – 
trong khóa đồ án tốt nghiệp Data Science and Machine Learning 2024 lớp DL07_K308 của nhóm 6.

Thành viên nhóm gồm có:
1. Vũ Thị Ngọc Anh  
2. Nguyễn Phạm Quỳnh Anh
''')

    yellow_pad_header('Mục tiêu của dự án')
    st.markdown("""
    **1. Xây dựng mô hình đề xuất thông minh:**
    - Đề xuất các mẫu xe máy tương đồng cho một mẫu được chọn hoặc theo từ khóa tìm kiếm.
    - Kết hợp nhiều nguồn thông tin (thông số kỹ thuật, hình ảnh, mô tả, giá, đánh giá) để tăng độ chính xác.

    **2. Phân khúc thị trường xe máy:**
    - Phân loại sản phẩm theo nhóm theo tệp giá, tuổi xe, khoảng giá tối thiểu/tối đa.
    - Hỗ trợ định giá và xây dựng chiến lược marketing hiệu quả hơn.
    """)

    yellow_pad_header('Phân công công việc')
    st.write("""
    - **Xử lý dữ liệu:** Ngọc Anh và Quỳnh Anh  
    - **Gợi ý xe máy bằng Gensim:** Quỳnh Anh  
    - **Gợi ý xe máy bằng Cosine similarity:** Ngọc Anh  
    - **Phân khúc xe máy bằng phương pháp phân cụm:** Ngọc Anh  
    - **Làm slide:** Ngọc Anh và Quỳnh Anh  
    - **Giao diện Streamlit:** Quỳnh Anh
    """)
    
elif page == 'Bài toán nghiệp vụ':
    st.markdown("""
    <h1 style='font-size:48px; font-weight:800; margin-bottom:8px;'>
        Bài toán nghiệp vụ
    </h1>
    <div style='width:90px; height:6px; background:#FF9A00; border-radius:3px; margin-bottom:24px;'></div>
""", unsafe_allow_html=True)
    # Function for light yellow pad header
    def yellow_pad_header(text):
        st.markdown(f"""
            <div style="
                background: #FFF4C2;
                border-left: 6px solid #FFDE37;
                padding: 12px 18px;
                border-radius: 6px;
                font-size: 24px;
                font-weight: bold;
                color: #333;
                margin: 15px 0 10px 0;
            ">
                {text}
            </div>
        """, unsafe_allow_html=True)

    yellow_pad_header('Vấn đề nghiệp vụ')
    st.markdown("""
        - Người dùng gặp khó khăn khi tìm xe phù hợp trong hàng trăm lựa chọn.
        - Chưa có hệ thống gợi ý xe tương tự khi người dùng chọn một mẫu cụ thể hoặc tìm kiếm theo từ khóa.
        - Thị trường xe máy rất đa dạng → khó nhận diện các phân khúc rõ ràng.
        - Cần hệ thống gợi ý & phân khúc tự động để hỗ trợ người dùng và đội ngũ phân tích.""")

    yellow_pad_header('Bài toán đặt ra')
    st.markdown("""
        1. Xây dựng mô hình **Gợi ý xe tương tự**
        - Sử dụng các đặc trưng từ mô tả xe và thông số kỹ thuật
        - Gợi ý các mẫu xe tương tự với xe được chọn hoặc theo từ khóa tìm kiếm.
        &nbsp;
        2. Xây dựng mô hình **Phân khúc thị trường xe bằng phương pháp phân cụm**
        - Phân cụm thị trường xe máy dựa các đặc trưng giá xe, tuổi xe, số km đã chạy, khoảng giá tối thiểu, tối đa.
        - Giúp nhận diện và phân loại xe theo các phân khúc khác nhau.
                """)
    
    yellow_pad_header('Phạm vi triển khai')
    st.markdown("""
        **1. Tiền xử lý dữ liệu và chuẩn hóa**:  
            - Chuẩn hóa các thông số của xe.  
            - Làm sạch dữ liệu và chuẩn hóa trường thông tin cho mô hình.  
                
        **2. Trích xuất đặc trưng văn bản và tính độ tương đồng**:  
            - Sử dụng **TF-IDF Vectorizer** để mã hóa mô tả và thông tin kỹ thuật.  
            - Tính độ tương đồng bằng **gensim similarity** và **cosine similarity**.  
            - Chọn phương pháp cho **điểm cao hơn** và **nghĩa đúng hơn** để đưa vào hệ thống gợi ý.  
                
        **3. Phân cụm thị trường (Clustering)**:  
            - Thử nghiệm trên các thuật toán: KMeans, Bisecting KMeans, Agglomerative Clustering  
            - Đánh giá bằng inertia, silhouette score, tính diễn giải.  
            - Chọn **KMeans** vì có hiệu suất ổn định, dễ diễn giải và ranh giới cụm phù hợp hơn với dữ liệu.

        **4. Xây dựng GUI trên Streamlit**:  
            - Cho phép người dùng **chọn xe trong danh sách** hoặc **nhập mô tả xe** → trả về **danh sách mẫu xe tương tự có trong sàn**.  
            - Cho phép **nhập tên xe** → hiển thị **xe thuộc cụm/phân khúc nào**.
                """)

    yellow_pad_header('Thu thập dữ liệu')
    st.markdown("""
        - Bộ dữ liệu gồm **7.208 tin đăng** với **18 thuộc tính** (thương hiệu, dòng xe, số km, năm đăng ký, giá niêm yết, mô tả, v.v…) được thu thập từ nền tảng **Chợ Tốt** (trước ngày 01/07/2025).
        - Bộ dữ liệu bao gồm các thông tin sau:
            - **id**: số thứ tự của sản phẩm trong bộ dữ liệu  
            - **Tiêu đề**: tựa đề bài đăng bán sản phẩm  
            - **Giá**: giá bán của xe máy  
            - **Khoảng giá min**: giá sàn ước tính của xe máy  
            - **Khoảng giá max**: giá trần ước tính của xe máy  
            - **Địa chỉ**: địa chỉ giao dịch (phường, quận, thành phố Hồ Chí Minh)  
            - **Mô tả chi tiết**: mô tả thêm về sản phẩm — đặc điểm nổi bật, tình trạng, thông tin khác  
            - **Thương hiệu**: hãng sản xuất (Honda, Yamaha, Piaggio, SYM…)  
            - **Dòng xe**: dòng xe cụ thể (Air Blade, Vespa, Exciter, LEAD, Vario, …)  
            - **Năm đăng ký**: năm đăng ký lần đầu của xe  
            - **Số km đã đi**: số kilomet xe đã vận hành  
            - **Tình trạng**: tình trạng hiện tại (ví dụ: đã sử dụng)  
            - **Loại xe**: Xe số, Tay ga, Tay côn/Moto  
            - **Dung tích xe**: dung tích xi-lanh (ví dụ: Dưới 50cc, 50–100cc, 100–175cc, …)  
            - **Xuất xứ**: quốc gia sản xuất (Việt Nam, Đài Loan, Nhật Bản, ...)  
            - **Chính sách bảo hành**: thông tin bảo hành nếu có  
            - **Trọng lượng**: trọng lượng ước tính của xe  
            - **Href**: đường dẫn tới bài đăng sản phẩm 
                """)

elif page == 'Đánh giá mô hình và Báo cáo':
    st.markdown("""
    <h1 style='font-size:48px; font-weight:800; margin-bottom:8px;'>
        Đánh giá mô hình và Báo cáo
    </h1>
    <div style='width:90px; height:6px; background:#FF9A00; border-radius:3px; margin-bottom:24px;'></div>
""", unsafe_allow_html=True)
    
    # Function for light yellow pad header
    def yellow_pad_header(text):
        st.markdown(f"""
            <div style="
                background: #FFF4C2;
                border-left: 6px solid #FFDE37;
                padding: 12px 18px;
                border-radius: 6px;
                font-size: 24px;
                font-weight: bold;
                color: #333;
                margin: 15px 0 10px 0;
            ">
                {text}
            </div>
        """, unsafe_allow_html=True) 

    yellow_pad_header('Thống kê mô tả sơ bộ')


    st.markdown("""        
    Bộ dữ liệu gồm **7.208 tin đăng** với **18 thuộc tính** (thương hiệu, dòng xe, số km, năm đăng ký, giá niêm yết, mô tả…) được thu thập từ nền tảng **Chợ Tốt** (trước ngày 01/07/2025).  
                """)

    # Hiển thị 4 biểu đồ dạng lưới 2x2
    col1, col2 = st.columns(2)
    with col1:
        st.image("brand_grouped_count.png")
        st.image("age_bin_stats.png")

    with col2:
        st.image("price_bin_stats.png")
        st.image("mileage_bin_stats.png")

    yellow_pad_header('Mô hình gợi ý xe máy tương tự')

    # with open("data/data_motobikes.xlsx", "rb") as f:
    #     st.download_button(
    #         label="📥 Tải xuống dữ liệu xe máy (Excel)",
    #         data=f,
    #         file_name="data_motobikes.xlsx",
    #         mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    #     )

    st.markdown('#### 1. Hướng xử lý')
    st.write('''
             - Chuẩn hóa và làm sạch dữ liệu.
             - Chia khoảng một số đặc trưng kiểu số để tạo thêm các đặc trưng phân loại mới (khoảng giá, tình trạng dựa theo số km chạy, tuổi xe, dung tích xe)
             - Gom các đặc trưng phân loại thành biến text
             - Làm sạch text và tokenize, xây dựng ma trận tương đồng (sparse matrix) giữa các văn bản để đánh giá mức độ giống nhau
             - Tính độ tương đồng bằng gensim và cosine similarity
                 - Trường hợp 1: gợi ý xe theo id sản phẩm được chọn
                    - Người dùng chọn xe từ danh sách xe trong tập dữ liệu
                    - Dựa trên ma trận tương đồng, tìm các xe có similarity score cao nhất.
                    - Tính độ tương đồng trung bình giữa 5 mẫu gợi ý cho một mẫu, sau đó áp dụng cho 7000 mẫu trong tập dữ liệu và tính trung bình.

                 - Trường hợp 2: gợi ý xe theo cụm từ khóa tìm kiểm (vd: “honda vision xanh dưới 15 triệu”)
                    - Người dùng nhập từ khóa tìm kiếm. 
                    - Xử lý từ khóa và chuyển từ khóa thành vector số dựa trên từ điển và TF-IDF
                    - Tính độ tương đồng giữa từ khóa và tất cả xe trong dữ liệu. 
                    - Sắp xếp và lấy ra 5 xe gợi ý phù hợp nhất.
                    - Cho danh sách 10 cụm từ khóa tìm kiếm. Tính độ tương đồng trung bình giữa 5 mẫu gợi ý cho một mẫu, sau đó áp dụng cho 10 cụm từ trên và tính trung bình
             ''')
    
    st.markdown('#### 2. Kết quả')
    st.write('Giữa 02 mô hình Gensim và Cosine similarity, Cosine similarity, trong cả 2 trường hợp chọn xe có sẵn hoặc tìm bằng từ khóa, cho điểm tương đồng trung bình cao hơn so với Gensim và cho các gợi ý sát nghĩa hơn Gensim.\nMô hình dùng để dự đoán xe trong ứng dụng này là Cosine similarity.') 

    yellow_pad_header('Mô hình phân khúc xe máy')
    
    st.markdown('#### 1. Xử lý dữ liệu')
    st.write('Dữ liệu được làm sạch, các đặc trưng biến số liên tục như giá, khoảng giá thấp nhất, lớn nhất, tuổi xe, số km đã đi được chọn để tạo mô hình phân cụm')

    st.markdown('#### 2. Phân cụm bằng các phương pháp khác nhau')
    st.write('''
    Mô hình phân cụm được xây dựng trên 02 môi trường: máy học truyền thống (sci-kit learn) và PySpark.
    - Máy học truyền thống: KMeans, Bisect Kmeans, Agglomerative clustering
    - PySpark: Kmeans, Bisecting Kmeans, GMM.

    ''')

    st.markdown('#### 3. Kết quả')


    st.markdown('''
    Số cụm được tạo thành trên mô hình máy học truyền thống: **03 cụm**
    Số cụm được tạo thành trên PySpark: **02 cụm**''')
    st.image("silhoutte_sklearn.png")                

    st.markdown('''      
    KMeans trên môi trường máy học truyền thống cho kết quả silhoutte score cao nhất và kết quả phân cụm dễ diễn giải hơn.
    
    **Phân loại phân khúc xe**:                
    1/ Cụm 0: Phân khúc Xe Phổ Thông – Trung cấp (Mid-range Popular Motorcycles): Xe tuổi trung bình, giá vừa phải, phù hợp đại đa số người mua.   
    2/ Cụm 1: Phân khúc Xe Cao Cấp – Premium / High-end Motorcycles: Tiêu biểu là các dòng SH, Vespa cao cấp, phân khối lớn, xe mới chạy ít.          
    3/ Cụm 2: Phân khúc Xe Cũ – Tiết Kiệm (Budget Used Motorcycles): Giá rẻ nhất, xe tuổi cao, chạy nhiều — phù hợp khách cần xe rẻ để di chuyển cơ bản.
    ''')


    st.write('''Trong 3 mô hình phân cụm KMeans, Bisect KMeans và Agglomerate thì KMeans với k = 3 cho kết quả phân cụm tốt nhất.
            nên mô hình phân cụm xe được sử dụng trong ứng dụng này là KMeans với k = 3.''')

    st.markdown('#### 4. Thống kê theo từng cụm:')

    st.write('Trực quan hóa')
    st.image('pca_clusters.png')

    cluster_summary = (
        df_cluster.groupby('cluster_label')
        .agg(
            count=('cluster_label', 'size'),
            avg_price=('price', 'mean'),
            avg_age=('age', 'mean'),
            avg_mileage=('mileage_km', 'mean')
        )
        .sort_values('cluster_label')
    )


    # Rename the index (cluster_label → Nhãn cụm xe)
    cluster_summary = cluster_summary.rename_axis("Nhãn cụm xe")

    # Rename columns
    cluster_summary = cluster_summary.rename(columns={
        "count": "Số lượng (xe)",
        "avg_price": "Giá trung bình (VND)",
        "avg_age": "Tuổi trung bình (năm)",
        "avg_mileage": "Số km trung bình (km)"
    })

    # Format số nguyên và thêm dấu phẩy
    cluster_summary["Giá trung bình (VND)"] = (
        cluster_summary["Giá trung bình (VND)"]
            .round(0).astype(int)
            .map(lambda x: f"{x:,}")
    )

    cluster_summary["Số km trung bình (km)"] = (
        cluster_summary["Số km trung bình (km)"]
            .round(0).astype(int)
            .map(lambda x: f"{x:,}")
    )

    st.dataframe(cluster_summary, width='stretch')


elif page == "Gợi ý mẫu xe tương tự":
    # Main page header
    st.markdown("""
    <h1 style='font-size:48px; font-weight:800; margin-bottom:8px;'>
        Gợi ý mẫu xe tương tự
    </h1>
    <div style='width:90px; height:6px; background:#FF9A00; border-radius:3px; margin-bottom:24px;'></div>
    """, unsafe_allow_html=True)

    # Prepare data + vector
    df_clean, X_final = df_clean, X_final

    # Styling and helpers
    def yellow_pad_header(text):
        st.markdown(f"""
            <div style="
                background: #FFF4C2;
                border-left: 6px solid #FFDE37;
                padding: 12px 18px;
                border-radius: 6px;
                font-size: 24px;
                font-weight: bold;
                color: #333;
                margin: 15px 0 10px 0;
            ">
                {text}
            </div>
        """, unsafe_allow_html=True)

    st.markdown("""
        <style>
        .card {
            border-radius: 10px;
            padding: 14px 16px;
            margin: 8px 0;
            border: 1px solid #eee;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
            background-color: #ffffff;
        }
        .bike-title {
            font-size: 16px;
            font-weight: 700;
            margin-bottom: 4px;
        }
        .bike-sub {
            font-size: 13px;
            color: #666666;
        }
        .small-muted {
            font-size: 12px;
            color: #777777;
        }
        </style>
    """, unsafe_allow_html=True)

    def display_bike_card(row):
        title = row.get('title', 'N/A')
        price = fmt_vnd(row.get('price', None))
        brand = row.get('brand', '-')
        model = row.get('model', '-')
        km = row.get('mileage_km', '-')
        year = row.get('registration_year', '-')
        year_shown = int(year) if str(year).isdigit() else year
        origin = row.get('origin', '-')
        desc = row.get('description', '')

        card_html = f"""
        <div class='card'>
            <div style='display:flex; gap:14px; align-items:center'>
                <div style='flex:1'>
                    <div class='bike-title'>{title}</div>
                    <div class='bike-sub'>{brand} — {model} • {origin}</div>
                    <div style='margin-top:6px'>{textwrap.shorten(str(desc), width=220)}</div>
                </div>
                <div style='text-align:right; min-width:150px'>
                    <div style='font-weight:700; font-size:16px'>{price}</div>
                    <div class='small-muted' style='margin-top:8px'>
                        Số km: {km}<br/>Năm: {year_shown}
                    </div>
                </div>
            </div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)

    # ✅ Main interaction
    yellow_pad_header("Gợi ý theo mẫu có sẵn")

    titles_list = df_clean['title'].unique().tolist()
    selected = st.selectbox("Chọn 1 mẫu trong danh sách", titles_list)

    if st.button("Gợi ý"):
        with st.spinner("🔎 Đang tìm mẫu tương tự..."):
            df_top, scores = get_top_n_similar_by_content(
                df_clean,
                X_final,
                title=selected,
                top_n=5
            )

        if df_top is None or len(df_top) == 0:
            st.warning("Không tìm thấy kết quả — kiểm tra lại dữ liệu.")
        else:
            st.success(f"Đã tìm {len(df_top)} mẫu tương tự ✅")

            # ✅ Show selected bike
            st.markdown("#### 🔶 Mẫu bạn đã chọn")
            selected_row = df_clean[df_clean["title"] == selected].iloc[0]
            display_bike_card(selected_row)

            # ✅ Show recommendations
            st.markdown("#### 🔶 Các mẫu tương tự")
            for _, row in df_top.iterrows():
                display_bike_card(row)
                st.caption(f"Similarity score: {row['similarity_score']:.3f}")

        
    # theo từ khóa
    yellow_pad_header("Tìm kiếm theo từ khóa")

    q = st.text_input('Nhập từ khóa tìm kiếm, ví dụ: "honda vision 2014 màu đỏ"')
    top_k = st.selectbox('Số kết quả trả về', [1, 3, 5, 10])

    if st.button('Tìm kiếm') and q.strip():
        with st.spinner('Đang xử lý từ khóa...'):

            # 1) Clean query like training data
            q_clean = clean_text(q)

            # 2) Vectorize cleaned query
            q_vec_tfidf = vectorizer.transform([q_clean])

            # 3) Pad numeric features with zeros
            num_dim = X_final.shape[1] - q_vec_tfidf.shape[1]
            q_num_zeros = np.zeros((1, num_dim))

            # 4) Combine TF-IDF + numeric zeros
            q_vec = hstack([q_vec_tfidf, q_num_zeros])

            # 5) Compute similarity
            sim_scores = cosine_similarity(q_vec, X_final).flatten()

            # 6) Select top results
            idxs = sim_scores.argsort()[::-1][:top_k]

            # 7) Select rows from cleaned DF
            res_df = df_clean.iloc[idxs].copy()
            res_df['similarity_score'] = sim_scores[idxs]

        st.success(f'Kết quả top {top_k} cho: "{q}"')

        # 8) Display
        for _, row in res_df.iterrows():
            display_bike_card(row)
            st.caption(f"Similarity score: {row['similarity_score']:.3f}")


elif page == "Phân cụm phân khúc xe máy":
    # Main page header
    st.markdown("""
    <h1 style='font-size:48px; font-weight:800; margin-bottom:8px;'>
        Phân cụm phân khúc xe máy
    </h1>
    <div style='width:90px; height:6px; background:#FF9A00; border-radius:3px; margin-bottom:24px;'></div>
    """, unsafe_allow_html=True)

    # Yellow pad header function (keep for consistent style)
    def yellow_pad_header(text):
        st.markdown(f"""
            <div style="
                background: #FFF4C2;
                border-left: 6px solid #FFDE37;
                padding: 12px 18px;
                border-radius: 6px;
                font-size: 24px;
                font-weight: bold;
                color: #333;
                margin: 15px 0 10px 0;
            ">
                {text}
            </div>
        """, unsafe_allow_html=True)

    # ----- Card CSS -----
    st.markdown("""
        <style>
        .card {
            border-radius: 10px;
            padding: 14px 16px;
            margin: 8px 0;
            border: 1px solid #eee;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
            background-color: #ffffff;
        }
        .bike-title {
            font-size: 16px;
            font-weight: 700;
            margin-bottom: 4px;
        }
        .bike-sub {
            font-size: 13px;
            color: #666666;
        }
        .small-muted {
            font-size: 12px;
            color: #777777;
        }
        </style>
    """, unsafe_allow_html=True)

    
    # ----- Main interaction -----
    yellow_pad_header("Phân cụm xe mới")


    st.markdown("""
    <style>
    .cluster-card {
        padding: 15px;
        border-radius: 12px;
        margin-top: 10px;
        margin-bottom: 15px;
        color: white;
        font-size: 16px;
    }
    .cluster-0 {
        background: linear-gradient(135deg, #4CAF50, #2E7D32);
    }
    .cluster-1 {
        background: linear-gradient(135deg, #1976D2, #0D47A1);
    }
    .cluster-2 {
        background: linear-gradient(135deg, #F57C00, #E65100);
    }
    .cluster-title {
        font-size: 20px;
        font-weight: 700;
        margin-bottom: 5px;
    }
    .cluster-desc {
        font-size: 15px;
    }
    </style>
    """, unsafe_allow_html=True)


    # # st.markdown("""
    # # - **Cụm 0:** Xe phổ thông – giá rẻ, tuổi xe trung bình, số km trung bình → **nhóm chiếm thị phần lớn nhất**.
    # # - **Cụm 1:** Xe mới hơn – giá cao hơn, chạy ít hơn → **phân khúc chất lượng tốt**.
    # # - **Cụm 2:** Xe rất cũ – giá thấp nhất, số km cực cao → **phân khúc xuống cấp hoặc dữ liệu km không chính xác**.
    # # """)

    # bike_labels = {0: "Xe phổ thông giá rẻ, tuổi xe trung bình",
    #                1: "Xe tương đối mới, phân khúc cao cấp",
    #                2: "Xe cũ xuống cấp hoặc dữ liệu cung cấp không chính xác"}


    # ====== CLUSTER NEW BIKE ======
    st.write("Vui lòng nhập các thông số của xe cần xác định")

    col1, col2 = st.columns(2)

    with col1:
        price = st.number_input("Giá xe (VND)", min_value=500_000, step=100_000, value=1_000_000)
        min_price = st.number_input("Khoảng giá min", min_value=500_000, step=100_000, value=800_000)

    with col2:
        max_price = st.number_input("Khoảng giá max", min_value=500_000, step=100_000, value=1_200_000)
        mileage_km = st.number_input("Số km đã đi", min_value=0, step=100, value=1000)

    registration_year = st.slider("Năm đăng ký", 1980, 2025)

    if st.button("Phân cụm"):
        X_new = preprocess_user_input(price, min_price, max_price, mileage_km, registration_year)
        cluster = int(kmeans.predict(X_new)[0])
        st.success(f"Xe thuộc cụm số **{cluster}**")

        # st.write(bike_labels.get(cluster, "Không có mô tả cho cụm này"))

        # ======= HIỂN THỊ THẺ GIẢI THÍCH CỤM THEO KẾT QUẢ =======

        cluster_cards = {
            0: """
                <div class="cluster-card cluster-0">
                    <div class="cluster-title">Cụm 0 – Xe phổ thông giá rẻ</div>
                    <div class="cluster-desc">
                        Giá thấp – tuổi xe trung bình – số km chạy vừa phải.<br>
                        Phân khúc xe phổ thông, phù hợp đa số người mua.
                    </div>
                </div>
            """,
            1: """
                <div class="cluster-card cluster-1">
                    <div class="cluster-title">Cụm 1 – Xe cao cấp / ít chạy</div>
                    <div class="cluster-desc">
                        Xe mới – ít km – giá cao.<br>
                        Các dòng SH, Vespa, xe cao cấp, tình trạng tốt.
                    </div>
                </div>
            """,
            2: """
                <div class="cluster-card cluster-2">
                    <div class="cluster-title">Cụm 2 – Xe cũ / giá rẻ</div>
                    <div class="cluster-desc">
                        Giá thấp nhất – km rất cao – tuổi xe lớn.<br>
                        Phân khúc xe đã cũ hoặc có dấu hiệu xuống cấp.
                    </div>
                </div>
            """
                }
        st.markdown("""
        <style>
        .cluster-card {
            border-radius: 10px;
            padding: 14px 18px;
            margin: 10px 0;
            border: 1px solid #E5C600;
            box-shadow: 0 2px 4px rgba(0,0,0,0.08);
            color: #000000;
        }

        .cluster-title {
            font-weight: 700;
            font-size: 18px;
            margin-bottom: 6px;
            color: #000000;
        }

        .cluster-desc {
            font-size: 14px;
            color: #000000;
            line-height: 1.4;
        }

        /* ✅ Different yellow for each cluster */
        .cluster-0 { background: #FFF7A6; }
        .cluster-1 { background: #FFE970; }
        .cluster-2 { background: #FFDE37; }
        </style>
        """, unsafe_allow_html=True)

        # Hiển thị card tương ứng
        st.markdown(cluster_cards.get(cluster, ""), unsafe_allow_html=True)

