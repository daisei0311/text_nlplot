import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import MeCab
import nlplot
from plotly.offline import iplot
import json
import logging
import plotly.express as px
import numpy as np

home_dir = "/home/dk/project/aitarentee/opwork_gui/src/"

def is_text_column_by_pattern(series):
    """正規表現でテキスト列を判定"""
    if series.dtype == 'object':
        sample_texts = series.dropna().head(10)  # 最初の10行をサンプルとして利用
        text_pattern = re.compile(r'[A-Za-z\s]')  # 英字またはスペースを含むかチェック
        matches = sample_texts.apply(lambda x: bool(text_pattern.search(str(x))))
        return matches.mean() > 0.7
    return False

def is_text_column_by_length(series, min_length=100):
    """文字列の長さを基準にテキスト列を判定"""
    if series.dtype == 'object':
        avg_length = series.apply(lambda x: len(str(x)) if isinstance(x, str) else 0).mean()
        return avg_length > min_length
    return False

def is_text_column(series):
    """列がテキストデータかどうかを判定"""
    text_ratio = series.apply(lambda x: isinstance(x, str)).mean()
    return text_ratio > 0.8

def is_text_column_combined(series):
    """複数の基準を組み合わせてテキスト列を判定"""
    return is_text_column(series) and is_text_column_by_length(series)

def setup_logger():
    if "logger" not in st.session_state:
        logger = logging.getLogger("streamlit_logger")
        logger.setLevel(logging.DEBUG)

        if not logger.hasHandlers():
            streamlit_handler = logging.StreamHandler()
            streamlit_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            streamlit_handler.setFormatter(formatter)
            logger.addHandler(streamlit_handler)

            file_handler = logging.FileHandler("app.log")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        st.session_state.logger = logger
    else:
        logger = st.session_state.logger
    return logger

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"\W", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def unique_preprocess_text(text):
    with open('removewords.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    common_words = data.get("common_words", [])
    for word in common_words:
        text = re.sub(word, "", text)
    return text.strip()

def mecab_text(text, pos_list=["名詞"]):
    parsed_text = tagger.parse(text)
    words = []
    for line in parsed_text.splitlines():
        if line == "EOS":
            break
        parts = line.split("\t")
        if len(parts) > 1:
            # featureをカンマで分割して最初の要素を取得（品詞）
            feature = parts[1].split(",")
            pos = feature[0]
            if pos in pos_list:
                words.append(parts[0])
    return words

def calculate_median_fixed(period):
    if "未満" in period:
        return 1.5
    elif "以上" in period:
        return 20
    else:
        start, end = map(int, period.replace("年", "").split("～"))
        return (start + end) / 2

def extract_text_data(df, pos_list=["名詞"]):
    """
    データ整形処理を行う関数。
    :param df: 元のデータフレーム
    :param pos_list: 抽出する品詞のリスト（例: ["名詞", "動詞", "形容詞"]）
    :return: 整形済みデータフレーム
    """
    try:
        text_columns_combined = [col for col in df.columns if is_text_column_combined(df[col])]
        print("Text Columns (Combined):", text_columns_combined)

        new_cols = []
        for text_col in text_columns_combined:
            df[text_col] = df[text_col].apply(preprocess_text).apply(unique_preprocess_text)
            # mecab_textにpos_listを渡す
            df[text_col + '_mecab'] = df[text_col].apply(lambda x: mecab_text(x, pos_list))
            df[text_col + '_mecab'] = df[text_col + '_mecab'].apply(
                lambda x: " ".join(x) if isinstance(x, list) else x
            )

            new_cols.append(text_col)
            new_cols.append(text_col + '_mecab')

        df_text = df[new_cols]

        st.session_state.cleaned = True
        st.success("データ整形が完了しました！")
        logger.info("Data cleaning completed successfully.")
        return df_text

    except Exception as e:
        logger.error(f"Data cleaning failed: {e}")
        raise ValueError(f"Data cleaning failed: {e}")

# ロガー & MeCab Tagger
logger = setup_logger()
tagger = MeCab.Tagger("-r /etc/mecabrc -d /var/lib/mecab/dic/debian")

# ページ設定
st.set_page_config(page_title="テキストデータ整形ツール", layout="wide")

# セッションステート初期化
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame()
if "cleaned" not in st.session_state:
    st.session_state.cleaned = False

# ---- ここがポイント：stopwords がまだ存在しない場合のみ JSON から読み込む ----
if "stopwords" not in st.session_state:
    with open(f'removewords.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
    # removewordsキーからデフォルトのストップワードを取得
    removewords = data.get("removewords", [])
    # セッションステートに格納（ユーザーが後で追加/削除可能）
    st.session_state.stopwords = removewords

# サイドバー: ページ選択
page = st.sidebar.selectbox(
    "ページを選択してください",
    [
        "データ & ストップワード設定", 
        "共起ネットワーク", 
        "N-gram Bar",
        "N-gram Treemap"
    ]
)

# =========================================================
# 1) データ & ストップワード設定 ページ
# =========================================================
if page == "データ & ストップワード設定":
    st.title("データアップロード & ストップワード設定ページ")

    # CSV アップロード
    uploaded_files = st.sidebar.file_uploader(
        "CSVファイルをアップロードしてください", 
        accept_multiple_files=True, 
        type="csv"
    )
    
    # 抽出したい品詞を選択
    pos_options = ["名詞", "動詞", "形容詞"]
    selected_pos = st.sidebar.multiselect("抽出する品詞を選択してください", options=pos_options, default=["名詞"])
    
    # アップロードされたファイルがあれば読み込み
    if uploaded_files:
        dfs = [pd.read_csv(file) for file in uploaded_files]
        df = pd.concat(dfs).reset_index(drop=True)
        st.session_state.df = df
        st.session_state.cleaned = False  # 新しいファイルをアップロードしたら再度False

        st.success("ファイルが正常に読み込まれました！")

    # データ整形ボタン
    if st.sidebar.button("データ整形"):
        if st.session_state.df.empty:
            st.error("最初にCSVファイルをアップロードしてください。")
        else:
            try:

                df_cleaned = extract_text_data(st.session_state.df, pos_list=selected_pos)
                # ファイルに保存せず、セッションステートに保存
                st.session_state.df_cleaned = df_cleaned
                st.session_state.cleaned = True
            except ValueError as e:
                st.error(f"エラーが発生しました: {e}")

    # 整形前後のデータを表示
    if "df_cleaned" not in st.session_state or not st.session_state.cleaned:
        if not st.session_state.df.empty:
            st.write("整形前データ:")
            st.dataframe(st.session_state.df.head(10))
    else:
        st.write("整形後データ:")
        st.dataframe(st.session_state.df_cleaned.head(10))

    # ----------------------------------------------
    # ストップワード設定
    # ----------------------------------------------
    st.write("## ストップワードの設定")
    st.write("現在のストップワード:")
    st.write(st.session_state.stopwords)

    new_stopword = st.text_input("新しいストップワードを追加:")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("追加"):
            if new_stopword and new_stopword not in st.session_state.stopwords:
                st.session_state.stopwords.append(new_stopword)
                st.success(f"'{new_stopword}' をストップワードに追加しました！")

    with col2:
        if st.session_state.stopwords:
            remove_stopword = st.selectbox("削除するストップワードを選択:", st.session_state.stopwords)
            if st.button("削除"):
                if remove_stopword in st.session_state.stopwords:
                    st.session_state.stopwords.remove(remove_stopword)
                    st.success(f"'{remove_stopword}' をストップワードから削除しました！")


# =========================================================
# 2) 共起ネットワーク ページ
# =========================================================
elif page == "共起ネットワーク":
    st.title("共起ネットワーク＆サンバーストチャート ページ")

    # すでに整形済みデータがセッションステートに存在するか確認
    if "df_cleaned" in st.session_state and st.session_state.cleaned:
        dfg = st.session_state.df_cleaned
        st.write("整形済みデータを使用します。")
    else:
        st.error("まず「データ & ストップワード設定」ページでデータを整形してください。")
        dfg = pd.DataFrame()
        
    if dfg.empty:
        st.error("整形済みデータが存在しません。")
        st.info("先に『データ & ストップワード設定』ページでデータ整形を完了してください。")
    else:
        # 形態素解析済みの列を探す
        mecab_cols = [col for col in dfg.columns if 'mecab' in col]
        if not mecab_cols:
            st.warning("形態素解析済みの列が見つかりません。")
        else:
            st.sidebar.write("### 共起ネットワーク 設定")
            target_col = st.sidebar.selectbox("解析対象の列を選択してください:", mecab_cols)
            top_n = st.sidebar.slider("トップNの単語数を選択してください (top_n):", 5, 100, 30)
            min_freq = st.sidebar.slider("最小頻度を選択してください (min_freq):", 1, 10, 5)
            min_edge_frequency = st.sidebar.slider("最小エッジ頻度を選択してください (min_edge_frequency):", 1, 10, 4)

            if st.button("Build 共起ネットワーク"):
                try:
                    logger.info(f"{top_n},{min_freq},{min_edge_frequency},{target_col},{dfg.shape}")
                    
                    npt = nlplot.NLPlot(dfg, target_col=target_col)
                    # 共起ネットワーク用のストップワード
                    # npt.get_stopword()で頻出単語を取得し、セッションのstopwordsと合体
                    stopwords_base = npt.get_stopword(top_n=top_n, min_freq=min_freq)
                    stopwords = stopwords_base + st.session_state.stopwords

                    with st.spinner("Building the 共起ネットワーク..."):
                        npt.build_graph(
                            stopwords=stopwords, 
                            min_edge_frequency=min_edge_frequency
                        )
                        fig_co_network = npt.co_network(
                            title="共起ネットワーク",
                            sizing=100,
                            node_size="adjacency_frequency",
                            color_palette="hls",
                            width=1200,
                            height=800,
                            save=False,
                        )
                        st.plotly_chart(fig_co_network)

                        logger.info("共起ネットワーク visualization rendered successfully.")
                                        # ここから下でサンバーストチャートを追加表示
                    with st.spinner("Building Sunburst chart from co-occurrence data..."):
                        # Sunburst可視化 (N=1固定で共起した単語群を可視化するイメージ)
                        fig_sun_co = npt.sunburst(
                            title='sunburst chart',
                            colorscale=True,
                            color_continuous_scale='Oryel',
                            width=1000,
                            height=800,
                            save=False
                        )
                        st.plotly_chart(fig_sun_co)

                except Exception as e:
                    st.error(f"エラーが発生しました: {e}")
                    logger.error(f"共起ネットワーク build failed: {e}")


# =========================================================
# 3) N-gram Bar ページ
# =========================================================
elif page == "N-gram Bar":
    st.title("N-gram Bar ページ")

    if "df_cleaned" in st.session_state and st.session_state.cleaned:
        dfg = st.session_state.df_cleaned
        st.write("整形済みデータを使用します。")
    else:
        st.error("まず「データ & ストップワード設定」ページでデータを整形してください。")
        dfg = pd.DataFrame()

    if dfg.empty:
        st.error("整形済みデータが存在しません。")
        st.info("先に『データ & ストップワード設定』ページでデータ整形を完了してください。")
    else:
        mecab_cols = [col for col in dfg.columns if 'mecab' in col]
        if not mecab_cols:
            st.warning("形態素解析済みの列が見つかりません。")
        else:
            st.sidebar.write("### N-gram Visualization 設定")
            target_col = st.sidebar.selectbox("解析対象の列を選択してください:", mecab_cols)
            n_gram = st.sidebar.slider("N-gram (1=unigram, 2=bigram, 3=trigram):", 1, 3, 1)
            top_ngram = st.sidebar.slider("表示するトップ数 (top_n):", 5, 100, 30)

            top_co = 30  
            min_freq = 3

            if st.button("Build N-gram Bar"):
                try:
                    with st.spinner("Generating N-gram charts..."):
                        npt = nlplot.NLPlot(dfg, target_col=target_col)

                        # N-gram 用のストップワード
                        stopwords_base = npt.get_stopword(top_n=top_co, min_freq=min_freq)
                        stopwords_ngram = stopwords_base + st.session_state.stopwords

                        # Bar Chart
                        fig_bar = npt.bar_ngram(
                            ngram=n_gram,
                            top_n=top_ngram,
                            stopwords=stopwords_ngram,
                            title=f"N-gram Bar Chart (n={n_gram})",
                            width=1000,
                            height=600
                        )
                        st.plotly_chart(fig_bar)

                        logger.info("N-gram Bar charts rendered successfully.")
                except Exception as e:
                    st.error(f"エラーが発生しました: {e}")
                    logger.error(f"N-gram chart generation failed: {e}")


# =========================================================
# 4) N-gram Treemap ページ
# =========================================================
elif page == "N-gram Treemap":
    st.title("N-gram Treemap ページ")

    if "df_cleaned" in st.session_state and st.session_state.cleaned:
        dfg = st.session_state.df_cleaned
        st.write("整形済みデータを使用します。")
    else:
        st.error("まず「データ & ストップワード設定」ページでデータを整形してください。")
        dfg = pd.DataFrame()

    if dfg.empty:
        st.error("整形済みデータが存在しません。")
        st.info("先に『データ & ストップワード設定』ページでデータ整形を完了してください。")
    else:
        mecab_cols = [col for col in dfg.columns if 'mecab' in col]
        if not mecab_cols:
            st.warning("形態素解析済みの列が見つかりません。")
        else:
            st.sidebar.write("### Treemap Visualization 設定")
            target_col = st.sidebar.selectbox("解析対象の列を選択してください:", mecab_cols)
            n_gram = st.sidebar.slider("N-gram (1=unigram, 2=bigram, 3=trigram):", 1, 3, 1)
            top_ngram = st.sidebar.slider("表示するトップ数 (top_n):", 5, 100, 30)

            top_co = 30  
            min_freq = 3
            
            if st.button("Build N-gram Treemap"):
                try:
                    with st.spinner("Generating N-gram Treemap..."):
                        npt = nlplot.NLPlot(dfg, target_col=target_col)

                        # N-gram 用のストップワード
                        stopwords_base = npt.get_stopword(top_n=top_co, min_freq=min_freq)
                        stopwords_ngram = stopwords_base + st.session_state.stopwords

                        # Treemap
                        fig_tree = npt.treemap(
                            ngram=n_gram,
                            top_n=top_ngram,
                            stopwords=stopwords_ngram,
                            title=f"N-gram Treemap (n={n_gram})",
                            width=1000,
                            height=600
                        )
                        st.plotly_chart(fig_tree)

                        logger.info("N-gram Treemap rendered successfully.")
                except Exception as e:
                    st.error(f"エラーが発生しました: {e}")
                    logger.error(f"N-gram Treemap generation failed: {e}")
