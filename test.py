import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import json
import re
from datetime import datetime, timedelta
import time
from textblob import TextBlob
from googleapiclient.discovery import build
from newsapi import NewsApiClient
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
from snownlp import SnowNLP
import jieba


# 頁面設定
st.set_page_config(page_title="新聞情感分析AI系統", page_icon="📰", layout="wide")

# 取得新聞的函式
def get_news(query, from_date, to_date, language='zh', sort_by='publishedAt'):
    try:
        #st.write(f"DEBUG: NewsAPI 查詢參數 from={from_date}, to={to_date}, language={language}, query={query}")
        # 初始化 NewsAPI
        newsapi = NewsApiClient(api_key=news_api_key)
        
        # 抓取新聞
        all_articles = newsapi.get_everything(
            q=query,
            from_param=from_date,
            to=to_date,
            language=language,
            sort_by=sort_by
        )
        
        # 整理資料
        articles = []
        for article in all_articles['articles']:
            #if articles:
                #st.write("DEBUG: 取得新聞發佈時間分布：")
                #st.write(pd.Series([a['發佈時間'] for a in articles]).value_counts())
            articles.append({
                '標題': article['title'],
                '來源': article['source']['name'],
                '作者': article['author'],
                '發佈時間': article['publishedAt'],
                '連結': article['url'],
                '內容': article['content'],
                '描述': article['description']
            })
        
        return pd.DataFrame(articles)
    except Exception as e:
        st.error(f"擷取新聞時發生錯誤：{e}")
        return pd.DataFrame()

# 搜尋 YouTube 影片的函式
def search_youtube_videos(query, language='zh', max_results=10):
    try:
        # 根據語言自動調整查詢關鍵字
        if language == 'zh':
            search_query = f"{query} 中文"
        else:
            search_query = f"{query} english"
        youtube = build('youtube', 'v3', developerKey=youtube_api_key)
        search_response = youtube.search().list(
            q=search_query,
            part='id,snippet',
            maxResults=max_results,
            type='video'
        ).execute()
        videos = []
        for item in search_response['items']:
            video_id = item['id'].get('videoId')
            if not video_id:
                continue
            videos.append({
                '標題': item['snippet']['title'],
                '頻道': item['snippet']['channelTitle'],
                '發佈時間': item['snippet']['publishedAt'],
                '影片ID': video_id,
                '連結': f'https://www.youtube.com/watch?v={video_id}'
            })
        return pd.DataFrame(videos)
    except Exception as e:
        st.error(f"搜尋 YouTube 影片時發生錯誤：{e}")
        return pd.DataFrame()

# 取得 YouTube 影片字幕的函式
def get_youtube_transcript(video_id, languages=['zh', 'en']):
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
        transcript_text = ' '.join([t['text'] for t in transcript_list])
        return transcript_text
    except Exception as e:
        st.warning(f"無法擷取此影片的字幕：{e}")
        return ""
# 取得 YouTube 影片留言的函式
def get_youtube_comments(video_id, api_key, max_results=100):
    youtube = build('youtube', 'v3', developerKey=api_key)
    comments = []
    try:
        response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100 if max_results > 100 else max_results,
            textFormat="plainText"
        ).execute()
        count = 0
        while response and count < max_results:
            for item in response['items']:
                snippet = item['snippet']['topLevelComment']['snippet']
                comments.append({
                    '留言內容': snippet['textDisplay'],
                    '留言時間': snippet['publishedAt'],
                    '影片ID': video_id
                })
                count += 1
                if count >= max_results:
                    break
            if 'nextPageToken' in response and count < max_results:
                response = youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    maxResults=100 if max_results - count > 100 else max_results - count,
                    pageToken=response['nextPageToken'],
                    textFormat="plainText"
                ).execute()
            else:
                break
    except Exception as e:
        st.warning(f"無法取得評論：{e}")
    return comments

# 使用 TextBlob 進行情感分析的函式
def analyze_sentiment(text):
    if not text or pd.isna(text):
        return {'polarity': 0, 'subjectivity': 0, 'sentiment': '中立'}
    
    # 簡單用正則判斷，如果「文字中含有中文」就用 SnowNLP，否則用 TextBlob
    if re.search(r'[\u4e00-\u9fff]', text):
        # 以 SnowNLP 處理中文；SnowNLP 的 sentiment 介於 [0,1]，>0.5 視為正面，<0.5 視為負面
        s = SnowNLP(text)
        polarity = (s.sentiments - 0.5) * 2  # 轉換成 [-1,1] 的尺度，方便畫圖或比較
        subjectivity = None  # SnowNLP 沒有明確主觀度指標，先設為 None
        if s.sentiments > 0.5:
            sentiment = '正面'
        elif s.sentiments < 0.5:
            sentiment = '負面'
        else:
            sentiment = '中立'
    else:
        # 如果沒有中文，就使用 TextBlob 分析（處理英文）
        tb = TextBlob(text)
        polarity = tb.sentiment.polarity
        subjectivity = tb.sentiment.subjectivity
        if polarity > 0:
            sentiment = '正面'
        elif polarity < 0:
            sentiment = '負面'
        else:
            sentiment = '中立'
    
    return {
        'polarity': polarity,
        'subjectivity': subjectivity,
        'sentiment': sentiment
    }

# 視覺化情感分佈的函式
def visualize_sentiment(df, language):
    if df.empty:
        st.warning("沒有可視覺化的資料")
        return

    # ----- 1. 繪製情感分佈圓餅圖 -----
    sentiment_counts = df['sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['情感', '數量']
    fig1 = px.pie(
        sentiment_counts,
        values='數量',
        names='情感',
        title='情感分佈',
        color='情感',
        color_discrete_map={'正面': 'green', '中立': 'blue', '負面': 'red'}
    )
    st.plotly_chart(fig1)

    # ----- 2. 繪製不同來源之情感長條圖 -----
    if '來源' in df.columns:
        top_sources = df.groupby(['來源', 'sentiment']).size().reset_index(name='count')
        fig2 = px.bar(
            top_sources,
            x='來源',
            y='count',
            color='sentiment',
            title='不同來源之情感分析',
            color_discrete_map={'正面': 'green', '中立': 'blue', '負面': 'red'}
        )
        st.plotly_chart(fig2)

    # ----- 3. 繪製時間序列趨勢圖：依日期看正、中、負的變化 -----
    #st.write("DEBUG: DataFrame columns:", df.columns.tolist())
    if '發佈時間' in df.columns:
        #st.write("DEBUG: Sample '發佈時間' values:", df['發佈時間'].head().tolist())
        try:
            # ...existing code...
            df['日期'] = pd.to_datetime(df['發佈時間'], errors='coerce').dt.date
            df = df[~df['日期'].isna()]
            df['sentiment'] = df['sentiment'].astype(str).str.strip()
            df = df[df['sentiment'].isin(['正面', '中立', '負面'])]

            # 先 groupby
            date_sentiment = df.groupby(['日期', 'sentiment']).size().reset_index(name='count')

            # 再顯示
            #st.dataframe(df[['日期', 'sentiment']].head(20))
            #st.dataframe(df['sentiment'].value_counts())
            #st.dataframe(date_sentiment)

            #st.write("DEBUG: date_sentiment shape:", date_sentiment.shape)
            #st.write("DEBUG: date_sentiment head:", date_sentiment.head(10))

            #st.write("DEBUG: 過濾後資料筆數：", len(df))
            #st.write("DEBUG: 日期欄 NaN 數量：", df['日期'].isna().sum())
            #st.write("DEBUG: Sample 'sentiment' values:", df['sentiment'].head(10).tolist())
            #st.write("DEBUG: sentiment dtype:", df['sentiment'].dtype)
            #st.write("DEBUG: sentiment unique:", df['sentiment'].unique())
            #st.write("DEBUG: sentiment value counts:", df['sentiment'].value_counts())

            if not date_sentiment.empty:
                fig3 = px.line(
                    date_sentiment,
                    x='日期',
                    y='count',
                    color='sentiment',
                    title='情感趨勢（依日期）',
                    color_discrete_map={'正面': 'green', '中立': 'blue', '負面': 'red'}
                )
                st.plotly_chart(fig3)
            else:
                st.info("DEBUG: 沒有情感趨勢資料可供繪製。")
            # ...existing code...
        except Exception as e:
            st.error(f"日期轉換或繪圖時發生錯誤：{e}")
    else:
        st.info("DEBUG: DataFrame 中沒有 '發佈時間' 欄。")

    # ----- 4. 產生詞雲 -----
    # 4.1 準備 text_corpus
    # 影片詞雲
    text_corpus = ""
    if language == 'zh':
        parts = []
        if '標題' in df.columns:
            parts.append(" ".join(df['標題'].dropna().astype(str).tolist()))
        if '描述' in df.columns:
            parts.append(" ".join(df['描述'].dropna().astype(str).tolist()))
        if '內容' in df.columns:
            parts.append(" ".join(df['內容'].dropna().astype(str).tolist()))
        # 新增留言摘要
        if '留言摘要' in df.columns:
            parts.append(" ".join(df['留言摘要'].dropna().astype(str).tolist()))
        raw = " ".join(parts)
        chinese_text = "".join(re.findall(r'[\u4e00-\u9fff]+', raw))
        text_corpus = chinese_text
    else:  # language == 'en'
        parts = []
        if '描述' in df.columns:
            parts.append(" ".join(df['描述'].dropna().astype(str).tolist()))
        if '內容' in df.columns:
            parts.append(" ".join(df['內容'].dropna().astype(str).tolist()))
        # 新增留言摘要
        if '留言摘要' in df.columns:
            parts.append(" ".join(df['留言摘要'].dropna().astype(str).tolist()))
        text_corpus = " ".join(parts)
    # 影片詞雲結束.

    # 新聞詞雲.
    text_corpus = ""
    if language == 'zh':
        # 把 標題、描述、內容 裡的中文都串起來
        parts = []
        if '標題' in df.columns:
            parts.append(" ".join(df['標題'].dropna().astype(str).tolist()))
        if '描述' in df.columns:
            parts.append(" ".join(df['描述'].dropna().astype(str).tolist()))
        if '內容' in df.columns:
            parts.append(" ".join(df['內容'].dropna().astype(str).tolist()))
        raw = " ".join(parts)

        # 只留下所有連續的漢字
        chinese_text = "".join(re.findall(r'[\u4e00-\u9fff]+', raw))
        text_corpus = chinese_text
    else:  # language == 'en'
        # 英文就抓描述+內容，直接拼起來
        parts = []
        if '描述' in df.columns:
            parts.append(" ".join(df['描述'].dropna().astype(str).tolist()))
        if '內容' in df.columns:
            parts.append(" ".join(df['內容'].dropna().astype(str).tolist()))
        text_corpus = " ".join(parts)

    # 4.2 如果有 text_corpus，繪製詞雲
    if text_corpus:
        #st.write(f"DEBUG: text_corpus 長度={len(text_corpus)}")
        #st.write(f"DEBUG: 前100字={text_corpus[:100]}")
        st.subheader("文字詞雲展示")
        if language == 'zh':
            font_path = "NotoSansTC-Regular.ttf"
            tokens = " ".join(jieba.cut(text_corpus))
            wc = WordCloud(
                font_path=font_path,
                width=800,
                height=400,
                background_color='white',
                max_words=100,
                stopwords=None,
                collocations=False
            ).generate(tokens)
        else:
            # 英文不需要指定 font_path
            wc = WordCloud(
                width=800,
                height=400,
                background_color='white',
                max_words=100,
                stopwords=None,
                collocations=False
            ).generate(text_corpus)

        fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
        ax_wc.imshow(wc, interpolation='bilinear')
        ax_wc.axis('off')
        st.pyplot(fig_wc)
    else:
        st.info("無可用文字生成詞雲。")
    # 新聞詞雲結束.

def build_analysis_context(all_results):
    context = ""
    for source_type, df in all_results.items():
        if not df.empty:
            context += f"【{source_type}分析摘要】\n"
            context += f"總數：{len(df)}\n"
            context += f"正面：{sum(df['sentiment']=='正面')}\n"
            context += f"中立：{sum(df['sentiment']=='中立')}\n"
            context += f"負面：{sum(df['sentiment']=='負面')}\n"
            context += f"平均極性：{df['polarity'].mean():.2f}\n"
            context += f"平均主觀性：{df['subjectivity'].mean():.2f}\n"
            context += "\n"
    return context

# 顯示分析結果的函式
def display_results(df, source_type):
    if df.empty:
        st.warning(f"找不到任何 {source_type}")
        return
    
    # 顯示基本資訊
    st.subheader(f"{source_type} 分析結果")
    st.write(f"共分析 {len(df)} 個 {source_type}")
    
    # 顯示情感分佈
    sentiment_distribution = df['sentiment'].value_counts()
    st.write("情感分佈：")
    st.write(sentiment_distribution)
    
    # 顯示資料表
    st.subheader(f"{source_type} 資料表")
    st.dataframe(df)
    
    # 情感視覺化
    st.subheader(f"{source_type} 情感視覺化")
    visualize_sentiment(df, language)
    
    # 顯示情感最極端的項目
    st.subheader(f"{source_type} 中情感最正向的項目")
    most_positive = df.loc[df['polarity'].idxmax()]
    st.write(f"標題：{most_positive.get('標題', '無')}")
    st.write(f"極性指數：{most_positive['polarity']:.4f}")
    st.write(f"連結：{most_positive.get('連結', '無')}")
    
    st.subheader(f"{source_type} 中情感最負向的項目")
    most_negative = df.loc[df['polarity'].idxmin()]
    st.write(f"標題：{most_negative.get('標題', '無')}")
    st.write(f"極性指數：{most_negative['polarity']:.4f}")
    st.write(f"連結：{most_negative.get('連結', '無')}")

# 主介面
st.title("📰 新聞情感分析AI系統")
st.markdown("""
本系統可搜尋並分析網路上的新聞內容（包含文字與影片），顯示其情感傾向。
""")

# --- 側邊欄 ---
with st.sidebar:
    st.header("API 設定")
    llm_api_key = st.text_input("LLM API 金鑰", type="password")
    news_api_key = st.text_input("NewsAPI 金鑰", type="password")
    youtube_api_key = st.text_input("YouTube API 金鑰", type="password")
    st.header("搜尋設定")
    query = st.text_input("搜尋關鍵字（中英文皆可）")
    col1, col2 = st.columns(2)
    with col1:
        days_ago = st.number_input("搜尋至幾天前的新聞", min_value=1, max_value=30, value=7)
    with col2:
        language = st.selectbox("語言", options=['zh', 'en'], index=0)
    max_results = st.slider("最多顯示影片數量", min_value=5, max_value=50, value=10)
    search_type = st.multiselect("搜尋類型", ['文字新聞', 'YouTube 影片'], default=['文字新聞'])
    analyze_button = st.button("開始分析")

# --- 切換語言時自動清空分析結果 ---
if "last_language" not in st.session_state:
    st.session_state["last_language"] = language
if language != st.session_state["last_language"]:
    st.session_state["all_results"] = {}
    st.session_state.messages = []
    st.session_state["last_language"] = language

# --- AI 助理小對話框 ---
with st.expander("💬 AI 助理 (Gemini)"):
    gemini_api_key = llm_api_key
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        else:
            st.chat_message("assistant").write(msg["content"])
    analysis_context = ""
    if "all_results" in st.session_state:
        analysis_context = build_analysis_context(st.session_state["all_results"])
    user_input = st.chat_input("請輸入您的問題...")
    if user_input and gemini_api_key:
        st.session_state.messages.append({"role": "user", "content": user_input})
        prompt = f"以下是分析資料摘要：\n{analysis_context}\n\n使用者提問：{user_input}"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={gemini_api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            reply = data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            reply = f"API 請求失敗: {e}"
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.rerun()

# --- 分析流程（只做資料處理與存檔，不顯示） ---
if analyze_button and query:
    st.session_state["all_results"] = {}
    st.session_state.messages = []
    all_results = {}
    to_date = datetime.now().strftime('%Y-%m-%d')
    from_date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')

    # 文字新聞
    if '文字新聞' in search_type and news_api_key:
        news_df = get_news(query, from_date, to_date, language)
        if not news_df.empty:
            sentiment_results = []
            progress_bar = st.progress(0)
            for i, (_, row) in enumerate(news_df.iterrows()):
                text_to_analyze = f"{row['標題']} {row['描述']} {row['內容']}"
                sentiment_data = analyze_sentiment(text_to_analyze)
                sentiment_results.append(sentiment_data)
                progress_bar.progress((i + 1) / len(news_df))
            progress_bar.empty()
            news_df['polarity'] = [r['polarity'] for r in sentiment_results]
            news_df['subjectivity'] = [r['subjectivity'] for r in sentiment_results]
            news_df['sentiment'] = [r['sentiment'] for r in sentiment_results]
            all_results['文字新聞'] = news_df

    # YouTube 影片
    if 'YouTube 影片' in search_type and youtube_api_key:
        videos_df = search_youtube_videos(query, language, max_results)
        if not videos_df.empty:
            progress_bar = st.progress(0)
            progress_text = st.empty()
            all_comments = []
            for i, (_, row) in enumerate(videos_df.iterrows()):
                progress_text.text(f"正在分析第 {i+1} 部影片，共 {len(videos_df)} 部")
                comments = get_youtube_comments(row['影片ID'], youtube_api_key, max_results=100)
                for c in comments:
                    c['影片標題'] = row['標題']
                    c['影片發佈時間'] = row['發佈時間']
                all_comments.extend(comments)
                progress_bar.progress((i + 1) / len(videos_df))
            progress_bar.empty()
            progress_text.empty()
            if all_comments:
                comments_df = pd.DataFrame(all_comments)
                # 先改欄位名稱
                comments_df.rename(columns={'留言內容': '內容', '留言時間': '發佈時間'}, inplace=True)
                sentiment_results = [analyze_sentiment(c) for c in comments_df['內容']]
                comments_df['polarity'] = [r['polarity'] for r in sentiment_results]
                comments_df['subjectivity'] = [r['subjectivity'] for r in sentiment_results]
                comments_df['sentiment'] = [r['sentiment'] for r in sentiment_results]
                all_results['YouTube 影片'] = comments_df

    st.session_state["all_results"] = all_results

# --- 顯示流程（唯一一份，顯示分頁、圖表、下載） ---
if "all_results" in st.session_state and st.session_state["all_results"]:
    all_results = st.session_state["all_results"]
    tabs = st.tabs([t for t in all_results.keys()] + ["總結"])
    for idx, (source_type, df) in enumerate(all_results.items()):
        with tabs[idx]:
            display_results(df, source_type)
    with tabs[-1]:
        st.header("情感分析總結")
        summary_data = []
        for source_type, df in all_results.items():
            if not df.empty:
                source_summary = {
                    '類型': source_type,
                    '總數': len(df),
                    '正面': sum(df['sentiment'] == '正面'),
                    '中立': sum(df['sentiment'] == '中立'),
                    '負面': sum(df['sentiment'] == '負面'),
                    '平均極性': df['polarity'].mean(),
                    '平均主觀性': df['subjectivity'].mean()
                }
                summary_data.append(source_summary)
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            st.write("總結資料：")
            st.dataframe(summary_df)
            # ...（你的長條圖、結論等照原本放這裡）...
        else:
            st.warning("目前沒有任何資料可供總結分析。")
    # 下載功能
    st.subheader("下載分析結果")
    for source_type, df in all_results.items():
        if not df.empty:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"下載 {source_type} 結果 (CSV)",
                data=csv,
                file_name=f"分析結果_{source_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
else:
    st.info("請輸入搜尋關鍵字並點擊「開始分析」以啟動系統。")
    with st.expander("使用說明"):
        st.markdown("""
            ### 🧠 如何使用「新聞情感分析 AI 系統」

            這是一套整合新聞、YouTube 留言、情感分析與 AI 助理的互動式系統，幫助你快速掌握熱門話題的公眾情緒傾向。

            ---

            #### 🔐 1. API 設定（必填才能使用）

            請先準備以下三組金鑰，並輸入於左側邊欄：

            - **NewsAPI 金鑰**：用來搜尋新聞內容（[註冊 NewsAPI](https://newsapi.org) 並取得 API Key）。
            - **YouTube API 金鑰**：用來搜尋 YouTube 影片並擷取留言（請到 [Google Cloud Console](https://console.cloud.google.com/) 建立專案並啟用 YouTube Data API）。
            - **LLM（AI 助理）API 金鑰**：可選。支援 Gemini AI 助理互動功能。請填入 Google Gemini 的 API 金鑰（如使用 OpenAI 可自行改程式支援）。

            ⚠️ **請妥善保存金鑰，避免洩漏或誤用。**

            ---

            #### 🔍 2. 搜尋設定（在左邊欄調整）

            - **搜尋關鍵字**：輸入想要分析的主題詞（可使用中文或英文）。
            - **搜尋天數**：系統會從今天往前計算，例如輸入「7」即代表搜尋最近 7 天的新聞/影片。
            - **語言選擇**：
            - `zh`：以中文搜尋新聞與影片，並進行中文情感分析。
            - `en`：以英文搜尋新聞與影片，並進行情感分析。
            - **最多顯示影片數量**：從 5 到 50 部影片之間調整。
            - **搜尋類型（可多選）**：
            - `文字新聞`：從新聞網站抓取文章資料。
            - `YouTube 影片`：抓取影片留言進行情感分析。

            設定完成後，請點擊 **「開始分析」** 按鈕，系統將開始運作。

            ---

            #### 🤖 3. AI 助理功能（Gemini）

            你可以打開下方的 **「💬 AI 助理 (Gemini)」** 對話框：

            - 啟用條件：已輸入 LLM API 金鑰。
            - 功能：AI 助理會根據你輸入的問題與分析結果，使用 Google Gemini 回覆建議與解釋。
            - 範例問題：
            - 「這次的關鍵字情感偏向如何？」
            - 「哪個來源的負面情緒最多？」
            - 「請幫我摘要這次分析重點。」

            ---

            #### 📊 4. 查看分析結果

            每個資料來源（文字新聞或影片）都會有獨立分頁，顯示：

            - **情感統計**：
            - 正面 / 中立 / 負面 數量
            - 情感極性指數與主觀性平均值
            - **可視化圖表**：
            - 圓餅圖：情感比例分佈
            - 長條圖：不同來源的情緒比對
            - 折線圖：隨時間的情緒變化趨勢
            - **文字雲（詞雲圖）**：
            - 直觀展示重要關鍵字
            - **極端情緒樣本**：
            - 正面與負面情緒最強的文章或留言內容與連結

            ---

            #### 📥 5. 資料總結與下載

            - 系統會自動整合總結表格（各來源的情緒統計），並可供 CSV 下載。
            - 每個資料來源（新聞或影片）都可分別匯出分析結果（含標題、來源、情感標註等）。

            ---

            #### 🧪 6. 範例關鍵字建議（中文 / 英文皆可）

            以下為你可以試用的熱門主題關鍵字：

            - 政治 (Politics)  
            - 經濟 (Economy)  
            - AI 技術 (Artificial Intelligence)  
            - 環保 (Environment)  
            - 健康 (Health)  
            - 教育 (Education)  
            - 電動車 (Electric Vehicle)  
            - ChatGPT、Gemini、OpenAI  

            ---

            #### 💡 提醒事項

            - 若系統顯示「無資料」，可能是：
            - 關鍵字過冷門，沒有新聞或影片。
            - API 金鑰未填或次數限制已達上限。
            - 建議選擇較熱門或時事性的主題詞作為關鍵字。

            """)
        
# 頁面底部
st.markdown("---")
st.markdown("© 2025 新聞情感分析 AI 系統 — 使用 Streamlit、NewsAPI、YouTube API、Gemini AI 開發。")
