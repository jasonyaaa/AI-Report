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
st.set_page_config(page_title="新聞情感分析系統", page_icon="📰", layout="wide")

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
        st.subheader("文字詞雲展示")

        if language == 'zh':
            # 只剩中文才用 jieba 分詞
            tokens = " ".join(jieba.cut(text_corpus))
            wc = WordCloud(
                font_path="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                width=800,
                height=400,
                background_color='white',
                max_words=100,
                stopwords=None,       # 如要加中文停用詞，在此傳入集合
                collocations=False
            ).generate(tokens)
        else:
            # 英文直接用原始文字
            wc = WordCloud(
                width=800,
                height=400,
                background_color='white',
                max_words=100,
                stopwords=None,       # 如要加英文停用詞，可傳 EN_STOPWORDS
                collocations=False
            ).generate(text_corpus)

        fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
        ax_wc.imshow(wc, interpolation='bilinear')
        ax_wc.axis('off')
        st.pyplot(fig_wc)
    else:
        st.info("無可用文字生成詞雲。")
    # 新聞詞雲結束.


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
st.title("📰 新聞情感分析系統")
st.markdown("""
本系統可搜尋並分析網路上的新聞內容（包含文字與影片），顯示其情感傾向。
""")

# 側邊欄：API 設定與搜尋參數
with st.sidebar:
    st.header("API 設定")
    llm_api_key = st.text_input("LLM API 金鑰", type="password")  # 新增這一行
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

# 顯示結果區域
if analyze_button and query:
    all_results = {}
    if not news_api_key and '文字新聞' in search_type:
        st.error("請輸入 NewsAPI 金鑰以搜尋文字新聞")
    
    if not youtube_api_key and 'YouTube 影片' in search_type:
        st.error("請輸入 YouTube API 金鑰以搜尋影片")
    
    if (news_api_key and '文字新聞' in search_type) or (youtube_api_key and 'YouTube 影片' in search_type):
        with st.spinner("正在搜尋與分析，請稍候..."):
            # 日期範圍
            to_date = datetime.now().strftime('%Y-%m-%d')
            from_date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')
            
            # 建立分頁
            tabs = st.tabs([t for t in search_type] + ["總結", "AI 助理"])
        
            
            # 處理文字新聞
            if '文字新聞' in search_type and news_api_key:
                with tabs[search_type.index('文字新聞')]:
                    st.header("文字新聞情感分析")
                    
                    news_df = get_news(query, from_date, to_date, language)
                    
                    if not news_df.empty:
                        sentiment_results = []
                        for _, row in news_df.iterrows():
                            text_to_analyze = f"{row['標題']} {row['描述']} {row['內容']}"
                            sentiment_data = analyze_sentiment(text_to_analyze)
                            sentiment_results.append(sentiment_data)
                        
                        news_df['polarity'] = [r['polarity'] for r in sentiment_results]
                        news_df['subjectivity'] = [r['subjectivity'] for r in sentiment_results]
                        news_df['sentiment'] = [r['sentiment'] for r in sentiment_results]
                        
                        display_results(news_df, "文字新聞")
                        all_results['文字新聞'] = news_df
                    else:
                        st.warning("找不到任何新聞資料，請嘗試其他關鍵字或延長時間範圍。")
            
            # 處理 YouTube 影片
            if 'YouTube 影片' in search_type and youtube_api_key:
                with tabs[search_type.index('YouTube 影片')]:
                    st.header("YouTube 影片留言情感分析")
                    st.info("⚠️ 本功能分析的是影片下方的留言評論，不是字幕。")

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
                            # 對每則留言做情感分析
                            sentiment_results = [analyze_sentiment(c) for c in comments_df['留言內容']]
                            comments_df['polarity'] = [r['polarity'] for r in sentiment_results]
                            comments_df['subjectivity'] = [r['subjectivity'] for r in sentiment_results]
                            comments_df['sentiment'] = [r['sentiment'] for r in sentiment_results]
                            comments_df.rename(columns={'留言內容': '內容', '留言時間': '發佈時間'}, inplace=True)
                            display_results(comments_df, "YouTube 留言")
                            all_results['YouTube 影片'] = comments_df
                        else:
                            st.warning("找不到任何留言資料。")
                        # <<< 這裡結束 >>>
            
            # 總結分頁
            with tabs[-2]:
                st.header("情感分析總結")
                
                if all_results:
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
                        
                        # 繪製總結長條圖
                        fig = go.Figure()
                        for source_type in summary_df['類型']:
                            row = summary_df[summary_df['類型'] == source_type].iloc[0]
                            fig.add_trace(go.Bar(
                                name=source_type,
                                x=['正面', '中立', '負面'],
                                y=[row['正面'], row['中立'], row['負面']],
                                marker_color=['green', 'blue', 'red']
                            ))
                        
                        fig.update_layout(
                            title='不同來源之情感分佈',
                            xaxis_title='情感類別',
                            yaxis_title='數量',
                            barmode='group'
                        )
                        st.plotly_chart(fig)
                        
                        # 最終結論
                        st.subheader("最終結論")
                        
                        total_positive = sum(row['正面'] for row in summary_data)
                        total_neutral = sum(row['中立'] for row in summary_data)
                        total_negative = sum(row['負面'] for row in summary_data)
                        total_items = sum(row['總數'] for row in summary_data)
                        
                        st.write(f"總共分析 {total_items} 筆資料：")
                        st.write(f"- {total_positive} ({total_positive/total_items*100:.1f}%) 為正面")
                        st.write(f"- {total_neutral} ({total_neutral/total_items*100:.1f}%) 為中立")
                        st.write(f"- {total_negative} ({total_negative/total_items*100:.1f}%) 為負面")
                        
                        if total_positive > (total_neutral + total_negative):
                            st.write(f"**結論：** 關鍵字「{query}」整體呈現**正面**情感傾向。")
                        elif total_negative > (total_neutral + total_positive):
                            st.write(f"**結論：** 關鍵字「{query}」整體呈現**負面**情感傾向。")
                        else:
                            st.write(f"**結論：** 關鍵字「{query}」整體情感傾向**中立或混合**。")
                else:
                    st.warning("目前沒有任何資料可供總結分析。")
            # AI 助理分頁
            with tabs[-1]:
                st.header("AI 助理")
                st.markdown("你可以詢問本次分析的任何問題，AI 會根據分析結果回覆。")
                user_question = st.text_area("請輸入你的問題", value="這次的關鍵詞情感分析結果你有甚麼看法？")
                ask_button = st.button("送出問題", key="ask_llm")

                if "llm_reply" not in st.session_state:
                    st.session_state.llm_reply = ""

                if ask_button and user_question and llm_api_key:
                    # 彙整所有分析結果
                    summary_text = ""
                    for source_type, df in all_results.items():
                        if not df.empty:
                            summary_text += f"\n【{source_type}】\n"
                            summary_text += df.head(10).to_markdown(index=False)
                    prompt = f"以下是本次情感分析的資料摘要：\n{summary_text}\n\n使用者問題：{user_question}\n請用繁體中文簡要回覆。"

                    headers = {
                        "Authorization": f"Bearer {llm_api_key}",
                        "Content-Type": "application/json"
                    }
                    data = {
                        "model": "gpt-3.5-turbo",
                        "messages": [
                            {"role": "system", "content": "你是一個新聞情感分析助理，請根據資料摘要回答問題。"},
                            {"role": "user", "content": prompt}
                        ]
                    }
                    try:
                        response = requests.post(
                            "https://api.openai.com/v1/chat/completions",
                            headers=headers,
                            json=data,
                            timeout=60
                        )
                        if response.status_code == 200:
                            ai_reply = response.json()["choices"][0]["message"]["content"]
                            st.session_state.llm_reply = ai_reply
                        else:
                            st.session_state.llm_reply = f"AI 回應失敗，狀態碼：{response.status_code}\n{response.text}"
                    except Exception as e:
                        st.session_state.llm_reply = f"AI 助理呼叫失敗：{e}"

                elif ask_button and not llm_api_key:
                    st.warning("請先輸入 LLM API 金鑰")

                # 顯示 LLM 回應
                if st.session_state.llm_reply:
                    st.success("AI 助理回覆：")
                    st.write(st.session_state.llm_reply)
        
        # 提供下載功能
        if all_results:
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
        ### 如何使用此系統
        
        1. **API 設定**  
           - 取得 [NewsAPI](https://newsapi.org) 的 API Key，以及在 Google Cloud Console 啟用 YouTube Data API 後取得 API Key  
           - 將金鑰分別貼到側邊欄的「NewsAPI 金鑰」與「YouTube API 金鑰」欄位
        
        2. **搜尋設定**  
           - 輸入欲搜尋的關鍵字（可同時使用中文與英文）  
           - 選擇要抓取多少天內的新聞  
           - 選擇語言（中文 or 英文）  
           - 選擇要分析的類型（文字新聞、YouTube 影片，或兩者）
        
        3. **查看結果**  
           - 觀察情感分佈（正面、中立、負面）  
           - 查看不同來源的情感分析  
           - 查看情感趨勢圖與最極端的正/負向項目
        
        **範例關鍵字：**  
        - 政治 (Politics)  
        - 經濟 (Economy)  
        - 科技 (Technology)  
        - 健康 (Health)  
        - 環保 (Environment)  
        """)
        
# 頁面底部
st.markdown("---")
st.markdown("© 2025 新聞情感分析系統 | 使用 Streamlit、NewsAPI 和 YouTube API 開發")
