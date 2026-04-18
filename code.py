import streamlit as st
import pandas as pd
import re
import string
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.calibration import CalibratedClassifierCV


def frontend():
    st.markdown("""
        <style>
        .stApp { background-color: #ffffff; }

        .block-container {
            padding: 2rem 2.5rem;
            max-width: 1400px;
        }

        .page-header {
            border-bottom: 2px solid #222222;
            padding-bottom: 1rem;
            margin-bottom: 1.75rem;
        }
        .page-title {
            font-size: 1.6rem;
            font-weight: 700;
            color: #111111;
            margin: 0;
        }
        .page-sub {
            font-size: 0.85rem;
            color: #666666;
            margin-top: 0.2rem;
        }

        .sec-label {
            font-size: 0.7rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            color: #888888;
            border-bottom: 1px solid #e4e4e4;
            padding-bottom: 0.4rem;
            margin-bottom: 0.75rem;
        }

        .result-card {
            padding: 1.25rem 1.5rem;
            margin: 1rem 0;
            border: 1px solid #e4e4e4;
            border-left: 4px solid;
            background: #fafafa;
        }
        .result-positive { border-left-color: #2e7d32; }
        .result-negative { border-left-color: #c62828; }
        .result-neutral  { border-left-color: #e65100; }

        .result-label {
            font-size: 1.25rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .label-positive { color: #2e7d32; }
        .label-negative { color: #c62828; }
        .label-neutral  { color: #e65100; }

        .result-conf {
            font-size: 0.82rem;
            color: #666666;
            margin-top: 0.2rem;
        }
        .result-conf b { color: #111111; }

        .stTextArea textarea {
            border: 1px solid #cccccc !important;
            border-radius: 4px !important;
            background: #ffffff !important;
            font-size: 0.9rem !important;
            padding: 0.75rem !important;
            color: #111111 !important;
            box-shadow: none !important;
        }
        .stTextArea textarea:focus {
            border-color: #111111 !important;
            outline: none !important;
        }

        .stButton > button {
            background: #111111 !important;
            color: #ffffff !important;
            font-size: 0.85rem !important;
            font-weight: 600 !important;
            padding: 0.6rem 1.5rem !important;
            border: none !important;
            border-radius: 4px !important;
            text-transform: uppercase !important;
            letter-spacing: 0.5px !important;
            width: 100% !important;
            transition: background 0.15s !important;
        }
        .stButton > button:hover { background: #333333 !important; }

        [data-testid="stSidebar"] {
            background: #f9f9f9;
            border-right: 1px solid #e4e4e4;
            overflow: hidden !important;
        }
        [data-testid="stSidebar"] > div:first-child {
            overflow-y: hidden !important;
        }
        [data-testid="stSidebar"] .stMetric {
            background: #ffffff;
            border: 1px solid #e4e4e4;
            border-left: 3px solid #111111;
            padding: 0.6rem 0.75rem;
            margin-bottom: 0.5rem;
        }

        [data-testid="stPlotlyChart"] {
            border: 1px solid #e4e4e4;
            background: #ffffff;
        }

        #MainMenu, footer { visibility: hidden; }
        </style>
    """, unsafe_allow_html=True)


def clean(text):
    text = text.lower()
    text = re.sub(r"can't", "can not", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"\bnot\s+(\w+)", r"not_\1", text)
    text = re.sub(r"\bno\s+(\w+)", r"no_\1", text)
    text = re.sub(r"\bnever\s+(\w+)", r"never_\1", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

@st.cache_resource
def train():
    data=pd.read_csv("Tweets.csv")
    data=data[['text', 'airline_sentiment']]
    data.rename(columns={'airline_sentiment': 'sentiment'}, inplace=True)
    data['clean']=data['text'].apply(clean)

    count=data['sentiment'].value_counts()
    percent=data['sentiment'].value_counts(normalize=True)*100

    vec=TfidfVectorizer(
    stop_words='english',
    max_features=9000,
    ngram_range=(1,3),
    min_df=2,
    max_df=0.9,
    sublinear_tf=True
    )
    X=vec.fit_transform(data["clean"])
    y=data["sentiment"]

    Xtr, Xte, ytr, yte=train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    labels=np.unique(ytr)
    weights=compute_class_weight('balanced', classes=labels, y=ytr)
    wmap=dict(zip(labels, weights))

    base=LinearSVC(class_weight=wmap, max_iter=2000, C=1)
    model=CalibratedClassifierCV(base, cv=3)
    model.fit(Xtr, ytr)

    preds=model.predict(Xte)
    acc=accuracy_score(yte, preds)
    report=classification_report(yte, preds, output_dict=True)
    cm=confusion_matrix(yte, preds, labels=labels)

    return model, vec, acc, count, percent, wmap, report, cm, labels


st.set_page_config(page_title="SENTIMENT ANALYSIS", page_icon="T", layout="wide", initial_sidebar_state="expanded")
frontend()

with st.spinner("Loading model..."):
    model, vec, acc, count, percent, wmap, report, cm, labels=train()

pos=percent.get('positive', 0)
neg=percent.get('negative', 0)
neu=percent.get('neutral', 0)
total=sum(count.values)

with st.sidebar:
    st.markdown("### Overview")
    st.metric("Model Accuracy", f"{acc*100:.1f}%")
    st.metric("Total Samples", f"{total:,}")
    st.divider()
    st.metric("Positive", f"{pos:.1f}%", delta=f"{count.get('positive', 0):,} tweets")
    st.metric("Negative", f"{neg:.1f}%", delta=f"{count.get('negative', 0):,} tweets")
    st.metric("Neutral", f"{neu:.1f}%", delta=f"{count.get('neutral', 0):,} tweets")

st.markdown("""
    <div class='page-header'>
        <div class='page-title'>SENTIMENT ANALYSIS</div>
    </div>
""", unsafe_allow_html=True)

left, right=st.columns([1, 1], gap="large")

with left:
    st.markdown("<div class='sec-label'>Analyze Text</div>", unsafe_allow_html=True)
    text=st.text_area(
        "Text input",
        height=130,
        placeholder="Paste a tweet, review, or comment...",
        label_visibility="collapsed"
    )
    run=st.button("Run Analysis")

    # result card appears below the button
    if run:
        if text.strip()=="":
            st.warning("Enter some text first.")
        else:
            with st.spinner("Analyzing..."):
                vec_in = vec.transform([clean(text)])
                
                probs = model.predict_proba(vec_in)[0]
                labels = model.classes_
                
                top_idx = np.argmax(probs)
                top_label = labels[top_idx]
                
                sorted_probs = np.sort(probs)
                margin = sorted_probs[-1] - sorted_probs[-2]
                
                if margin < 0.15:
                    pred = 'neutral'
                else:
                    pred = top_label
                
                conf = probs[top_idx] * 100
                pmap = dict(zip(labels, probs * 100))

            st.markdown("<div class='sec-label' style='margin-top:1rem;'>Result</div>", unsafe_allow_html=True)
            st.markdown(f"""
                <div class='result-card result-{pred}'>
                    <div class='result-label label-{pred}'>{pred.capitalize()}</div>
                    <div class='result-conf'>Confidence: <b>{conf:.2f}%</b></div>
                </div>
            """, unsafe_allow_html=True)

with right:
    # confidence bar appears on the right, only after analysis runs
    if run and text.strip()!="":
        st.markdown("<div class='sec-label'>Confidence Breakdown</div>", unsafe_allow_html=True)
        df=pd.DataFrame({
            'Sentiment': [s.capitalize() for s in pmap.keys()],
            'Probability': list(pmap.values())
        })
        bar=px.bar(
            df, x='Sentiment', y='Probability', color='Sentiment',
            color_discrete_map={'Positive': '#2e7d32', 'Negative': '#c62828', 'Neutral': '#e65100'},
            text='Probability'
        )
        bar.update_traces(texttemplate='%{text:.1f}%', textposition='outside', marker_line_width=0)
        bar.update_layout(
            showlegend=False,
            yaxis_title="Confidence (%)",
            xaxis_title="",
            height=373,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='#fafafa',
            margin=dict(t=25, b=5, l=5, r=5),
            font=dict(size=12, color='#111111'),
            yaxis=dict(gridcolor='#eeeeee'),
            xaxis=dict(showgrid=False)
        )
        st.plotly_chart(bar, use_container_width=True)

st.divider()

a, b, c=st.columns(3, gap="medium")

with a:
    st.markdown("<div class='sec-label'>Sentiment Distribution</div>", unsafe_allow_html=True)
    pie=go.Figure(data=[go.Pie(
        labels=[s.capitalize() for s in count.index],
        values=count.values,
        hole=0.45,
        marker=dict(colors=['#2e7d32', '#c62828', '#e65100'], line=dict(color='#ffffff', width=2)),
        textinfo='label+percent',
        textfont=dict(size=12),
        hovertemplate='%{label}: %{value:,} tweets<extra></extra>'
    )])
    pie.update_layout(
        showlegend=False,
        height=300,
        margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#111111')
    )
    st.plotly_chart(pie, use_container_width=True)

with b:
    st.markdown("<div class='sec-label'>Confusion Matrix</div>", unsafe_allow_html=True)
    heatmap=px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=[s.capitalize() for s in labels],
        y=[s.capitalize() for s in labels],
        color_continuous_scale=[[0, '#f0f0f0'], [1, '#1565c0']],
        text_auto=True
    )
    heatmap.update_layout(
        height=300,
        margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12, color='#111111'),
        coloraxis_showscale=False
    )
    st.plotly_chart(heatmap, use_container_width=True)

with c:
    st.markdown("<div class='sec-label'>Precision / Recall / F1 by Class</div>", unsafe_allow_html=True)
    rows=[]
    for s in labels:
        if s in report:
            rows.append({
                'Class': s.capitalize(),
                'Precision': round(report[s]['precision']*100, 2),
                'Recall': round(report[s]['recall']*100, 2),
                'F1': round(report[s]['f1-score']*100, 2)
            })
    df=pd.DataFrame(rows)
    melt=df.melt(id_vars='Class', var_name='Metric', value_name='Score')
    metric=px.bar(
        melt, x='Class', y='Score', color='Metric', barmode='group',
        color_discrete_map={'Precision': '#1565c0', 'Recall': '#6a1b9a', 'F1': '#2e7d32'},
        text='Score'
    )
    metric.update_traces(texttemplate='%{text:.1f}%', textposition='outside', marker_line_width=0)
    metric.update_layout(
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#fafafa',
        margin=dict(t=25, b=5, l=5, r=5),
        font=dict(size=12, color='#111111'),
        yaxis=dict(gridcolor='#eeeeee', title='Score (%)'),
        xaxis=dict(showgrid=False, title=''),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    st.plotly_chart(metric, use_container_width=True)