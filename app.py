import os
import pandas as pd
import streamlit as st
from pain_rag_prototype import load_data, build_index, retrieve, format_output

st.set_page_config(page_title="痛みを察知する暗黙知RAG教材", layout="wide")

st.title("痛みを察知する暗黙知RAG教材")
st.caption("看護学生向けプロトタイプ")

implicit_df, case_df = load_data()
vectorizer, matrix = build_index(implicit_df)

case_id = st.selectbox("ケースを選択してください", case_df["ケースID"].tolist(), index=0)
case_row = case_df[case_df["ケースID"] == case_id].iloc[0]

col1, col2 = st.columns([1.3, 1])
with col1:
    st.subheader("事例")
    st.write(case_row["事例本文"])
    st.markdown(f"**主疾患**: {case_row['主疾患']}")
    st.markdown(f"**状況**: {case_row['状況']}")
    st.markdown(f"**バイタル**: {case_row['バイタル']}")
    st.markdown(f"**検査データ**: {case_row['検査データ']}")

with col2:
    st.subheader("学生の質問")
    question = st.text_area(
        "自由に入力してください",
        value="この患者さんは痛いと訴えていません。どこに注目すべきですか。",
        height=120,
    )
    top_k = st.slider("参照する暗黙知件数", min_value=2, max_value=6, value=4)

if st.button("RAGで考える", type="primary"):
    retrieved = retrieve(case_row, question, implicit_df, vectorizer, matrix, top_k=top_k)
    output = format_output(case_row, retrieved)

    st.subheader("出力")
    st.text(output)

    st.subheader("参照した暗黙知")
    show_cols = ["ID", "大カテゴリ", "元の表現", "言い換え", "観察対象", "学生への問い", "score"]
    st.dataframe(retrieved[show_cols], use_container_width=True)

with st.expander("暗黙知データベースを見る"):
    st.dataframe(implicit_df, use_container_width=True)

with st.expander("事例データベースを見る"):
    st.dataframe(case_df, use_container_width=True)
