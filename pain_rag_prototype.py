import argparse
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_data():
    implicit_df = pd.read_csv(os.path.join(BASE_DIR, "implicit_knowledge.csv"))
    case_df = pd.read_csv(os.path.join(BASE_DIR, "case_bank.csv"))
    return implicit_df, case_df


def build_index(implicit_df):
    docs = (
        implicit_df["大カテゴリ"].fillna("") + " " +
        implicit_df["サブカテゴリ"].fillna("") + " " +
        implicit_df["元の表現"].fillna("") + " " +
        implicit_df["言い換え"].fillna("") + " " +
        implicit_df["観察対象"].fillna("") + " " +
        implicit_df["解釈の方向性"].fillna("") + " " +
        implicit_df["背景にある暗黙知"].fillna("") + " " +
        implicit_df["学生への問い"].fillna("") + " " +
        implicit_df["関連する病態・データ"].fillna("") + " " +
        implicit_df["関わり方の視点"].fillna("") + " " +
        implicit_df["事例タグ"].fillna("")
    ).tolist()
    # Japanese-friendly retrieval using character n-grams
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
    matrix = vectorizer.fit_transform(docs)
    return vectorizer, matrix


def retrieve(case_row, question, implicit_df, vectorizer, matrix, top_k=4):
    query = " ".join([
        str(case_row["事例本文"]),
        str(case_row["主疾患"]),
        str(case_row["状況"]),
        str(case_row["バイタル"]),
        str(case_row["検査データ"]),
        str(case_row["表情"]),
        str(case_row["動き"]),
        str(case_row["発言"]),
        str(case_row["普段との違い"]),
        str(case_row["学習テーマ"]),
        question,
    ])
    qv = vectorizer.transform([query])
    sims = cosine_similarity(qv, matrix).flatten()
    top_idx = sims.argsort()[::-1][:top_k]
    result = implicit_df.iloc[top_idx].copy()
    result["score"] = sims[top_idx]
    return result


def format_output(case_row, retrieved_df):
    signs = []
    differences = []
    pathophys = []
    caring = []
    next_checks = []
    questions = []

    for _, r in retrieved_df.iterrows():
        obs = str(r["観察対象"])
        interp = str(r["解釈の方向性"])
        related = str(r["関連する病態・データ"])
        care = str(r["関わり方の視点"])
        q = str(r["学生への問い"])
        if obs not in signs:
            signs.append(obs)
        if "変化" in interp or "いつも" in str(r["元の表現"]) or "違う" in str(r["元の表現"]):
            differences.append(interp)
        if related not in pathophys:
            pathophys.append(f"{related}との関連を考える")
        if care not in caring:
            caring.append(care)
        if obs not in next_checks:
            next_checks.append(obs)
        if q not in questions:
            questions.append(q)

    text = []
    text.append("① 気になるサイン")
    # 先に事例の具体所見を出す
    text.append(f"- 表情: {case_row['表情']}")
    text.append(f"- 動き: {case_row['動き']}")
    text.append(f"- 発言: {case_row['発言']}")
    text.append(f"- 様子: {case_row['様子']}")
    for s in signs[:2]:
        text.append(f"- 観察の視点: {s}")
    text.append("")
    text.append("② いつもとの違い")
    text.append(f"- {case_row['普段との違い']}")
    for d in differences[:2]:
        text.append(f"- {d}")
    text.append("")
    text.append("③ 病態からの推理")
    text.append(f"- 主疾患は{case_row['主疾患']}、状況は{case_row['状況']}です。")
    text.append(f"- バイタルは{case_row['バイタル']}、検査データは{case_row['検査データ']}です。")
    for p in pathophys[:3]:
        text.append(f"- {p}")
    text.append("")
    text.append("④ 看護としての受け止め")
    text.append(f"- 『{case_row['発言']}』という言葉だけで判断せず、表情・動き・姿勢の情報も統合して受け止めます。")
    for c in caring[:3]:
        text.append(f"- {c}")
    text.append("")
    text.append("⑤ 次に確認すること")
    for n in next_checks[:3]:
        text.append(f"- {n}を追加で確認する")
    text.append("- 痛みの強さ、性質、きっかけ、緩和因子を本人の言葉で確認する")
    text.append("")
    text.append("⑥ 学生への問い")
    for q in questions[:3]:
        text.append(f"- {q}")
    text.append("")
    text.append("参考にした暗黙知")
    for _, r in retrieved_df.iterrows():
        text.append(f"- {r['ID']} {r['元の表現']}（score={r['score']:.3f}）")
    return "\n".join(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, help="例: CASE-01")
    parser.add_argument("--question", required=True, help="学生の質問")
    args = parser.parse_args()

    implicit_df, case_df = load_data()
    vectorizer, matrix = build_index(implicit_df)

    case_match = case_df[case_df["ケースID"] == args.case]
    if case_match.empty:
        raise ValueError("ケースIDが見つかりません。")
    case_row = case_match.iloc[0]

    retrieved = retrieve(case_row, args.question, implicit_df, vectorizer, matrix)
    output = format_output(case_row, retrieved)
    print(output)


if __name__ == "__main__":
    main()
