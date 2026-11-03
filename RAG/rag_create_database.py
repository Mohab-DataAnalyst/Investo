"""
rag_create_database.py
──────────────────────
Investo RAG — Vector Database Builder

Loads only the fields needed for the investor-facing RAG system,
builds text documents, embeds them with Gemini, and saves them to Chroma.
"""

import os
import shutil
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────
CHROMA_PATH     = "chroma"
EMBEDDING_MODEL = "gemini-embedding-2"


# ── Database Connection ───────────────────────────────────────────────
def get_engine():
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL not found in .env file.")
    return create_engine(database_url)


def load_data_from_db() -> tuple:
    """
    Load only public, non-sensitive columns for the investor-facing RAG.

    STARTUPS  — all public project fields (LEFT JOIN so projects without
                business data still appear).
    INVESTORS — preference/matching fields only; no personal or financial data.
                Excluded: name, email, phone, password, tokens,
                          min_investment, max_investment (financial prefs
                          excluded to prevent cross-investor data leakage
                          via vector similarity retrieval).
    INTERACTIONS — investment relationships only (APPROVED status, joined
                   to get startup title not raw project UUID).
                   Excluded: amounts, timestamps, payment details.
    """
    print("Connecting to PostgreSQL...")
    engine = get_engine()

    startups_df = pd.read_sql("""
        SELECT
            pp.title            AS startup_name,
            pp.industry,
            pp.country,
            pp.funding_stage,
            pp.founded_year,
            pp.tech_stack,
            pp.success_score,
            bd.annual_revenue,
            bd.total_funding,
            bd.no_employees,
            bd.valuation,
            bd.customer_base,
            bd.social_media_followers,
            bd.ipo,
            bd.acquired
        FROM projects_project pp
        LEFT JOIN projects_businessdata bd ON pp.id = bd.project_id
    """, engine)

    investors_df = pd.read_sql("""
        SELECT
            user_id                     AS investor_id,
            preferred_industry,
            country_preference,
            risk_tolerance,
            min_success_score_preferred
        FROM investors_investorprofile
    """, engine)

    # JOIN to projects_project to get pp.title (startup name) instead of
    # the raw project_id UUID. WHERE ia.status = 'APPROVED' — only real
    # investments, not pending or rejected applications.
    interactions_df = pd.read_sql("""
        SELECT
            ia.investor_id,
            pp.title    AS startup_name,
            1           AS invested
        FROM projects_investmentapplication ia
        JOIN projects_project pp ON ia.project_id = pp.id
        WHERE ia.status = 'APPROVED'
    """, engine)

    print(f"  Loaded {len(startups_df)} startups")
    print(f"  Loaded {len(investors_df)} investors")
    print(f"  Loaded {len(interactions_df)} interactions")

    return startups_df, investors_df, interactions_df


# ══════════════════════════════════════════════════════════════════════
#  DOCUMENT BUILDERS
# ══════════════════════════════════════════════════════════════════════

def build_startup_documents(df: pd.DataFrame) -> list[Document]:
    docs = []
    curr_year = datetime.now().year

    for _, row in df.iterrows():
        founded = row.get("founded_year")
        startup_age = (curr_year - int(founded)) if pd.notna(founded) else "N/A"

        text = f"""
STARTUP PROFILE
Name:                   {row['startup_name']}
Industry:               {row['industry']}
Country:                {row['country']}
Funding Stage:          {row['funding_stage']}
Founded Year:           {founded} ({startup_age} years old)
Tech Stack:             {row['tech_stack']}
Annual Revenue:         ${row['annual_revenue']}M
Total Funding:          ${row['total_funding']}M
Employees:              {row['no_employees']}
Valuation:              ${row['valuation']}B
Customer Base:          {row['customer_base']}M
Social Media Followers: {row['social_media_followers']}
Success Score:          {row['success_score']}/9
IPO:                    {'Yes' if row['ipo'] else 'No'}
Acquired:               {'Yes' if row['acquired'] else 'No'}
""".strip()

        docs.append(Document(
            page_content=text,
            metadata={
                "type":          "startup",
                "name":          row["startup_name"],
                "industry":      row["industry"],
                "country":       row["country"],
                "funding_stage": row["funding_stage"],
                "success_score": int(row["success_score"]) if pd.notna(row["success_score"]) else 0,
            }
        ))

    print(f"  Built {len(docs)} startup documents.")
    return docs


def build_investor_documents(df: pd.DataFrame) -> list[Document]:
    docs = []

    for _, row in df.iterrows():
        text = f"""
INVESTOR PROFILE
Preferred Industry:  {row['preferred_industry']}
Country Preference:  {row['country_preference']}
Risk Tolerance:      {row['risk_tolerance']}
Min Success Score:   {row['min_success_score_preferred']}/9
""".strip()

        docs.append(Document(
            page_content=text,
            metadata={
                "type":               "investor",
                "investor_id":        str(row["investor_id"]),
                "preferred_industry": row["preferred_industry"],
                "risk_tolerance":     row["risk_tolerance"],
            }
        ))

    print(f"  Built {len(docs)} investor documents.")
    return docs


def build_market_summary_documents(df: pd.DataFrame) -> list[Document]:
    docs = []

    for industry, group in df.groupby("industry"):
        avg_revenue   = round(group["annual_revenue"].mean(), 2)
        avg_funding   = round(group["total_funding"].mean(), 2)
        avg_success   = round(group["success_score"].mean(), 2)
        avg_employees = round(group["no_employees"].mean(), 0)
        top_countries = group["country"].value_counts().head(3).to_dict()
        stage_dist    = group["funding_stage"].value_counts(normalize=True).round(3).to_dict()
        success_rate  = round(len(group[group["success_score"] >= 7]) / len(group) * 100, 1)

        text = f"""
MARKET SUMMARY — INDUSTRY: {industry}
Total Startups:     {len(group)}
Avg Success Score:  {avg_success}/9
Success Rate (≥7):  {success_rate}%
Avg Annual Revenue: ${avg_revenue}M
Avg Total Funding:  ${avg_funding}M
Avg Employees:      {int(avg_employees)}
Top Countries:      {top_countries}
Funding Stage Mix:  {stage_dist}
""".strip()

        docs.append(Document(
            page_content=text,
            metadata={"type": "market_summary", "industry": industry}
        ))

    for country, group in df.groupby("country"):
        avg_success    = round(group["success_score"].mean(), 2)
        avg_funding    = round(group["total_funding"].mean(), 2)
        top_industries = group["industry"].value_counts().head(3).to_dict()
        success_rate   = round(len(group[group["success_score"] >= 7]) / len(group) * 100, 1)

        text = f"""
MARKET SUMMARY — COUNTRY: {country}
Total Startups:    {len(group)}
Avg Success Score: {avg_success}/9
Success Rate (≥7): {success_rate}%
Avg Total Funding: ${avg_funding}M
Top Industries:    {top_industries}
""".strip()

        docs.append(Document(
            page_content=text,
            metadata={"type": "market_summary", "country": country}
        ))

    print(f"  Built {len(docs)} market summary documents.")
    return docs


def build_interaction_documents(
    interactions_df: pd.DataFrame,
    startups_df:     pd.DataFrame,
    investors_df:    pd.DataFrame,
) -> list[Document]:
    docs = []

    for investor_id, group in interactions_df.groupby("investor_id"):
        invested_names = group["startup_name"].tolist()
        portfolio      = startups_df[startups_df["startup_name"].isin(invested_names)]

        if portfolio.empty:
            continue

        industries = portfolio["industry"].value_counts().to_dict()
        stages     = portfolio["funding_stage"].value_counts().to_dict()
        countries  = portfolio["country"].value_counts().to_dict()
        avg_score  = round(portfolio["success_score"].mean(), 2)

        text = f"""
INVESTOR PORTFOLIO HISTORY
Portfolio Size:    {len(invested_names)} startups
Invested Startups: {', '.join(invested_names[:10])}{'...' if len(invested_names) > 10 else ''}
Industry Mix:      {industries}
Stage Mix:         {stages}
Country Mix:       {countries}
Avg Success Score: {avg_score}/9
""".strip()

        inv_row = investors_df[investors_df["investor_id"] == investor_id]
        if not inv_row.empty:
            inv   = inv_row.iloc[0]
            text += f"""
Preferred Industry: {inv['preferred_industry']}
Risk Tolerance:     {inv['risk_tolerance']}
""".strip()

        docs.append(Document(
            page_content=text,
            metadata={
                "type":        "interaction",
                "investor_id": str(investor_id),
            }
        ))

    print(f"  Built {len(docs)} interaction documents.")
    return docs


# ══════════════════════════════════════════════════════════════════════
#  PIPELINE
# ══════════════════════════════════════════════════════════════════════

def build_all_documents(startups_df, investors_df, interactions_df) -> list[Document]:
    print("Building documents...")
    docs  = []
    docs += build_startup_documents(startups_df)
    docs += build_investor_documents(investors_df)
    docs += build_market_summary_documents(startups_df)
    docs += build_interaction_documents(interactions_df, startups_df, investors_df)
    print(f"Total documents: {len(docs)}")
    return docs


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")
    return chunks


def save_to_chroma(chunks: list[Document]):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print("Cleared existing Chroma DB.")

    embedding_function = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
    )

    print("Embedding and saving to Chroma (this may take a few minutes)...")
    db = Chroma.from_documents(
        chunks,
        embedding_function,
        persist_directory=CHROMA_PATH,
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to '{CHROMA_PATH}'.")


def main():
    startups_df, investors_df, interactions_df = load_data_from_db()
    documents = build_all_documents(startups_df, investors_df, interactions_df)
    chunks    = split_documents(documents)
    save_to_chroma(chunks)
    print("\nDatabase ready.")


if __name__ == "__main__":
    main()
