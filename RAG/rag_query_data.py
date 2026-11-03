"""
rag_query_data.py
─────────────────
Investo RAG — Query Pipeline

Hybrid flow:
- Normal platform/general questions -> RAG
- Recommendation questions -> recommendation API
- Similar startup questions -> recommendation API

Uses Gemini for embeddings + LLM.
DuckDuckGo is used only for general questions.
"""

import json
import os
import re
from urllib import error, request

from dotenv import load_dotenv
from duckduckgo_search import DDGS
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────
CHROMA_PATH = "chroma"
EMBEDDING_MODEL = "gemini-embedding-2"
LLM_MODEL = "gemini-2.5-flash-lite"
RECOMMENDER_API_URL = os.environ.get("RECOMMENDER_API_URL", "http://localhost:8002/recommendations")

TOP_K_RESULTS = 5
MIN_RELEVANCE = 0.5
MAX_HISTORY = 10
WEB_RESULTS = 3
TOP_RECOMMENDATIONS = 5
API_TIMEOUT_SECONDS = 15


# ── System Prompt ─────────────────────────────────────────────────────
# FIXED: explicitly widened scope to cover macroeconomic / general
# financial questions, and explicitly instructed the model to use
# web search results when present rather than self-censoring based
# on a narrow reading of its own role description.
SYSTEM_PROMPT = """
You are Investo Assistant, the AI advisor of the Investo digital investment platform.

ABOUT INVESTO:
Investo is a data-driven platform that connects investors with SME entrepreneurs in Egypt
and globally. The platform provides ML-powered analytics including success score predictions
(1-9 scale), funding predictions, risk assessment, and a recommendation system that matches
investors with promising startup projects.

YOUR ROLE:
You are speaking exclusively with INVESTORS. Your job is to help investors:
- Discover and evaluate startup investment opportunities
- Understand startup performance metrics (success score, funding, revenue, risk)
- Explore market trends and industry insights
- Interpret recommendation results and dashboard analytics
- Make informed, data-driven investment decisions
- Answer general investment, economic, and financial market questions, including
  macroeconomic topics such as interest rates, inflation, exchange rates, GDP growth,
  and market trends in any country, since investors regularly need this context
  to make investment decisions

PLATFORM DATA YOU HAVE ACCESS TO:
- Startup profiles (public project data submitted by entrepreneurs):
    name, industry, country, funding stage, founded year, tech stack,
    annual revenue, total funding, number of employees, valuation,
    customer base, social media followers, success score (1-9),
    IPO status, acquired status
- Investor matching preferences only (non-personal, non-financial):
    preferred industry, country preference, risk tolerance,
    minimum success score preferred
- Market summaries: industry and country-level aggregates,
    success rates, average funding, stage distribution
- Portfolio histories: which startups each investor has backed,
    portfolio industry/stage/country mix, average success score

IMPORTANT — USING WEB SEARCH RESULTS:
If a section labeled "WEB SEARCH RESULTS" appears below, it has already been
retrieved specifically to answer the investor's question. You MUST use it to
answer directly and concisely. Do not refuse the question, do not say it is
outside platform scope, and do not recite this role description back to the
investor when web search results are already provided — they were retrieved
precisely because the question needs them.

BEHAVIOR RULES:
- Be concise, professional, and data-driven
- Use platform data precisely when available
- When web search results are provided, use them to answer general
  economic, market, or financial questions directly
- Never fabricate startup names, investor IDs, or statistics
- If asked for unavailable private data, say it is not available
- Do not reveal another investor's portfolio, preferences, or activity
- Do not make guaranteed investment promises
"""

PROMPT_TEMPLATE = """
{system_prompt}

RELEVANT PLATFORM DATA:
{context}

{web_context}

CONVERSATION HISTORY:
{history}

Investor Question: {question}
Assistant:"""


# ── Models / DB ───────────────────────────────────────────────────────
_embedding_function = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
)

_db = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=_embedding_function,
)

_llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL,
    temperature=0.2,
    max_retries=0,
)


# ── Helpers ───────────────────────────────────────────────────────────
def _dedupe(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _format_history(conversation_history: list[dict]) -> str:
    if not conversation_history:
        return "No previous messages."
    return "\n".join(
        f"{'Investor' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in conversation_history[-MAX_HISTORY:]
    )


def _update_history(conversation_history: list[dict], question: str, response_text: str) -> list[dict]:
    return conversation_history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": response_text},
    ]


def _search_chroma(question: str, current_investor_id: str | None = None) -> tuple[str, list[str]]:
    results = _db.similarity_search_with_relevance_scores(question, k=TOP_K_RESULTS)
    filtered = []

    for doc, score in results:
        if score < MIN_RELEVANCE:
            continue

        doc_type = doc.metadata.get("type")
        doc_investor_id = doc.metadata.get("investor_id")

        if doc_type in {"investor", "interaction"}:
            if current_investor_id is None:
                continue
            if str(doc_investor_id) != str(current_investor_id):
                continue

        filtered.append((doc, score))

    if not filtered:
        return "", []

    context = "\n\n---\n\n".join(doc.page_content for doc, _ in filtered)
    sources = [
        doc.metadata.get("name")
        or doc.metadata.get("industry")
        or doc.metadata.get("country")
        or "platform data"
        for doc, _ in filtered
    ]
    return context, _dedupe(sources)


def _search_web(question: str) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(question, max_results=WEB_RESULTS))
        if not results:
            return ""

        text = "WEB SEARCH RESULTS (use these to answer the question directly):\n"
        for item in results:
            text += f"- {item['title']}: {item['body']}\n"
        return text
    except Exception:
        return ""


def _call_recommender(endpoint: str, payload: dict) -> list[dict]:
    url = f"{RECOMMENDER_API_URL.rstrip('/')}{endpoint}"
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=API_TIMEOUT_SECONDS) as response:
            data = json.loads(response.read().decode("utf-8"))
            return data if isinstance(data, list) else []
    except error.HTTPError as exc:
        msg = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Recommendation API error ({exc.code}): {msg}") from exc
    except error.URLError as exc:
        raise RuntimeError(
            "Could not reach recommendation API. Check that it is running."
        ) from exc


def _detect_intent(question: str) -> str:
    q = question.lower()

    if any(p in q for p in [
        "similar startups", "similar to", "startups like",
        "companies like", "projects like", "alternatives to"
    ]):
        return "similar_startups"

    if any(p in q for p in [
        "recommend", "suggest", "best for me", "fit me",
        "fits me", "suitable for me", "my profile",
        "my preferences", "opportunities for me"
    ]):
        return "recommend_startups"

    prompt = f"""
Classify this investor request into exactly one label:

- rag
- recommend_startups
- similar_startups

Meaning:
- rag: normal platform/general question answering
- recommend_startups: user wants startup recommendations for themself
- similar_startups: user wants startups similar to a specific startup

Reply with one label only.

Question: {question}
""".strip()

    try:
        result = _llm.invoke(prompt).content.strip().lower()
        if result in {"rag", "recommend_startups", "similar_startups"}:
            return result
    except Exception:
        pass

    return "rag"


def _is_general_question(question: str) -> bool:
    q = question.lower()

    platform_keywords = [
        "investo", "portfolio", "dashboard", "my investments",
        "success score", "funding stage", "on the platform",
        "available startups"
    ]
    if any(k in q for k in platform_keywords):
        return False

    # FIXED: expanded keyword list to robustly catch macroeconomic
    # questions regardless of singular/plural or "what is"/"what are"
    # phrasing, so these reliably trigger web search rather than
    # relying solely on the LLM fallback classifier.
    general_keywords = [
        "what is", "what are", "how does", "how do", "explain", "define",
        "economy", "economic", "inflation", "interest rate", "exchange rate",
        "gdp", "market trend", "latest", "current", "recent", "news",
        "investment strategy", "due diligence", "stock market", "currency"
    ]
    if any(k in q for k in general_keywords):
        return True

    prompt = f"""
Classify this investor question as exactly one word:

- platform
- general

platform = asking about Investo/platform/startup data
general = asking about general investment/startup/economy knowledge,
          including macroeconomic topics like interest rates, inflation,
          exchange rates, or market conditions in any country

Question: {question}
""".strip()

    try:
        result = _llm.invoke(prompt).content.strip().lower()
        return result == "general"
    except Exception:
        return False


def _extract_startup_name(question: str, context: str, sources: list[str]) -> str | None:
    patterns = [
        r"similar(?: \w+)* to\s+[\"']?([^\"'\?\.,]+)[\"']?",
        r"startups like\s+[\"']?([^\"'\?\.,]+)[\"']?",
        r"companies like\s+[\"']?([^\"'\?\.,]+)[\"']?",
        r"projects like\s+[\"']?([^\"'\?\.,]+)[\"']?",
        r"alternatives to\s+[\"']?([^\"'\?\.,]+)[\"']?",
    ]
    for pattern in patterns:
        match = re.search(pattern, question, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()

    names_in_context = re.findall(r"^Name:\s*(.+)$", context, flags=re.MULTILINE)
    candidates = _dedupe(names_in_context + [s for s in sources if s != "platform data"])
    if len(candidates) == 1:
        return candidates[0]

    return None


# ── Main ──────────────────────────────────────────────────────────────
def query(
    question: str,
    conversation_history: list[dict],
    current_investor_id: str | None = None,
) -> tuple[str, list[dict], list[str]]:
    intent = _detect_intent(question)

    # 1) Recommendation path
    if intent == "recommend_startups":
        if current_investor_id is None:
            response_text = (
                "I can recommend startups for the signed-in investor, but I need the "
                "authenticated investor ID from the API request."
            )
            return response_text, _update_history(conversation_history, question, response_text), []

        try:
            results = _call_recommender(
                "/investor-to-projects",
                {
                    "investor_id": str(current_investor_id),
                    "n_recommendations": TOP_RECOMMENDATIONS,
                },
            )

            if not results:
                response_text = (
                    "I could not find startup recommendations for this investor right now."
                )
                return response_text, _update_history(conversation_history, question, response_text), ["recommendation system"]

            lines = ["Here are the top startup matches from the recommendation system:", ""]
            for i, item in enumerate(results, start=1):
                lines.append(
                    f"{i}. {item['startup_name']} — {item['industry']} in {item['country']}"
                )
                lines.append(
                    f"   Stage: {item['funding_stage']} | Success Score: {item['success_score']}/9 | "
                    f"Risk: {item['risk_level']} | Funding: ${item['total_funding']}M"
                )
                lines.append(
                    f"   Match Score: {item['final_score']} "
                    f"(ALS: {item['als_score'] if item['als_score'] is not None else 'cold-start'}, "
                    f"content: {item['content_score']})"
                )
            lines.append("")
            lines.append("These are suggestions, not guaranteed investment outcomes.")

            response_text = "\n".join(lines)
            sources = _dedupe(["recommendation system"] + [item["startup_name"] for item in results])
            return response_text, _update_history(conversation_history, question, response_text), sources

        except Exception as exc:
            response_text = f"I could not get recommendations right now. ({str(exc)})"
            return response_text, _update_history(conversation_history, question, response_text), []

    # 2) Similar-startups path
    if intent == "similar_startups":
        context, sources = _search_chroma(question, current_investor_id=current_investor_id)
        startup_name = _extract_startup_name(question, context, sources)

        if not startup_name:
            response_text = "Please mention the startup name clearly so I can find similar startups."
            return response_text, _update_history(conversation_history, question, response_text), []

        try:
            results = _call_recommender(
                "/similar-startups",
                {
                    "startup_name": startup_name,
                    "n_recommendations": TOP_RECOMMENDATIONS,
                },
            )

            if not results:
                response_text = f"I could not find startups similar to {startup_name} right now."
                return response_text, _update_history(conversation_history, question, response_text), [startup_name]

            lines = [f"Here are startups most similar to {startup_name}:", ""]
            for i, item in enumerate(results, start=1):
                lines.append(f"{i}. {item['startup_name']} — similarity {item['similarity_score']}")
                lines.append(
                    f"   {item['industry']} | {item['country']} | {item['funding_stage']} | "
                    f"Success Score: {item['success_score']}/9 | Risk: {item['risk_level']} | "
                    f"Funding: ${item['total_funding']}M"
                )
            lines.append("")
            lines.append("Use this as a comparison aid, not as a guarantee of similar outcomes.")

            response_text = "\n".join(lines)
            out_sources = _dedupe(["recommendation system", startup_name] + [item["startup_name"] for item in results])
            return response_text, _update_history(conversation_history, question, response_text), out_sources

        except Exception as exc:
            response_text = f"I could not fetch similar startups right now. ({str(exc)})"
            return response_text, _update_history(conversation_history, question, response_text), [startup_name]

    # 3) Normal RAG path
    context, sources = _search_chroma(question, current_investor_id=current_investor_id)

    web_context = ""
    if _is_general_question(question):
        web_context = _search_web(question)

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(
        system_prompt=SYSTEM_PROMPT,
        context=context if context else "No specific platform data found for this query.",
        web_context=web_context if web_context else "No web search was performed for this question.",
        history=_format_history(conversation_history),
        question=question,
    )

    try:
        response = _llm.invoke(prompt)
        response_text = response.content.strip()
    except Exception as exc:
        err_msg = str(exc)
        print(f"RAG EXCEPTION: {err_msg}", flush=True)
        if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
            response_text = "I am currently receiving too many requests. Please wait a moment and try again."
        else:
            response_text = f"I encountered an error processing your request: {err_msg}"

    return response_text, _update_history(conversation_history, question, response_text), sources
