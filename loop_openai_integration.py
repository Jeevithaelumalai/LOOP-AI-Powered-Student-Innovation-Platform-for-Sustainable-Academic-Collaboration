"""
LOOP — AI-Powered Student Recommendation System
OpenAI API Integration (Embeddings + LLM Re-ranking)

Pipeline:
  1. Embed student profiles using text-embedding-ada-002
  2. Find similar profiles using cosine similarity
  3. Re-rank shortlist using GPT-4 for intelligent matching
  4. Return ranked matches with explanations
"""

import json
from openai import OpenAI
import numpy as np

# ── Setup ──────────────────────────────────────────────────────────────────────
client = OpenAI(api_key="your-api-key-here")  # Replace with your key

# ── Step 1: Embed a student profile ───────────────────────────────────────────
def get_embedding(text: str) -> list[float]:
    """
    Converts a student profile text into a vector using OpenAI embeddings.
    Model: text-embedding-ada-002  →  returns 1536-dimensional vector
    """
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding


# ── Step 2: Cosine similarity between two vectors ─────────────────────────────
def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Measures how similar two embeddings are.
    Score: 1.0 = identical, 0.0 = unrelated, -1.0 = opposite
    """
    a = np.array(vec_a)
    b = np.array(vec_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ── Step 3: Find top-N similar students ───────────────────────────────────────
def find_similar_students(
    query_profile: dict,
    all_profiles: list[dict],
    top_n: int = 5
) -> list[dict]:
    """
    Embeds the query profile and scores it against all stored profiles.
    Returns the top-N most similar students with their scores.
    """
    # Build a natural language description from structured profile data
    query_text = build_profile_text(query_profile)
    query_embedding = get_embedding(query_text)

    scored = []
    for profile in all_profiles:
        if profile["id"] == query_profile["id"]:
            continue  # Skip self

        profile_text = build_profile_text(profile)
        profile_embedding = get_embedding(profile_text)
        score = cosine_similarity(query_embedding, profile_embedding)

        scored.append({
            "profile": profile,
            "similarity_score": round(score, 4)
        })

    # Sort by highest similarity
    scored.sort(key=lambda x: x["similarity_score"], reverse=True)
    return scored[:top_n]


# ── Helper: Convert profile dict → natural language text ──────────────────────
def build_profile_text(profile: dict) -> str:
    """
    Converts structured student data into a rich natural language string.
    This is fed to the embedding model.

    NOTE: Plain comma-separated skills give poor embeddings.
    Full sentences give much richer, more meaningful vectors.
    """
    return (
        f"Student named {profile['name']}. "
        f"Skills: {', '.join(profile['skills'])}. "
        f"Interests: {', '.join(profile['interests'])}. "
        f"Project goal: {profile['project_goal']}. "
        f"Experience level: {profile['experience_level']}."
    )


# ── Step 4: LLM Re-ranking + Explanation ──────────────────────────────────────
def llm_rerank(
    query_profile: dict,
    candidates: list[dict]
) -> list[dict]:
    """
    Sends the shortlisted candidates to GPT-4 for intelligent re-ranking.
    The LLM understands context: complementary skills, project fit, etc.
    Returns candidates re-ranked with a plain-language explanation for each.
    """
    candidates_text = "\n".join([
        f"{i+1}. {build_profile_text(c['profile'])} "
        f"(similarity score: {c['similarity_score']})"
        for i, c in enumerate(candidates)
    ])

    prompt = f"""
You are a student collaboration matchmaker for a platform called LOOP.

A student is looking for teammates:
{build_profile_text(query_profile)}

Here are the top candidate matches (pre-ranked by embedding similarity):
{candidates_text}

Re-rank these candidates based on:
- Complementary skills (not just identical skills)
- Project goal alignment
- Experience level compatibility

Respond ONLY with valid JSON in this format (no markdown, no preamble):
{{
  "ranked": [
    {{
      "rank": 1,
      "name": "candidate name",
      "reason": "one sentence explaining why this is a good match"
    }}
  ]
}}
"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You are an expert at matching students for project collaboration."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3  # Lower = more consistent, deterministic output
    )

    raw = response.choices[0].message.content
    result = json.loads(raw)

    # Merge LLM ranking with original candidate data
    name_to_candidate = {c["profile"]["name"]: c for c in candidates}
    ranked_output = []
    for item in result["ranked"]:
        candidate = name_to_candidate.get(item["name"], {})
        ranked_output.append({
            "rank": item["rank"],
            "profile": candidate.get("profile", {}),
            "similarity_score": candidate.get("similarity_score", 0),
            "reason": item["reason"]
        })

    return ranked_output


# ── Full pipeline ──────────────────────────────────────────────────────────────
def recommend_teammates(
    query_profile: dict,
    all_profiles: list[dict],
    top_n: int = 5
) -> list[dict]:
    """
    End-to-end recommendation pipeline:
      embed → similarity search → LLM re-rank → return results
    """
    print(f"\n[1] Finding matches for: {query_profile['name']}")
    candidates = find_similar_students(query_profile, all_profiles, top_n)
    print(f"[2] Retrieved {len(candidates)} candidates via embedding similarity")

    print("[3] Re-ranking with GPT-4...")
    ranked = llm_rerank(query_profile, candidates)
    print("[4] Done.\n")

    return ranked


# ── Sample data & demo ─────────────────────────────────────────────────────────
if __name__ == "__main__":

    # The student looking for teammates
    query_student = {
        "id": "s001",
        "name": "Jeevitha",
        "skills": ["Python", "Machine Learning", "FastAPI"],
        "interests": ["AI", "recommendation systems", "data pipelines"],
        "project_goal": "Build an AI-powered student collaboration platform",
        "experience_level": "intermediate"
    }

    # Other students in the platform
    student_pool = [
        {
            "id": "s002",
            "name": "Arjun",
            "skills": ["React", "JavaScript", "UI/UX"],
            "interests": ["frontend", "design systems", "web apps"],
            "project_goal": "Build beautiful user interfaces for AI tools",
            "experience_level": "intermediate"
        },
        {
            "id": "s003",
            "name": "Priya",
            "skills": ["Java", "Spring Boot", "REST APIs"],
            "interests": ["backend systems", "microservices", "cloud"],
            "project_goal": "Design scalable backend for collaborative platforms",
            "experience_level": "advanced"
        },
        {
            "id": "s004",
            "name": "Rahul",
            "skills": ["Python", "NLP", "transformers"],
            "interests": ["large language models", "text processing"],
            "project_goal": "Research LLM applications in education",
            "experience_level": "advanced"
        },
        {
            "id": "s005",
            "name": "Sneha",
            "skills": ["SQL", "Power BI", "data analysis"],
            "interests": ["analytics", "business intelligence"],
            "project_goal": "Visualise student engagement data",
            "experience_level": "beginner"
        }
    ]

    results = recommend_teammates(query_student, student_pool)

    print("=" * 55)
    print("LOOP — Recommended Teammates for Jeevitha")
    print("=" * 55)
    for r in results:
        print(f"\nRank #{r['rank']}  {r['profile'].get('name', 'Unknown')}")
        print(f"  Skills : {', '.join(r['profile'].get('skills', []))}")
        print(f"  Score  : {r['similarity_score']}")
        print(f"  Why    : {r['reason']}")
