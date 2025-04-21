import cohere
import os
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()
API_KEY = os.getenv("COHERE_API_KEY")

co = cohere.Client(API_KEY)

def get_embedding(text):
    response = co.embed(
        texts=[text],
        model="embed-english-v3.0",
        input_type="search_document"
    )
    return response.embeddings[0]

def get_similarity_score(resume_text, jd_text):
    resume_vec = get_embedding(resume_text)
    jd_vec = get_embedding(jd_text)
    score = cosine_similarity([resume_vec], [jd_vec])[0][0]
    return score

def generate_resume_suggestions(resume, jd):
    prompt = (
        "You are a career advisor. Suggest specific improvements to this resume "
        "so it better matches the following job description.\n\n"
        f"Resume:\n{resume}\n\n"
        f"Job Description:\n{jd}\n\n"
        "Suggestions:"
    )

    response = co.generate(
        model="command",  # free-tier friendly model
        prompt=prompt,
        max_tokens=300,
        temperature=0.7,
    )
    return response.generations[0].text.strip()
