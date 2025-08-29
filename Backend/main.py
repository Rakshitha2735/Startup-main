from fastapi import FastAPI, HTTPException, status, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
import os
import jwt
import requests
import asyncio
from datetime import datetime, timedelta
from typing import List, Optional
from pymongo import errors
from bson import ObjectId
import httpx
from dotenv import load_dotenv
from xml.etree import ElementTree
import re
from database import users_collection

# Database imports
from database import (
    hash_password, verify_password, create_access_token,
    get_user_by_id, get_user_profile, update_user_profile,
    create_roadmap, get_roadmap_by_id, get_user_roadmaps,
    update_roadmap, delete_roadmap, get_user_by_id,
    profiles_collection,ideas_collection
)

load_dotenv()

# Constants
DEVELOPER_EMAILS = {"ry352004@gmail.com"}
JWT_SECRET = os.environ.get("JWT_SECRET", "fallback_secret")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")  # Add this to your .env file
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/paper/search"
ARXIV_API = "https://export.arxiv.org/api/query"
CROSSREF_API = "https://api.crossref.org/works"

# Initialize FastAPI
app = FastAPI(
    title="Research Advisor API",
    description="API for fetching academic research papers",
    version="1.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer(auto_error=False)

# Models
class RegisterIn(BaseModel):
    name: str
    email: EmailStr
    password: str
    confirm_password: str

class LoginIn(BaseModel):
    email: EmailStr
    password: str

class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"

class IdeaInput(BaseModel):
    prompt: str

class IdeaResponse(BaseModel):
    prompt: str
    validation: str
    created_at: datetime

# Update the ProfileBase model to match your frontend changes
class ProfileBase(BaseModel):
    name: str
    email: EmailStr
    role: Optional[str] = None
    skills: Optional[List[str]] = None
    interests: Optional[List[str]] = None
    preferred_role: Optional[str] = None  # Changed from role_preference
    experience: Optional[str] = None      # Changed from experience_level
    availability: Optional[str] = None    # Changed from int to string
    location: Optional[str] = None

class ProfileCreate(ProfileBase):
    pass

class ProfileResponse(ProfileBase):
    user_id: str
    updated_at: datetime

class ResearchPaper(BaseModel):
    title: str
    authors: List[str]
    abstract: str
    published_date: str
    source: str
    url: str
    doi: Optional[str] = None

class ResearchRequest(BaseModel):
    idea: str
    max_results: int = 10

class ResearchResponse(BaseModel):
    papers: List[ResearchPaper]
    search_terms: List[str]

# Helper Functions
def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not credentials or not credentials.credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

def create_access_token_helper(subject: str, role: str = "user"):
    payload = {
        "sub": subject,
        "exp": datetime.utcnow() + timedelta(days=1),
        "role": role
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def call_groq_validation(prompt: str) -> str:
    if not GROQ_API_KEY:
        return f"Mock validation for: {prompt[:100]}... This startup idea has potential in the current market. Consider focusing on user acquisition and product-market fit."

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "system", "content": "You are a startup idea validation assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        if response.status_code != 200:
            return f"Validation service temporarily unavailable. Mock validation: {prompt[:100]}..."
        
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Groq API error: {e}")
        return f"Mock validation for: {prompt[:100]}... This startup idea shows promise. Focus on market research and MVP development."
def generate_search_terms(idea: str) -> List[str]:
    """Generate more precise search terms from the startup idea"""
    if not GROQ_API_KEY:
        # Enhanced fallback with domain-specific terms
        words = re.findall(r'\b\w{3,}\b', idea.lower())
        stop_words = {'the', 'and', 'for', 'with', 'that', 'this', 'your', 'have', 'from'}
        filtered_words = [word for word in words if word not in stop_words]
        
        # Add domain-specific terms based on context
        domain_terms = []
        idea_lower = idea.lower()
        
        if any(term in idea_lower for term in ['ai', 'artificial', 'machine learning', 'ml']):
            domain_terms.extend(['artificial intelligence', 'machine learning', 'neural networks'])
        if any(term in idea_lower for term in ['agriculture', 'farming', 'crop', 'soil']):
            domain_terms.extend(['agriculture', 'precision farming', 'crop yield'])
        if any(term in idea_lower for term in ['health', 'medical', 'patient', 'diagnosis']):
            domain_terms.extend(['healthcare', 'medical technology', 'clinical'])
        
        return (domain_terms + filtered_words)[:5]
    
    # Use Groq API for better term extraction
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # More specific prompt for academic research
    prompt = f"""
    Extract 3-5 precise technical and academic search terms from this startup idea: {idea}
    Focus on terms that would be effective for searching academic databases like Semantic Scholar, arXiv, and CrossRef.
    Return ONLY the terms separated by commas, no explanations or extra text.
    """
    
    payload = {
        "model": "llama3-8b-8192",
        "messages": [
            {
                "role": "system", 
                "content": "You are an expert research assistant. Extract precise academic search terms that would return highly relevant research papers."
            },
            {
                "role": "user", 
                "content": prompt
            }
        ],
        "temperature": 0.1,
        "max_tokens": 50
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        if response.status_code == 200:
            content = response.json()["choices"][0]["message"]["content"].strip()
            # Clean up the response
            terms = [term.strip().strip('"').strip("'") for term in content.split(",")]
            clean_terms = []
            for term in terms:
                if (term and len(term) > 2 and not term.isdigit() and 
                    not term.startswith("Here") and not term.lower() in ['the', 'and', 'for']):
                    clean_terms.append(term)
            
            if clean_terms:
                return clean_terms[:5]
    except Exception as e:
        print(f"Error generating search terms with Groq: {e}")
    
    # Fallback to the enhanced method
    return generate_search_terms.fallback(idea)
import asyncio
import httpx
from typing import List, Optional
import xml.etree.ElementTree as ET
import re
from datetime import datetime

async def fetch_semantic_scholar(search_terms: List[str], max_results: int) -> List[ResearchPaper]:
    """Fetch papers from Semantic Scholar with improved error handling"""
    try:
        query = " ".join(search_terms[:2])
        params = {
            "query": query,
            "limit": min(max_results, 10),
            "fields": "title,authors,abstract,year,url,externalIds,citationCount",
            "sort": "relevance"
        }

        headers = {
            "User-Agent": "Research-Advisor-API/1.0 (contact@researchadvisor.com)"
        }

        if SEMANTIC_SCHOLAR_API_KEY:
            headers["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(SEMANTIC_SCHOLAR_API, params=params, headers=headers)
            
            if response.status_code != 200:
                print(f"Semantic Scholar error {response.status_code}: {response.text[:200]}")
                return []
            
            data = response.json()
            papers = []
            
            for item in data.get("data", []):
                try:
                    title = item.get("title", "").strip()
                    if not title:
                        continue
                        
                    authors = [author.get("name", "") for author in item.get("authors", [])]
                    abstract = (item.get("abstract") or "")[:500] + "..." if item.get("abstract") else "No abstract available"
                    
                    papers.append(ResearchPaper(
                        title=title,
                        authors=authors,
                        abstract=abstract,
                        published_date=str(item.get("year", "")),
                        source="Semantic Scholar",
                        url=item.get("url", ""),
                        doi=item.get("externalIds", {}).get("DOI")
                    ))
                except Exception as e:
                    print(f"Error processing Semantic Scholar paper: {e}")
                    continue
                    
            return papers
            
    except Exception as e:
        print(f"Semantic Scholar fetch failed: {e}")
        return []
ARXIV_API = "https://export.arxiv.org/api/query"

async def fetch_arxiv(search_terms: List[str], max_results: int) -> List[ResearchPaper]:
    """Fetch papers from arXiv with query mapping + fallback"""
    try:
        if not search_terms:
            return []

        # --- Step 1: Build query ---
        query = "+OR+".join([f'all:{word}' for term in search_terms[:3] for word in term.split()[:2]])
        params = {
            "search_query": query,
            "start": 0,
            "max_results": min(max_results, 20),
            "sortBy": "relevance",
            "sortOrder": "descending"
        }

        headers = {
            "User-Agent": "Research-Advisor-API/1.0 (contact@researchadvisor.com)"
        }

        async with httpx.AsyncClient(timeout=45.0, follow_redirects=True) as client:
            response = await client.get(ARXIV_API, params=params, headers=headers)
            print("arXiv query:", params["search_query"])  # debug line

            if response.status_code != 200:
                print(f"arXiv error {response.status_code}: {response.text[:200]}")
                return []

            papers = parse_arxiv_response(response.text, max_results)

            # --- Step 3: Fallback if nothing found ---
            if not papers:
                print("No arXiv results → retrying with fallback terms...")
                fallback_terms = ["optimization", "reinforcement learning", "machine learning"]
                query = "+OR+".join([f'all:{t}' for t in fallback_terms])
                params["search_query"] = query
                response = await client.get(ARXIV_API, params=params, headers=headers)
                print("arXiv fallback query:", params["search_query"])
                if response.status_code == 200:
                    papers = parse_arxiv_response(response.text, max_results)

            return papers[:max_results]

    except Exception as e:
        print(f"arXiv fetch failed: {e}")
        return []


def parse_arxiv_response(xml_text: str, max_results: int) -> List[ResearchPaper]:
    """Helper to parse arXiv XML into ResearchPaper objects"""
    root = ET.fromstring(xml_text)
    papers = []

    # ✅ Use raw namespace, no "atom:" prefix
    for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
        try:
            title_elem = entry.find('{http://www.w3.org/2005/Atom}title')
            title = title_elem.text.strip() if title_elem is not None else "No title"

            summary_elem = entry.find('{http://www.w3.org/2005/Atom}summary')
            abstract = summary_elem.text.strip() if summary_elem is not None else "No abstract available"
            if len(abstract) > 500:
                abstract = abstract[:500] + "..."

            authors = []
            for author_elem in entry.findall('{http://www.w3.org/2005/Atom}author'):
                name_elem = author_elem.find('{http://www.w3.org/2005/Atom}name')
                if name_elem is not None and name_elem.text:
                    authors.append(name_elem.text.strip())

            published_elem = entry.find('{http://www.w3.org/2005/Atom}published')
            published_date = published_elem.text if published_elem is not None else ""

            id_elem = entry.find('{http://www.w3.org/2005/Atom}id')
            url = id_elem.text if id_elem is not None else ""

            papers.append(ResearchPaper(
                title=title,
                authors=authors,
                abstract=abstract,
                published_date=published_date,
                source="arXiv",
                url=url
            ))

        except Exception as e:
            print(f"Error processing arXiv entry: {e}")
            continue

    return papers[:max_results]


async def fetch_crossref(search_terms: List[str], max_results: int) -> List[ResearchPaper]:
    """Fetch papers from CrossRef with improved error handling"""
    try:
        query = " ".join(search_terms[:2])
        params = {
            "query": query,
            "rows": min(max_results, 20),
            "sort": "relevance",
            "select": "title,author,abstract,created,URL,DOI,published-print,published-online"
        }
        
        headers = {
            "User-Agent": "Research-Advisor-API/1.0 (mailto:contact@researchadvisor.com)"
        }
        
        async with httpx.AsyncClient(timeout=45.0) as client:
            response = await client.get(CROSSREF_API, params=params, headers=headers)
            
            if response.status_code != 200:
                print(f"CrossRef error {response.status_code}: {response.text[:200]}")
                return []
                
            data = response.json()
            papers = []
            
            for item in data.get("message", {}).get("items", []):
                try:
                    # Handle title (can be a list)
                    title_list = item.get("title", [])
                    title = " ".join(title_list) if isinstance(title_list, list) else str(title_list)
                    title = title.strip()
                    if not title:
                        continue
                    
                    # Handle abstract
                    abstract = item.get("abstract", "")
                    if not abstract:
                        abstract = "No abstract available"
                    if len(abstract) > 500:
                        abstract = abstract[:500] + "..."
                    
                    # Handle authors
                    authors = []
                    for author in item.get("author", []):
                        given = author.get("given", "")
                        family = author.get("family", "")
                        author_name = f"{given} {family}".strip()
                        if author_name:
                            authors.append(author_name)
                    
                    # Handle publication date
                    pub_date = ""
                    date_fields = ["published-print", "published-online", "created"]
                    for field in date_fields:
                        if field in item and "date-parts" in item[field]:
                            date_parts = item[field]["date-parts"][0]
                            if date_parts:
                                pub_date = "-".join(str(part) for part in date_parts[:3])
                                break
                    
                    papers.append(ResearchPaper(
                        title=title,
                        authors=authors,
                        abstract=abstract,
                        published_date=pub_date,
                        source="CrossRef",
                        url=item.get("URL", ""),
                        doi=item.get("DOI")
                    ))
                    
                except Exception as e:
                    print(f"Error processing CrossRef item: {e}")
                    continue
                    
            return papers
            
    except Exception as e:
        print(f"CrossRef fetch failed: {e}")
        return []

@app.post("/research-papers", response_model=ResearchResponse)
async def get_research_papers(request: ResearchRequest, current_user=Depends(get_current_user)):
    """Enhanced research papers endpoint enforcing ~15 results (5 from each source)"""
    print(f"Research request received: {request.idea[:50]}...")

    if not request.idea or not request.idea.strip():
        raise HTTPException(status_code=400, detail="Idea cannot be empty")

    try:
        # Generate search terms
        search_terms = generate_search_terms(request.idea)
        if not search_terms:
            search_terms = re.findall(r'\w{4,}', request.idea)[:3]  # Fallback

        print(f"Search terms: {search_terms}")

        # We want 5 per source → total 15
        per_source = max(1, request.max_results // 3)

        try:
            semantic_task = asyncio.create_task(fetch_semantic_scholar(search_terms, per_source))
            arxiv_task = asyncio.create_task(fetch_arxiv(search_terms, per_source))
            crossref_task = asyncio.create_task(fetch_crossref(search_terms, per_source))

            semantic_papers, arxiv_papers, crossref_papers = await asyncio.gather(
                semantic_task, arxiv_task, crossref_task,
                return_exceptions=True
            )

            if isinstance(semantic_papers, Exception):
                print(f"Semantic Scholar failed: {semantic_papers}")
                semantic_papers = []
            if isinstance(arxiv_papers, Exception):
                print(f"arXiv failed: {arxiv_papers}")
                arxiv_papers = []
            if isinstance(crossref_papers, Exception):
                print(f"CrossRef failed: {crossref_papers}")
                crossref_papers = []

        except asyncio.TimeoutError:
            print("Timeout fetching papers from all sources")
            semantic_papers, arxiv_papers, crossref_papers = [], [], []

        print(f"Results - Semantic: {len(semantic_papers)}, arXiv: {len(arxiv_papers)}, CrossRef: {len(crossref_papers)}")

        # Combine & deduplicate
        all_papers = semantic_papers + arxiv_papers + crossref_papers
        seen_titles = set()
        unique_papers = []
        for paper in all_papers:
            normalized_title = paper.title.lower().strip()
            if normalized_title and normalized_title not in seen_titles:
                seen_titles.add(normalized_title)
                unique_papers.append(paper)

        # Limit to requested max (default 15)
        final_papers = unique_papers[:request.max_results]

        print(f"Final papers: {len(final_papers)}")

        return ResearchResponse(
            papers=final_papers,
            search_terms=search_terms
        )

    except Exception as e:
        print(f"Error in research papers endpoint: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to fetch research papers: {str(e)}")

    """
    Check if a paper is relevant based on its abstract and search terms
    """
    if not abstract or not search_terms:
        return True  # Default to including if we can't determine
        
    abstract_lower = abstract.lower()
    term_matches = 0
    
    for term in search_terms:
        term_lower = term.lower()
        # Check for exact matches or close matches
        if term_lower in abstract_lower:
            term_matches += 1
        # Also check for word boundaries for better matching
        elif re.search(r'\b' + re.escape(term_lower) + r'\b', abstract_lower):
            term_matches += 1
    
    # Consider paper relevant if at least one search term matches
    # or if it's from a short list of terms, require more matches
    if len(search_terms) <= 3:
        return term_matches >= 1
    else:
        return term_matches >= 2
from typing import List, Optional
import datetime
from pymongo import MongoClient
from pymongo.errors import PyMongoError

# MongoDB Configuration
MONGODB_URI = "mongodb://localhost:27017/"  # Default local connection
DATABASE_NAME = "research"                  # As seen in your Compass screenshot
COLLECTION_NAME = "papers"                  # Customizable collection name

async def store_papers(papers: List[ResearchPaper]) -> bool:
    """Store papers in MongoDB with error handling"""
    try:
        client = MongoClient(MONGODB_URI)
        db = client[DATABASE_NAME]
        collection = db[COLLECTION_NAME]
        
        documents = [{
            "title": paper.title,
            "authors": paper.authors,
            "abstract": paper.abstract,
            "published_date": paper.published_date,
            "source": paper.source,
            "url": paper.url,
            "doi": paper.doi,
            "timestamp": datetime.datetime.now()
        } for paper in papers]
        
        if documents:
            result = collection.insert_many(documents)
            print(f"📀 Stored {len(result.inserted_ids)} papers in MongoDB")
            return True
        return False
        
    except PyMongoError as e:
        print(f"❌ MongoDB storage error: {e}")
        return False
    finally:
        client.close()

async def fetch_papers(
    search_terms: List[str],
    max_results: int = 15,
    store: bool = False
) -> List[ResearchPaper]:
    """
    Unified fetch function ensuring ~15 results (5 from each source).
    """

    per_source = max_results // 3  # e.g. 15 // 3 = 5

    # Fetch concurrently
    semantic, arxiv, crossref = await asyncio.gather(
        fetch_semantic_scholar(search_terms, per_source),
        fetch_arxiv(search_terms, per_source),
        fetch_crossref(search_terms, per_source)
    )

    all_papers = semantic + arxiv + crossref

    # Deduplicate by title
    seen_titles = set()
    unique_papers = []
    for paper in all_papers:
        title_norm = paper.title.strip().lower()
        if title_norm and title_norm not in seen_titles:
            seen_titles.add(title_norm)
            unique_papers.append(paper)

    # Limit to max_results (default 15)
    final_papers = unique_papers[:max_results]

    # Optional: store in MongoDB
    if store and final_papers:
        await store_papers(final_papers)

    print(f"✅ Returning {len(final_papers)} papers "
          f"(Semantic: {len(semantic)}, arXiv: {len(arxiv)}, CrossRef: {len(crossref)})")

    return final_papers

@app.post("/register", status_code=status.HTTP_201_CREATED)
def register(user: RegisterIn):
    if user.password != user.confirm_password:
        raise HTTPException(status_code=400, detail="Passwords do not match")

    hashed = hash_password(user.password)
    doc = {
        "name": user.name,
        "email": user.email.lower(),
        "password_hash": hashed,
        "created_at": datetime.utcnow()
    }
    try:
        res = users_collection.insert_one(doc)
        return {"id": str(res.inserted_id), "email": user.email}
    except errors.DuplicateKeyError:
        raise HTTPException(status_code=400, detail="Email already registered")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Registration failed")

@app.post("/login", response_model=TokenOut)
def login(credentials: LoginIn):
    user = users_collection.find_one({"email": credentials.email.lower()})
    if not user or not verify_password(credentials.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    role = "developer" if credentials.email.lower() in DEVELOPER_EMAILS else "user"
    token = create_access_token_helper(subject=str(user["_id"]), role=role)
    return {"access_token": token}

from datetime import datetime
from fastapi import Depends, HTTPException
from bson import ObjectId

@app.get("/dashboard-data")
def get_dashboard_data(current_user=Depends(get_current_user)):
    # First verify developer access
    if current_user.get("email") != "ry352004@gmail.com":
        raise HTTPException(
            status_code=403,
            detail="Access forbidden. Only ry352004@gmail.com can access this endpoint."
        )

    try:
        # Get non-developer users
        users_cursor = users_collection.find({
            "email": {"$ne": "ry352004@gmail.com"}
        })

        users_list = []
        for user in users_cursor:
            user_id = user["_id"]
            profile = profiles_collection.find_one({"user_id": user_id}) or {}
            validations = list(ideas_collection.find(
                {"user_id": user_id},
                {"_id": 0, "prompt": 1, "validation": 1, "created_at": 1}
            ).sort("created_at", -1).limit(10))  # Limit to 10 most recent

            # Convert datetime objects to ISO format strings
            users_list.append({
                "id": str(user_id),
                "name": user.get("name", "Unknown"),
                "email": user.get("email", ""),
                "created_at": user.get("created_at", datetime.utcnow()).isoformat(),
                "profile_data": {
                    "skills": profile.get("skills", []),
                    "interests": profile.get("interests", []),
                    "experience": profile.get("experience", ""),
                    "availability": profile.get("availability", ""),
                    "location": profile.get("location", ""),
                    "updated_at": profile.get("updated_at", datetime.utcnow()).isoformat() if profile.get("updated_at") else None
                },
                "validation_history": [
                    {
                        "prompt": v["prompt"],
                        "validation": v["validation"],
                        "created_at": v["created_at"].isoformat() if v.get("created_at") else None
                    } for v in validations
                ]
            })

        return {"users": users_list}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred: {str(e)}"
        )

    except Exception as e:
        print(f"Error fetching dashboard data: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to fetch dashboard data"
        )

# AI Validation + Suggestions (No Authentication)
# --------------------------------------------------

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ---------------- Existing Imports ----------------
# (your login/authentication imports here)
# --------------------------------------------------

# ---------------- New Imports ----------------
import json, re, requests, os
from datetime import datetime
from typing import List
from pydantic import BaseModel, Field
# ----------------------------------------------


# Environment variable for GROQ API (you'll need to set this)
import os
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Make sure to set this in your environment

# Pydantic models
class IdeaInput(BaseModel):
    prompt: str = Field(..., min_length=30, max_length=2000)

class ValidationScores(BaseModel):
    overall: int
    feasibility: int
    marketDemand: int
    uniqueness: int
    strength: int
    riskFactors: int

class ValidationDetails(BaseModel):
    verdict: str
    feasibility: str
    marketDemand: str
    uniqueness: str
    strength: str
    riskFactors: str
    existingCompetitors: str

class Suggestions(BaseModel):
    critical: List[str]
    recommended: List[str]
    optional: List[str]

class ValidationResponse(BaseModel):
    prompt: str
    validation: ValidationDetails
    scores: ValidationScores
    suggestions: Suggestions
    created_at: datetime

def call_groq_validation(prompt: str) -> dict:
    """
    Enhanced validation function using the comprehensive AI prompt system
    """
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set in environment")

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    system_prompt = """You are an AI Startup Validator for "Startup GPS". 
Your role is to provide dynamic, detailed, and actionable startup validation reports based on the user's idea. 
Do not use any static or placeholder data. Always analyze the user's input deeply.

### Instructions:
1. Analyze the startup idea comprehensively across these dimensions:
   - **Overall Validation Score (0-100%)**
   - **Feasibility** (technical & operational)
   - **Market Demand**
   - **Uniqueness / Differentiation**
   - **Strengths**
   - **Risk Factors**

2. For each dimension:
   - Give a **score (0-100)**  
   - Write a **detailed explanation** with specific insights relevant to the idea.

3. Provide three clear sections of suggestions:
   - **Critical Improvements (⚠ must-do fixes)** – 3 to 5 items
   - **Recommended Enhancements (✓ should-do improvements)** – 3 to 5 items
   - **Optional Considerations (• nice-to-have ideas)** – 2 to 4 items

4. Add detailed analysis for each dimension:
   - Feasibility (timeline, complexity, scalability)
   - Market Demand (target audience, adoption signals, growth potential)
   - Uniqueness (differentiation vs competitors, barriers to entry)
   - Strength (value proposition, monetization, scalability potential)
   - Risk Factors (competition, adoption, finance, tech)
   - Existing Competitors (real names where possible, differentiation opportunities)

5. Always adapt output **directly to the user's startup idea**.
   - Do NOT give generic or repeated responses.
   - Each section must be grounded in the specific industry/domain of the idea.
   - Avoid filler text.

### Response Format (JSON):
{
  "overall_score": 85,
  "scores": {
    "feasibility": 78,
    "market_demand": 82,
    "uniqueness": 65,
    "strength": 81,
    "risk_factors": 74
  },
  "analysis": {
    "verdict": "Detailed overall assessment...",
    "feasibility": "Technical and operational analysis...",
    "market_demand": "Market size, audience, adoption potential...",
    "uniqueness": "Differentiation analysis...",
    "strength": "Core advantages and value proposition...",
    "risk_factors": "Key risks and mitigation strategies...",
    "existing_competitors": "Real competitor analysis..."
  },
  "suggestions": {
    "critical": ["Item 1", "Item 2", "Item 3"],
    "recommended": ["Item 1", "Item 2", "Item 3"],
    "optional": ["Item 1", "Item 2"]
  }
}

Provide ONLY the JSON response with no additional text."""

    user_prompt = f"Please validate this startup idea comprehensively: {prompt}"

    payload = {
        "model": "llama3-70b-8192",  # Using more powerful model for better analysis
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.3,  # Lower temperature for more consistent analysis
        "max_tokens": 4000
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Failed to get response from Groq API")

        data = response.json()
        ai_text = data["choices"][0]["message"]["content"]
        
        # Clean up the response to ensure it's valid JSON
        ai_text = ai_text.strip()
        if ai_text.startswith("```json"):
            ai_text = ai_text[7:]
        if ai_text.endswith("```"):
            ai_text = ai_text[:-3]
        ai_text = ai_text.strip()
        
        # Parse JSON response
        try:
            result = json.loads(ai_text)
        except json.JSONDecodeError:
            # Fallback: try to extract data using regex patterns
            result = parse_fallback_response(ai_text)
        
        return result
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"API request failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation processing failed: {str(e)}")

def parse_fallback_response(text: str) -> dict:
    """
    Fallback parser in case JSON parsing fails
    """
    # Extract scores using regex patterns
    overall_score = extract_score(text, r"overall[_\s]*score[\"']?\s*:\s*(\d+)")
    feasibility_score = extract_score(text, r"feasibility[\"']?\s*:\s*(\d+)")
    market_score = extract_score(text, r"market[_\s]*demand[\"']?\s*:\s*(\d+)")
    uniqueness_score = extract_score(text, r"uniqueness[\"']?\s*:\s*(\d+)")
    strength_score = extract_score(text, r"strength[\"']?\s*:\s*(\d+)")
    risk_score = extract_score(text, r"risk[_\s]*factors[\"']?\s*:\s*(\d+)")
    
    # Extract analysis sections
    verdict = extract_section(text, r"verdict[\"']?\s*:\s*[\"'](.*?)[\"']", "Strong potential identified with key areas for development.")
    feasibility = extract_section(text, r"feasibility[\"']?\s*:\s*[\"'](.*?)[\"']", "Technical implementation appears feasible with proper planning.")
    market_demand = extract_section(text, r"market[_\s]*demand[\"']?\s*:\s*[\"'](.*?)[\"']", "Market shows promising demand indicators.")
    uniqueness = extract_section(text, r"uniqueness[\"']?\s*:\s*[\"'](.*?)[\"']", "Concept demonstrates notable differentiation opportunities.")
    strength = extract_section(text, r"strength[\"']?\s*:\s*[\"'](.*?)[\"']", "Core strengths provide solid foundation for growth.")
    risk_factors = extract_section(text, r"risk[_\s]*factors[\"']?\s*:\s*[\"'](.*?)[\"']", "Manageable risks identified with mitigation strategies available.")
    competitors = extract_section(text, r"competitors[\"']?\s*:\s*[\"'](.*?)[\"']", "Competitive landscape analysis reveals positioning opportunities.")
    
    # Extract suggestions
    critical = extract_suggestions(text, "critical")
    recommended = extract_suggestions(text, "recommended")
    optional = extract_suggestions(text, "optional")
    
    return {
        "overall_score": overall_score,
        "scores": {
            "feasibility": feasibility_score,
            "market_demand": market_score,
            "uniqueness": uniqueness_score,
            "strength": strength_score,
            "risk_factors": risk_score
        },
        "analysis": {
            "verdict": verdict,
            "feasibility": feasibility,
            "market_demand": market_demand,
            "uniqueness": uniqueness,
            "strength": strength,
            "risk_factors": risk_factors,
            "existing_competitors": competitors
        },
        "suggestions": {
            "critical": critical,
            "recommended": recommended,
            "optional": optional
        }
    }

def extract_score(text: str, pattern: str) -> int:
    """Extract numeric score from text using regex pattern"""
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return min(100, max(0, int(match.group(1))))
    return 70  # Default score

def extract_section(text: str, pattern: str, default: str) -> str:
    """Extract analysis section from text"""
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()[:500]  # Limit length
    return default

def extract_suggestions(text: str, category: str) -> List[str]:
    """Extract suggestion items from text"""
    # Look for the category followed by suggestions
    pattern = rf"{category}[\"']?\s*:\s*\[(.*?)\]"
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    
    if match:
        suggestions_text = match.group(1)
        # Parse individual suggestions
        suggestions = re.findall(r'["\']([^"\']+)["\']', suggestions_text)
        return suggestions[:5]  # Limit to 5 suggestions
    
    # Fallback: look for bullet points or numbered lists
    lines = text.split('\n')
    suggestions = []
    in_category = False
    
    for line in lines:
        if category.lower() in line.lower():
            in_category = True
            continue
        if in_category:
            if line.strip().startswith(('-', '•', '*')) or re.match(r'^\d+\.', line.strip()):
                suggestion = re.sub(r'^[-•*\d\.\s]+', '', line.strip())
                if suggestion:
                    suggestions.append(suggestion[:100])  # Limit length
            elif line.strip() and not line.startswith(' '):
                break
    
    return suggestions[:5] if suggestions else [f"No specific {category} suggestions identified"]

# API Endpoints

@app.post("/validate-idea", response_model=ValidationResponse)
async def validate_idea(idea: IdeaInput):
    """
    Enhanced idea validation endpoint with comprehensive AI analysis (No Authentication Required)
    """
    try:
        # Get AI validation
        ai_result = call_groq_validation(idea.prompt)
        
        # Structure the response to match frontend expectations
        validation_response = ValidationResponse(
            prompt=idea.prompt,
            validation=ValidationDetails(
                verdict=ai_result["analysis"]["verdict"],
                feasibility=ai_result["analysis"]["feasibility"],
                marketDemand=ai_result["analysis"]["market_demand"],
                uniqueness=ai_result["analysis"]["uniqueness"],
                strength=ai_result["analysis"]["strength"],
                riskFactors=ai_result["analysis"]["risk_factors"],
                existingCompetitors=ai_result["analysis"]["existing_competitors"]
            ),
            scores=ValidationScores(
                overall=ai_result["overall_score"],
                feasibility=ai_result["scores"]["feasibility"],
                marketDemand=ai_result["scores"]["market_demand"],
                uniqueness=ai_result["scores"]["uniqueness"],
                strength=ai_result["scores"]["strength"],
                riskFactors=ai_result["scores"]["risk_factors"]
            ),
            suggestions=Suggestions(
                critical=ai_result["suggestions"]["critical"],
                recommended=ai_result["suggestions"]["recommended"],
                optional=ai_result["suggestions"]["optional"]
            ),
            created_at=datetime.utcnow()
        )
        
        # You can optionally save to database here if needed
        # For now, just return the response
        
        return validation_response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

@app.post("/validate-idea", response_model=IdeaResponse)
def validate_idea(idea: IdeaInput, current_user=Depends(get_current_user)):
    ai_result = call_groq_validation(idea.prompt)

    idea_doc = {
        "user_id": current_user["_id"],
        "prompt": idea.prompt,
        "validation": ai_result["validation"],
        "score": ai_result["score"],
        "suggestions": ai_result["suggestions"],
        "created_at": datetime.utcnow()
    }
    ideas_collection.insert_one(idea_doc)

    return IdeaResponse(
        prompt=idea.prompt,
        validation=ai_result["validation"],
        score=ai_result["score"],
        suggestions=ai_result["suggestions"],
        created_at=idea_doc["created_at"],
    )


@app.post("/profile", response_model=ProfileResponse)
async def create_or_update_profile(
    profile: ProfileCreate,
    current_user=Depends(get_current_user)
):
    profile_data = profile.dict()
    profile_data.update({
        "user_id": current_user["_id"],
        "updated_at": datetime.utcnow()
    })
    
    try:
        update_user_profile(current_user["_id"], profile_data)
        updated_profile = get_user_profile(current_user["_id"])
        if not updated_profile:
            raise HTTPException(status_code=400, detail="Profile not saved correctly")
        updated_profile["user_id"] = str(updated_profile["user_id"])
        return updated_profile
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/profile", response_model=ProfileResponse)
async def edit_profile(
    profile: ProfileCreate,
    current_user=Depends(get_current_user)
):
    existing_profile = get_user_profile(current_user["_id"])
    if not existing_profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    
    profile_data = profile.dict()
    profile_data.update({
        "user_id": current_user["_id"],
        "updated_at": datetime.utcnow()
    })
    
    try:
        update_user_profile(current_user["_id"], profile_data)
        updated_profile = get_user_profile(current_user["_id"])
        updated_profile["user_id"] = str(updated_profile["user_id"])
        return updated_profile
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/profile", response_model=ProfileResponse)
async def get_profile(current_user=Depends(get_current_user)):
    profile = get_user_profile(current_user["_id"])
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    profile["user_id"] = str(profile["user_id"])
    return profile

@app.get("/health")
def health_check():
    """Health check endpoint - no authentication required"""
    return {
        "status": "healthy",
        "message": "Research Advisor API is running",
        "timestamp": datetime.utcnow().isoformat(),
        "endpoints": {
            "research_papers": "POST /research-papers (requires auth)",
            "debug_sources": "GET /debug/sources (requires auth)",
            "test_sources": "GET /debug/test-sources (requires auth)",
            "health": "GET /health (public)"
        }
    }

@app.get("/debug/research-sources")
async def debug_research_sources(
    query: str = Query("artificial intelligence agriculture"),
    max_results: int = Query(5),
    current_user=Depends(get_current_user)
):
    """Debug endpoint to test each research source individually"""
    
    search_terms = query.split()[:3]
    
    results = {}
    
    # Test each source
    results["semantic_scholar"] = await fetch_semantic_scholar(search_terms, max_results)
    results["arxiv"] = await fetch_arxiv(search_terms, max_results)
    results["crossref"] = await fetch_crossref(search_terms, max_results)
    
    return {
        "search_terms": search_terms,
        "results": {
            source: {
                "count": len(papers),
                "papers": [{"title": p.title, "source": p.source} for p in papers[:3]]
            } for source, papers in results.items()
        }
    }

import asyncio
import re
import logging
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
from pydantic import BaseModel, Field
import requests
from fastapi import HTTPException, Depends

# Enhanced Pydantic models
class RoadmapInput(BaseModel):
    prompt: str
    timeframe: str

class RoadmapResponse(BaseModel):
    id: str
    prompt: str
    timeframe: str
    roadmap: str
    created_at: datetime
    updated_at: datetime
    user_id: str

class RoadmapUpdate(BaseModel):
    prompt: Optional[str] = None
    timeframe: Optional[str] = None
    roadmap: Optional[str] = None

# Add these new endpoints to your existing FastAPI app
@app.post("/roadmaps", response_model=RoadmapResponse)
async def create_roadmap_endpoint(
    roadmap_input: RoadmapInput,
    current_user: dict = Depends(get_current_user)
):
    # Call AI
    roadmap_text = call_groq_roadmap(roadmap_input.prompt, roadmap_input.timeframe)

    # Save to DB
    roadmap_data = {
        "prompt": roadmap_input.prompt,
        "timeframe": roadmap_input.timeframe,
        "roadmap": roadmap_text,
        "user_id": current_user["_id"]
    }

    roadmap_id = create_roadmap(str(current_user["_id"]), roadmap_data)


    # Return response
    return RoadmapResponse(
        id=str(roadmap_id),
        prompt=roadmap_input.prompt,
        timeframe=roadmap_input.timeframe,
        roadmap=roadmap_text,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        user_id=str(current_user["_id"])
    )


@app.get("/roadmaps/{roadmap_id}", response_model=RoadmapResponse)
async def get_roadmap(
    roadmap_id: str,
    current_user: dict = Depends(get_current_user)
):
    roadmap = get_roadmap_by_id(roadmap_id)
    if not roadmap:
        raise HTTPException(status_code=404, detail="Roadmap not found")
    
    # Verify the requesting user owns this roadmap
    if roadmap["user_id"] != str(current_user["_id"]):
        raise HTTPException(status_code=403, detail="Not authorized to access this roadmap")
    
    return roadmap

@app.get("/users/{user_id}/roadmaps", response_model=List[RoadmapResponse])
async def get_user_roadmaps(
    user_id: str,
    current_user: dict = Depends(get_current_user)
):
    # Verify the requesting user is accessing their own roadmaps
    if user_id != str(current_user["_id"]):
        raise HTTPException(status_code=403, detail="Not authorized to access these roadmaps")
    
    roadmaps = get_user_roadmaps(user_id)
    return roadmaps

@app.put("/roadmaps/{roadmap_id}", response_model=RoadmapResponse)
async def update_roadmap_endpoint(
    roadmap_id: str,
    update_data: RoadmapUpdate,
    current_user: dict = Depends(get_current_user)
):
    # First verify the roadmap exists and belongs to this user
    roadmap = get_roadmap_by_id(roadmap_id)
    if not roadmap:
        raise HTTPException(status_code=404, detail="Roadmap not found")
    if roadmap["user_id"] != str(current_user["_id"]):
        raise HTTPException(status_code=403, detail="Not authorized to update this roadmap")
    
    # Perform the update
    updated = update_roadmap(roadmap_id, update_data.dict(exclude_unset=True))
    if updated.modified_count == 0:
        raise HTTPException(status_code=404, detail="Roadmap not found or no changes made")
    
    # Return the updated roadmap
    updated_roadmap = get_roadmap_by_id(roadmap_id)
    return updated_roadmap

@app.delete("/roadmaps/{roadmap_id}")
async def delete_roadmap_endpoint(
    roadmap_id: str,
    current_user: dict = Depends(get_current_user)
):
    # First verify the roadmap exists and belongs to this user
    roadmap = get_roadmap_by_id(roadmap_id)
    if not roadmap:
        raise HTTPException(status_code=404, detail="Roadmap not found")
    if roadmap["user_id"] != str(current_user["_id"]):
        raise HTTPException(status_code=403, detail="Not authorized to delete this roadmap")
    
    # Perform the deletion
    deleted = delete_roadmap(roadmap_id)
    if deleted.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Roadmap not found")
    
    return {"message": "Roadmap deleted successfully"}

# Add this helper function
def call_groq_roadmap(prompt: str, timeframe: str) -> str:
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set in environment")

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    system_prompt = """You are a startup roadmap specialist. Generate a detailed, actionable roadmap based on the provided idea and timeframe. Structure your response EXACTLY as follows:

Overview:
[Provide a 3-4 sentence high-level summary of the entire roadmap]

Phase 1: [Phase Name] - [Brief one-line description]
Tasks:
- Task 1 (specific action item)
- Task 2
- Task 3
- Task 4
Implementation:
- How to accomplish this phase (3-4 specific steps)
- Resources needed
- Team members involved
- Potential challenges

Phase 2: [Phase Name] - [Brief one-line description]
Tasks:
- Task 1
- Task 2
- Task 3
- Task 4
Implementation:
- How to accomplish this phase (3-4 specific steps)
- Resources needed
- Team members involved
- Potential challenges

[Continue with Phase 3, 4, etc. as needed based on timeframe]

Each phase should have:
1. A clear name and one-line description
2. 4-5 specific tasks (bullet points under "Tasks")
3. 3-4 implementation details (bullet points under "Implementation")
4. Adjust the number of phases according to the specified timeframe (3 months = 3 phases, 6 months = 5 phases, etc.)"""

    payload = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Create a roadmap for: {prompt}\nTimeframe: {timeframe}"}
        ],
        "temperature": 0.6
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Groq API request failed: {str(e)}")


@app.get("/")
def root():
    return {
        "message": "Startup GPS backend is running with authentication!",
        "status": "healthy",
        "version": "1.0.0"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)