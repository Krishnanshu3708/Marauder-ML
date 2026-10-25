"""Skill gap analysis module for CareerFit ML.

Provides maintainable skill taxonomy, alias normalization, and skill gap
comparison between candidates and job descriptions.
"""

from dataclasses import asdict, dataclass
import re
from typing import Any, Optional, Sequence, Union


# Maintainable dictionary of canonical skills mapped by domain
SKILL_TAXONOMY: dict[str, list[str]] = {
    "programming_languages": [
        "python", "javascript", "typescript", "java", "c++", "c#", "go",
        "rust", "ruby", "php", "swift", "kotlin", "scala", "r", "sql", "html", "css"
    ],
    "frameworks_and_libraries": [
        "react", "angular", "vue", "nextjs", "nodejs", "express", "fastapi",
        "django", "flask", "spring boot", "asp.net", "tensorflow", "pytorch",
        "scikit-learn", "keras", "pandas", "numpy", "tailwind css", "bootstrap"
    ],
    "cloud_and_devops": [
        "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ansible",
        "jenkins", "ci/cd", "linux", "git", "github actions", "prometheus", "grafana"
    ],
    "databases_and_storage": [
        "postgresql", "mysql", "mongodb", "redis", "elasticsearch", "sqlite",
        "oracle", "dynamodb", "cassandra", "neo4j", "snowflake"
    ],
    "ai_and_data_science": [
        "machine learning", "deep learning", "natural language processing",
        "computer vision", "data analysis", "data engineering", "big data",
        "spark", "hadoop", "llm", "generative ai", "reinforcement learning"
    ],
    "software_engineering_practices": [
        "rest api", "graphql", "microservices", "agile", "scrum", "unit testing",
        "system design", "object oriented programming", "data structures"
    ],
}

# Flat set of all canonical skills
ALL_CANONICAL_SKILLS: set[str] = {
    skill for domain_skills in SKILL_TAXONOMY.values() for skill in domain_skills
}

# Alias mapping from abbreviations and colloquial variants to canonical skill names
SKILL_ALIASES: dict[str, str] = {
    # Programming languages
    "js": "javascript",
    "py": "python",
    "python3": "python",
    "ts": "typescript",
    "golang": "go",
    "cpp": "c++",
    "c plus plus": "c++",
    "csharp": "c#",
    "c sharp": "c#",
    ".net": "asp.net",
    "dotnet": "asp.net",
    
    # Web & Frameworks
    "react.js": "react",
    "reactjs": "react",
    "vue.js": "vue",
    "vuejs": "vue",
    "next.js": "nextjs",
    "node": "nodejs",
    "node.js": "nodejs",
    "express.js": "express",
    "expressjs": "express",
    "spring": "spring boot",
    "springboot": "spring boot",
    "sklearn": "scikit-learn",
    "scikit": "scikit-learn",
    "tf": "tensorflow",
    "torch": "pytorch",
    "tailwind": "tailwind css",

    # Cloud & DevOps
    "amazon web services": "aws",
    "google cloud": "gcp",
    "google cloud platform": "gcp",
    "microsoft azure": "azure",
    "k8s": "kubernetes",
    "kube": "kubernetes",
    "cicd": "ci/cd",
    "continuous integration": "ci/cd",

    # Databases
    "postgres": "postgresql",
    "psql": "postgresql",
    "mongo": "mongodb",
    "elastic": "elasticsearch",

    # AI & ML
    "ml": "machine learning",
    "dl": "deep learning",
    "nlp": "natural language processing",
    "cv": "computer vision",
    "genai": "generative ai",
    "large language models": "llm",

    # Software Engineering
    "rest": "rest api",
    "restful": "rest api",
    "restful api": "rest api",
    "rest apis": "rest api",
    "oop": "object oriented programming",
    "dsa": "data structures",
}


def normalize_skill(skill: Optional[str]) -> str:
    """Normalize a raw skill string to its canonical taxonomy form.

    Args:
        skill: Raw skill string (e.g. 'React.js', 'K8s', 'py').

    Returns:
        Canonical lowercase skill name or normalized string.
    """
    if not skill or not isinstance(skill, str):
        return ""
    
    cleaned = skill.strip().lower()
    # Check direct alias
    if cleaned in SKILL_ALIASES:
        return SKILL_ALIASES[cleaned]
    # Check if already canonical
    if cleaned in ALL_CANONICAL_SKILLS:
        return cleaned

    return cleaned


def extract_skills(
    text: Optional[str],
    custom_skills: Optional[Sequence[str]] = None,
) -> set[str]:
    """Extract recognized technical skills from unstructured text.

    Uses word-boundary regex matching against canonical skills and aliases.

    Args:
        text: Raw text from resume or job description.
        custom_skills: Optional additional skill keywords to scan for.

    Returns:
        Set of canonical skill names found in the text.
    """
    if not text or not isinstance(text, str):
        return set()

    found_skills: set[str] = set()
    text_lower = text.lower()

    # Build lookup dictionary combining aliases, canonical skills, and custom skills
    lookup: dict[str, str] = {}
    for canonical in ALL_CANONICAL_SKILLS:
        lookup[canonical] = canonical
    for alias, canonical in SKILL_ALIASES.items():
        lookup[alias] = canonical
    if custom_skills:
        for skill in custom_skills:
            norm = normalize_skill(skill)
            if norm:
                lookup[norm] = norm

    # Sort candidates by length descending to match multi-word phrases first
    # e.g., 'machine learning' before 'machine'
    candidates = sorted(lookup.keys(), key=len, reverse=True)

    for candidate in candidates:
        # Regex boundary check that safely handles special characters like c++, c#, .net
        escaped = re.escape(candidate)
        pattern = rf"(?<!\w){escaped}(?!\w)"
        if re.search(pattern, text_lower):
            found_skills.add(lookup[candidate])

    return found_skills


@dataclass
class SkillGapResult:
    """Structured report of skill gap analysis."""

    matched_skills: list[str]       # Skills present in both resume and job
    missing_skills: list[str]       # Required job skills absent from resume
    additional_skills: list[str]    # Candidate skills not required by job
    matched_count: int              # Total count of matched skills
    missing_count: int              # Total count of missing skills
    total_job_skills: int           # Total skills identified in job description
    match_percentage: float         # Percentage of job skills met (0.0 to 100.0)
    readiness_level: str            # High, Moderate, Low, or No Skills Found

    def to_dict(self) -> dict[str, Any]:
        """Convert result to standard dictionary."""
        return asdict(self)


def analyze_skill_gap(
    resume_input: Union[str, Sequence[str]],
    job_input: Union[str, Sequence[str]],
    custom_skills: Optional[Sequence[str]] = None,
) -> SkillGapResult:
    """Analyze the skill gap between a resume and a job description.

    Supports inputs as either raw unstructured text or pre-extracted skill lists.

    Args:
        resume_input: Resume text or list of candidate skills.
        job_input: Job description text or list of required job skills.
        custom_skills: Optional extra skills to track.

    Returns:
        SkillGapResult detailing matched, missing, and additional skills.
    """
    # Extract/normalize resume skills
    if isinstance(resume_input, str):
        resume_skills = extract_skills(resume_input, custom_skills=custom_skills)
    else:
        resume_skills = {normalize_skill(s) for s in resume_input if normalize_skill(s)}

    # Extract/normalize job skills
    if isinstance(job_input, str):
        job_skills = extract_skills(job_input, custom_skills=custom_skills)
    else:
        job_skills = {normalize_skill(s) for s in job_input if normalize_skill(s)}

    matched = sorted(resume_skills.intersection(job_skills))
    missing = sorted(job_skills.difference(resume_skills))
    additional = sorted(resume_skills.difference(job_skills))

    total_job = len(job_skills)
    matched_count = len(matched)
    missing_count = len(missing)

    if total_job > 0:
        match_percentage = round((matched_count / total_job) * 100.0, 2)
    else:
        match_percentage = 0.0

    if total_job == 0:
        readiness = "No Skills Specified"
    elif match_percentage >= 75.0:
        readiness = "High Fit"
    elif match_percentage >= 50.0:
        readiness = "Moderate Fit"
    else:
        readiness = "Skill Gap Identified"

    return SkillGapResult(
        matched_skills=matched,
        missing_skills=missing,
        additional_skills=additional,
        matched_count=matched_count,
        missing_count=missing_count,
        total_job_skills=total_job,
        match_percentage=match_percentage,
        readiness_level=readiness,
    )
