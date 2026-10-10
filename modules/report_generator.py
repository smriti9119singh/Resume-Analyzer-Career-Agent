"""
Generates a downloadable Markdown career analysis report.
"""
from typing import Dict


def build_markdown_report(candidate_name: str, match_result: Dict, feedback: Dict, agent_data: Dict) -> str:
    """
    Formats the full analysis into a structured Markdown report for download.
    """
    lines = [
        f"# AI Resume Analyzer & Career Agent — Report for {candidate_name}",
        "",
        "## 1. Machine Learning Alignment Summary",
        f"- **Overall Alignment Score**: {match_result.get('overall_match_score', 0)}%",
        f"- **TF-IDF Cosine Similarity**: {match_result.get('tfidf_cosine_similarity', 0)}%",
        f"- **Deterministic Skill Overlap**: {match_result.get('skill_overlap_score', 0)}%",
        f"- **Matched Skills**: {', '.join(match_result.get('matched_skills', [])) or 'None'}",
        f"- **Missing Required Skills**: {', '.join(match_result.get('missing_skills', [])) or 'None'}",
        "",
        f"> **Disclaimer**: {match_result.get('disclaimer', '')}",
        "",
        "## 2. Grounded Professional Summary",
        feedback.get("grounded_summary", ""),
        "",
        "## 3. Prioritized Learning Roadmap",
    ]
    for item in agent_data.get("roadmap", []):
        lines.append(
            f"- **Week {item['week']} ({item['focus_skill']} — {item['allocated_hours']} hrs)**: {item['deliverable']}"
        )

    lines.append("")
    lines.append("## 4. Agent Tool Execution Trace")
    for step in agent_data.get("tool_trace", []):
        lines.append(f"- `{step['tool']}` ({step['type']}): {step['summary']}")

    return "\n".join(lines)
