"""
Matching engine for user-meeting recommendations.
Implements 3-stage filtering: Hard Filter → Soft Score → Top-N.
"""
import random
from typing import List, Dict, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import CATEGORIES, TAG_DIMENSIONS


def generate_dummy_db(count: int = 50) -> List[Dict]:
    """
    Generate dummy meeting database with random tags.

    Args:
        count: Number of meetings to generate

    Returns:
        List of meeting dictionaries with id, category, title, and tags
    """
    db = []
    for i in range(count):
        cat = random.choice(CATEGORIES)

        tags = {
            key: random.choice(values)
            for key, values in TAG_DIMENSIONS.items()
        }

        title = f"{cat} 모임 {i+1}호"

        db.append({
            "id": i,
            "category": cat,
            "title": title,
            "tags": tags
        })
    return db


class MatchingEngine:
    """
    3-stage matching engine for user-meeting recommendations.

    Stage 1: Hard Filter - Category must match exactly
    Stage 2: Soft Score - Tag matching (25 points per tag, max 100)
    Stage 3: Top-N - Return highest scoring meetings
    """

    def __init__(self, meeting_db: List[Dict]):
        """
        Initialize matching engine with meeting database.

        Args:
            meeting_db: List of meeting dictionaries
        """
        self.meeting_db = meeting_db
        self.tag_keys = list(TAG_DIMENSIONS.keys())
        self.points_per_tag = 25

    def match(self, user_profile: Dict, top_n: int = 3) -> List[Dict]:
        """
        Run 3-stage matching algorithm.

        Args:
            user_profile: User profile with category and tags
            top_n: Number of top results to return

        Returns:
            List of matched meetings with scores and details
        """
        if not user_profile:
            return []

        # Stage 1: Hard Filter - Category matching
        target_category = user_profile.get('category')
        if not target_category:
            return []

        filtered_db = [
            m for m in self.meeting_db
            if m['category'] == target_category
        ]

        if not filtered_db:
            return []

        # Stage 2: Soft Score - Tag matching
        candidates = []
        user_tags = user_profile.get('tags', {})

        for meeting in filtered_db:
            score = 0
            match_details = []
            meeting_tags = meeting.get('tags', {})

            for key in self.tag_keys:
                if user_tags.get(key) == meeting_tags.get(key):
                    score += self.points_per_tag
                    match_details.append(f"{key}(O)")
                else:
                    match_details.append(f"{key}(X)")

            candidates.append({
                "info": meeting,
                "score": score,
                "details": match_details
            })

        # Stage 3: Top-N - Sort by score (descending)
        candidates.sort(key=lambda x: x['score'], reverse=True)

        return candidates[:top_n]

    def get_category_meetings(self, category: str) -> List[Dict]:
        """Get all meetings in a specific category."""
        return [m for m in self.meeting_db if m['category'] == category]

    def get_statistics(self) -> Dict:
        """Get database statistics by category."""
        stats = {cat: 0 for cat in CATEGORIES}
        for meeting in self.meeting_db:
            cat = meeting.get('category')
            if cat in stats:
                stats[cat] += 1
        return stats
