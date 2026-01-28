"""
UI generation utilities for web preview.
"""
from pathlib import Path
from typing import List, Dict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import CATEGORIES, TAG_DIMENSIONS


class UIGenerator:
    """Generate HTML UI for matching results visualization."""

    @staticmethod
    def generate_matching_result_html(
        user_profile: Dict,
        recommendations: List[Dict],
        output_path: Path
    ) -> str:
        """
        Generate HTML page showing matching results.

        Args:
            user_profile: User's analyzed profile
            recommendations: List of recommended meetings
            output_path: Path to save HTML file

        Returns:
            Generated HTML content
        """
        # Build recommendations HTML
        recs_html = ""
        for i, rec in enumerate(recommendations):
            meeting = rec.get('info', {})
            score = rec.get('score', 0)
            details = rec.get('details', [])

            medal = ["🥇", "🥈", "🥉"][i] if i < 3 else f"{i+1}"

            recs_html += f"""
            <div class="recommendation-card">
                <div class="rank">{medal}</div>
                <div class="meeting-info">
                    <h3>{meeting.get('title', 'Unknown')}</h3>
                    <div class="category-badge">{meeting.get('category', '')}</div>
                    <div class="score">매칭률: {score}%</div>
                    <div class="tags">
                        {' '.join(f'<span class="tag">{t}</span>' for t in meeting.get('tags', {}).values())}
                    </div>
                    <div class="details">{', '.join(details)}</div>
                </div>
            </div>
            """

        html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>모임 추천 결과</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f5f5;
            padding: 20px;
        }}
        .container {{ max-width: 600px; margin: 0 auto; }}
        .header {{
            background: #ff6f0f;
            color: white;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 20px;
        }}
        .header h1 {{ font-size: 24px; margin-bottom: 10px; }}
        .user-profile {{
            background: white;
            padding: 16px;
            border-radius: 12px;
            margin-bottom: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .user-profile h2 {{ font-size: 16px; color: #333; margin-bottom: 12px; }}
        .profile-item {{ margin-bottom: 8px; font-size: 14px; color: #666; }}
        .recommendation-card {{
            background: white;
            padding: 16px;
            border-radius: 12px;
            margin-bottom: 12px;
            display: flex;
            align-items: flex-start;
            gap: 16px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .rank {{
            font-size: 32px;
            min-width: 50px;
            text-align: center;
        }}
        .meeting-info {{ flex: 1; }}
        .meeting-info h3 {{ font-size: 16px; margin-bottom: 8px; }}
        .category-badge {{
            display: inline-block;
            background: #ff6f0f;
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            margin-bottom: 8px;
        }}
        .score {{ font-size: 14px; color: #ff6f0f; font-weight: bold; margin-bottom: 8px; }}
        .tags {{ display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 8px; }}
        .tag {{
            background: #e8e8e8;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
        }}
        .details {{ font-size: 12px; color: #999; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🥕 당근 모임 추천</h1>
            <p>당신에게 딱 맞는 모임을 찾았어요!</p>
        </div>

        <div class="user-profile">
            <h2>📊 분석 결과</h2>
            <div class="profile-item">카테고리: <strong>{user_profile.get('category', '-')}</strong></div>
            <div class="profile-item">추천 태그: {' '.join(user_profile.get('tags', {}).values())}</div>
        </div>

        <h2 style="margin-bottom: 12px; font-size: 18px;">🎯 추천 모임 TOP {len(recommendations)}</h2>
        {recs_html}
    </div>
</body>
</html>"""

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)

        return html

    @staticmethod
    def generate_category_summary_html(
        category_stats: Dict[str, int],
        output_path: Path
    ) -> str:
        """
        Generate HTML summary of category distribution.

        Args:
            category_stats: Dictionary of category counts
            output_path: Path to save HTML file

        Returns:
            Generated HTML content
        """
        total = sum(category_stats.values())

        bars_html = ""
        for cat in CATEGORIES:
            count = category_stats.get(cat, 0)
            pct = (count / total * 100) if total > 0 else 0
            bars_html += f"""
            <div class="bar-row">
                <div class="bar-label">{cat}</div>
                <div class="bar-container">
                    <div class="bar-fill" style="width: {pct}%"></div>
                </div>
                <div class="bar-value">{count} ({pct:.1f}%)</div>
            </div>
            """

        html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>카테고리 분포</title>
    <style>
        body {{ font-family: sans-serif; padding: 20px; max-width: 600px; margin: 0 auto; }}
        h1 {{ color: #ff6f0f; }}
        .bar-row {{ display: flex; align-items: center; margin: 10px 0; }}
        .bar-label {{ width: 100px; font-size: 14px; }}
        .bar-container {{ flex: 1; height: 24px; background: #eee; border-radius: 4px; overflow: hidden; }}
        .bar-fill {{ height: 100%; background: #ff6f0f; transition: width 0.3s; }}
        .bar-value {{ width: 100px; text-align: right; font-size: 14px; }}
    </style>
</head>
<body>
    <h1>📊 카테고리 분포</h1>
    <p>총 {total}개 데이터</p>
    {bars_html}
</body>
</html>"""

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)

        return html
