import json
from tabulate import tabulate # 표 예쁘게 출력하는 라이브러리 (없으면 pip install tabulate)

# 파일 경로
FILE_QWEN25 = "./data/processed/user_profiles_vectorized.json"
# FILE_QWEN3 = "./data/processed/user_profiles_qwen3_result.json"
FILE_QWEN3 = "./data/processed/user_profiles_qwen3_result_1.json"

def load_data(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return {item['user_id']: item for item in json.load(f)}
    except FileNotFoundError:
        print(f"❌ 파일을 찾을 수 없음: {filepath}")
        return {}

OUTPUT_MD = "./data/processed/comparison_result.md"

def main():
    data_25 = load_data(FILE_QWEN25)
    data_3 = load_data(FILE_QWEN3)

    if not data_25 or not data_3:
        return

    # 비교 테이블 생성
    table_data = []

    # 공통된 유저 ID만 비교 (상위 5명만 샘플링)
    common_ids = list(set(data_25.keys()) & set(data_3.keys()))[:5]

    # 마크다운 내용 생성
    md_lines = []
    md_lines.append(f"# 📊 모델 성능 비교: Qwen 2.5 vs Qwen 3\n")
    md_lines.append(f"**총 {len(common_ids)}명 샘플**\n")
    md_lines.append("---\n")

    print(f"\n📊 [모델 성능 비교] Qwen 2.5 vs Qwen 3 (총 {len(common_ids)}명 샘플)")
    print("=" * 80)

    for uid in common_ids:
        items = data_25[uid]['items']
        res_25 = data_25[uid].get('inferred_interests', 'N/A')
        res_3 = data_3[uid].get('inferred_interests', 'N/A')

        # Qwen 3의 <think> 태그가 있다면 제거하고 핵심만 보여주기 (옵션)
        if "<think>" in res_3:
            # 생각 부분은 너무 기니까 잘라내고 답변만 보여줌 (필요시 수정)
            pass

        print(f"👤 유저: {uid}")
        print(f"🛒 구매 물품: {items}")
        print(f"🤖 Qwen 2.5: {res_25}")
        print(f"🧠 Qwen 3  : {res_3}")
        print("-" * 50)

        # 마크다운에 추가
        md_lines.append(f"## 👤 유저: {uid}\n")
        md_lines.append(f"**🛒 구매 물품:** {items}\n")
        md_lines.append(f"**🤖 Qwen 2.5:**\n> {res_25}\n")
        md_lines.append(f"**🧠 Qwen 3:**\n> {res_3}\n")
        md_lines.append("---\n")

    # 마크다운 파일 저장
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"\n✅ 마크다운 파일 저장 완료: {OUTPUT_MD}")

if __name__ == "__main__":
    main()