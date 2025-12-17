from typing import List

QUESTION_JSON_SCHEMA = '''請以純 JSON 格式回傳題目，僅輸出 JSON（不要任何額外說明、文字或註解）。
輸出 JSON 必須為一個物件，並符合下列規範與欄位：

必填欄位（所有題型皆須包含）
- `question`: 題目文字（繁體中文，15-40 字為佳）
- `type`: 題型，固定為 `single`（單選題）
- `options`: 選項陣列（需提供 3-6 個選項）
- `options_score`: 與 `options` 等長的數值陣列，對應每個選項的分數
    （請使用 0.0 到 1.0 的範圍；數值越大表示該選項在該題目維度上的強度或偏好越高）。
- `dimension`: 題目維度，請使用以下其中一個值：`emotion`、`stress`、`term_pref`、`risk`、`decision`

回傳範例（必須遵守結構）：
{
    "question": "題目描述文字...",
    "type": "single",
    "options": ["選項A", "選項B", "選項C", ...],
    "options_score": [選項A分數, 選項B分數, ...],
    "dimension": "risk|stress|term_pref|emotion|decision"
}

注意事項：
- 一律回傳 JSON 物件，不要額外輸出任何純文字，避免破壞 JSON 格式。
- 一律使用繁體中文。
- 請使用「投資相關」的題目情境與描述。
- 請使用「您」作為主詞，讓題目更具個人化與互動性，避免使用第三人稱。
- 請確保題目和選項具備邏輯一致性且符合投資情境，避免矛盾或不合理的描述。
- dimension 欄位必須與題目內容相符，且只能使用指定的五個維度之一，只能回覆一個維度而已。
- 請為每個選項評分，並在輸出的 JSON 中以 `options_score` 欄位提供與 `options` 等長的數字陣列
    （範圍 0.0–1.0，若使用 1–5 或其他量表請標準化為 0–1）。
'''


def build_question_prompt(
        qtype: str, current_number: int, total_questions: int) -> str:
    """Generates a well-formed prompt
    for a single question generation request."""
    instruct = {
        "emotion": (
            "請生成一個描述投資相關的情境式題目，用以評估情緒反應。"
        ),
        "stress": (
            "請生成一個描述投資過程中壓力感受的題目，說明可能引發壓力的情境或感受。"
        ),
        "risk": (
            "請生成一個與投資風險承受度相關的題目，用以評估投資者對風險的態度。"
        ),
        "term_pref": (
            "請生成一個評估投資偏好（短期 vs 長期）的題目，用以判斷投資者的時間偏好。"
        ),
        "decision_impulse": (
            "請生成一個評估決策衝動性或理性程度的題目，描述是否容易衝動或會衡量風險。"
        ),
        "decision": (
            "請生成一個描述投資決策習慣的題目，描繪投資決策中的常見行為模式。"
        ),
    }

    if qtype not in instruct:
        q_instruct = instruct["decision"]
    else:
        q_instruct = instruct[qtype]

    return (
        f"你是一位理財顧問與心理評估專家。請根據要求生成第{current_number}題（共{total_questions}題）：\n"
        f"{q_instruct}\n\n"
        + QUESTION_JSON_SCHEMA
    )


def build_advice_prompt(
        avg_negative: float, avg_neutral: float, avg_positive: float,
        detail_lines: List[str], stress_index: int, time_horizon: int
        ) -> str:
    """Builds the advice/instruction prompt for the LLM.
        Returns detailed instruction string."""
    summary = "\n".join(detail_lines)
    # Ask for a concise, structured advice output with recommended sections.
    return f"""
請根據以下使用者在心理問卷中的情緒分析結果與心理壓力指數，提供個人化的投資心理檢測與投資建議：

整體平均情緒分析結果：
- 平均負面情緒: {avg_negative:.3f}
- 平均中性情緒: {avg_neutral:.3f}
- 平均正面情緒: {avg_positive:.3f}
心理壓力指數: {stress_index}
時域偏好數值: {time_horizon}

詳細問答與分析：
{summary}

基於上述資訊，請提供以下內容：
- 針對關鍵心理特徵（2-3 行）簡述重點。
- 具體且可執行的投資策略或操作建議（3-5 條，每條 1-2 行）。
- 針對心理與市場風險的管理建議（1-2 條）。
- 使用者可執行的跟進步驟（1-3 條簡短步驟）。
- 一句短的鼓勵或同理心話語（1 行）。

請使用繁體中文，確保建議內容具體且實用，並以同理心的語氣撰寫。
請使用「您」作為主詞，讓建議更具個人化與互動性，避免使用第三人稱。
請務必使用 Markdown 格式回覆。
"""
