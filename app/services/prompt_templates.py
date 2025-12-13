from typing import List


QUESTION_JSON_SCHEMA = '''請以純 JSON 格式回傳題目，僅輸出 JSON（不要任何額外說明、文字或註解）。
輸出 JSON 必須為一個物件，並符合下列規範與欄位：

必填欄位（所有題型皆須包含）
- `question`: 題目文字（繁體中文，15-40 字為佳）
- `type`: 題型，值為 `mc`（多選）或 `likert`（尺度題）或 `open`（開放式）
- `option_type`: 前端顯示用之題型標籤，請使用以下其中一個文字：「選擇題」、「Likert 1-5」、「開放式題目」
- `dimension`: 題目維度，請使用以下其中一個值：`emotion`、`stress`、`time`、`risk`、`decision`

若 `type` 為 `mc`（多選）時，請同時包含：
- `options`: 字串陣列，包含 3 到 5 個選項字串（每個選項最適 5-20 字），選項不得包含「A.」或「B.」等字首。

若 `type` 為 `likert` 時，請同時包含：
- `likert_range`: 範圍字串（固定為 `1-5`）
- `likert_option`: 長度為 5 的字串陣列，代表每個評分標籤。
- 例如：['完全不同意', '不同意', '中立', '同意', '完全同意']。
- 每個描述建議 2-6 字。

回傳範例（必須遵守結構）：
{
  "question": "題目描述文字...",
    "type": "mc|likert|open",
    "option_type": "選擇題|Likert 題|開放式題目",
  "options": ["選項A", "選項B", "選項C", ...],
  "likert_range": "1-5",
  "likert_option": ["選項1", "選項2", "選項3", "選項4", "選項5"],
  "dimension": "risk|stress|time|emotion|decision"
}

注意事項：
- 一律回傳 JSON 物件，不要額外輸出任何純文字（例如「第 1 題」或「請選擇」）。若需額外說明，請放入 `note` 欄位，避免破壞 JSON 格式。
- 選項或 likert 描述應儘量簡短且易於 UI 呈現，不要使用行內標記或 HTML。
- 一律使用繁體中文。
'''


def build_question_prompt(
        qtype: str, current_number: int, total_questions: int) -> str:
    """Generates a well-formed prompt
    for a single question generation request."""
    instruct = {
        "emotion_mc": (
            "請生成一個情境式多選題（mc），描述一個投資相關的情境。"
        ),
        "stress_likert": (
            "請生成一個壓力感知題（likert），描述一個投資相關的壓力情境，"
            "提示使用 Likert 1 到 5 評分（1代表無壓力，5代表極大壓力），"
        ),
        "risk_mc": (
            "請生成一個風險偏好多選題（mc），提供 3 到 5 個選項描述不同的風險承受程度，"
            "每個選項代表不同風險承受力（保守/中庸/積極等）。"
        ),
        "time_pref_likert": (
            "請生成一個時間偏好題（likert），提示使用 Likert 1 到 5 評分，"
            "範圍提示為 1（偏好短期回報）到 5（偏好長期回報），"
        ),
        "decision_impulse": (
            "請生成一個評估決策衝動性或理性程度的多選題（mc），說明是否容易衝動或會衡量風險。"
        ),
        "decision_mc": (
            "請生成一個決策習慣多選題（mc），描述在投資決策中常見的行為模式。"
        ),
    }

    if qtype not in instruct:
        q_instruct = instruct["decision_mc"]
    else:
        q_instruct = instruct[qtype]

    # Provide a small, explicit wrapper so the model focuses on JSON only.
    return (
        f"你是一位理財顧問與心理評估專家。請根據要求生成第{current_number}題（共{total_questions}題）：\n"
        f"{q_instruct}\n\n"
        "請直接輸出一個乾淨的 JSON 物件（不要額外說明或多行文字）。\n"
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
"""
