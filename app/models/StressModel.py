# Stub: stress model 已停用，保留簡單介面以避免呼叫端錯誤

class StressModel:
    def __init__(self):
        # 不再載入大型模型，僅提示已停用
        print("⚠️ StressModel disabled: "
              "stress analysis removed from the project.")

    def analyze(self, text_zh):
        """
        回傳空的分析結果結構（與原有模型回傳格式相容的最小表現）。
        這樣呼叫端仍可處理返回值而不會拋例外。
        """
        return []  # 原本模型回傳 list 結構，保留一致性
