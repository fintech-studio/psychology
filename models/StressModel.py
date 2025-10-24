from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from utils.Translate import Translator

class StressModel:
    def __init__(self):
        self.model_name = "dstefa/roberta-base_stress_classification"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.classifier = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer, top_k=None,device=0)
        self.translator = Translator()

    def analyze(self, text_zh):
        text_en = self.translator.translate_zn_en(text_zh)
        # print("翻譯後的英文文本:", text_en)
        result = self.classifier(text_en)
        return result

if __name__ == "__main__":
    model = StressModel()
    text = input("請輸入中文文本進行壓力分析: ")
    res = model.analyze(text)
    print(res)