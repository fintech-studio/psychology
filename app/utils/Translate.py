from transformers import pipeline

class Translator:
    def init__(self):
        pass

    def translate_zn_en(self, text):
        translator = pipeline("translation", model="Helsinki-NLP/opus-mt-zh-en")
        result = translator(text)
        return result[0]['translation_text']
    
    def translate_en_zn(self, text):
        translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-zh")
        result = translator(text)
        return result[0]['translation_text']