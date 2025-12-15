# Models package

from .SentimentModel import SentimentModel
# from .StressModel import StressModel

# Defer heavy model initialization until startup to reduce import-time cost
sentimentModel = None
# stressModel = None


def init_models():
    global sentimentModel
    if sentimentModel is None:
        sentimentModel = SentimentModel()
    return sentimentModel
