from models.StressModel import StressModel
from models.SentimentModel import SentimentModel

# This file is just for test ,if you want to run either model from this script
# Enter 1 for StressModel, 2 for SentimentModel
choice = int(input("Enter 1 for StressModel, 2 for SentimentModel: "))

if choice == 1:
    stressmodel = StressModel()
    print("Using StressModel")
    text = input("輸入中文文本: ")
    result = stressmodel.analyze(text)
    print(result)

elif choice == 2:
    sentimentmodel = SentimentModel()
    print("Using SentimentModel")
    text = input("輸入中文文本: ")
    result = sentimentmodel.analyze(text)
    print(result)
