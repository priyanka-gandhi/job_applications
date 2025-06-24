from nltk.parse import CoreNLPParser
from pycorenlp import StanfordCoreNLP
import xlrd
import pandas as pd

df = pd.read_excel("/Users/priyanka/Downloads/reinvitationtointerviewmpulsemobilebdsdeveloperr/Sample Message Data.xlsx")
df = df.replace('\n','', regex=True)
result = CoreNLPParser(url='http://localhost:9000')
nlp = StanfordCoreNLP('http://localhost:9000')

sentiment=[]
#text = "This movie was actually neither that funny, nor super witty. The movie was meh. I liked watching that movie. If I had a choice, I would not watch that movie again."
for text in df["content"]:
    result = nlp.annotate(text,
                   properties={
                       'annotators': 'sentiment, ner, pos',
                       'outputFormat': 'json',
                        "ssplit.eolonly": "true",
                       'timeout': 10000,
                   })

    #if len(result["sentences"]) > 1:
    #        print(text)
    #        exit(0)

    for s in result["sentences"]:
    #
        print("{}: '{}': (Sentiment Value) {}".format(s["index"]," ".join([t["word"] for t in s["tokens"]]), s["sentiment"]))
        sentiment.append(s["sentiment"])
print(len(sentiment))
print(len(df))
df["Stanford CoreNLP Result"] = sentiment
df.to_csv("/Users/priyanka/Downloads/reinvitationtointerviewmpulsemobilebdsdeveloperr/StanfordCoreNLP_results.csv", index=False)
