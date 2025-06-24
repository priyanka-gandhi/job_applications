import nltk
'''
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('maxent_ne_chunker')
'''
#s=input()
#print(s)

text = nltk.word_tokenize("And now for something completely different")

#to get pos tags of the sentence
pairs = nltk.pos_tag(text)
print(pairs[0][1])
print("--------------------------------------")
#to find the missing ?? pos tags in the sentence
print("Filling Missing POS Values")
text ="The/DT planet/NN Jupiter/NNP and/CC its/PPS moons/NNS are/VBP in/IN effect/NN a/DT minisolar/JJ system/?? ,/, and/CC Jupiter/NNP itself/PRP is/VBZ often/RB called/VBN a/DT star/?? that/IN never/RB caught/??? fire/NN ./."
words = text.split()
words = [i.split("/")[0] for i in words]
text = ' '.join(words)
text = nltk.word_tokenize(text)
pairs = nltk.pos_tag(text)

text=''
for i in range(len(pairs)):
    text = text + pairs[i][0] +"/"+pairs[i][1]+" "
print(text)

print("--------------------------------------")
#to find Nouns in the text
print("Finding Nouns")
words=[]
for i in range(len(pairs)):
    if 'NN' in pairs[i][1]:
        words.append(pairs[i][0])
print(words)

print("--------------------------------------")
#chunking
print("Chunking Text")
#Our chunk pattern consists of one rule, that a noun phrase, NP, should be formed whenever the chunker finds an optional determiner, DT, followed by any number of adjectives, JJ, and then a noun, NN.
pattern = 'NP: {<DT>?<JJ>*<NN>}'
text = "European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in the mobile phone market and ordered the company to alter its practices"

text = nltk.word_tokenize(text)
text = nltk.pos_tag(text)
cp = nltk.RegexpParser(pattern)
cs = cp.parse(text)
print(cs)

print("--------------------------------------")
# BIO Tagging
print("IOB  Tagging")
from nltk.chunk import conlltags2tree, tree2conlltags
from pprint import pprint
iob_tagged = tree2conlltags(cs)
print(iob_tagged)

print("--------------------------------------")
#NER
#2 Sets - finding matching sentence combination
set_A = "Delhi (also known as the National Capital Territory of India) is a metropolitan region in India that includes the national capital city, New Delhi. With a population of 22 million in 2011, it is the world's second most populous city and the largest city in India in terms of area. The NCT and its urban region have been given the special status of National Capital Region (NCR) under the Constitution of India's 69th amendment act of 1991. The NCR includes the neighbouring cities of Baghpat, Gurgaon, Sonepat, Faridabad, Ghaziabad, Noida, Greater Noida and other nearby towns, and has nearly 22.2 million residents. Seattle is a coastal seaport city and the seat of King County, in the U.S. state of Washington. With an estimated 634,535 residents as of 2012, Seattle is the largest city in the Pacific Northwest region of North America and one of the fastest-growing cities in the United States. The Seattle metropolitan area of around 4 million inhabitants is the 15th largest metropolitan area in the nation.[6] The city is situated on a narrow isthmus between Puget Sound (an inlet of the Pacific Ocean) and Lake Washington, about 100 miles (160 km) south of the Canada–United States border. A major gateway for trade with Asia, Seattle is the 8th largest port in the United States and 9th largest in North America in terms of container handling. Martin Luther OSA (10 November 1483 – 18 February 1546) was a German monk, Catholic priest, professor of theology and seminal figure of a reform movement in 16th century Christianity, subsequently known as the Protestant Reformation.[1] He strongly disputed the claim that freedom from God's punishment for sin could be purchased with money. He confronted indulgence salesman Johann Tetzel, a Dominican friar, with his Ninety-Five Theses in 1517. His refusal to retract all of his writings at the demand of Pope Leo X in 1520 and the Holy Roman Emperor Charles V at the Diet of Worms in 1521 resulted in his excommunication by the Pope and condemnation as an outlaw by the Emperor."
#set_B = "The Seattle area had been inhabited by Native Americans for at least 4,000 years before the first permanent European settlers. Arthur A. Denny and his group of travelers, subsequently known as the Denny Party, arrived at Alki Point on November 13, 1851. The settlement was moved to its current site and named 'Seattle' in 1853, after Chief Si'ahl of the local Duwamish and Suquamish tribes. Although technically a federally administered union territory, the political administration of the NCT of Delhi today more closely resembles that of a state of India, with its own legislature, high court and an executive council of ministers headed by a Chief Minister. New Delhi is jointly administered by the federal government of India and the local government of Delhi, and is the capital of the NCT of Delhi.Luther taught that salvation and subsequently eternity in heaven is not earned by good deeds but is received only as a free gift of God's grace through faith in Jesus Christ as redeemer from sin and subsequently eternity in hell. His theology challenged the authority of the Pope of the Roman Catholic Church by teaching that the Bible is the only source of divinely revealed knowledge from God and opposed sacerdotalism by considering all baptized Christians to be a holy priesthood. Those who identify with these, and all of Luther's wider teachings, are called Lutherans."


#set_A = 'Fortunately, there are “low-chill” apple varieties for temperate climates. (Chilling hours are defined as nonconsecutive hours of winter temperatures below 45 degrees.) As a general guide, if you live on or near the coast, your garden gets only 100 to 200 chilling hours. Inland San Diego gardens get about 400 to 500 chilling hours — still considered “low chill.”'
set_A = nltk.word_tokenize(set_A)
set_A = nltk.pos_tag(set_A)
print(nltk.ne_chunk(set_A))
#print(nltk.ne_chunk(text))

# WORD SENSE DISAMBIGUITY THROUGH pywsd library
print("--------------------------------------")
import re
#count 'a','an', 'the' and dates
text = ["Delhi, is a metropolitan and the capital region of India which includes the national capital city, New Delhi. It is the second most populous metropolis in India after Mumbai and the largest city in terms of area.","Mumbai, also known as Bombay, is the capital city of the Indian state of Maharashtra. It is the most populous city in India, and the fourth most populous city in the world, with a total metropolitan area population of approximately 20.5 million.","New York is a state in the Northeastern region of the United States. New York is the 27th-most extensive, the 3rd-most populous, and the 7th-most densely populated of the 50 United States.","The Indian Rebellion of 1857 began as a mutiny of sepoys of the East India Company's army on 10 May 1857, in the town of Meerut, and soon escalated into other mutinies and civilian rebellions largely in the upper Gangetic plain and central India, with the major hostilities confined to present-day Uttar Pradesh, Bihar, northern Madhya Pradesh, and the Delhi region.","The Boston Tea Party (referred to in its time simply as 'the destruction of the tea' or by other informal names and so named until half a century later,[2]) was a political protest by the Sons of Liberty in Boston, a city in the British colony of Massachusetts, against the tax policy of the British government and the East India Company that controlled all the tea imported into the colonies. On December 16, 1773, after officials in Boston refused to return three shiploads of taxed tea to Britain, a group of colonists boarded the ships and destroyed the tea by throwing it into Boston Harbor. The incident remains an iconic event of American history, and other political protests often refer to it."]

for line in text:
    a_count=0
    an_count=0
    the_count=0

    dates_count = len(re.findall(r'\d+\S\d+\S\d+', line))
    line = line.split()

    for word in line:
        if word == 'a':
            a_count +=1
        elif word == 'an':
            an_count += 1
        elif word == 'the':
            the_count += 1
        else:
            pass
    print(a_count)
    #print(an_count)
    #print(the_count)
    #print(dates_count)

print("--------------------------------------")
#spelling checker
from textblob import TextBlob

a = "cmputrs have made life easier"           # incorrect spelling
print("original text: "+str(a))

b = TextBlob(a)

# prints the corrected spelling
print("corrected text: "+str(b.correct()))

'''
print("--------------------------------------")
#Language detection
from langdetect import detect

lang = detect("Ein, zwei, drei, vier")

'''
print("--------------------------------------")
#Similarity Score

import math
import re
from collections import Counter

WORD = re.compile(r"\w+")


def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
    sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def text_to_vector(text):
    words = WORD.findall(text)
    return Counter(words)


text1 = "This is a foo bar sentence ."
text2 = "This sentence is similar to a foo bar sentence ."

vector1 = text_to_vector(text1)
vector2 = text_to_vector(text2)
print(vector1)
print(vector2)
cosine = get_cosine(vector1, vector2)

print("Cosine:", cosine)


