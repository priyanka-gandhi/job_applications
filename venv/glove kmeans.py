from gensim.models import KeyedVectors
# Load the Stanford GloVe model

model = KeyedVectors.load_word2vec_format('/Users/priyanka/PycharmProjects/bilstm+crf_basic/data/example/glove.840B.300d.txt', binary=False)
dataset = pandas.read_excel('/Users/priyanka/Downloads/reinvitationtointerviewmpulsemobilebdsdeveloperr/Sample Message Data.xlsx', encoding = 'utf-8')

non_glove_words_df = get_non_glove_words(dataset['content'], model = model)
print(len(non_glove_words_df))
