import torch
import torch.nn.functional as F
import re
from nltk.corpus import stopwords
import string
import preprocessor as p # To Process the tweets, clean, et cetera
from torchtext import data
import pickle


class SentimentPredictor():

    def __init__(self, model, model_params_dict):

        # Happy Emoticons
        self.__emoticon_happy = set([
        ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
        ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
        '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
        'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
        '<3'])

        # Sad Emoticons
        self.__emoticon_sad = set([
        ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
        ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
        ':c', ':{', '>:\\', ';('])

        # Compiler to Delete Symbols
        self.__emoji_pattern = re.compile("["
         u"\U0001F600-\U0001F64F"  # emoticons
         u"\U0001F300-\U0001F5FF"  # symbols & pictographs
         u"\U0001F680-\U0001F6FF"  # transport & map symbols
         u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
         u"\U00002702-\U000027B0"
         u"\U000024C2-\U0001F251""]+", flags=re.UNICODE)

        # Italian Stop Words
        self.stop_words = set(stopwords.words('italian'))

        # Join All the Emoticons in one object
        self.emoticons = self.__emoticon_happy.union(self.__emoticon_sad)

        # Tokenizer rule
        self.WORD = re.compile(r'\w+')

        # LOAD Torch Model - Our Classifier
        self.model = model
        self.model.load_state_dict(torch.load(model_params_dict))

        # Text field
        self.TEXT = pickle.load(open('/Volumes/Transcend/CADD/covid-sentiment/sentiment-predictions/analyzer/TEXT.pickle', 'rb'))

        # Device
        self.__device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



    def regTokenize(self, text):
        """Method to Tokenize"""
        words = self.WORD.findall(text)
        return words



    def clean_tweets(self, tweet):
        """Tweet Cleaner"""
        # PREPROCESS
        tweet = tweet.lower() # Lowercase everything
        tweet = tweet.split() # Split tweet
        tweet = [t for t in tweet if len(t) > 1] # remove letters and one length charachters
        tweet = ' '.join(tweet) # Join tweet tokens


        # Clean With LIBRARY
        tweet = p.clean(tweet)

        # FURTHER MANUAL CLEANING

        word_tokens = self.regTokenize(tweet) # Tokenize

        #after tweepy preprocessing the colon symbol left remain after      #removing mentions
        tweet = re.sub(r':', '', tweet)
        tweet = re.sub(r'‚Ä¶', '', tweet)
        #replace consecutive non-ASCII characters with a space
        tweet = re.sub(r'[^\x00-\x7F]+',' ', tweet)
        #remove emojis from tweet
        tweet = self.__emoji_pattern.sub(r'', tweet)
        #filter using NLTK library append it to a string
        filtered_tweet = [w for w in word_tokens if not w in self.stop_words]
        filtered_tweet = []
        #looping through conditions
        for w in word_tokens:
            #check tokens against stop words , emoticons and punctuations
            if w not in self.stop_words and w not in self.emoticons and w not in string.punctuation:
                filtered_tweet.append(w)


        return ' '.join(filtered_tweet)


    def predict_sentiment(self, sentence, clean = False, min_len = 2):
        self.model.eval()
        if clean:
            sentence = self.clean_tweets(sentence)
        tokenized = self.regTokenize(sentence)
        if len(tokenized) < min_len:
            tokenized += ['<pad>'] * (min_len - len(tokenized))
        indexed = [self.TEXT.vocab.stoi[t] for t in tokenized]
        tensor = torch.LongTensor(indexed).to(self.__device)
        tensor = tensor.unsqueeze(0)
        #prediction = torch.sigmoid(self.model(tensor))
        #print(F.softmax(self.model(tensor), dim = 1))
        prediction = F.softmax(self.model(tensor))
        return prediction.argmax().item()


    def predict_prob_sentiment(self, sentence, clean = False, min_len = 2):
        self.model.eval()
        if clean:
            sentence = self.clean_tweets(sentence)
        tokenized = self.regTokenize(sentence)
        if len(tokenized) < min_len:
            tokenized += ['<pad>'] * (min_len - len(tokenized))
        indexed = [self.TEXT.vocab.stoi[t] for t in tokenized]
        tensor = torch.LongTensor(indexed).to(self.__device)
        tensor = tensor.unsqueeze(0)
        prediction = F.softmax(self.model(tensor))
        return prediction.max().item()




if __name__ == '__main__':
    from model_code import CNN_bi_uni_gram
    model = CNN_bi_uni_gram(30002, 128, 1, 3)
    #text_field = '/Volumes/Transcend/CADD/covid-sentiment/predictions/analyzer/TEXT.pickle'
    params = open('/Volumes/Transcend/CADD/covid-sentiment/sentiment-predictions/analyzer/cnn-uni-classifier.pt', 'rb')
    tweet2 = 'Ho letto il decreto legge di conte, tutto nella norma.'
    test = SentimentPredictor(model, params)
    print(test.predict_sentiment(tweet2, True))
    print(test.predict_prob_sentiment(tweet2, True))
