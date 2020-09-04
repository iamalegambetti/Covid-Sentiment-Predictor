# Covid-Sentiment-Predictor

PyTorch Implementaion of a Sentiment Analysis Classifier for Italian Tweets related to the Italian Politics during the Lockdown period March-May 2020.

<h3> File Explanation </h3> 
<ul> 
  <li> train.py consists of the model creation and its training on pretrained word2vec embeddings pretrained by the Italian Nlp Association.</li> 
  <li> predictor.py consists of an implementation to clean new tweets and apply the model saved in train.py to new tweets, eventually giving them a score either positive, negative or neutral. </li> 
</ul>


<h3> Further Explanations </h3> 
<ul> 
  <li> The model achieves a ~77% test set accuracy on 3 target categories. </li>
  <li> Data has been collected from this <a href = "https://github.com/charlesmalafosse/open-dataset-for-sentiment-analysis"> GitHub Repo from Charles Malfosse </a>
    Sentiment is classified to either positive, negative, neutral, or mixed. For clarity reasons, mixed sentiment tweets, which were one of the least popular category, were discarded.
  <li> Pretrained word2vec embedding were downloaded from <a href = "http://www.italianlp.it/resources/italian-word-embeddings/"> here</a>.</li>
  <li> Model consists of two Convolutions on an unigram (a word) first and a bi-gram (a combination of two words) later and seeks to learn embedding representations for our training dataset. </li> 
  <li> If you are really interested in the application of the model you can read the report <a href = "https://drive.google.com/file/d/1iEAgCbi5aXCOJaIm0WMTb10Qov8pgCE3/view?usp=sharing"> here</a>. </li>
</ul>

Data and embeddings are not shared for privacy reasons. For any enquiries please contact me at alessandro.gambetti4@gmail.com. Thanks!
