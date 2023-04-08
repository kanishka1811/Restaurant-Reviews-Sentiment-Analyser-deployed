from flask import Flask, render_template, request
import pickle
import numpy as np
import re
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer



model = pickle.load(open('foodReviews.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def form():
    review = request.form.get('text1')
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    new_corpus = [review]
    new_X_test = vectorizer.transform(new_corpus).toarray()
    # prediction

    result = model.predict(new_X_test)


    if result == 1:
        res = "positive";
    else:
        res = "negative"


    return render_template('index.html', prediction_text=f'The review is {res}.')


if __name__ == '__main__':
    app.run( debug=True )

