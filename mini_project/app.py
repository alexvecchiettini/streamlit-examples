import streamlit as st
from keras import models
import re
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
# Import sentiment analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def main():
    
    st.title('Sentiment analysis')

    input_text = st.text_input("Evaluate this", disabled=False, placeholder="This model ain't that good")
    # text = preprocess_text(input_text)
    # tokenizer = Tokenizer(num_words=5000)
    # tokenizer.fit_on_texts(text)
    # text = tokenizer.text_to_sequences(text)
    # text = pad_sequences(text, padding='post', maxlen=100)
    click = st.button('Run on text')

    if click:
        st.write('Calculating results...')
        # Initialize model
        analyzer = SentimentIntensityAnalyzer()
        # Analyze the text with polarityScores
        result = analyzer.polarity_scores(input_text)

        # string_result = 'negative' if result < .5 else 'positive'
        st.write(f"This text is {result}")
    




# def preprocess_text(sentence):
#     # Remove punctuations and numbers
#     sentence = re.sub('[^a-zA-Z]', ' ', sentence)

#     # Single character removal
#     sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

#     # Removing multiple spaces
#     sentence = re.sub(r'\s+', ' ', sentence)

#     return sentence



# @st.cache(allow_output_mutation=True)
# def load_model():
#     model = models.load_model('models/model_32_GRU_32')
#     return model

# def main():
#     st.title('Sentiment analysis')
#     with st.spinner('Loading model...'):
#         model = load_model()

#     input_text = st.text_input("Evaluate this", disabled=False, placeholder="This model ain't that good")
#     text = preprocess_text(input_text)
#     tokenizer = Tokenizer(num_words=5000)
#     tokenizer.fit_on_texts(text)
#     text = tokenizer.text_to_sequences(text)
#     text = pad_sequences(text, padding='post', maxlen=100)
#     click = st.button('Run on text')

#     if click:
#         st.write('Calculating results...')
#         result = model.predict(text)
#         result = .5
#         string_result = 'negative' if result < .5 else 'positive'
#         st.write(f"This text is {string_result}")

#         # predict(model, categories, image)


if __name__ == '__main__':
    main()





# import streamlit as st
# import pandas as pd
# import numpy as np

# st.title('Uber pickups in NYC')

# DATE_COLUMN = 'date/time'
# DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
#             'streamlit-demo-data/uber-raw-data-sep14.csv.gz')

# @st.cache_data
# def load_data(nrows):
#     data = pd.read_csv(DATA_URL, nrows=nrows)
#     lowercase = lambda x: str(x).lower()
#     data.rename(lowercase, axis='columns', inplace=True)
#     data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
#     return data

# data_load_state = st.text('Loading data...')
# data = load_data(10000)
# data_load_state.text("Done! (using st.cache_data)")

# if st.checkbox('Show raw data'):
#     st.subheader('Raw data')
#     st.write(data)

# st.subheader('Number of pickups by hour')
# hist_values = np.histogram(data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]
# st.bar_chart(hist_values)

# # Some number in the range 0-23
# hour_to_filter = st.slider('hour', 0, 23, 17)
# filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]

# st.subheader('Map of all pickups at %s:00' % hour_to_filter)
# st.map(filtered_data)