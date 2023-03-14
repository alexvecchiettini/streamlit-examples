import streamlit as st
# Import sentiment analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def main():
    
    st.title('Sentiment analysis')

    input_text = st.text_input("Evaluate this", disabled=False, placeholder="This model ain't that good")

    click = st.button('Run on text')

    if click:
        st.write('Calculating results...')
        # Initialize model
        analyzer = SentimentIntensityAnalyzer()
        # Analyze the text with polarityScores
        result = analyzer.polarity_scores(input_text)

        # string_result = 'negative' if result < .5 else 'positive'
        st.write(f"This text is evaluated as follows:")
        st.write(f"Positive - {result['pos']}")
        st.write(f"Neutral  - {result['neu']}")
        st.write(f"Negative - {result['neg']}")


if __name__ == '__main__':
    main()