import streamlit as st
import keras


# def load_image():
#     uploaded_file = st.file_uploader(label='Pick an image to test')
#     if uploaded_file is not None:
#         image_data = uploaded_file.getvalue()
#         st.image(image_data)
#         return Image.open(io.BytesIO(image_data))
#     else:
#         return None


# def load_model():
#     model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
#     model.eval()
#     return model


# def load_labels():
#     labels_path = 'https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt'
#     labels_file = os.path.basename(labels_path)
#     if not os.path.exists(labels_file):
#         wget.download(labels_path)
#     with open(labels_file, "r") as f:
#         categories = [s.strip() for s in f.readlines()]
#         return categories


# def predict(model, categories, image):
#     preprocess = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])
#     input_tensor = preprocess(image)
#     input_batch = input_tensor.unsqueeze(0)

#     with torch.no_grad():
#         output = model(input_batch)

#     probabilities = torch.nn.functional.softmax(output[0], dim=0)

#     top5_prob, top5_catid = torch.topk(probabilities, 5)
#     for i in range(top5_prob.size(0)):
#         st.write(categories[top5_catid[i]], top5_prob[i].item())
###################################################################
# @st.cache(allow_output_mutation=True)
# def load_model():
#     model = keras.models.load_model('models/model_32_GRU_32')
#     return model

# def main():
#     st.title('Sentiment analysis')
#     with st.spinner('Loading model...'):
#         model = load_model()

#     # model = keras.models.load_model('models/model_32_GRU_32')
#     input_text = st.text_input("Evaluate this", disabled=False, placeholder="This model ain't that good")
    
#     click = st.button('Run on text')

#     if click:
#         st.write('Calculating results...')
#         # result = model.predict(input_text)
#         result = .5
#         string_result = 'negative' if result < .5 else 'positive'
#         st.write(f"This text is {string_result}")

#         # predict(model, categories, image)


# if __name__ == '__main__':
#     main()





import streamlit as st
import pandas as pd
import numpy as np

st.title('Uber pickups in NYC')

DATE_COLUMN = 'date/time'
DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
            'streamlit-demo-data/uber-raw-data-sep14.csv.gz')

@st.cache_data
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data

data_load_state = st.text('Loading data...')
data = load_data(10000)
data_load_state.text("Done! (using st.cache_data)")

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)

st.subheader('Number of pickups by hour')
hist_values = np.histogram(data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]
st.bar_chart(hist_values)

# Some number in the range 0-23
hour_to_filter = st.slider('hour', 0, 23, 17)
filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]

st.subheader('Map of all pickups at %s:00' % hour_to_filter)
st.map(filtered_data)