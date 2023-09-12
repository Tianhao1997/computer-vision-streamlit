import streamlit as st
from streamlit_option_menu import option_menu
from video_object_detection import VideoObjectDetection
from image_object_detection import ImageObjectDetection
from facial_emotion_recognition import FacialEmotionRecognition
from hand_gesture_classification import HandGestureClassification
from image_optical_character_recgonition import ImageOpticalCharacterRecognition
from image_classification import ImageClassification
from video_utils import create_video_frames
import plotly.express as px
from PIL import Image
from io import BytesIO
import base64
import json
import os
import cv2
import numpy as np
from streamlit_webrtc import (
    RTCConfiguration,
    WebRtcMode,
    WebRtcStreamerContext,
    webrtc_streamer,
)

# Hide warnings to make it easier to locate
# errors in logs, should they show up
import warnings
warnings.filterwarnings("ignore")

# Hide Streamlit logo
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Make Radio buttons horizontal
st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

# Functions to load models
@st.cache(allow_output_mutation=True)
def load_video_object_detection():
    return VideoObjectDetection()

@st.cache(allow_output_mutation=True)
def load_image_object_detection():
    return ImageObjectDetection()

@st.cache(allow_output_mutation=True)
def load_image_classifier():
    return ImageClassification()

@st.cache(allow_output_mutation=True)
def load_facial_emotion_classifier():
    return FacialEmotionRecognition()

@st.cache(allow_output_mutation=True)
def load_hand_gesture_classifier():
    return HandGestureClassification()

@st.cache(allow_output_mutation=True)
def load_image_optical_character_recognition():
    return ImageOpticalCharacterRecognition()


# Load models and store in cache
video_object_detection = load_video_object_detection()
image_object_detection = load_image_object_detection()
facial_emotion_classifier = load_facial_emotion_classifier()
hand_gesture_classifier = load_hand_gesture_classifier()
image_optical_character_recognition = load_image_optical_character_recognition()
image_classifier = load_image_classifier()


#############################################
@st.cache(allow_output_mutation=True)
def gesture_classification(frame):
    """
    Perform hand detection, classification inference, and annotate frame.

    Parameters:
        frame (av.VideoFrame): frame from webcam
    Returns:
        frame (av.VideoFrame): annotated video frame
    """
    x, y, c = frame.shape

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmark prediction
    result = self.hands.process(framergb)

    className = ''

    # post process the result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)

                landmarks.extend([lmx, lmy])

            # Drawing landmarks on frames
            self.mpDraw.draw_landmarks(frame, handslms, self.mpHands.HAND_CONNECTIONS,
                                       self.mpDraw.DrawingSpec(color=(48, 255, 48), thickness=2, circle_radius=3),
                                       self.mpDraw.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=3))

            # Predict gesture
            prediction = model_xgb.predict(np.array(landmarks).reshape(1, -1))
            predicted_names = [k for k, v in gesture_names.items() if v == prediction]
            className = str(predicted_names[0])

            # Get text cooorinates
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            text_width, text_height = cv2.getTextSize(className, font, fontScale, cv2.LINE_AA)[0]
            CenterCoordinates = (int(frame.shape[1] / 2) - int(text_width / 2) - 25, 50)

            # show the prediction on the frame
            cv2.putText(frame, className, CenterCoordinates, cv2.FONT_HERSHEY_SIMPLEX,
                        1, (48, 255, 48), 2, cv2.LINE_AA)
    else:
        text = "Waiting for hand gesture..."

        # show the prediction on the frame
        cv2.putText(frame, text, (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)

    return frame
@st.cache(allow_output_mutation=True)
def callback(frame: av.VideoFrame) -> av.VideoFrame:
    """
    Callback for hand gesture classification through webcam.

    Parameters:
        frame (av.VideoFrame): video frame taken from webcam
    Returns:
        annotated_frame (av.VideoFrame): video frame with annotations included
    """

    image = frame.to_ndarray(format="bgr24")
    annotated_image = gesture_classification(image)

    return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")


model_xgb = XGBClassifier()
model_xgb.load_model("model.json")
gesture_names = {'A': 0,
                 'B': 1,
                 'C': 2,
                 'D': 3,
                 'E': 4,
                 'F': 5,
                 'G': 6,
                 'H': 7,
                 'I': 8,
                 'J': 9,
                 'K': 10,
                 'L': 11,
                 'M': 12,
                 'N': 13,
                 'O': 14,
                 'P': 15,
                 'Q': 16,
                 'R': 17,
                 'S': 18,
                 'T': 19,
                 'U': 20,
                 'V': 21,
                 'W': 22,
                 'X': 23,
                 'Y': 24,
                 'Z': 25}
#################################################################

# Paths for image examples
image_examples = {'Traffic': 'examples/Traffic.jpeg',
                  'Barbeque': 'examples/Barbeque.jpeg',
                  'Home Office': 'examples/Home Office.jpeg',
                  'Car': 'examples/Car.jpeg',
                  'Dog': 'examples/Dog.jpeg',
                  'Tropics': 'examples/Tropics.jpeg',
                  'Quick Brown Dog': 'examples/Quick Brown Dog.png',
                  'Receipt': 'examples/Receipt.png',
                  'Street Sign': 'examples/Street Sign.jpeg',
                  'Kanye': 'examples/Kanye.png',
                  'Shocked': 'examples/Shocked.png',
                  'Yelling': 'examples/Yelling.jpeg'}

# Paths for video examples
video_examples = {'Traffic': 'examples/Traffic.mp4',
                  'Elephant': 'examples/Elephant.mp4',
                  'Airport': 'examples/Airport.mp4',
                  'Kanye': 'examples/Kanye.mp4',
                  'Laughing Guy': 'examples/Laughing Guy.mp4',
                  'Parks and Recreation': 'examples/Parks and Recreation.mp4'}

# Create streamlit sidebar with options for different tasks
with st.sidebar:
    page = option_menu(menu_title='Menu',
                       menu_icon="robot",
                       options=["Welcome!",
                                "Object Detection",
                                "Facial Emotion Recognition",
                                "Hand Gesture Classification",
                                "Optical Character Recognition",
                                "Image Classification"],
                       icons=["house-door",
                              "search",
                              "emoji-smile",
                              "hand-thumbs-up",
                              "eyeglasses",
                              "check-circle"],
                       default_index=0,
                       )

    # Make sidebar slightly larger to accommodate larger names
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 350px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


st.title('Computer Vision Demos for IDE cources')

# Load and display local gif file
file_ = open("resources/camera-robot-eye.gif", "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")
file_.close()

# Page Definitions
if page == "Welcome!":

    # Page info display
    st.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
        unsafe_allow_html=True,
    )

    st.subheader('Quickstart')
    st.write(
        """
        Flip through the pages in the menu on the left hand side bar to perform CV tasks on-demand!
        
        Run computer vision tasks on:
        
            * Images
                * Examples
                * Upload your own
            * Video
                * Webcam
                * Examples
                * Upload your own
        """
    )

if page == "Object Detection":

    # Page info display
    st.header('Object Detection')
    st.markdown("![Alt Text](https://media.giphy.com/media/vAvWgk3NCFXTa/giphy.gif)")
    st.write("This object detection app uses a pretrained YOLOv5 model which was trained to recognize the labels contained within the COCO dataset. More info [here](https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/) on the classes this app can detect.")

    # User selected option for data type
    data_type = st.radio(
        "Select Data Type",
        ('Webcam', 'Video', 'Image'))

    # If data type is Webcam use streamlit_webrtc to connect, use callback function for inference
    if data_type == 'Webcam':
        RTC_CONFIGURATION = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )
        webrtc_ctx = webrtc_streamer(
            key="object-detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_frame_callback=video_object_detection.callback,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    # If data type is Video provide option to use example or upload your own
    elif data_type == 'Video':
        input_type = st.radio(
            "Use example or upload your own?",
            ('Example', 'Upload'))

        # Load in example or uploaded video
        if input_type == 'Example':
            option = st.selectbox(
                'Which example would you like to use?',
                (['Traffic',
                  'Elephant',
                  'Airport']))
            uploaded_file = video_examples[option]
            vid = uploaded_file
        else:
            uploaded_file = st.file_uploader("Choose a file", type=['mp4'])

        # Create video frames and run detection when user clicks run!
        if st.button('🔥 Run!'):
            # Stop according to user input
            if st.button('STOP'):
                pass
            # Throw error if there is no file
            if uploaded_file is None:
                st.error("No file uploaded yet.")
            else:
                # Create file if user uploads their own
                if uploaded_file and input_type == 'Upload':
                    vid = uploaded_file.name
                    with open(vid, mode='wb') as f:
                        f.write(uploaded_file.read())

                # Create video frames
                with st.spinner("Creating video frames..."):
                    frames, fps = create_video_frames(vid)

                # Run Object detection
                with st.spinner("Running object detection..."):
                    st.subheader("Object Detection Predictions")
                    video_object_detection.static_vid_obj(frames, fps)
                    if input_type == 'Upload':
                        # Delete uploaded video after annotation is complete
                        if vid:
                            os.remove(vid)

                # Provide download option
                video_file=open('outputs/annotated_video.mp4', 'rb')
                video_bytes = video_file.read()
                st.download_button(
                    label="Download annotated video",
                    data=video_bytes,
                    file_name='annotated_video.mp4',
                    mime='video/mp4'
                )

    # If data type is Image provide option to use example or upload your own
    elif data_type == 'Image':
        input_type = st.radio(
            "Use example or upload your own?",
            ('Example', 'Upload'))

        # Load in example or uploaded image
        if input_type == 'Example':
            option = st.selectbox(
                'Which example would you like to use?',
                ('Home Office', 'Traffic', 'Barbeque'))
            uploaded_file = image_examples[option]
        else:
            uploaded_file = st.file_uploader("Choose a file", type=['jpg', 'jpeg', 'png'])

        # Run detection and provide download options when user clicks run!
        if st.button('🔥 Run!'):
            # Throw error if there is no file
            if uploaded_file is None:
                st.error("No file uploaded yet.")
            else:
                # Run object detection
                with st.spinner("Running object detection..."):
                    img = Image.open(uploaded_file)
                    labeled_image, detections = image_object_detection.classify(img)

                # Provide download options if objects were detected
                if labeled_image and detections:
                    # Create image buffer and download
                    buf = BytesIO()
                    labeled_image.save(buf, format="PNG")
                    byte_im = buf.getvalue()

                    # Download annotated image options
                    st.subheader("Object Detection Predictions")
                    st.image(labeled_image)
                    st.download_button('Download Image', data=byte_im,file_name="image_object_detection.png", mime="image/jpeg")

                    # Create json and download button
                    st.json(detections)
                    st.download_button('Download Predictions', json.dumps(detections), file_name='image_object_detection.json')

elif page == 'Facial Emotion Recognition':

    # Page info display
    st.header('Facial Emotion Recognition')
    st.markdown("![Alt Text](https://media.giphy.com/media/bnhtSlVeo7BxC/giphy.gif)")
    st.write('This app can classify seven different emotions including: Neutral, Happiness, Surprise, Sadness, Anger, Disgust, and Fear. Try it out!')

    # User selected option for data type
    data_type = st.radio(
        "Select Data Type",
        ('Webcam', 'Video', 'Image'))

    # If data type is Webcam use streamlit_webrtc to connect, use callback function for inference
    if data_type == 'Webcam':
        RTC_CONFIGURATION = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )
        webrtc_ctx = webrtc_streamer(
            key="facial-emotion-recognition",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_frame_callback=facial_emotion_classifier.callback,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    elif data_type == 'Video':
        # Option to use example video or upload your own
        input_type = st.radio(
            "Use example or upload your own?",
            ('Example', 'Upload'))

        # Load in example or uploaded video
        if input_type == 'Example':
            option = st.selectbox(
                'Which example would you like to use?',
                (['Laughing Guy',
                  'Parks and Recreation',
                  'Kanye']))
            uploaded_file = video_examples[option]
            vid = uploaded_file
        else:
            uploaded_file = st.file_uploader("Choose a file", type=['mp4'])

        # Create video frames and run recognition when user clicks run!
        if st.button('🔥 Run!'):
            # Stop according to user input
            if st.button('STOP'):
                pass
            # Throw error if there is no file
            if uploaded_file is None:
                st.error("No file uploaded yet.")
            else:
                # Create file when user uploads their own video
                if uploaded_file and input_type == 'Upload':
                    vid = uploaded_file.name
                    with open(vid, mode='wb') as f:
                        f.write(uploaded_file.read())

                # Create video frames
                with st.spinner("Creating video frames..."):
                    frames, fps = create_video_frames(vid)

                # Run emotion recognition
                with st.spinner("Running emotion recognition..."):
                    st.subheader("Emotion Recognition Predictions")
                    facial_emotion_classifier.static_vid_fer(frames, fps)
                    if input_type == 'Upload':
                        # Delete uploaded video after annotation is complete
                        if vid:
                            os.remove(vid)

                # Provide download options
                video_file=open('outputs/annotated_video.mp4', 'rb')
                video_bytes = video_file.read()
                st.download_button(
                    label="Download annotated video",
                    data=video_bytes,
                    file_name='annotated_video.mp4',
                    mime='video/mp4'
                )

    # If data type is Image provide option to use example or upload your own
    elif data_type == 'Image':
        input_type = st.radio(
            "Use example or upload your own?",
            ('Example', 'Upload'))

        # If data type is Image provide option to use example or upload your own
        if input_type == 'Example':
            option = st.selectbox(
                'Which example would you like to use?',
                ('Kanye', 'Shocked', 'Yelling'))
            uploaded_file = image_examples[option]
        else:
            uploaded_file = st.file_uploader("Choose a file", type=['jpg', 'jpeg', 'png'])

        if st.button('🔥 Run!'):
            # Throw error if there is no file
            if uploaded_file is None:
                st.error("No file uploaded yet.")
            else:
                # Run emotion recognition
                with st.spinner("Running emotion recognition..."):
                    img = cv2.imread(uploaded_file)
                    labeled_image, detections = facial_emotion_classifier.prediction_label(img)

                    # Format output to rgb for display
                    labeled_image = labeled_image[..., ::-1]
                    labeled_image = Image.fromarray(np.uint8(labeled_image))

                # Provide download options if objects were detected
                if labeled_image is not None and detections is not None:
                    # Create image buffer and download
                    buf = BytesIO()
                    labeled_image.save(buf, format="PNG")
                    byte_im = buf.getvalue()

                    # Provide download option annotated image
                    st.subheader("Emotion Recognition Predictions")
                    st.image(labeled_image)
                    st.download_button('Download Image', data=byte_im,file_name="image_emotion_recognition.png", mime="image/jpeg")

                    # Provide download option for predictions
                    st.json(detections)
                    st.download_button('Download Predictions', json.dumps(str(detections)), file_name='image_emotion_recognition.json')
                else:
                    # Display warning when no face is detected in the image
                    st.image(img)
                    st.warning('No faces recognized in this image...')

elif page == 'Hand Gesture Classification':

    # Page info display
    st.header('Hand Gesture Classification')
    st.markdown("![Alt Text](https://media.giphy.com/media/tIeCLkB8geYtW/giphy.gif)")
    st.write('This app can classify ten different hand gestures including: Okay, Peace, Thumbs Up, Thumbs Down, Hang Loose, Stop, Rock On, Star Trek, Fist, Smile Sign. Try it out!')
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    # webrtc_ctx = webrtc_streamer(
    #     key="hand-gesture-classification",
    #     mode=WebRtcMode.SENDRECV,
    #     rtc_configuration=RTC_CONFIGURATION,
    #     video_frame_callback=hand_gesture_classifier.callback,
    #     media_stream_constraints={"video": True, "audio": False},
    #     async_processing=True,
    # )

    webrtc_ctx = webrtc_streamer(
        key="hand-gesture-classification",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

elif page == 'Optical Character Recognition':

    # Page info display
    st.header('Image Optical Character Recognition')
    st.markdown("![Alt Text](https://media.giphy.com/media/JIX9t2j0ZTN9S/giphy.gif)")

    st.warning("Developing...")

    # # User selected option for data type
    # input_type = st.radio(
    #     "Use example or upload your own?",
    #     ('Example', 'Upload'))

    # # Provide option to use example or upload your own
    # if input_type == 'Example':
    #     option = st.selectbox(
    #         'Which example would you like to use?',
    #         ('Quick Brown Dog', 'Receipt', 'Street Sign'))
    #     uploaded_file = image_examples[option]
    # else:
    #     uploaded_file = st.file_uploader("Choose a file", type=['jpg', 'jpeg', 'png'])

    # if st.button('🔥 Run!'):
    #     # Run OCR
    #     with st.spinner("Running optical character recognition..."):
    #         annotated_image, text = image_optical_character_recognition.image_ocr(uploaded_file)

    #     # Create image buffer and download
    #     buf = BytesIO()
    #     annotated_image.save(buf, format="PNG")
    #     byte_im = buf.getvalue()

    #     # Display and provide download option for annotated image
    #     st.subheader("Captioning Prediction")
    #     st.image(annotated_image)
    #     if text == '':
    #         st.wite("No text in this image...")
    #     else:
    #         st.write(text)

    #         st.download_button('Download Text', data=text, file_name='outputs/ocr_pred.txt')

elif page == 'Image Classification':

    # Page info display
    st.header('Image Classification')
    st.markdown("![Alt Text](https://media.giphy.com/media/Zvgb12U8GNjvq/giphy.gif)")

    st.warning("Developing...")
    # # User selected option for data type
    # input_type = st.radio(
    #     "Use example or upload your own?",
    #     ('Example', 'Upload'))

    # # Provide option to use example or upload your own
    # if input_type == 'Example':
    #     option = st.selectbox(
    #         'Which example would you like to use?',
    #         ('Car', 'Dog', 'Tropics'))
    #     uploaded_file = image_examples[option]
    # else:
    #     uploaded_file = st.file_uploader("Choose a file", type=['jpg', 'jpeg', 'png'])

    # if st.button('🔥 Run!'):
    #     # Throw error if there is no file
    #     if uploaded_file is None:
    #         st.error("No file uploaded yet.")
    #     else:
    #         # Run classification
    #         with st.spinner("Running classification..."):
    #             img = Image.open(uploaded_file)
    #             preds = image_classifier.classify(img)

    #         # Display image
    #         st.subheader("Classification Predictions")
    #         st.image(img)
    #         fig = px.bar(preds.sort_values("Pred_Prob", ascending=True), x='Pred_Prob', y='Class', orientation='h')
    #         st.write(fig)

    #         # Provide download option for predictions
    #         st.write("")
    #         csv = preds.to_csv(index=False).encode('utf-8')
    #         st.download_button('Download Predictions',csv,
    #                            file_name='classification_predictions.csv')
