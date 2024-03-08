import cv2
import streamlit as st
from st_btn_select import st_btn_select
import numpy as np
from services.video_processing import VideoProcessing, BBOX_GREEN
from services.face_detection import FaceDetection, preprocess_image

if "faces" not in st.session_state:
    st.session_state.faces = []


def stream_video():
    run = True
    conf = st.sidebar.slider('Confidence: ', 0.0, 1.0, 0.5)
    iou = st.sidebar.slider('IOU Threshold: ', 0.0, 1.0, 0.3)

    selection = st_btn_select(('Run', 'Stop'))

    if selection == 'Run':
        run = True
    if selection == 'Stop':
        run = False

    if run:
        frame_window = st.empty()
        video_processing = VideoProcessing(0)
        video_processing.pipeline_to_streamlit(frame_window)
    else:
        st.write('Stopped')


def take_photo():
    detector = FaceDetection()
    detector.load_model()

    picture = st.camera_input("Take a picture")

    conf = st.sidebar.slider('Confidence: ', 0.0, 1.0, 0.5)
    iou = st.sidebar.slider('IOU Threshold: ', 0.0, 1.0, 0.3)

    if picture:
        bytes_data = picture.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(
            bytes_data, np.uint8), cv2.IMREAD_COLOR)

        image = preprocess_image(cv2_img)

        boxes, _, _ = detector.detect(cv2_img, image)

        for bbox in range(boxes.shape[0]):
            box = boxes[bbox, :]
            # cv2.rectangle(cv2_img, (box[0], box[1]), (box[2], box[3]), BBOX_GREEN, 4)
            st.session_state.faces.append(cv2.cvtColor(cv2_img[box[1]:box[3], box[0]:box[2]], cv2.COLOR_BGR2RGB))


def main():
    page_names_to_funcs = {
        # "From Uploaded Picture": from_picture,
        "Stream": stream_video,
        "Single Shoot": take_photo
    }

    if len(st.session_state.faces):
        for face in st.session_state.faces:
            st.sidebar.image(face)

    selected_page = st.sidebar.selectbox(
        "Select mode", page_names_to_funcs.keys())
    page_names_to_funcs[selected_page]()


if __name__ == "__main__":
    st.set_page_config(
        page_title="Face Recognition Market", page_icon=":pencil2:"
    )
    st.title("Face Recognition Market")
    st.sidebar.subheader("Configuration")
    main()
