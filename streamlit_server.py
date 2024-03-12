import cv2
import numpy as np
import streamlit as st
from qdrant_client.models import PointStruct
from st_btn_select import st_btn_select

from services.database import upload_embedding
from services.face_detection import FaceDetection, preprocess_image
from services.face_recognition import FaceRecognition
from services.video_processing import VideoProcessing

if "face_id" not in st.session_state:
    st.session_state.face_id = 0


def stream_video():
    run = True
    conf = st.sidebar.slider('Confidence: ', 0.0, 1.0, 0.9)
    iou = st.sidebar.slider('IOU Threshold: ', 0.0, 1.0, 0.7)

    selection = st_btn_select(('Run', 'Stop'))

    if selection == 'Run':
        run = True
    if selection == 'Stop':
        run = False

    if run:
        frame_window = st.empty()
        video_processing = VideoProcessing(0)
        video_processing.pipeline_to_streamlit(frame_window, conf, iou)
    else:
        st.write('Stopped')


def take_photo():
    detector = FaceDetection()
    detector.load_model()

    face_recognition = FaceRecognition()

    conf = st.sidebar.slider('Confidence: ', 0.0, 1.0, 0.9)
    iou = st.sidebar.slider('IOU Threshold: ', 0.0, 1.0, 0.7)

    with st.form("face_enrollment_form"):
        st.write("Register your face")
        fullname = st.text_input(
            label="fullname",
            placeholder="fullname",
            label_visibility="hidden"
        )
        age = st.text_input(label="age", placeholder="age", label_visibility="hidden")

        picture = st.camera_input("Take a picture")

        if picture:
            bytes_data = picture.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(
                bytes_data, np.uint8), cv2.IMREAD_COLOR)

            image = preprocess_image(cv2_img)

            boxes, _, _ = detector.detect(cv2_img, image, conf, iou)

            for bbox in range(boxes.shape[0]):
                box = boxes[bbox, :]
                extracted_face = cv2_img[box[1]:box[3], box[0]:box[2]]
                face_embedding = face_recognition.generate_embedding(
                    extracted_face
                )

                st.session_state.face_id += 1
                face_id = st.session_state.face_id

                face_data = {
                    "id": face_id,
                    "payload": {
                        "fullname": fullname,
                        "age": age,
                    },
                    "embedding": face_embedding
                }

        submitted = st.form_submit_button(label="Submit")

        if submitted:
            with st.spinner("Enrolling..."):
                upload_embedding(vector_data=[
                    PointStruct(
                        id=face_data["id"],
                        vector=face_data["embedding"],
                        payload=face_data["payload"]
                    )
                ])
            st.success("Enroll Succeed!!!")


def main():
    page_names_to_funcs = {
        # "From Uploaded Picture": from_picture,
        "Stream": stream_video,
        "Single Shoot": take_photo
    }

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
