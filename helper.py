from ultralytics import YOLO
from segment_anything import SamPredictor
import streamlit as st
import pandas as pd
import base64
import cv2
import numpy as np
import supervision as sv
from supervision.draw.color import Color, ColorPalette

import settings

# def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
#     sam_predictor.set_image(image)
#     result_masks = []
#     for box in xyxy:
#         masks, scores, logits = sam_predictor.predict(
#             box=box,
#             multimask_output=True
#         )
#         index = np.argmax(scores)
#         result_masks.append(masks[index])
#     return np.array(result_masks)


def click_detect():
    st.session_state['detect'] = True
    st.session_state['predicted'] = False
def click_download():
    st.session_state['download'] = True

def download_boxes(selected_boxes):
    # Create a DataFrame to hold the selected bounding box data
    df = pd.DataFrame(selected_boxes, columns=["Data"])

    # Convert the DataFrame to a CSV string
    csv_string = df.to_csv().encode('utf-8')
    return csv_string
    # Create a download link for the CSV file
    # b64 = base64.b64encode(csv_string.encode()).decode()
    # href = f'<a href="data:file/csv;base64,{b64}" download="selected_boxes.csv">Download Selected Boxes Data</a>'
    # st.markdown(href, unsafe_allow_html=True)

@st.cache_data
def predict(_model, _uploaded_image, confidence):
    res = _model.predict(_uploaded_image, conf=confidence)
    boxes = res[0].boxes
    masks = res[0].masks
    st.session_state['predicted'] = True
    detections = sv.Detections.from_yolov8(res[0])
    classes = ["Sea Urchin"]
    labels = [
        f"{idx} {classes[class_id]} {confidence:0.2f}"
        for idx, [_, _, confidence, class_id, _] in enumerate(detections)
        ]
    box_annotator = sv.BoxAnnotator(text_scale=3, text_thickness=4, thickness=4, text_color=Color.white())
    annotated_image = box_annotator.annotate(scene=np.array(_uploaded_image), detections=detections, labels=labels)
    sv.plot_image(annotated_image, (16, 16))
    st.image(annotated_image, caption='Detected Image', use_column_width=True)

    #Segmentation
    # boxes.mask = segment(
    #     sam_predictor=sam_predictor,
    #     image=cv2.cvtColor(_uploaded_image, cv2.COLOR_BGR2RGB),
    #     xyxy=boxes.xyxy
    # )

    # # annotate image with detections
    # box_annotator = sv.BoxAnnotator()
    # mask_annotator = sv.MaskAnnotator()
    # annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
    # # annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
    # sv.plot_image(annotated_image, (16, 16))
    return boxes, labels



def show_detection_results(boxes, labels):
    selected_boxes = []
    try:
        with st.expander("Detection Results"):
            for idx, box in enumerate(boxes):
                checkbox_label = f"{labels[idx]}"
                checkbox_state = st.checkbox(checkbox_label, value=True)
                if checkbox_state:
                    #TODO: Format this result for csv!
                    selected_boxes.append(f"{box.xyxy}")
    except Exception as ex:
        st.write("No image is uploaded yet!")
    return selected_boxes









def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model


def display_tracker_options():
    display_tracker = st.radio("Display Tracker", ('Yes', 'No'))
    is_display_tracker = True if display_tracker == 'Yes' else False
    if is_display_tracker:
        tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))
        return is_display_tracker, tracker_type
    return is_display_tracker, None


def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=None, tracker=None):
    """
    Display the detected objects on a video frame using the YOLOv8 model.

    Args:
    - conf (float): Confidence threshold for object detection.
    - model (YoloV8): A YOLOv8 object detection model.
    - st_frame (Streamlit object): A Streamlit object to display the detected video.
    - image (numpy array): A numpy array representing the video frame.
    - is_display_tracking (bool): A flag indicating whether to display object tracking (default=None).

    Returns:
    None
    """

    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720*(9/16))))

    # Display object tracking, if specified
    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True, tracker=tracker)
    else:
        # Predict the objects in the image using the YOLOv8 model
        res = model.predict(image, conf=conf)

    # # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )


def play_stored_video(conf, model):
    """
    Plays a stored video file. Tracks and detects objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_vid = st.sidebar.selectbox(
        "Choose a video...", settings.VIDEOS_DICT.keys())

    is_display_tracker, tracker = display_tracker_options()

    with open(settings.VIDEOS_DICT.get(source_vid), 'rb') as video_file:
        video_bytes = video_file.read()
    if video_bytes:
        st.video(video_bytes)

    if st.sidebar.button('Detect Video Objects'):
        try:
            vid_cap = cv2.VideoCapture(
                str(settings.VIDEOS_DICT.get(source_vid)))
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))