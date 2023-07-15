import math
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
import streamlit as st
import pandas as pd
import os
import cv2
import numpy as np
import ffmpegcv
import supervision as sv
from supervision.draw.color import Color, ColorPalette

import settings

@st.cache_data
def init_models():
    SAM_ENCODER_VERSION = "vit_b"
    SAM_CHECKPOINT_PATH = "weights/sam_vit_b_01ec64.pth"
    sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(device=settings.DEVICE)
    global sam_predictor 
    sam_predictor = SamPredictor(sam)

def init_func():
    init_models()
    # global file_name 
    # file_name = ""
    st.session_state['initialized'] = True


# Segment Anything Model
# Runs the segmenter, with proper data formatting
def segment(image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    global sam_predictor
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)


# Checks that a new image is loaded
# Changes the session state accordingly
def change_image(img):
    if img.name != st.session_state.image_name:
        st.session_state['detect'] = False
        st.session_state['predicted'] = False
        st.session_state.image_name = img.name

# Use this to repridict IMMEDIATELY, 
# Detect Does not have to be pressed again
def repredict():
    st.session_state['predicted'] = False
    st.write("Repredict, Predicted is false")


# Use this to repredict AFTER pressing detect
def redetect():
    st.session_state['predicted'] = False
    st.session_state['detect'] = False
    st.write("Redetect, Predicted is false")

# Detect Button 
def click_detect():
    st.session_state['detect'] = True
    


def download_boxes(selected_boxes, img_name):
    #Detections in YOLO format
    return 


# Predict Function
# Performs the object detection and image segmentation
def predict(_model, _uploaded_image, confidence, detect_type):
    boxes = []
    labels = []
    if st.session_state['predicted'] == False:
        res = _model.predict(_uploaded_image, conf=confidence)
        boxes = res[0].boxes
        masks = res[0].masks
        detections = sv.Detections.from_yolov8(res[0])
        classes = res[0].names
        if(detections is not None):
            labels = [
                f"{idx} {classes[class_id]} {confidence:0.2f}"
                for idx, [_, _, confidence, class_id, _] in enumerate(detections)
                ]
        
        box_annotator = sv.BoxAnnotator(text_scale=3, text_thickness=4, thickness=4, text_color=Color.white())
        annotated_image = box_annotator.annotate(scene=np.array(_uploaded_image), detections=detections, labels=labels)
        st.image(annotated_image, caption='Detected Image', use_column_width=True)
        
        #Segmentation
        if detect_type == "Objects + Segmentation":
            with st.spinner('Running Segmenter...'):
                #Do the Segmentation
                detections.mask = segment(
                    image=cv2.cvtColor(np.array(_uploaded_image), cv2.COLOR_BGR2RGB),
                    xyxy=detections.xyxy
                )
                # annotate image with detections
                box_annotator = sv.BoxAnnotator()
                mask_annotator = sv.MaskAnnotator()
                annotated_image = mask_annotator.annotate(scene=np.array(_uploaded_image), detections=detections)
                # annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
                st.image(annotated_image, caption='Segmented Image', use_column_width=True)
                # results_math(detections, _uploaded_image, classes)
        st.session_state['predicted'] = True
        st.session_state.results = [boxes, detections, classes, labels]
    else:
        box_annotator = sv.BoxAnnotator(text_scale=3, text_thickness=4, thickness=4, text_color=Color.white())
        annotated_image = box_annotator.annotate(scene=np.array(_uploaded_image), detections=st.session_state.results[1], labels=st.session_state.results[3])
        st.image(annotated_image, caption='Detected Image', use_column_width=True)
        if detect_type == "Objects + Segmentation":
            box_annotator = sv.BoxAnnotator()
            mask_annotator = sv.MaskAnnotator()
            annotated_image = mask_annotator.annotate(scene=np.array(_uploaded_image), detections=st.session_state.results[1])
            st.image(annotated_image, caption='Segmented Image', use_column_width=True)
    

#Results Calculations
def results_math( _image, detect_type):
    _, detections, classes ,_ = st.session_state.results

    if detect_type == "Objects + Segmentation":
        segmentation_mask = detections.mask
        class_id_list = detections.class_id
        binary_mask = np.where(segmentation_mask > 0.5, 1, 0)

        white_background = np.ones_like(_image) * 255
        new_images = white_background * (1 - binary_mask[..., np.newaxis]) + _image * binary_mask[..., np.newaxis]
    
    # Initialize empty lists to store data
    index_list = []
    class_id_list = []
    result_list = []
    confidence_list = []
    select_list = []

    for idx, [_, _, confidence, class_id, _] in enumerate(detections):
        if detect_type == "Objects + Segmentation":
            result = np.sum(new_images[idx] != 255) / new_images[idx].size * 100
            result_list.append(result)
        select = True
        # Append values to respective lists
        index_list.append(idx)
        class_id_list.append(classes[class_id])
        confidence_list.append(confidence)
        select_list.append(select)
    # Create DataFrame
    if detect_type == "Objects + Segmentation":
        data = {
            'Index': index_list,
            'class_id': class_id_list,
            'Coverage (%)': result_list,
            'Confidence': confidence_list,
            'Select': select_list
        }
    else:
        data = {
            'Index': index_list,
            'class_id': class_id_list,
            'Confidence': confidence_list,
            'Select': select_list
        }

    df = pd.DataFrame(data)

    # Set class_id as the index
    df.set_index('class_id', inplace=True)

    st.write("Image Detection Results")
    if detect_type == "Objects + Segmentation":
        edited_df = st.data_editor(df, disabled=["Index", "class_id", "Coverage (%)", "Confidence"])
    else:
        edited_df = st.data_editor(df, disabled=["Index", "class_id", "Confidence"])
    
    #Manual Substrate Selection
    substrate = substrate_selection()

    #Making the dataframe for an excel sheet
    excel = {}
    excel['Image'] = st.session_state.image_name
    for cl in classes:
        col1 = f"(#) " + classes[cl]
        excel[col1] = 0
        if detect_type == "Objects + Segmentation":
            col2 = f"(%) " + classes[cl]
            excel[col2] = 0.00
    
    excel['Substrate'] = substrate
    dfex = pd.DataFrame(excel, index=[st.session_state.image_name])

    #Put data into the excel dataframe
    for index, row in edited_df.iterrows():
        #Only add data if row is selected
        if(row['Select'] == True):
            id = index
            class_num = f"(#) " + id
            #Increment number of class
            dfex.loc[st.session_state.image_name, class_num] += 1
            if detect_type == "Objects + Segmentation":
                coverage = row['Coverage (%)']
                class_per = f"(%) " + id
                #Add to total coverage
                dfex.loc[st.session_state.image_name, class_per] += coverage

    return dfex

def add_to_list(data):
    #TODO: Update or overwrite list of same images
    if st.session_state.list is not None:
        frames = [st.session_state.list, data]
        st.session_state.list = pd.concat(frames)
    else:
        st.session_state.list = data
    st.session_state.add_to_list = True
    # st.dataframe(st.session_state.list)




# def show_detection_results():
#     boxes, _, _, labels = st.session_state.results

#     #Making the dataframe for an excel sheet
#     excel = {}
#     excel['Image'] = st.session_state.image_name
#     for cl in classes:
#         col1 = f"(#) " + classes[cl]
#         excel[col1] = 0
#     excel['Substrate'] = substrate
#     # st.write(excel)
#     dfex = pd.DataFrame(excel, index=[st.session_state.image_name])
#     # st.dataframe(dfex)

#     #Put data into the excel dataframe
#     for index, row in edited_df.iterrows():
#         #Only add data if row is selected
#         if(row['Select'] == True):
#             id = index
#             coverage = row['Coverage (%)']
#             class_num = f"(#) " + id
#             class_per = f"(%) " + id
#             #Increment number of class
#             dfex.loc[st.session_state.image_name, class_num] += 1
#             #Add to total coverage
#             dfex.loc[st.session_state.image_name, class_per] += coverage



#     selected_boxes = []
#     try:
#         with st.expander("Detection Results"):
#             for idx, box in enumerate(boxes):
#                 checkbox_label = f"{labels[idx]}"
#                 checkbox_state = st.checkbox(checkbox_label, value=True)
#                 if checkbox_state:
#                     #TODO: Format this result for csv!
#                     selected_boxes.append(f"{box.xyxy}")
#     except Exception as ex:
#         st.write("No image is uploaded yet!")




#     return selected_boxes

def substrate_selection():
    data_df = pd.DataFrame(
        {
            "Substrate":[
                "Sandy",
            ],
        }
    )
    res = st.data_editor(
        data_df,
        column_config={
            "Substrate": st.column_config.SelectboxColumn(
                "Substrate",
                help = "Manual Substrate Selection",
                width = "medium",
                options = [
                    "Sandy",
                    "Mixed",
                    "Rocky",
                ],
            )
        },
        hide_index = True,
    )
    return res.loc[0]["Substrate"]







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

def preview_video_upload(video_name,data):
    with open(video_name, 'wb') as video_file:
        video_file.write(data)
        
    with open(video_name, 'rb') as video_file:
        video_bytes = video_file.read()
    if video_bytes:
        st.video(video_bytes)
    return video_name

def preview_finished_capture(video_name):
    with open(video_name, 'rb') as video_file:
        video_bytes = video_file.read()
    if video_bytes:
        st.video(video_bytes)

def capture_uploaded_video(conf, model, fps,  source_vid, destination_path):
    """
    Plays a stored video file. Tracks and detects objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.
        fps: Frame rate to sample the input video at.
        source_path: Path/input.[MP4,MPEG]
        destinantion_path: Path/output.[MP4,MPEG]

    Returns:
        None

    Raises:
        None
    """
    with st.spinner("Processing Video Capture..."):
        _, tracker = display_tracker_options()

        if st.sidebar.button('Detect Video Objects'):
            try:
                vid_cap = ffmpegcv.VideoCapture(source_vid)
                video_out = ffmpegcv.VideoWriter(destination_path, 'h264', vid_cap.fps*fps)
                if video_out is None:
                    raise Exception("Error creating VideoWriter")
                Urchins = [0,0]
                frame_count = 0
                with vid_cap, video_out:
                    for frame in vid_cap:
                        frame_count = frame_count + 1
                        results = model.track(frame, conf=conf, iou=0.2, persist=True, tracker=tracker, device=settings.DEVICE)[0]

                        if results.boxes.id is not None:
                            boxes = results.boxes.xyxy.cpu().numpy().astype(int)
                            ids = results.boxes.id.cpu().numpy().astype(int)
                            clss = results.boxes.cls.cpu().numpy().astype(int)
                            confs = results.boxes.conf.cpu().numpy().astype(float)

                            # Label Detection
                            for box_num  in range(len(boxes)):
                                color =  (0, 255, 0)
                                if clss[box_num] == 1:
                                    if ids[box_num]>Urchins[1]:
                                        if frame_count%10==0:
                                            Urchins[0]=  Urchins[0] +1
                                            Urchins[1] = ids[box_num]
                                            color =  (255, 0, 255)
                                        else:
                                            color =  (163, 255, 163) 

                                cv2.rectangle( 
                                    frame,
                                    (boxes[box_num][0], boxes[box_num][1]),
                                    (boxes[box_num][2], boxes[box_num][3]),
                                    color,
                                    5)
                                cv2.putText(
                                    frame,
                                    f" Id:{ids[box_num]} Class:{clss[box_num]}; Conf:{round(confs[box_num],2)} ",
                                    (boxes[box_num][0], boxes[box_num][1]),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    2,
                                    color,
                                    2)

                        cv2.putText(
                            frame,
                            f"Urchins: {Urchins[0]}",
                            (20,90),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            5,
                            (0, 255, 255),
                            2)    
                        video_out.write(frame)
                vid_cap.release()
                video_out.release()
                if os.path.exists(destination_path):
                    print("Capture Done. " + str(Urchins[0]) + " of Urchins found.")
                    return True
            except Exception as e:
                import traceback
                st.sidebar.error("Error loading video: " + str(e))
                traceback.print_exc()
    return True
