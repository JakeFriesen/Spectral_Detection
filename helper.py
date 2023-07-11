import math
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
import streamlit as st
import pandas as pd
import base64
import cv2
import numpy as np
import supervision as sv
import torch
from supervision.draw.color import Color

import settings

@st.cache_data
def init_models():
    SAM_ENCODER_VERSION = "vit_b"
    SAM_CHECKPOINT_PATH = "weights/sam_vit_b_01ec64.pth"
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(device=DEVICE)
    global sam_predictor 
    sam_predictor = SamPredictor(sam)

def init_func():
    init_models()
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
def change_image(img_list):
    if st.session_state.next_img == True:
        st.session_state.next_img = False
        st.session_state['detect'] = False
        st.session_state['predicted'] = False
        st.session_state.img_num += 1
        if(st.session_state.img_num >= len(img_list)):
            st.write("At the end of image list! Upload more images")
            st.session_state.img_num = len(img_list)-1
    if img_list:
        img = img_list[st.session_state.img_num]
    else:
        img = None
    if img.name != st.session_state.image_name:
        st.session_state['detect'] = False
        st.session_state['predicted'] = False
        st.session_state.image_name = img.name
    

# Use this to repredict IMMEDIATELY, 
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
        
        box_annotator = sv.BoxAnnotator(text_scale=2, text_thickness=3, thickness=3, text_color=Color.white())
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
    diameter_list = []

    #Side length of PVC box in cm
    #TODO: Get actual number for this
    side_length_PVC = 50

    for idx, [_, _, confidence, class_id, _] in enumerate(detections):
        if detect_type == "Objects + Segmentation":
            #Get % of non white pixels inside box (assumed box height is height of image)
            percentage_of_box = np.sum(new_images[idx] != 255) / (new_images[idx].shape[0]*new_images[idx].shape[0]) * 100
            #Area of mask is area of PVC * percentage_of_box / 100
            result = side_length_PVC * side_length_PVC * percentage_of_box / 100
            #Calculate diameter
            diameter = 2 * np.sqrt(result / np.pi)

            result_list.append(result)
            diameter_list.append(diameter)
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
            'Area (cm^2)': result_list,
            'Diameter (cm)': diameter_list,
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
        edited_df = st.data_editor(df, disabled=["Index", "class_id", "Area (cm^2)", "Diameter (cm)", "Confidence"])
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
            col2 = f"Total " + classes[cl] + f" Area (cm^2) " 
            col3 = f"Average " + classes[cl] + f" Diameter (cm)"
            excel[col2] = 0.00
            excel[col3] = 0.00
    
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
                coverage = row['Area (cm^2)']
                class_per = f"Total " + id + f" Area (cm^2) " 
                #Add to total coverage
                dfex.loc[st.session_state.image_name, class_per] += coverage

                #Get Average diameter - Take previous average, and use:
                #  avg_new = ((n-1)*avg_old + d_new)/n
                class_diameter = f"Average " + id + f" Diameter (cm)"
                d_new = row['Diameter (cm)']
                avg_old = dfex.loc[st.session_state.image_name, class_diameter]
                n = dfex.loc[st.session_state.image_name, class_num]
                avg_new = ((n-1)*avg_old + d_new)/n
                dfex.loc[st.session_state.image_name, class_diameter] = avg_new

    #Return Excel Dataframe
    return dfex

def add_to_list(data):
    if st.session_state.list is not None:
        #Check for duplicates
        for index, row in st.session_state.list.iterrows():
            if row['Image'] == data['Image'][0]:
                st.session_state.list= st.session_state.list.drop(index)

        frames = [st.session_state.list, data]
        st.session_state.list = pd.concat(frames)
    else:
        st.session_state.list = data
    st.session_state.add_to_list = True

def clear_image_list():
    st.session_state.list = None
    st.session_state.add_to_list = False
    st.experimental_rerun()

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

