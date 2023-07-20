import math
from ultralytics import YOLO
import streamlit as st
import pandas as pd
import base64
import cv2
import numpy as np
import supervision as sv
from supervision.draw.color import Color
from streamlit_image_annotation import detection
import os, shutil
import zipfile
from pathlib import Path

import settings


def clear_folder(folder):
    if os.path.exists(folder):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

def init_func():
    # init_models()
    st.session_state['initialized'] = True
    #remove detected images
    clear_folder(settings.RESULTS_DIR)
    clear_folder(settings.IMAGES_DIR)
    clear_folder(settings.DATA_DIR)


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
    st.session_state.segmented = False


# Use this to repredict AFTER pressing detect
def redetect():
    st.session_state['predicted'] = False
    st.session_state['detect'] = False
    st.session_state.segmented = False

# Detect Button 
def click_detect():
    st.session_state['detect'] = True
    


# Predict Function
# Performs the object detection and image segmentation
def predict(_model, _uploaded_image, confidence, detect_type):
    boxes = []
    labels = []
    col1, col2 = st.columns(2)
    # Detection Stage
    if st.session_state['predicted'] == False:
        #TODO: Use sv.Detections.merge([detections1, detections2]) for multiple confidence level support
        # Check if masks are none FIRST
        # If either are None, only keep one
        # Else, merge and take both afterwards
        if st.session_state.model_type == "Built-in":
            res = _model.predict(_uploaded_image, conf=confidence, classes = [0,2,3])
            res1 = _model.predict(_uploaded_image, conf=st.session_state.kelp_conf, classes = [1])
            # boxes = res[0].boxes
            classes = res[0].names
            detections1 = sv.Detections.from_yolov8(res[0])
            detections2 = sv.Detections.from_yolov8(res1[0])
            detections = sv.Detections.merge([detections2, detections1])
            # res = np.concatenate((res.numpy(), res1.numpy()), axis=0)
            # boxes = np.concatenate((res[0].boxes.xyxy.numpy(), res1[0].boxes.xyxy.numpy()), axis = 0)
            if detections1.mask is None:
                detections.mask = detections2.mask
            elif detections2.mask is None:
                detections.mask = detections1.mask
            boxes = detections.xyxy
        else:
            res = _model.predict(_uploaded_image, conf=confidence)
            classes = res[0].names
            detections = sv.Detections.from_yolov8(res[0])
            boxes = detections.xyxy

        # res = _model.predict(_uploaded_image, conf=confidence)
        # boxes = res[0].boxes
        # detections = sv.Detections.from_yolov8(res[0])
        if(detections is not None):
            labels = [
                f"{idx} {classes[class_id]} {confidence:0.2f}"
                for idx, [_, _, confidence, class_id, _] in enumerate(detections)
                ]
        
        box_annotator = sv.BoxAnnotator(text_scale=2, text_thickness=3, thickness=3, text_color=Color.white())
        annotated_image = box_annotator.annotate(scene=np.array(_uploaded_image), detections=detections, labels=labels)
        with col1:
            st.image(annotated_image, caption='Detected Image', use_column_width=True)
        st.session_state.results = [boxes, detections, classes, labels, annotated_image]
    #Interactive Detection Stage
    if interactive_detections():
        #Need to re-run segmenter, the bounding boxes have changed
        st.session_state.segmented = False

    #Segmentation Stage
    if detect_type == "Objects + Segmentation" and st.session_state.segmented == False:
        with col2:
            with st.spinner('Running Segmenter...'):
                #Show the Segmentation
                new_boxes = np.array(st.session_state['result_dict'][st.session_state.image_name]['bboxes'])
                new_boxes = np.floor(new_boxes)
                # Only choose the detection masks that have the same boxes as new_boxes
                cur_boxes = st.session_state.results[0]
                cur_boxes = np.floor(cur_boxes)
                for idx, [_, _, confidence, class_id, _] in enumerate(detections):
                    if cur_boxes[idx] not in new_boxes:
                        detections.mask[idx] = None

                # annotate image with detections
                box_annotator = sv.BoxAnnotator()
                mask_annotator = sv.MaskAnnotator()
                annotated_image = mask_annotator.annotate(scene=np.array(_uploaded_image), detections=detections)
                
                st.image(annotated_image, caption='Segmented Image', use_column_width=True)
                st.session_state.segmented = True
    st.session_state['predicted'] = True
    st.session_state.results[1] = detections  
    

#Results Calculations
def results_math( _image, detect_type):
    boxes, detections, classes ,_ ,_  = st.session_state.results

    if detect_type == "Objects + Segmentation" and detections.mask is not None:
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
    diameter_list = []

    # formatted boxes from manual annotator
    new_boxes = [[b[0], b[1], b[2]+b[0], b[3]+b[1]] for b in st.session_state['result_dict'][st.session_state.image_name]['bboxes']]
    new_boxes = np.array(new_boxes)

    if st.session_state.drop_quadrat == "Area (Drop Quadrat)":
        #Side length of PVC box in cm - Taken from the user
        side_length_PVC = st.session_state.side_length

    detected_boxes = boxes
    detected_boxes = np.floor(detected_boxes)
    new_boxes = np.floor(new_boxes)

    for idx, [_, _, confidence, class_id, _] in enumerate(detections):
        if detected_boxes[idx] in new_boxes:
            if detect_type == "Objects + Segmentation":
                if st.session_state.drop_quadrat == "Area (Drop Quadrat)":
                    #Get % of non white pixels inside box (assumed box height is height of image)
                    percentage_of_box = np.sum(new_images[idx] != 255) / (new_images[idx].shape[0]*new_images[idx].shape[0]) * 100
                    #Area of mask is area of PVC * percentage_of_box / 100
                    result = side_length_PVC * side_length_PVC * percentage_of_box / 100
                    #Calculate diameter
                    diameter = 2 * np.sqrt(result / np.pi)
                    diameter_list.append(diameter)
                elif st.session_state.drop_quadrat == "Percentage":
                    #Just percentage, no diameter
                    result = np.sum(new_images[idx] != 255) / (new_images[idx].size) * 100
                result_list.append(result)
            # select = True
            # Append values to respective lists
            index_list.append(idx)
            class_id_list.append(st.session_state.class_list[class_id])
            confidence_list.append(confidence)
            # select_list.append(select)
    #Add any boxes from manual annotator
    for idx, box in enumerate(new_boxes):
        if box not in detected_boxes:
            if detect_type == "Objects + Segmentation":
                result_list.append(0)
                if st.session_state.drop_quadrat == "Area (Drop Quadrat)":
                    diameter_list.append(0)
            #This is a new box
            index_list.append(idx)
            class_id_list.append(st.session_state.class_list[st.session_state['result_dict'][st.session_state.image_name]['labels'][idx]])
            confidence_list.append(1)
            # select_list.append(True)

    # Create DataFrame
    if detect_type == "Objects + Segmentation":
        if st.session_state.drop_quadrat == "Area (Drop Quadrat)":
            data = {
                'Index': index_list,
                'class_id': class_id_list,
                'Area (cm^2)': result_list,
                'Diameter (cm)': diameter_list,
                'Confidence': confidence_list
            }
        elif st.session_state.drop_quadrat == "Percentage":
            data = {
                'Index': index_list,
                'class_id': class_id_list,
                'Coverage (%)': result_list,
                'Confidence': confidence_list
            }
    else:
        data = {
            'Index': index_list,
            'class_id': class_id_list,
            'Confidence': confidence_list
        }

    df = pd.DataFrame(data)

    # Set class_id as the index
    df.set_index('class_id', inplace=True)

    st.write("Image Detection Results")
    if detect_type == "Objects + Segmentation":
        if st.session_state.drop_quadrat == "Area (Drop Quadrat)":
            edited_df = st.data_editor(df, disabled=["Index", "class_id", "Area (cm^2)", "Diameter (cm)", "Confidence"])
        else:
            edited_df = st.data_editor(df, disabled=["Index", "class_id", "Coverage (%)", "Confidence"])
    else:
        edited_df = st.data_editor(df, disabled=["Index", "class_id", "Confidence"])
    
    #Manual Substrate Selection
    substrate = substrate_selection()

    #Making the dataframe for an excel sheet
    excel = {}
    excel['Image'] = st.session_state.image_name
    for cl in st.session_state.class_list:
        col1 = f"(#) " + cl
        excel[col1] = 0
        if detect_type == "Objects + Segmentation":
            if st.session_state.drop_quadrat == "Area (Drop Quadrat)":
                col2 = f"Total " + cl + f" Area (cm^2) " 
                col3 = f"Average " + cl + f" Diameter (cm)"
                excel[col2] = 0.00
                excel[col3] = 0.00
            else:
                col2 = cl + f" Coverage(%)" 
                excel[col2] = 0.00
            
    
    excel['Substrate'] = substrate
    dfex = pd.DataFrame(excel, index=[st.session_state.image_name])

    #Put data into the excel dataframe
    for index, row in edited_df.iterrows():
        #Only add data if row is selected
        # if(row['Select'] == True):
        id = index
        class_num = f"(#) " + id
        #Increment number of class
        dfex.loc[st.session_state.image_name, class_num] += 1
        if detect_type == "Objects + Segmentation":
            if st.session_state.drop_quadrat == "Area (Drop Quadrat)":
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
            else:
                coverage = row['Coverage (%)']
                class_per = id + f" Coverage(%)" 
                #Add to total coverage
                dfex.loc[st.session_state.image_name, class_per] += coverage

    #Return Excel Dataframe
    return dfex

def add_to_list(data, _image):
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

    #Save the detected image result
    image_path = Path(settings.RESULTS_DIR, st.session_state.image_name)
    #Make a new image with the manual annotations
    saved_image = np.array(_image.copy())
    new_boxes = np.floor(np.array([[b[0], b[1], b[2]+b[0], b[3]+b[1]] for b in st.session_state['result_dict'][st.session_state.image_name]['bboxes']]))
    labels = st.session_state['result_dict'][st.session_state.image_name]['labels']
    for idx, box in enumerate(new_boxes):
        saved_image = cv2.rectangle(saved_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), settings.COLOR_LIST[labels[idx]], 3)
    cv2.imwrite(str(image_path), cv2.cvtColor(saved_image, cv2.COLOR_RGB2BGR))
    #Make the data dump text file here as well
    dump_data()


def clear_image_list():
    st.session_state.list = None
    st.session_state.add_to_list = False
    st.experimental_rerun()
    clear_folder(settings.RESULTS_DIR)

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

def zip_images():
    if not os.path.exists('Detected_Images'):
        os.mkdir('Detected_Images')

    if os.path.exists("Detected_Images/Detection_Images.zip"):
        os.remove("Detected_Images/Detection_Images.zip")
    file_paths = get_all_file_paths("Detected_Images")
    with zipfile.ZipFile('Detected_Images/Detection_Images.zip', 'w') as img_zip:
        for file in file_paths:
            img_zip.write(file)
    with open("Detected_Images/Detection_Images.zip", 'rb') as fp:
        st.download_button( label = "Download Images",
                            help = "Download detection result images",
                            data = fp,
                            file_name = "Detection_Images.zip",
                            mime='text/zip')

def get_all_file_paths(directory):
    # initializing empty file paths list
    file_paths = []
  
    # crawling through directory and subdirectories
    for root, directories, files in os.walk(directory):
        for filename in files:
            # join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)
  
    # returning all file paths
    return file_paths   


def interactive_detections():
    #Grab the list of classes for this detection
    label_list = st.session_state.class_list + list(st.session_state.results[2].values())
    if st.session_state.manual_class != "":
        label_list += [st.session_state.manual_class]
    #Remove duplicates
    label_list = list(dict.fromkeys(label_list))
    st.session_state.class_list = label_list

    bboxes = []
    labels = []

    if 'result_dict' not in st.session_state:
        result_dict = {}
        st.session_state['result_dict'] = result_dict.copy()
    if st.session_state.image_name not in st.session_state.result_dict:
        st.session_state['result_dict'][st.session_state.image_name] = {'bboxes': bboxes,'labels':labels}

    #This is the first run, take the results from the initial detection
    if st.session_state['predicted'] == False:
        for box in st.session_state.results[0]:
            width = box[2] - box[0]
            height = box[3] - box[1]
            bboxes.append([box[0], box[1], width, height])
            # top_coord = [box[0] - (box[2]/2), box[1] - (box[3]/2)]
            # bboxes.append([top_coord[0], top_coord[1], box[2], box[3]])
        for detections in st.session_state.results[1]:
            labels.append(int(detections[3])) 
        st.session_state['result_dict'][st.session_state.image_name] = {'bboxes': bboxes,'labels':labels}
    else:
        bboxes = st.session_state['result_dict'][st.session_state.image_name]['bboxes']
        labels = st.session_state['result_dict'][st.session_state.image_name]['labels']

    target_image_path = Path(settings.IMAGES_DIR , st.session_state.image_name)
    new_labels = detection(image_path=target_image_path, 
                        bboxes=bboxes, 
                        labels=labels, 
                        label_list=label_list, 
                        height = 1080,
                        width = 1920)
    if new_labels is not None:
        st.session_state['result_dict'][st.session_state.image_name]['labels'] = [v['label_id'] for v in new_labels]
        st.session_state['result_dict'][st.session_state.image_name]['bboxes'] = [v['bbox'] for v in new_labels]
        return True
    else:
        return False


def load_model(model_path):
    model = YOLO(model_path)
    return model

def dump_data():
    #Text files are normalized center point, normalized width/height
    #index x y w h 
    #TODO: Put the images in a folder
    # Download the folder on click
    # Make the YAML file
    if not os.path.exists('Dump'):
        os.mkdir('Dump')
    boxes, _, classes, labels, _ = st.session_state.results
    h, w, x = st.session_state.results[4].shape
    file_name = "Dump/"  + st.session_state.image_name[:-3] + "txt"
    with open(file_name, 'a') as f:
        for idx, box in enumerate(boxes):
            wn = float(box[2]-box[0]) / w
            hn = float(box[3] - box[1]) / h
            x1n = float(box[0] + float(box[2]-box[0])/2) / w
            y1n = float(box[1] + float(box[3] - box[1])/2) / h
            cl = st.session_state['result_dict'][st.session_state.image_name]['labels'][idx]
            text_str = f'{cl} {x1n:.3f} {y1n:.3f} {wn:.3f} {hn:.3f} \n'
            f.write(text_str)
    #YAML file:
    #nc:{number of classes}
    #names:['','','']

def dump_data_button():
    if os.path.exists("Dump/data.yaml"):
        os.remove("Dump/data.yaml")     
    #Make the YAML file
    str1 = f'nc: {len(st.session_state.results[2])}\n'
    str2 = f"names: ["
    for name in st.session_state.results[2].values():
        str2 += f"'{name}', "
    str2 = str2[:-2] + "]"
    with open("Dump/data.yaml", 'w') as fp:
        fp.write(str1)
        fp.write(str2)

    #Zip the Data
    if os.path.exists("Dump/Detection_Data.zip"):
        os.remove("Dump/Detection_Data.zip")
    file_paths = get_all_file_paths("Dump")
    with zipfile.ZipFile('Dump/Detection_Data.zip', 'w') as img_zip:
        for file in file_paths:
            img_zip.write(file)
    with open("Dump/Detection_Data.zip", 'rb') as fp:
        st.download_button( label = "Detection Data Dump",
                            help = "Dump all YOLO Detection data, which can be used to train future models.",
                            data = fp,
                            file_name = "Detection_Data.zip",
                            mime='text/zip')

    return