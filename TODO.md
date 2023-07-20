## TODO:
- [ ] User Interface (Jake)
    - [x] Upload your own model
    - [x] Choose image/video, transect/drop quadrat, confidence
    - [x] Manual selection of detections (check how intuitive this is)
    - [ ] Slider for video slicing
    - [x] Batch detection workflow
    - [x] Manual substrate classifier
- [ ] Functional UI Changes
    - [ ] Use form for sidebar toggles
    - [x] Add Substrate to table
    - [x] Add detection results path (no coverage)
    - [x] Add a button to clear the image list
- [x] SAM Integration into website (Jake)
- [ ] Video parsing function
    - [ ] Selectable frame sequence (transects)
    - [ ] Individual tracking after detection (transects)
    - [ ] Finding optimal frame (drop quadrat)
- [x] Results
    - [x] Area calculation inside pvc pipe (need dimensions)
    - [x] Diameter of urchins, stars? (need dimensions)
    - [x] Spreadsheet formatting
- [X] Model (Matt)
    - [x] Better kelp model
- [X] Manual addition of detections
    - [x] https://discuss.streamlit.io/t/new-component-streamlit-image-annotation-a-easy-way-to-annotate-images-using-streamlit/41146
- [ ] Documentation
    - [x] How to use in the app
    - [ ] Add it to the GitHub: https://github.com/JakeFriesen/Spectral_Detection
    - [ ] Collect all data, and give it to Navneet after project is over


Week of July 10th:
- [x] Get Area results calculated (Estimated numbers for now)
    - [x] Area covered by each species (cm^2)
    - [x] Diameter of everything (no discrimination for now)
- [X] Manual Detection Working
    - [x] Get the detection bounding boxes to show in the live editor
    - [X] Get the list of resulting bounding boxes into a useable format at the end (for segmentation)
    - [X] Remove selection in the results list
- [x] Get YOLO segmentation masks working
    - [x] Temporarily remove SAM segmentation
    - [x] Use mask results in the same way as SAM
- [x] Saving Result Images
    - [x] Add to List should also save the detection/segmentation image (with updated selections)
    - [x] Extra button to download the images
- [ ] Installation Documentation
    - [x] Remove SAM model from GitHub, add description about where to add model (unless unused at this point)
    - [x] Talk to Sam about other issues he found when installing it (pip version?)
    - [ ] Remove unused requirements



## Final Stretch Stuff
- [ ] Manual Annotation fixing
    - [x] Get redetection to override existing boxes (e.g. changing confidence should completely redo the boxes)
    - [ ] Test reclassifying boxes - Doesn't work, but solution is to delete and make a new one (possibly fine?)
- [x] Radio buttons for "Drop Quadrat" or not
    - [x] Text input for PVC side length
    - [x] Change results to percentage when not "Drop Quadrat"
- [x] Add classes to manual annotator - Text input to update labels list (cached?)
- [ ] Multiple confidence levels for the model (hardcoded)
- [X] Download Images should reflect manual annotations
- [ ] Formatted detection data dump for future trained models (must include manual annotator data!)
- [ ] Integrate Video stuff into main - get hosted on web site
- [ ] Documentation
    - [ ] Update "About" Tab
    - [ ] Get Installation Guide finished and accurate (incl. video requirements)