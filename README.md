# Thermoreflectance Calibration Module
Developed for calibrating the laser spike annealing system in Michael O. Thompsons lab at Cornell University.

# Workflow
1. Produces shell script that contains the necessary conditions for collecting all the laser data
2. Run shell script to collect data
3. Meanwhile run gen\_summery.py to generate summary images and start to write up yaml files (that designates the usable frames)
4. Run preprocess\_script.py to process the data and produces fitting npys
5. Measure melting power for different materials and put it into temp\_surface.py script
6. Run temp\_surface.py to get the surface fitting

## TODOs
- [x] Reading yaml to select images
- [x] Quick image selection method
- [x] Stacking images correclty
- [ ] Width fit! 
- [ ] On-the-fly summerizing data

## Concept
To be done

## Development log

### 6/4
* Allow choosing the frames you want by specifying it in yaml file
* Generate summary images for quick frame selection manually

### 6/7 
* Modification on shell script code generation
* Ran test run and found that the wafer was melted at the power that should not have melt the wafer according to previous data
* Moving the position after each scan will only overlap the old images

### 6/8 
* Investigate the data that I got on 6/7
* Find the melting power at each dwell (How to define melt)
* Max's fix 
* Start another data collection

### 6/9
* Refine yaml frame choosing scheme
* Refine script for running the preprocess
* Finish data collection

### 6/10
* Get first scan averaged (Finally!!)
* Summary of all conditions

### 6/11
* Get aveage image for all conditions
* Generate yaml files 
* For some reason does not run well on jugaur. Decided to run locally.

### 6/14
* Keep running
* The low power part is not working well

### 6/15
* Use high power position to give good estimate of the peak position for the lower ones
* The pattern in the background comes from the imperfection of blank image. Should use blank for each condition.

### 6/16
* Crop the image to use smaller region for fitting. Improved accuracy and efficiency.

### 6/18, 19
* Redo the anneals using new blank collection scheme
* Changed structure of Block and Calibmnger to accept new form of data


### Some Remarks
* Calibration procedure should start from finding the melting power at each dwell first
