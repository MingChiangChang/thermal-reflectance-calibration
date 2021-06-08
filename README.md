# Thermoreflectance Calibration Module
Developed for calibrating the laser spike annealing system in Michael O. Thompsons lab at Cornell University.

## TODOs
- [x] Reading yaml to select images
- [x] Quick image selection method
- [ ] Stacking images correclty
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

### 6/8 Plans
* Investigate the data that I got on 6/7
* Find the melting power at each dwell (How to define melt)
* Max's fix 
