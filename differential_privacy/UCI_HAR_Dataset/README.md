Additional Readme to the 

Human Activity Recognition Using Smartphones Dataset
Version 1.0

more details regarding the original data can be refer to README.txt


Originally for each record it is provided:
======================================

- Triaxial acceleration from the accelerometer (total acceleration) and the estimated body acceleration.
- Triaxial Angular velocity from the gyroscope. 
- A 561-feature vector with time and frequency domain variables. 
- Its activity label. 
- An identifier of the subject who carried out the experiment.

Further cleaning for the measurements
=======================================
The data then has been processed to, for each activity and each subject, find the average of the mean and standard deviation for each measurement. For the list of the processed measurement, please refer to CodeBook.md

The dataset includes the following files:
=========================================

- 'run_analysis.R': R script that does the following.

  1. Merges the training and the test sets to create one data set.
  2. Extracts only the measurements on the mean and standard   deviation for each measurement.
  3. Uses descriptive activity names to name the activities in the data set
  4. Appropriately labels the data set with descriptive variable names.
  5. From the data set in step 4, creates a second, independent tidy data set with the average of each variable for each activity and each subject.

-'tidydata.txt': the result of the tidy data from step 5 of run_analysis.R. Best view with the following script in R

 tidydata <- read.table("tidydata.txt")
 View(tidydata)

- 'README.txt'

- 'features_info.txt': Shows information about the variables used on the feature vector.

- 'features.txt': List of all features.

- 'activity_labels.txt': Links the class labels with their activity name.

- 'train/X_train.txt': Training set.

- 'train/y_train.txt': Training labels.

- 'test/X_test.txt': Test set.

- 'test/y_test.txt': Test labels.

The following files are available for the train and test data. Their descriptions are equivalent. 

- 'train/subject_train.txt': Each row identifies the subject who performed the activity for each window sample. Its range is from 1 to 30. 

 



