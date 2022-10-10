
#read files and merge the same type of training and test data
#into oone column using rbind 

x <- read.table("train/X_train.txt")
x <- rbind(x,read.table("test/X_test.txt"))
y <- read.table("train/y_train.txt")
y <- rbind(y,read.table("test/y_test.txt"))
subject <- read.table("train/subject_train.txt")
subject <- rbind(subject,read.table("test/subject_test.txt"))

#read the measurement features
feature <- read.table("features.txt")

#Extracts only the measurements on the mean and standard deviation
#for each measurement. That is, all mean() and std() elements in feature

MeanStdL <- grepl("mean()|std()",feature[,2])
x <- x[,MeanStdL]

#Appropriately labels the columns with descriptive variable names.

feature <- feature[MeanStdL,]
names(x) <- feature[,2]
names(y) <- "activity"
names(subject) <- "subject_id"

#merge all columns into a data frame using cbind
data <- cbind(subject,y,x)

#Uses descriptive activity names to name the activities in the data set

label <- read.table("activity_labels.txt")
data$activity <- as.character(data$activity)
for (i in (1:6)) data$activity <- sub(i,label[i,2],data$activity)

#creates a second, independent tidy data set with the average of each variable
#for each activity and each subject.

new <- aggregate(x,list(activity = data$activity,subject_id = data$subject_id),mean)

#write the result in a file 

write.table(new, file = "tidydata.txt",row.names = FALSE)

