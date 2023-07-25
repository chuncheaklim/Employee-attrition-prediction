# Import all necessary packages
library(dplyr) # To do data manipulation
library (DataExplorer) 
library(VIM) # To handle missing data
library(missForest) # To handle missing data
library(caret) # To perform machine learning
library(pscl) # To produce goodness of fit metrics for logistic regression
library(reshape2)
library(ggplot2) # To do visualization
library(zeallot) # To unpack assignments into multiple variables
library(caTools) # To split data
library(naivebayes) # To run naive bayes model
library(randomForest) # To run random forest model
library(ROCR) # To plot ROCAUC
library(smotefamily) # To balance class imbalance
library(car)


# Suppress warning
options(warn=-1)


#########################################################################
#                           Data Importing and Inspection
#########################################################################

# Import data from directory and assign to variable df
df = read.csv("C:\\Users\\oks_o\\Desktop\\Applied Machine Learning\\employee_attrition.csv",  stringsAsFactors = TRUE)

# Inspect head of dataframe
head(df)

# Inspect number of observations and variables and datatype of variables
str(df)

# Inspect dimension of dataset
dim(df)

# Inspect statistic summary of variables and identify any redundant variables
summary(df)

# Drop EmployeeNumber, EmployeeCount, Over18, StandardHours columns as they are either single value or factor and ID
df = subset(df, select = -c(EmployeeNumber, EmployeeCount, Over18, StandardHours))

# Rename Age column
colnames(df)[1] = "Age"

# Inspect column names of df again
colnames(df)


#########################################################################
#                           Data Cleaning
#########################################################################

# Check presence of missing value
table(is.na(df))

# Exclude target variable first before introducing NA into dataset
new.df <- subset(df, select = -2)

# Introduce 1% of NA to every columns into dataset
new.df <- prodNA(new.df, noNA = 0.01)

# Add target variable back to dataframe
new.df$Attrition <- df$Attrition

# Identify missing values in every columns
sapply(new.df, function(new.df) sum(is.na(new.df)))

# Visualize any missing value in dataset
plot_missing(new.df)

# Impute missing values by using hot deck imputation
# Set domain_var equal to "Attrition" avoid break of relation (refer to domain_var before impute)
# Set imp_var equal to FALSE to avoid generation of imp_var columns
imputed_df <- hotdeck(new.df, domain_var = "Attrition", imp_var = FALSE)

# Identify any missing values in every columns after hot deck imputation
sapply(imputed_df, function(imputed_df) sum(is.na(imputed_df)))

# Visualize presence of missing value after imputation
plot_missing(imputed_df)

#########################################################################
#                           Exploratory Data Analysis
#########################################################################

## Plot a lower triangle correlation coefficient matrix
#Select numerical column
df_with_numeric = imputed_df %>% select(where(is.numeric))

#Calculate correlation coefficient
cormat <- round(cor(df_with_numeric),2)
head(cormat)

#Melt the correlation matrix 
melted_cormat <- melt(cormat)
head(melted_cormat)

#Plot the melted correlation matrix
ggplot(data = melted_cormat, aes(x=Var1, y=Var2, fill=value)) + 
  geom_tile()

#Create function to get upper triangle of the correlation matrix
get_upper_tri <- function(cormat){
  cormat[lower.tri(cormat)]<- NA
  return(cormat)
}
upper_tri <- get_upper_tri(cormat)
head(upper_tri)


#Create function to reorder correlation matrix
reorder_cormat <- function(cormat){
  # Use correlation between variables as distance
  dd <- as.dist((1-cormat)/2)
  hc <- hclust(dd)
  cormat <-cormat[hc$order, hc$order]
}


# Melt the correlation matrix
melted_cormat <- melt(upper_tri, na.rm = TRUE)
# Create a ggheatmap
ggheatmap <- ggplot(melted_cormat, aes(Var2, Var1, fill = value))+
  geom_tile(color = "white")+
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name="Pearson\nCorrelation") +
  theme_minimal()+ # minimal theme
  theme(axis.text.x = element_text(angle = 45, vjust = 1, 
                                   size = 12, hjust = 1))+
  coord_fixed()

#Input the correlation coefficient values into heatmap
ggheatmap + 
  geom_text(aes(Var2, Var1, label = value), color = "black", size = 4) +
  theme(
    axis.title.x = element_blank(),
    axis.title.y = element_blank(),
    panel.grid.major = element_blank(),
    panel.border = element_blank(),
    panel.background = element_blank(),
    axis.ticks = element_blank(),
    legend.justification = c(1, 0),
    legend.position = c(0.6, 0.7),
    legend.direction = "horizontal")+
  guides(fill = guide_colorbar(barwidth = 7, barheight = 1,
                               title.position = "top", title.hjust = 0.5))

# Plot boxplot for multiple columns to detect outliers
m1 <- melt(imputed_df)
ggplot(m1,aes(x = variable,y = value)) + facet_wrap(~variable, scales = 'free') + geom_boxplot()

# Identify if different job level will affect attrition
ggplot(imputed_df, aes(x = JobLevel, fill = Attrition)) + 
  geom_bar(position = "dodge") +
  geom_text(aes(label = ..count..), stat = "count", vjust = -1.0, colour = "black", position=position_dodge(width=0.9)) +
  labs(title = "Attrition rate for different Job Level") +
  theme_classic()+ theme(plot.title = element_text(hjust = 0.5))

# Identify overtime rate across different job level
ggplot(imputed_df, aes(x = JobLevel, fill = OverTime)) + 
  geom_bar(position = "dodge") +
  geom_text(aes(label = ..count..), stat = "count", vjust = -1.0, colour = "black", position=position_dodge(width=0.9)) +
  labs(title = "OverTime status across different job level") +
  theme_classic()+ theme(plot.title = element_text(hjust = 0.5))

# Identify if satisfaction level of environment will affect attrition
ggplot(imputed_df, aes(x = EnvironmentSatisfaction, fill = Attrition)) + 
  geom_bar(position = "dodge") +
  geom_text(aes(label = ..count..), stat = "count", vjust = -1.0, colour = "black", position=position_dodge(width=0.9)) +
  labs(title = "Attrition rate for different environment satisfaction") +
  theme_classic()+ theme(plot.title = element_text(hjust = 0.5))

# Identify if OverTime will effect Attrition
ggplot(imputed_df, aes(x = OverTime, fill = Attrition)) + 
  geom_bar(position = "dodge") +
  geom_text(aes(label = ..count..), stat = "count", vjust = -1.0, colour = "black", position=position_dodge(width=0.9)) +
  labs(title =  "Attrition rate for different OverTime status") +
  theme_classic()+ theme(plot.title = element_text(hjust = 0.5))

# Identify distribution of monthly income in different attrition rate
ggplot(imputed_df, aes(x = Attrition, y = MonthlyIncome, fill = Attrition)) +
  geom_violin() +
  labs(title =  "Monthly Income distribution for different attrition status") +
  theme_classic()+ theme(plot.title = element_text(hjust = 0.5))


#########################################################################
#                           Data Preprocessing
#########################################################################

as.data.frame(colnames(imputed_df))

## Preprocess dataset by using one hot encoding and min max scaler
# Remove Attrition column (target variable) indexed at 31 before passed into preprocessing methods
ds <- subset(imputed_df, select = -31)

# Filter numerical variables with small rating scale and outliers by using their index
filtered_ds <- subset(ds, select = -c(16, 18, 21:30))
rating_outliers_ds <- subset(ds, select = c(16, 18, 21:30))

# Filter numerical variables with rating by using their index
without_rating_df <- subset(rating_outliers_ds, select = -c(3:5, 8))
rating_df <- subset(rating_outliers_ds, select = c(3:5, 8))

# Standardize numerical columns with MinMaxScaler
process <- preProcess(as.data.frame(filtered_ds), method=c("range"))
mm_scaled_df <- predict(process, as.data.frame(filtered_ds))

# Standardize numerical columns with outliers with RobustScaler
robustscaler <- function(X){
  med <- median(X)
  iqr <- IQR(X)
  scaled_X <- (X - med)/iqr
  return(scaled_X)
}

rs_scaled_df <- as.data.frame(lapply(without_rating_df, robustscaler))

# Combine output of MinMaxScaler and RobustScaler and rating_df
final_scaled_df <- cbind(mm_scaled_df, rs_scaled_df, rating_df)

# One hot encode categorical variable and assign to variable: dummy
# Set fullRank = T to remove redundant columns and to avoid perfect collinearity
dummy <- dummyVars(" ~ .", data=final_scaled_df, fullRank = T)
preprocessed_df <- data.frame(predict(dummy, newdata=final_scaled_df))

# Add Attrition column back into preprocessed dataset
preprocessed_df$Attrition <- df$Attrition

# Convert "No"/"Yes" of Attrition column into 0 and 1.
preprocessed_df$Attrition <- factor(preprocessed_df$Attrition, levels=c("No", "Yes"), labels=c("0", "1"))

str(preprocessed_df)

#########################################################################
#                       Data Splitting and Sampling
#########################################################################


# Build train_test_split function to split data

train_test_split <- function(dataframe, vector, seed = 42, split_ratio = 0.8){
  
  #' Take a dataset and split it into train and test data and fit it into logistic regression model
  #' Dataframe is dataset to apply splitting
  #' Seed is the seed used for reproducibility
  #' split_ratio is the ratio for splitting
  #' Vector refers to target data label of the dataset
  
  # Set seed to maintain reproducibility
  set.seed(seed)
  
  # Split data into train and test
  sample <- sample.split(vector, SplitRatio = split_ratio)
  train = subset(dataframe, sample == TRUE)
  test = subset(dataframe, sample == FALSE)
  
  # Return train and test dataset
  list(train, test)
}

c(train, test) %<-% train_test_split(preprocessed_df, preprocessed_df$Attrition)
dim(train)
dim(test)

# Training Set 1: Slightly imbalanced original dataset
train1 <- as.data.frame(train)
table(train1$Attrition)

# Training Set 2: Balanced dataset

smote_train <- SMOTE(train[-45], train$Attrition, K=3, dup_size = 2)
train2 <- smote_train$data

# Rename target variable into Attrition to standardize
colnames(train2)[45] = "Attrition"

table(train2$Attrition)

train2$Attrition <- as.factor(train2$Attrition)
str(train2)

#########################################################################
#           User Defined Function for Model Implementation
#########################################################################

## Build function for model implementation and evaluation to simplify code 
# Build predict_train_and_test_data function to predict train and test data for logistic regression
predict_train_and_test_data_for_lr <- function(model, train, test){
  
  #' Pass training and test dataset into model and predict the target 
  #' Model is model built to predict outcome
  #' Train is train dataset
  #' Test is test dataset
  
  # Predict class for train data from model built
  pred_prob_train <- predict(model, type = "response", train)
  pred_prob_train
  pred_class_train <- ifelse(pred_prob_train > 0.5, 1, 0)
  pred_class_train
  
  # Predict class for test data from model built
  pred_prob_test <- predict(model, type = "response", test)
  pred_prob_test
  pred_class_test <- ifelse(pred_prob_test > 0.5, 1, 0)
  pred_class_test
  
  # Return training class prediction and test class prediction
  list(pred_class_train, pred_class_test)
}


# Build predict_train_and_test_data function to predict train and test data 
predict_train_and_test_data <- function(model, train, test){
  
  #' Pass training and test dataset into model and predict the target 
  #' Model is model built to predict outcome
  #' Train is train dataset
  #' Test is test dataset
  
  # Predict class for train data from model built
  pred_train <- predict(model, train)
  pred_train
  
  # Predict class for test data from model built
  pred_test <- predict(model, test)
  pred_test
  
  # Return training class prediction and test class prediction
  list(pred_train, pred_test)
}


# Define build_confusion_matrix to create confusion matrix for test data
build_confusion_matrix_for_train_and_test <- function(actual_train, predicted_train, actual_test, predicted_test){
  
  #' Build confusion matrix for train and test data
  #' actual_train is the vector of actual target in train data
  #' predicted_train is the vector of predicted class in train data by model
  #' actual_test is the vector of actual target in test data
  #' predicted_test is the vector of predicted class in test data by model
  
  # Build confusion matrix for train data
  cm_train <- table(actual_train, predicted_train)
  
  cm_stat_train <- confusionMatrix(cm_train, mode = "everything")
  
  # Build confusion matrix for test data
  cm_test <- table(actual_test, predicted_test)
  
  cm_stat_test <- confusionMatrix(cm_test, mode = "everything")
  
  # Return confusion matrix for train and test data
  list(cm_train, cm_stat_train, cm_test, cm_stat_test)
}

# Build function to calculate metrics from confusion matrix
calc_test_metrics <- function(true_positive, true_negative, false_positive, false_negative){
  
  #' Calculate accuracy, precision, recall and F1_score metrics from confusion matrix
  #' true_positive refers to first row and first column in 2x2 confusion matrix
  #' true_negative refers to second row and second column in 2x2 confusion matrix
  #' false_positive refers to first row and second column in 2x2 confusion matrix
  #' false_negative refers to second row and first column in 2x2 confusion matrix 
  #' Return a metrics table to evaluate model performance
  
  accuracy <- round((true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative) * 100, 2)
  precision <- round(true_positive / (true_positive + false_positive) * 100, 2)
  recall <- round(true_positive / (true_positive + false_negative) * 100, 2)
  F1_score <- round(((2 * precision * recall) / (precision + recall)) , 2)
  
  tab <- matrix(c(accuracy, precision, recall, F1_score), ncol=4, byrow=TRUE)
  
  colnames(tab) <- c("Accuracy", "Precision", "Recall", "F1_score")
  
  tab <- as.table(tab)
  
  return(tab)
}

## Logistic Regression Function Without Hyperparameter Tuning
# Build fit_logreg function to fit logistic regression model into training data
fit_logreg <- function(data, formula){
  
  #' Data refers to dataset passed into training model
  #' Formula is formula used to build model
  
  # Fit train data into logistic regression model
  model <- glm(formula,family=binomial(link='logit'),data=data)
  
  # Return model built
  return(model)
}

## Logistic Regression Function With Hyperparameter Tuning
# Define function for hyperparameter tuning for logistic regression
logreg_hyperparameter_tuning <- function(seed = 1234 , formula, data){
  
  #' A simplified function to fit gridsearchcv with selected hyperparameter of logistic regression (as local variable)
  #' Data refers to dataset passed into cross validation function
  #' Formula is formula used to build model
  
  # Set seed to maintain reproducibility
  set.seed(seed = seed)
  
  # Custom Control Parameters with GridSearchCV
  custom_lr = trainControl(method = "repeatedcv",
                           number = 10, repeats = 5, search = 'grid',
                           verboseIter = F)
  
  # Define GridSearchCV for logistic regression with range of hyperparameters
  en <- train(formula,
              data,
              method = 'glmnet', family = 'binomial',
              metric = "Accuracy",
              tuneGrid = expand.grid(alpha = seq (0, 1, length = 10) ,
                                     lambda = seq(0.001, 1, length = 5)),
              trControl = custom_lr)
}

## Naive Bayes Function Without Hyperparameter Tuning
# Build fit_nb function to fit naive bayes model into training data
fit_nb <- function(data, formula){
  
  #' Data refers to dataset passed into training model
  #' Formula is formula used to build model
  
  # Fit train data into logistic regression model
  model <- naive_bayes(formula, data)
  
  # Return model built
  return(model)
}

## Naive Bayes Function With Hyperparameter Tuning
# Define function for hyperparameter tuning for Naive Bayes
nb_hyperparameter_tuning <- function(seed = 1234 , formula, data){
  
  #' A simplified function to fit gridsearchcv with selected hyperparameter of naive bayes (as local variable)
  #' Data refers to dataset passed into cross validation function
  #' Formula is formula used to build model
  
  # Set seed to maintain reproducibility
  set.seed(seed = seed)
  
  # Custom Control Parameters
  custom_nb = trainControl(method = "repeatedcv",
                           number = 4, repeats = 2, search = 'grid',
                           verboseIter = F)
  
  # Define variable for tuneGrid
  search_grid_nb <- expand.grid(
    usekernel = c(TRUE, FALSE),
    fL = 0:5,
    adjust = seq(0, 5, by = 1)
  )
  
  # Define GridSearchCV for naive bayes with range of hyperparameters
  nb <- train(formula,
              data,
              method = 'nb',
              metric = "Accuracy",
              tuneGrid = search_grid_nb,
              trControl = custom_nb)
}

## Random Forest Function Without Hyperparameter Tuning
# Build fit_rf function to fit random forest model into training data
fit_rf <- function(formula, data){
  
  #' Data refers to dataset passed into training model
  #' Formula is formula used to build model
  
  # Fit train data into random forest model
  rf_model <- randomForest(formula, data)
  
  # Return model built
  return(rf_model)
}

## Random Forest Function With Hyperparameter Tuning
# Custom random forest model for multi hyperparameter tuning 
# Without custom only can tune "mtry" hyperparameter
# Parameters, grid, fit, predict, prob, sort and levels need to be defined to enable customRF to use
# Mtry and ntree hyperparameters are considered in this random forest hyperparameter tuning
customRF <- list(type = "Classification", library = "randomForest", loop = NULL) # define customRF as global variable
customRF$parameters <- data.frame(parameter = c("mtry", "ntree"), class = rep("numeric", 2), label = c("mtry", "ntree"))
customRF$grid <- function(x, y, len = NULL, search = "grid") {}
customRF$fit <- function(x, y, wts, param, lev, last, weights, classProbs, ...) {
  randomForest(x, y, mtry = param$mtry, ntree=param$ntree, ...)
}
customRF$predict <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
  predict(modelFit, newdata)
customRF$prob <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
  predict(modelFit, newdata, type = "prob")
customRF$sort <- function(x) x[order(x[,1]),]
customRF$levels <- function(x) x$classes

# Define function for hyperparameter tuning for random forest
rf_hyperparameter_tuning <- function(seed = 1234 , formula, data){
  
  # Set seed to maintain reproducibility
  set.seed(seed = seed)
  
  # Custom control parameters
  custom_rf <- trainControl(method="repeatedcv", number=2, repeats=1, verboseIter = F, search = "grid")
  
  # Define variable for tuneGrid
  search_grid_rf <- expand.grid(
    .mtry= seq(2, 50, 2),
    .ntree=c(1000, 1500, 2000, 2500)
  )
  
  # Define GridSearchCV for random forest with range of hyperparameters
  rf <- train(formula,
              data,
              method = customRF,
              metric = "Accuracy",
              tuneGrid = search_grid_rf,
              trControl = custom_rf)
}

# Define function for plotting ROC and calculate AUC
roc_plot_and_auc_calc <- function(model, data,  target_vector, title){
  
  #' Plot ROC curve and calculate AUC score of models other than logistic regression
  #' Could be applied on logistic regression after hyperparameter tuning by Grid or Random Search
  #' Model refers to name of model that want to plot ROC and calculate AUC
  #' data is the dataframe of training or test set
  #' target vector is the target variable in the dataframe
  #' title refer to the title set for ROC curve
  #' Return AUC score 

  pred <- as.data.frame(predict(model , type = "prob", data))

  pred_roc = prediction(pred$`1`, target_vector)
  perf = performance(pred_roc, 'tpr', 'fpr')
  plot(perf, colorize = T,
     main = title,
     ylab = "Sensitivity (True Positive Rate)",
     xlab = '1-Specificity (False Positive Rate)',
     print.cutoffs.at = seq(0, 1, 0.1),
     text.adj = c(-0.2, 1.7))


  auc <- round(as.numeric(performance(pred_roc, "auc")@y.values), 3)

  return(auc)
}

#########################################################################
#       Model Implementation and Validation for Training Set 1
#########################################################################

## Logistic Regression
# Without hyperparameter tuning

# Fit logistic regression into train1
lr1 <- fit_logreg(train1, Attrition ~.)
summary(lr1)
pR2(lr1)

# Determine feature importance of lr1 
imp <- as.data.frame(varImp(lr1))
imp <- data.frame(names   = rownames(imp), overall = imp$Overall)
head(imp[order(imp$overall,decreasing = T),], 5) # Sort feature importance descendingly and show first 5

# Predict train and test data with model trained (lr1)
c(lr_train_pred1, lr_test_pred1) %<-% predict_train_and_test_data_for_lr(lr1, train1[,-45], test[,-45])

# Plot ROC curve and calculate AUC
pred_prob1 <- predict(lr1, type = "response" , test)
pred1 = prediction(pred_prob1, test$Attrition)
perf1 <- performance(pred1, "tpr", 'fpr')
plot(perf1, colorize = T,
     main = "ROC curve for logistic regression prediction on test (training set 1)",
     ylab = "Sensitivity (True Positive Rate)",
     xlab = "1-Specificity (False Positive Rate)",
     print.cutoffs.at = seq(0, 1, 0.1),
     text.adj = c(-0.2,1.7))

auc <- round(as.numeric(performance(pred1, "auc")@y.values), 3)
auc

# Build confusion matrix for train and test prediction against actual class
c(lr_train_cm1, lr_train_stat1, lr_test_cm1, lr_test_stat1) %<-% 
  build_confusion_matrix_for_train_and_test(train1$Attrition, lr_train_pred1, test$Attrition, lr_test_pred1)

# Training set accuracy
table(lr_train_pred1)
lr_train_stat1

# Test set accuracy
table(lr_test_pred1)
lr_test_stat1

# With Hyperparameter Tuning
# Perform GridSearchCV for logistic regression 
lr_tune1 <- logreg_hyperparameter_tuning(formula = Attrition ~., data = train1)
lr_tune1$bestTune


# Determine feature importance of lr_tune1
varImp(lr_tune1, scale = TRUE)

# Predict train and test data with model trained (lr_tune1)
c(tune_lr_train_pred1, tune_lr_test_pred1) %<-% predict_train_and_test_data(lr_tune1, train1[,-45], test[-45])

# Plot Roc and calculate Auc for tuned logistic regression
roc_plot_and_auc_calc(lr_tune1, test, test$Attrition, 
         "ROC curve for tuned logistic regression prediction on test (training set 1)")


# Build confusion matrix for train and test prediction against actual class
c(tune_lr_train_cm1, tune_lr_train_stat1, tune_lr_test_cm1, tune_lr_test_stat1) %<-% 
  build_confusion_matrix_for_train_and_test(train1$Attrition, tune_lr_train_pred1, test$Attrition, tune_lr_test_pred1)

# Training set accuracy
table(tune_lr_train_pred1)
tune_lr_train_stat1

# Test set accuracy
table(tune_lr_test_pred1)
tune_lr_test_stat1


# Compare result of dataset 1 in logistic regression with or without hyperparameter tuning
lr_metrics_table1 <- calc_test_metrics(lr_test_cm1[1, 1], lr_test_cm1[2,2], lr_test_cm1[1, 2], lr_test_cm1[2,1])
lr_metrics_table_tune1 <- calc_test_metrics(tune_lr_test_cm1[1, 1], tune_lr_test_cm1[2,2], tune_lr_test_cm1[1, 2], tune_lr_test_cm1[2,1])
lr_ds1_result <- rbind(lr_metrics_table1, lr_metrics_table_tune1)
rownames(lr_ds1_result) <- c("LR Without Tuning in Train1", "LR With Hyperparameter Tuning in Train1")
lr_ds1_result


## Naive bayes 
# Without Hyperparameter Tuning
# Fit naive bayes into train1
nb1 <- fit_nb(data = train1, formula = Attrition ~.)
summary(nb1)

# Predict train and test data with model trained (nb1)
c(nb_train_pred1, nb_test_pred1) %<-% predict_train_and_test_data(nb1, train1[,-45], test[,-45])

# Build confusion matrix for train and test prediction against actual class
c(nb_train_cm1, nb_train_stat1, nb_test_cm1, nb_test_stat1) %<-% 
  build_confusion_matrix_for_train_and_test(train1$Attrition, nb_train_pred1, test$Attrition, nb_test_pred1)

# Training set accuracy
table(nb_train_pred1)
nb_train_stat1

# Test set accuracy
table(nb_test_pred1)
nb_test_stat1

# Plot Roc and calculate Auc for naive bayes
roc_plot_and_auc_calc(nb1, test, test$Attrition, 
                      "ROC curve for naive bayes prediction on test (training set 1)")


# With Hyperparameter Tuning
# Perform GridSearchCV for naive bayes
nb_tune1 <- nb_hyperparameter_tuning(formula = Attrition ~., data = train1)
nb_tune1$bestTune

# Determine feature importance of nb_tune1
varImp(nb_tune1)

# Predict train and test data with model trained (nb_tune1)
c(tune_nb_train_pred1, tune_nb_test_pred1) %<-% predict_train_and_test_data(nb_tune1, train1[,-45], test[,-45])

# Build confusion matrix for train and test prediction against actual class
c(tune_nb_train_cm1, tune_nb_train_stat1, tune_nb_test_cm1, tune_nb_test_stat1) %<-% 
  build_confusion_matrix_for_train_and_test(train1$Attrition, tune_nb_train_pred1, test$Attrition, tune_nb_test_pred1)

# Training set accuracy
table(tune_nb_train_pred1)
tune_nb_train_stat1

# Test set accuracy
table(tune_nb_test_pred1)
tune_nb_test_stat1

# Plot Roc and calculate Auc for tuned naive bayes
roc_plot_and_auc_calc(nb_tune1, test, test$Attrition, 
                      "ROC curve for tuned naive bayes prediction on test (training set 1)")

# Compare result of dataset 1 in logistic regression with or without hyperparameter tuning
nb_metrics_table1 <- calc_test_metrics(nb_test_cm1[1, 1], nb_test_cm1[2,2], nb_test_cm1[1, 2], nb_test_cm1[2,1])
nb_metrics_table_tune1 <- calc_test_metrics(tune_nb_test_cm1[1, 1], tune_nb_test_cm1[2,2], tune_nb_test_cm1[1, 2], tune_nb_test_cm1[2,1])
nb_ds1_result <- rbind(nb_metrics_table1, nb_metrics_table_tune1)
rownames(nb_ds1_result) <- c("NB Without Tuning in Train1", "NB With Hyperparameter Tuning in Train1")
nb_ds1_result


## Random Forest
# Without Hyperparameter Tuning
# Fit random forest into train1
rf1 <- fit_rf(data = train1, formula = Attrition ~.)
summary(rf1)
rf1$ntree
rf1$mtry

# Determine feature importance of rf1
imp_rf1 <- as.data.frame(varImp(rf1))
imp_rf1 <- data.frame(names   = rownames(imp_rf1), overall = imp_rf1$Overall)
head(imp_rf1[order(imp_rf1$overall,decreasing = T),], 5) # Sort feature importance descendingly and show first 5

# Plot feature importance of rf1
varImpPlot(rf1)

# Predict train and test data with model trained (rf1)
c(rf_train_pred1, rf_test_pred1) %<-% predict_train_and_test_data(rf1, train1[,-45], test[,-45])

# Build confusion matrix for train and test prediction against actual class
c(rf_train_cm1, rf_train_stat1, rf_test_cm1, rf_test_stat1) %<-% 
  build_confusion_matrix_for_train_and_test(train1$Attrition, rf_train_pred1, test$Attrition, rf_test_pred1)

# Training set accuracy
table(rf_train_pred1)
rf_train_stat1

# Test set accuracy
table(rf_test_pred1)
rf_test_stat1

# Plot Roc and calculate Auc for random forest
roc_plot_and_auc_calc(rf1, test, test$Attrition, 
                      "ROC curve for random forest prediction on test (training set 1)")


# Random Forest with Hyperparameter Tuning
# Perform GridSearchCV for random forest
rf_tune1 <- rf_hyperparameter_tuning(data = train1, formula = Attrition ~.)
print(rf_tune1)
rf_tune1$bestTune

# Determine feature importance of rf_tune1
varImp(rf_tune1)

# Predict train and test data with model trained (rf_tune1)
c(tune_rf_train_pred1, tune_rf_test_pred1) %<-% predict_train_and_test_data(rf_tune1, train1[,-45], test[,-45])

# Build confusion matrix for train and test prediction against actual class
c(tune_rf_train_cm1, tune_rf_train_stat1, tune_rf_test_cm1, tune_rf_test_stat1) %<-% 
  build_confusion_matrix_for_train_and_test(train1$Attrition, tune_rf_train_pred1, test$Attrition, tune_rf_test_pred1)

# Training set accuracy
table(tune_rf_train_pred1)
tune_rf_train_stat1

# Test set accuracy
table(tune_rf_test_pred1)
tune_rf_test_stat1

# Plot Roc and calculate Auc for random forest
roc_plot_and_auc_calc(rf_tune1, test, test$Attrition, 
                      "ROC curve for tuned random forest prediction on test (training set 1)")

# Compare result of training set 1 in random forest with or without hyperparameter tuning
rf_metrics_table1 <- calc_test_metrics(rf_test_cm1[1, 1], rf_test_cm1[2,2], rf_test_cm1[1, 2], rf_test_cm1[2,1])
rf_metrics_table_tune1 <- calc_test_metrics(tune_rf_test_cm1[1, 1], tune_rf_test_cm1[2,2], tune_rf_test_cm1[1, 2], tune_rf_test_cm1[2,1])
rf_ds1_result <- rbind(rf_metrics_table1, rf_metrics_table_tune1)
rownames(rf_ds1_result) <- c("RF Without Tuning in Train1", "RF With Hyperparameter Tuning in Train1")
rf_ds1_result

#########################################################################
#       Model Implementation and Validation for Training Set 2
########################################################################


## Logistic Regression
# Without hyperparameter tuning
# Fit logistic regression into train2
lr2 <- fit_logreg(train2, Attrition ~.)
summary(lr2)
pR2(lr2)

# Determine feature importance of lr2
imp_lr2 <- as.data.frame(varImp(lr2))
imp_lr2 <- data.frame(names   = rownames(imp_lr2), overall = imp_lr2$Overall)
head(imp_lr2[order(imp_lr2$overall,decreasing = T),], 5) # Sort feature importance descendingly and show first 5

# Predict train and test data with model trained (lr2)
c(lr_train_pred2, lr_test_pred2) %<-% predict_train_and_test_data_for_lr(lr2, train2[,-45], test[,-45])

# Build confusion matrix for train and test prediction against actual class
c(lr_train_cm2, lr_train_stat2, lr_test_cm2, lr_test_stat2) %<-% 
  build_confusion_matrix_for_train_and_test(train2$Attrition, lr_train_pred2, test$Attrition, lr_test_pred2)

# Training set accuracy
table(lr_train_pred2)
lr_train_stat2

# Test set accuracy
table(lr_test_pred2)
lr_test_stat2

# Compute prediction probabilities of logistic regression for training set 2
pred_prob2 <- predict(lr2, type = "response", test)

# Plot ROC curve for pred
pred2 = prediction(pred_prob2, test$Attrition)
perf2 = performance(pred2, 'tpr', 'fpr')
plot(perf2, colorize = T,
     main = 'ROC curve for logistic regression prediction on test (training set 2)',
     ylab = "Sensitivity",
     xlab = '1-Specificity',
     print.cutoffs.at = seq(0, 1, 0.1),
     text.adj = c(-0.2, 1.7))

# Compute AUC
auc2 <- round(as.numeric(performance(pred2, "auc")@y.values), 3)
auc2


# With hyperparameter tuning
# Perform GridSearchCV for logistic regression 
lr_tune2 <- logreg_hyperparameter_tuning(formula = Attrition ~., data = train2)
lr_tune2$bestTune

varImp(lr_tune2, scale = TRUE)

# Predict train and test data with model trained (lr_tune2)
c(tune_lr_train_pred2, tune_lr_test_pred2) %<-% predict_train_and_test_data(lr_tune2, train2[,-45], test[,-45])

# Build confusion matrix for train and test prediction against actual class
c(tune_lr_train_cm2, tune_lr_train_stat2, tune_lr_test_cm2, tune_lr_test_stat2) %<-% 
  build_confusion_matrix_for_train_and_test(train2$Attrition, tune_lr_train_pred2, test$Attrition, tune_lr_test_pred2)

# Training set accuracy
table(tune_lr_train_pred2)
tune_lr_train_stat2

# Test set accuracy
table(tune_lr_test_pred2)
tune_lr_test_stat2

# Plot Roc and calculate Auc for logistic regression
roc_plot_and_auc_calc(lr_tune2, test, test$Attrition, 
                      "ROC curve for tuned logistic regression prediction on test (training set 2)")


# Compare result of train2 in logistic regression with or without hyperparameter tuning
lr_metrics_table2 <- calc_test_metrics(lr_test_cm2[1, 1], lr_test_cm2[2,2], lr_test_cm2[1, 2], lr_test_cm2[2,1])
lr_metrics_table_tune2 <- calc_test_metrics(tune_lr_test_cm2[1, 1], tune_lr_test_cm2[2,2], tune_lr_test_cm2[1, 2], tune_lr_test_cm2[2,1])
lr_ds2_result <- rbind(lr_metrics_table2, lr_metrics_table_tune2)
rownames(lr_ds2_result) <- c("LR Without Tuning in Train2", "LR With Hyperparameter Tuning in Train2")
lr_ds2_result


## Naive Bayes
# Without Hyperparameter Tuning
# Fit naive bayes into train2
nb2 <- fit_nb(data = train2, formula = Attrition ~.)
summary(nb2)

# Predict train and test data with model trained (nb2)
c(nb_train_pred2, nb_test_pred2) %<-% predict_train_and_test_data(nb2, train2[,-45], test[,-45])

# Build confusion matrix for train and test prediction against actual class
c(nb_train_cm2, nb_train_stat2, nb_test_cm2, nb_test_stat2) %<-% 
  build_confusion_matrix_for_train_and_test(train2$Attrition, nb_train_pred2, test$Attrition, nb_test_pred2)

# Training set accuracy
table(nb_train_pred2)
nb_train_stat2

# Test set accuracy
table(nb_test_pred2)
nb_test_stat2

# Plot Roc and calculate Auc for naive bayes
roc_plot_and_auc_calc(nb2, test, test$Attrition, 
                      "ROC curve for naive bayes prediction on test (training set 2)")


# With hyperparameter tuning
# Perform GridSearchCV with naive bayes
nb_tune2 <- nb_hyperparameter_tuning(formula = Attrition ~., data = train2)
nb_tune2$bestTune

varImp(nb_tune2)


# Predict train and test data with model trained (nb_tune2)
c(tune_nb_train_pred2, tune_nb_test_pred2) %<-% predict_train_and_test_data(nb_tune2, train2[,-45], test[,-45])

# Build confusion matrix for train and test prediction against actual class
c(tune_nb_train_cm2, tune_nb_train_stat2, tune_nb_test_cm2, tune_nb_test_stat2) %<-% 
  build_confusion_matrix_for_train_and_test(train2$Attrition, tune_nb_train_pred2, test$Attrition, tune_nb_test_pred2)

# Training set accuracy
table(tune_nb_train_pred2)
tune_nb_train_stat2

# Test set accuracy
table(tune_nb_test_pred2)
tune_nb_test_stat2

# Plot Roc and calculate Auc for tuned naive bayes
roc_plot_and_auc_calc(nb_tune2, test, test$Attrition, 
                      "ROC curve for tuned naive bayes prediction on test (training set 2)")


# Compare result of training set 2 in naive bayes with or without hyperparameter tuning
nb_metrics_table2 <- calc_test_metrics(nb_test_cm2[1, 1], nb_test_cm2[2,2], nb_test_cm2[1, 2], nb_test_cm2[2,1])
nb_metrics_table_tune2 <- calc_test_metrics(tune_nb_test_cm2[1, 1], tune_nb_test_cm2[2,2], tune_nb_test_cm2[1, 2], tune_nb_test_cm2[2,1])
nb_ds2_result <- rbind(nb_metrics_table2, nb_metrics_table_tune2)
rownames(nb_ds2_result) <- c("NB Without Tuning in Train2", "NB With Hyperparameter Tuning in Train2")
nb_ds2_result


## Random Forest
# Without Hyperparameter tuning
# Fit random forest into train2
rf2 <- fit_rf(data = train2, formula = Attrition ~.)
summary(rf2)

# Default parameters for rf2
rf2$ntree
rf2$mtry

# Determine feature importance of rf2
imp_rf2 <- as.data.frame(varImp(rf2))
imp_rf2 <- data.frame(names   = rownames(imp_rf2), overall = imp_rf2$Overall)
head(imp_rf2[order(imp_rf2$overall,decreasing = T),], 5) # Sort feature importance descendingly and show first 5


varImpPlot(rf2)

# Predict train and test data with model trained (rf2)
c(rf_train_pred2, rf_test_pred2) %<-% predict_train_and_test_data(rf2, train2[,-45], test[,-45])

# Build confusion matrix for train and test prediction against actual class
c(rf_train_cm2, rf_train_stat2, rf_test_cm2, rf_test_stat2) %<-% 
  build_confusion_matrix_for_train_and_test(train2$Attrition, rf_train_pred2, test$Attrition, rf_test_pred2)

# Training set accuracy
table(rf_train_pred2)
rf_train_stat2

# Test set accuracy
table(rf_test_pred2)
rf_test_stat2

# Plot Roc and calculate Auc for random forest
roc_plot_and_auc_calc(rf2, test, test$Attrition, 
                      "ROC curve for random forest prediction on test (training set 2)")


# Define function for hyperparameter tuning for random forest
rf_hyperparameter_tuning2 <- function(seed = 1234 , formula, data){
  
  # Set seed to maintain reproducibility
  set.seed(seed = seed)
  
  # Custom control parameters
  custom_rf <- trainControl(method="repeatedcv", number=2, repeats=1, verboseIter = F, search = "grid")
  
  # Define variable for tuneGrid
  search_grid_rf <- expand.grid(
    .mtry= seq(1, 10, 1),
    .ntree=c(50, 700, 50)
  )
  
  # Define GridSearchCV for random forest with range of hyperparameters
  rf <- train(formula,
              data,
              method = customRF,
              metric = "Accuracy",
              tuneGrid = search_grid_rf,
              trControl = custom_rf)
}


# With hyperparameter tuning
# Perform GridSearchCV with random forest
rf_tune2 <- rf_hyperparameter_tuning2(data = train2, formula = Attrition ~.)
print(rf_tune2)

rf_tune2$bestTune

# Top 5 feature importance
varImp(rf_tune2)

# Predict train and test data with model trained (rf_tune2)
c(tune_rf_train_pred2, tune_rf_test_pred2) %<-% predict_train_and_test_data(rf_tune2, train2[,-45], test[,-45])

# Build confusion matrix for train and test prediction against actual class
c(tune_rf_train_cm2, tune_rf_train_stat2, tune_rf_test_cm2, tune_rf_test_stat2) %<-% 
  build_confusion_matrix_for_train_and_test(train2$Attrition, tune_rf_train_pred2, test$Attrition, tune_rf_test_pred2)

# Training set accuracy
table(tune_rf_train_pred2)
tune_rf_train_stat2

# Test set accuracy
table(tune_rf_test_pred2)
tune_rf_test_stat2

# Plot Roc and calculate Auc for tuned random forest
roc_plot_and_auc_calc(rf_tune2, test, test$Attrition, 
                      "ROC curve for tuned random forest prediction on test (training set 2)")

# Compare result of training set 2 in random forest with or without hyperparameter tuning
rf_metrics_table2 <- calc_test_metrics(rf_test_cm2[1, 1], rf_test_cm2[2,2], rf_test_cm2[1, 2], rf_test_cm2[2,1])
rf_metrics_table_tune2 <- calc_test_metrics(tune_rf_test_cm2[1, 1], tune_rf_test_cm2[2,2], tune_rf_test_cm2[1, 2], tune_rf_test_cm2[2,1])
rf_ds2_result <- rbind(rf_metrics_table2, rf_metrics_table_tune2)
rownames(rf_ds2_result) <- c("RF Without Tuning in Train2", "RF With Hyperparameter Tuning in Train2")
rf_ds2_result

########################################################################
#         Solving Overfitting issue of Random Forest
########################################################################

# Solving overfitting of random forest for training set 1
# Set seed to maintain reproducibility
set.seed(seed = 1234)

# Custom control parameters
custom_rf1 <- trainControl(method="repeatedcv", number=4, repeats=2, verboseIter = F, search = "grid")

# Define variable for tuneGrid
search_grid_rf1 <- expand.grid(
  .mtry= seq(1, 5, 1),
  .ntree=seq(50, 300, 50)
)

# Define GridSearchCV for random forest with range of hyperparameters
rf <- train(Attrition ~.,
            train1,
            method = customRF,
            metric = "Accuracy",
            tuneGrid = search_grid_rf1,
            trControl = custom_rf1)


c(tune_rf_train_pred_trial, tune_rf_test_pred_trial) %<-% predict_train_and_test_data(rf, train1[,-45], test[,-45])

c(tune_rf_train_cm_trial, tune_rf_train_stat_trial, tune_rf_test_cm_trial, tune_rf_test_stat_trial) %<-% 
  build_confusion_matrix_for_train_and_test(train1$Attrition, tune_rf_train_pred_trial, test$Attrition, tune_rf_test_pred_trial)

# Training set accuracy
table(tune_rf_train_pred_trial)
tune_rf_train_stat_trial

# Test set accuracy
table(tune_rf_test_pred_trial)
tune_rf_test_stat_trial

# Solving overfitting issue of random forest for training set 2
# Calculate vif of every columns and return it as dataframe
vif_values <- as.data.frame(vif(lr2))

# Reset index of vif dataframe
vif_val <- cbind(newColName = rownames(vif_values), vif_values)
rownames(vif_val) <- 1:nrow(vif_values)
vif_val

# Rename columns of dataframe
colnames(vif_val)[1] = "Columns_name"
colnames(vif_val)[2] = "vif_values"

# Plot barplot of VIF 
ggplot(data = vif_val, aes(reorder(Columns_name,
                   vif_values),
           vif_values))+
  geom_col() +
  geom_abline(slope=0, intercept=5,  col = "red",lty=2) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))

as.data.frame(colnames(train2))

# Remove columns from train2 and test set based on their respective index except MonthlyIncome.
train2_dropped <- subset(train2, select = -c(5, 6, 9, 10, 11, 18, 19, 25))
test_dropped <- subset(test, select = -c(5, 6, 9, 10, 11, 18, 19, 25))


# Random Forest on dropped features
rf_tune_trial2 <- rf_hyperparameter_tuning2(data = train2_dropped, formula = Attrition ~.)
print(rf_tune_trial2)

rf_tune_trial2$bestTune

# Top 5 feature importance
varImp(rf_tune_trial2)

# Predict train and test data with model trained (rf_tune2)
c(tune_rf_train_pred_trial2, tune_rf_test_pred_trial2) %<-% predict_train_and_test_data(rf_tune_trial2, train2_dropped[,-37], test_dropped[,-37])

# Build confusion matrix for train and test prediction against actual class
c(tune_rf_train_cm_trial2, tune_rf_train_stat_trial2, tune_rf_test_cm_trial2, tune_rf_test_stat_trial2) %<-% 
  build_confusion_matrix_for_train_and_test(train2_dropped$Attrition, tune_rf_train_pred_trial2, test_dropped$Attrition, tune_rf_test_pred_trial2)

# Training set accuracy
table(tune_rf_train_pred_trial2)
tune_rf_train_stat_trial2

# Test set accuracy
table(tune_rf_test_pred_trial2)
tune_rf_test_stat_trial2

#########################################################################
#   Feature selection on Original training set to run random forest
########################################################################

# Show the index of every columns
as.data.frame(colnames(preprocessed_df))

# Select variables that listed in feature importances table
selected_df <- subset(preprocessed_df, select = c(1, 2, 4, 7, 14, 17, 27,30, 31, 33, 34, 35, 37, 45))

# Train test split
c(sel_train, sel_test) %<-% train_test_split(selected_df, selected_df$Attrition)

# Perform random forest and make prediction
rf_fs1 <- fit_rf(data = sel_train, formula = Attrition~.)

c(rf_train_pred_fs1, rf_test_pred_fs1) %<-% predict_train_and_test_data(rf_fs1, sel_train[-14], sel_test[-14])

c(rf_train_cm_fs1, rf_train_stat_fs1, rf_test_cm_fs1, rf_test_stat_fs1) %<-% 
  build_confusion_matrix_for_train_and_test(sel_train$Attrition, rf_train_pred_fs1, sel_test$Attrition, rf_test_pred_fs1)

# Statistic of train confusion matrix
rf_train_stat_fs1

# Statistic of test confusion matrix
rf_test_stat_fs1

# Show index of every columns in selected_Df
as.data.frame(colnames(train1))

# Select DistanceFromHome, EnvironmentSatisfaction, OverTime.Yes, YearsAtcompany and Attrition based on columns index
selected_df2 <- subset(selected_df, select = c(4, 5, 9, 13, 14))

# Train test split
c(sel_train2, sel_test2) %<-% train_test_split(selected_df2, selected_df2$Attrition)

# Perform random forest without tuning and make prediction
rf_fs2 <- fit_rf(data = sel_train2, formula = Attrition ~.)

c(rf_train_pred_fs2, rf_test_pred_fs2) %<-% predict_train_and_test_data(rf_fs2, sel_train2[,-5], sel_test2[,-5])

c(rf_train_cm_fs2, rf_train_stat_fs2, rf_test_cm_fs2, rf_test_stat_fs2) %<-% 
  build_confusion_matrix_for_train_and_test(sel_train2$Attrition, rf_train_pred_fs2, sel_test2$Attrition, rf_test_pred_fs2)

# Statistic of second trial of train confusion matrix
rf_train_stat_fs2

# Statistic of second trial of test confusion matrix
rf_test_stat_fs2

# Hyperparameter tuning on remaining variables set
rf_fs2_tuned <- rf_hyperparameter_tuning(formula = Attrition~. ,data = sel_train2)

c(rf_train_pred_fs2_tuned, rf_test_pred_fs2_tuned) %<-% predict_train_and_test_data(rf_fs2_tuned, sel_train2[,-5], sel_test2[,-5])

c(rf_train_cm_fs2_tune, rf_train_stat_fs2_tune, rf_test_cm_fs2_tune, rf_test_stat_fs2_tune) %<-% 
  build_confusion_matrix_for_train_and_test(sel_train2$Attrition, rf_train_pred_fs2_tuned, sel_test2$Attrition, rf_test_pred_fs2_tuned)

# Statistic of tuned second trial of train confusion matrix
rf_train_stat_fs2_tune

# Statistic of tuned second trial of test confusion matrix
rf_test_stat_fs2_tune



#########################################################################
#   Feature selection on balanced training set to run random forest
########################################################################

as.data.frame(colnames(preprocessed_df))
selected_df2 <- subset(preprocessed_df, select = c(2, 17, 31, 35, 37, 45))

c(sel_train3, sel_test3) %<-% train_test_split(selected_df2, selected_df2$Attrition)

smote_train2 <- SMOTE(sel_train3[-6], sel_train3$Attrition, K=3, dup_size = 2)
train2_fs <- smote_train2$data
head(train2_fs)


# Rename target variable into Attrition to standardize
colnames(train2_fs)[6] = "Attrition"
head(train2_fs)

table(train2_fs$Attrition)
dim(train2_fs)

train2_fs$Attrition <- as.factor(train2_fs$Attrition)

rf_fs3 <- fit_rf(Attrition~., train2_fs)

c(rf_bal_train_pred_fs3, rf_bal_test_pred_fs3) %<-% predict_train_and_test_data(rf_fs3, train2_fs, sel_test3)

c(rf_bal_train_fs3_cm, rf_bal_train_fs3_cm_stat, rf_bal_test_fs3_cm, rf_bal_test_fs3_cm_stat) %<-% build_confusion_matrix_for_train_and_test(train2_fs$Attrition, rf_bal_train_pred_fs3, sel_test2$Attrition, rf_bal_test_pred_fs3)


rf_bal_train_fs3_cm_stat
rf_bal_test_fs3_cm_stat
