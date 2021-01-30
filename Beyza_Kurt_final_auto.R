library(caret)
library(skimr)
library(rlang)
library(RANN)
library(randomForest)
library(fastAdaboost)
library(gbm)
library(xgboost)
library(caretEnsemble)
library(C50)
library(earth)
library(e1071)
library(naivebayes)
library(mda)
library(zeallot)

# warning message not necessary for this script
options(warn = -1)
writeLines("")

# Import Dataset
import_df <- function() {
  writeLines("IMPORT DATASET\n")
  writeLines("Enter the dataset's path for 'read.csv()' command: online_shoppers_intention.csv \n(full path, name or link)\n")
  
  ds_path <- "online_shoppers_intention.csv"
  df <- read.csv(ds_path)
  
  return(df)
}


# Data Exploration
exploration <- function(df, n_col) {
  writeLines("\nDATA EXPLORATION\n")
  
  # The internal structure of dataset 
  print(str(df))
  
  writeLines("\nFirst 4 columns and their first 6 rows:\n")
  print(head(df[, 1:4]))
  
  n_col_3 <- n_col - 3
  
  writeLines("\nLast 4 columns and their first 6 rows:\n")
  print(head(df[, n_col_3 : n_col]))
  
  writeLines("")
}


# Data Preparation
preparation <- function(df, y_column_no) {
  writeLines("\n\nDATA PREPARATION")
  
  set.seed(100)
  
  # Step 1: Get row numbers for the train data
  trainRowNumbers <- createDataPartition(df[,y_column_no], p=0.8, list=FALSE)
  
  # Step 2: Create the training  dataset
  trainData <- df[trainRowNumbers,]
  
  # Step 3: Create the test dataset
  testData <- df[-trainRowNumbers,]
  
  # Store Y for later use
  y = as.factor(trainData[,y_column_no])
  
  # overview of a data frame
  skimmed <- skim(trainData)
  writeLines("\nOverview of The Training Data:")
  print(skimmed)
  
  writeLines("")
  
  return(list(trainData, testData, y))
}

# Data Preprocessing
preprocessing <- function(df, trainData, y, columns, y_column_no) {
  writeLines("\nDATA PREPROCESSING\n")
  
  # Missing Data Imputation
  writeLines("Missing Data Imputation\n")
  
  methods <- '  "BoxCox", "YeoJohnson", "expoTrans", "center", "scale", "range",
  "knnImpute", "bagImpute", "medianImpute", "pca", "ica",
  "spatialSign", "corr", "zv", "nzv", "conditionalX"'
  writeLines(methods)
  writeLines("  Enter the preprocessing method from above list: knnImpute\n")
  
  preprocess_method <- "knnImpute"
  
  # With the selected method, a preprocessing model is created to find missing data
  preProcess_missingdata_model <- preProcess(trainData, method=paste(preprocess_method))
  
  writeLines("About Preproccessing Model")
  print(preProcess_missingdata_model)
  
  # Model is applied to our training dataset
  trainData <- predict(preProcess_missingdata_model, newdata = trainData)
  
  writeLines("After using this model to predict the missing values in trainData,")
  writeLines(paste("Is there any missing data?", anyNA(trainData)))
  writeLines("")
  
  
  # Converting Categorical Variables
  writeLines("\nConverting Categorical Variables\n")
  
  # Dummy variables is converting a categorical variable to 
  # as many binary variables as here are categories
  dummies_model <- dummyVars(as.formula( paste(columns[y_column_no], " ~ .") ), data = trainData)
  writeLines("Dummy variables that is converting a categorical variable to as many binary variables as\nhere are categories, are created.\n")
  
  # Model is applied to our training dataset
  trainData_mat <- predict(dummies_model, newdata = trainData)
  trainData <- data.frame(trainData_mat)
  
  # The internal structure of the train data
  writeLines("The Internal Structure of The Train Data:")
  print(str(trainData))
  writeLines("")
  
  
  # Transform The Data
  writeLines("\nTransform The Data\n")
  
  methods <- '  1. range: Normalize values so it ranges between 0 and 1
  2. center: Subtract Mean
  3. scale: Divide by standard deviation
  4. BoxCox: Remove skewness leading to normality. Values must be > 0
  5. YeoJohnson: Like BoxCox, but works for negative values.
  6. expoTrans: Exponential transformation, works for negative values.
  7. pca: Replace with principal components'
  writeLines(methods)
  
  writeLines("  Enter the name of the transformation method from above list: range\n")
  
  transform_method <- "range"
  
  # With the selected method, a preprocessing model is created to transform the data
  preProcess_transformation_model <- preProcess(trainData, method=paste(transform_method))
  
  # Model is applied to our training dataset
  trainData <- predict(preProcess_transformation_model, newdata = trainData)
  
  # The final grade label is added to training dataset
  trainData[[ columns[y_column_no] ]] <- y
  
  # The internal structure of the train data
  writeLines("The Internal Structure of The Train Data:")
  print(str(trainData))
  writeLines("")
  
  return(list(trainData, preProcess_missingdata_model, dummies_model, preProcess_transformation_model))
}


show_feature_plots <- function(trainData, columns, y_column_no) {
  # plot except the final grade label
  n_col_train <- ncol(trainData)
  n_col_train_1 <- n_col_train - 1
  
  writeLines("\nBox plot of The Train Data is loading...")
  show(featurePlot(x = trainData[, 1:n_col_train_1], 
                   y = trainData[[ columns[y_column_no] ]], 
                   plot = "box",
                   title("title"),
                   strip=strip.custom(par.strip.text=list(cex=.7)),
                   scales = list(x = list(relation="free"), 
                                 y = list(relation="free"))))
  
  writeLines("")
  
  writeLines("Density plot of The Train Data is loading...")
  show(featurePlot(x = trainData[, 1:n_col_train_1], 
                   y = trainData[[ columns[y_column_no] ]], 
                   plot = "density",
                   strip=strip.custom(par.strip.text=list(cex=.7)),
                   scales = list(x = list(relation="free"), 
                                 y = list(relation="free"))))
  
  writeLines("")
}


# Training
training <- function(trainData, training_method, columns, y_column_no) {
  set.seed(100)
  
  methods <- '  "knn", "naive_bayes", "lda", "rpart", "adaboost",
  "bagFDAGCV", "parRF", "rf", "rfRules", "RRF", "gbm", 
  "xgbDART", "xgbLinear", "xgbTree", "bag", "C5.0"
  "C5.0Cost", "C5.0Rules", "C5.0Tree", "fda", "bagEarth", 
  "bagEarthGCV", "earth", "gcvEarth", "bagFDA", "ranger",
  "svmLinear2", "svmLinearWeights", "ordinalRF", "treebag"'
  writeLines(methods)
  
  writeLines(paste("  Enter the training method from above list:", paste(training_method)))
  writeLines("  (There are more than these but this script's libraries contains them)")
  
  writeLines(paste("\nThe", toupper(paste(training_method)) ,"model is being trained..."))
  training_model = train(as.formula( paste(columns[y_column_no], " ~ .") ), data=trainData, method=paste(training_method))
  
  writeLines("")
  writeLines(paste(toupper(paste(training_method)), "Model Accuracies plot loading..."))
  try(show(plot(training_model, main="Model Accuracies")))
  # Some models does not have this
  # So error will appear
  # But when enter is pressed, the code continues to run
  
  writeLines("")
  
  writeLines(paste(toupper(paste(training_method)), "Variable Importance plot loading..."))
  varimp <- varImp(training_model)
  show(plot(varimp, main="Variable Importance"))
  
  writeLines("")
  
  return(training_model)
}


prediction <- function(testData, preProcess_missingdata_model, dummies_model, preProcess_transformation_model) {
  writeLines("PREDICTION\n")
  
  # Step 1: Impute missing values 
  testData2 <- predict(preProcess_missingdata_model, testData)
  
  # Step 2: Create one-hot encodings (dummy variables)
  testData2 <- predict(dummies_model, testData2)
  
  # Step 3: Transform the features with selected method
  testData2 <- predict(preProcess_transformation_model, testData2)
  
  # The internal structure of the test data
  writeLines("Test data after preprocessing models are applied:")
  print(str(data.frame(testData2)))
  
  writeLines("")
  
  return(testData2)
}


performance_evaluation <- function(testData, testData2, training_model, columns, y_column_no) {
  # Test data is predicted by using training model
  fitted <- predict(training_model, testData2)
  
  # Calculates a cross-tabulation of observed and predicted classes with associated statistics
  show(caret::confusionMatrix(reference = as.factor(testData[[ columns[y_column_no] ]]), data = fitted, mode='everything'))
  writeLines("")
}


comparison <- function(all_models) {
  # Model performances are compared
  models_compare <- resamples(all_models)
  
  # Summary of the models performances
  show(summary(models_compare))
  
  # Box plots to compare models
  scales <- list(x=list(relation="free"), y=list(relation="free"))
  writeLines("Model Compare plot is loading...")
  show(bwplot(models_compare, scales=scales))
  
  writeLines("")
}


# Run all functions
main <- function() {
  # Import the Dataset
  df <- import_df()
  
  n_col <- ncol(df) # to be used in many functions
  exploration(df, n_col)   # Data Exploration
  
  writeLines("  Enter the number of the column to be used as the final grade label: 18 
  For example 18 for 'online_shoppers_intention.csv'")
  y_column_no <- 18
  
  # Data Preparation
  c(trainData, testData, y) %<-% preparation(df, y_column_no)
  
  # all names according to the location of the original data set
  # This is used when specific column names are required
  columns <- names(df)
  
  # Data Preprocessing
  c(trainData, preProcess_missingdata_model, dummies_model, preProcess_transformation_model) %<-% preprocessing(df, trainData, y, columns, y_column_no)
  
  # Feature Plots
  writeLines("Do you want to see feature plots (y/n): y")
  show_feature_plots(trainData, columns, y_column_no)
  
  writeLines("  Would you like to see the Model Accuracies and Variable Importance plots
  of the training models? yes, one-by-one (1) or no, direct comparison (2)? 1\n")
  
  # to keep all training models
  all_models <- list()      
  # to keep all training models names
  all_models_names <- c("knn", "naive_bayes", "lda", "earth", "fda")
  
  # for all models
  writeLines("TRAINING\n")
  for (i in 1:length(all_models_names)) {
    # Training
    training_model <- training(trainData, all_models_names[[i]], columns, y_column_no)
    
    # add new training model to list
    all_models <- c(all_models, list(training_model))
  }
  
  # Names in the list are replaced with the names of models
  names(all_models) <- all_models_names
  
  # Comparison of Models
  comparison(all_models)
  
  # Application of preprocessing steps to the test data
  testData2 <- prediction(testData, preProcess_missingdata_model, dummies_model, preProcess_transformation_model)
  
  # Predictions
  writeLines("")
  writeLines(toupper("Predicting Test Data with All Models"))
  for (i in 1:length(all_models)) {
    writeLines(paste(i, ". For", toupper(all_models_names[[i]]), "model:"))
    performance_evaluation(testData, testData2, all_models[[i]], columns, y_column_no)
  }
}

main()