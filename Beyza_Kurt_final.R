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

# wait for user
# for giving the user time to review the output
wait_user <- function() {
  readline(prompt = "\n(Press enter to skip)\n")
}

# Import Dataset
import_df <- function() {
  ds_path = readline(prompt = "Enter the dataset's path for 'read.csv()' command: \n(full path, name or link) ");
  writeLines("")
   
  df <- read.csv(ds_path)
  return(df)
}


# Data Exploration
exploration <- function(df, n_col) {
  # The internal structure of dataset 
  print(str(df))
  
  wait_user()
  
  writeLines("First 4 columns and their first 6 rows:\n")
  print(head(df[, 1:4]))
  
  n_col_3 <- n_col - 3
  
  writeLines("\nLast 4 columns and their first 6 rows:\n")
  print(head(df[, n_col_3 : n_col]))
  
  wait_user()
}


# Data Preparation
preparation <- function(df, y_column_no) {
  set.seed(100)
  
  # Step 1: Get row numbers for the train data
  trainRowNumbers <- createDataPartition(df[,y_column_no], p=0.8, list=FALSE)
  
  # Step 2: Create the training  dataset
  trainData <- df[trainRowNumbers,]
  
  # Step 3: Create the test dataset
  testData <- df[-trainRowNumbers,]
  
  # Store Y for later use
  y = as.factor(trainData[,y_column_no])
  
  skimmed <- skim(trainData)
  writeLines("\nFor The Training Data:")
  print(skimmed)
  
  wait_user()
  
  return(list(trainData, testData, y))
}

# Data Preprocessing
preprocessing <- function(trainData, y, columns, y_column_no) {
  # Missing Data Imputation
  
  methods <- '  "BoxCox", "YeoJohnson", "expoTrans", "center", "scale", "range",
  "knnImpute", "bagImpute", "medianImpute", "pca", "ica",
  "spatialSign", "corr", "zv", "nzv", "conditionalX"'
  writeLines(methods)
  preprocess_method = readline(prompt = "Enter the preprocessing method from above list: ");
  writeLines("")
  
  # With the selected method, a preprocessing model is created to find missing data
  preProcess_missingdata_model <- preProcess(trainData, method=paste(preprocess_method))
  
  writeLines("About Preproccessing Model")
  print(preProcess_missingdata_model)
  wait_user()
  
  # Model is applied to our training dataset
  trainData <- predict(preProcess_missingdata_model, newdata = trainData)
  
  writeLines("After using this model to predict the missing values in trainData,")
  writeLines(paste("Is there any missing data?", anyNA(trainData)))
  
  wait_user()
  
  
  # Converting Categorical Variables
  
  # Dummy variables are converting a categorical variable to 
  # as many binary variables as here are categories
  dummies_model <- dummyVars(as.formula( paste(columns[y_column_no], " ~ .") ), data = trainData)
  writeLines("Dummy variables that is converting a categorical variable to as many binary variables as\nhere are categories, are created.\n")
  
  # Model is applied to our training dataset
  trainData_mat <- predict(dummies_model, newdata = trainData)
  trainData <- data.frame(trainData_mat)
  
  # The internal structure of the train data
  print(str(trainData))
  
  wait_user()
  
  
  # Transform The Data
  
  methods <- '  1. range: Normalize values so it ranges between 0 and 1
  2. center: Subtract Mean
  3. scale: Divide by standard deviation
  4. BoxCox: Remove skewness leading to normality. Values must be > 0
  5. YeoJohnson: Like BoxCox, but works for negative values.
  6. expoTrans: Exponential transformation, works for negative values.
  7. pca: Replace with principal components'
  writeLines(methods)
  
  transform_method = readline(prompt = "Enter the name of the transformation method from above list: ");

  # With the selected method, a preprocessing model is created to transform the data
  preProcess_transformation_model <- preProcess(trainData, method=paste(transform_method))
  
  # Model is applied to our training dataset
  trainData <- predict(preProcess_transformation_model, newdata = trainData)
  
  # The final grade label is added to training dataset
  trainData[[ columns[y_column_no] ]] <- y
  
  writeLines("")
  # The internal structure of the train data
  print(str(trainData))
  
  wait_user()
  
  return(list(trainData, preProcess_missingdata_model, dummies_model, preProcess_transformation_model))
}

# Feature Plots
show_feature_plots <- function(trainData, columns, y_column_no) {
  # plot except the final grade label
  n_col_train <- ncol(trainData)
  n_col_train_1 <- n_col_train - 1
  
  writeLines("\nBox plot is loading...")
  show(featurePlot(x = trainData[, 1:n_col_train_1], 
              y = trainData[[ columns[y_column_no] ]], 
              plot = "box",
              strip=strip.custom(par.strip.text=list(cex=.7)),
              scales = list(x = list(relation="free"), 
                            y = list(relation="free"))))
  
  wait_user()
  
  writeLines("Density plot is loading...")
  show(featurePlot(x = trainData[, 1:n_col_train_1], 
              y = trainData[[ columns[y_column_no] ]], 
              plot = "density",
              strip=strip.custom(par.strip.text=list(cex=.7)),
              scales = list(x = list(relation="free"), 
                            y = list(relation="free"))))
  
  wait_user()
}


# Training
training <- function(trainData, answer, all_models_names, columns, y_column_no) {
  set.seed(100)
  
  methods <- '  "knn", "naive_bayes", "lda", "rpart", "adaboost",
  "bagFDAGCV", "parRF", "rf", "rfRules", "RRF", "gbm", 
  "xgbDART", "xgbLinear", "xgbTree", "bag", "C5.0"
  "C5.0Cost", "C5.0Rules", "C5.0Tree", "fda", "bagEarth", 
  "bagEarthGCV", "earth", "gcvEarth", "bagFDA", "ranger",
  "svmLinear2", "svmLinearWeights", "ordinalRF", "treebag"'
  writeLines(methods)
  
  training_method = readline(prompt = "  Enter the training method from above list: 
  (There are more than these but this script's libraries contains them)");
  
  writeLines("\nThe model is being trained...")
  training_model = train(as.formula( paste(columns[y_column_no], " ~ .") ), data=trainData, method=paste(training_method))
  
  # For being able to name the models in the list that contains all models
  all_models_names <- c(all_models_names, paste(training_method))
  
  if (answer == 1){
    writeLines("\nModel Accuracies plot loading...")
    try(show(plot(training_model, main="Model Accuracies")))
    # Some models does not have this
    # So error will appear
    # But when enter is pressed, the code continues to run
    
    wait_user()
    
    writeLines("Variable Importance plot loading...")
    varimp <- varImp(training_model)
    show(plot(varimp, main="Variable Importance"))
    
    wait_user()
  } else {
    writeLines("")
  }
  
  return(list(training_model, all_models_names))
}


# Prediction
prediction <- function(testData, preProcess_missingdata_model, dummies_model, preProcess_transformation_model, answer) {
  # Step 1: Impute missing values 
  testData2 <- predict(preProcess_missingdata_model, testData)
  
  # Step 2: Create one-hot encodings (dummy variables)
  testData2 <- predict(dummies_model, testData2)
  
  # Step 3: Transform the features with selected method
  testData2 <- predict(preProcess_transformation_model, testData2)
  
  # The internal structure of the test data
  writeLines("Test data after preprocessing models are applied:")
  print(str(data.frame(testData2)))
  
  wait_user()
  
  return(testData2)
}


# Performance Evaluation
performance_evaluation <- function(testData, testData2, training_model, columns, y_column_no) {
  # Test data is predicted by using training model
  fitted <- predict(training_model, testData2)
  
  # Calculates a cross-tabulation of observed and predicted classes with associated statistics
  show(caret::confusionMatrix(reference = as.factor(testData[[ columns[y_column_no] ]]), data = fitted, mode='everything'))
  wait_user()
}


# Comparison
comparison <- function(all_models) {
  # Model performances are compared
  models_compare <- resamples(all_models)
  
  # Summary of the models performances
  show(summary(models_compare))
  
  wait_user()
  
  # Box plots to compare models
  scales <- list(x=list(relation="free"), y=list(relation="free"))
  writeLines("\nModel Compare plot is loading...")
  show(bwplot(models_compare, scales=scales))
  
  wait_user()
}


# Run all functions
main <- function() {
  # Import the Dataset
  df <- import_df()
  
  n_col <- ncol(df) # to be used in many functions
  exploration(df, n_col)   # Data Exploration
  
  y_column_no = readline(prompt = "  Enter the number of the column to be used as the final grade label: 
  For example 18 for 'online_shoppers_intention.csv'");
  y_column_no = as.integer(y_column_no);
  
  # Data Preparation
  c(trainData, testData, y) %<-% preparation(df, y_column_no)
  
  # all names according to the location of the original data set
  # This is used when specific column names are required
  columns <- names(df)
  
  # Data Preprocessing
  c(trainData, preProcess_missingdata_model, dummies_model, preProcess_transformation_model) %<-% preprocessing(trainData, y, columns, y_column_no)
  
  answer = readline(prompt = "Do you want to see feature plots (y/n):")
  answer <- "no"
  if (toupper(answer) == "Y" || toupper(answer) == "YES") {
    show_feature_plots(trainData, columns, y_column_no)
  } else {
    writeLines("")
  }
  
  answer = readline("  Would you like to see the Model Accuracies and Variable Importance plots
  of the training models? yes, one-by-one (1) or no, direct comparison (2)? ")
  answer <- as.integer(answer)
  writeLines("")
  
  all_models <- list()      # to keep all training models
  all_models_names <- c()   # to keep all training models names
  
  cont <- "yes"
  # To create the desired number of training models
  while (toupper(cont) == "Y" || toupper(cont) == "YES") {
    # Training
    c(training_model, all_models_names) %<-% training(trainData, answer, all_models_names, columns, y_column_no)
    
    # add new training model to list
    all_models <- c(all_models, list(training_model))
    
    # Comparison can be made after at least 3 models
    if (length(all_models) > 2) {
      cont = readline(prompt = "Do you want to continue for more models (y/n):")
    }
  }
  
  # Names in the list are replaced with the names of models
  names(all_models) <- all_models_names
  
  # Comparison of Models
  comparison(all_models)
  
  # Application of preprocessing steps to the test data
  testData2 <- prediction(testData, preProcess_missingdata_model, dummies_model, preProcess_transformation_model)
  
  # Predictions
  writeLines(toupper("Predicting Test Data with All Models"))
  for (i in 1:length(all_models)) {
    writeLines(paste(i, ". For", toupper(all_models_names[[i]]), "model:"))
    performance_evaluation(testData, testData2, all_models[[i]], columns, y_column_no)
  }
}

main()
