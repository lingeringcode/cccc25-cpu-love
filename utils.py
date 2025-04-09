'''
  utils.py - Exported functions for this logistic regression project.

  Logistic regression returns a probability that something may or may not happen or "be". We can use this probability as a single value to assess the likelihood of the event/being, such as "This X headline is of Y news genre".

  So, let's say our LR model returns a value of 0.990 for a particular headline's news genre as being "POLITICS". This probability score is very likely to accurately predict that this headline is indeed an article about POLITICS. Conversely, another headline with a prediction score of 0.005 on that same logistic regression model is very likely not about POLITICS. Yet, what about a headline with a prediction score of 0.6? 

  In this lesson, our LR model uses those probability estimates as a binary category. To do so, we must decide what's called a "classification threshold" or "decision threshold". Any value above that threshold indicates a headline is about POLITICS, and any value below the threshold indicates that the headline is not POLITICS, but some other news genre.

  The default decision threshold in the scikit-learn code library that we will use is 0.5. But, this library also enables us to "tune" the LR model based on our problem-dependency / context, as well as take the best/top probability score to predict the news genre of the input headline.
'''
import pandas as pd
import re
import numpy as np
import logging

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns
import mplcyberpunk

# ML Modeling
from sklearn.metrics import precision_recall_fscore_support,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import RocCurveDisplay,roc_curve
from sklearn.preprocessing import normalize
# Saving and importing trained models
import pickle

'''
  # MODEL TRAINING FUNCTIONS
  The functions below help us create a systematic and reproducable workflow to train the data.
  Be sure to check out my videos that walk through an overview of what they do.
'''
def _reciprocal_rank(true_genre_labels: list, machine_predicted_genre_labels: list):
    '''
    ## Purpose
    Compute the reciprocal rank (RR) at cutoff k.

    ## Parameters
        - `true_genre_labels` (List): List of actual news genre labels
        - `machine_predicted_genre_labels` (List): List of news genre labels predicted by the LR algorithm
    
    ## Return Values
        - `recip_rank` (Float): Reciprocal rank
    '''
    
    # add index to list only if machine predicted label exists in true labels
    tp_pos_list = [(idx + 1) for idx, r in enumerate(machine_predicted_genre_labels) if r in true_genre_labels]

    recip_rank = 0
    if len(tp_pos_list) > 0:
        # for reciprocal rank we must find the position of the first **correctly labeled** item
        first_pos_list = tp_pos_list[0]
        
        # recip_rank = 1/rank
        recip_rank = 1 / float(first_pos_list)

    return recip_rank

def compute_mrr_at_k(eval_news_category_items:list):
    '''
    # compute_mrr_at_k()

    Computes the MRR (average RR) at cutoff k.

    MRR evaluates any process that produces a list of possible responses to a sample of queries, ordered by probability of correctness, e.g., 1 for first place, 1⁄2 for second place, 1⁄3 for third place and so on. Review this ["Mean reciprocal rank" wikipedia article](https://en.wikipedia.org/wiki/Mean_reciprocal_rank) for a simple explainer.

    ## Parameters
    - `eval_news_category_items` (List): List that contains 2 values
        1. String - Actual news genre category
        2. List of strings - Predicted news genre category in order by estimated probability to be returned by the model.
            - The example below shows how 
                - `'HEALTHY LIVING'` was the actual label, but it was third in 'reciprocal rank' with a value of 1/3
                - `'WORLDPOST'` was the actual label, and it was first in 'reciprocal rank' with a value of 1
                
                [
                    [
                        ['HEALTHY LIVING'], ['POLITICS', 'ENTERTAINMENT', 'HEALTHY LIVING']
                    ], 
                    [
                        ['WORLDPOST'], ['WORLDPOST', 'MEDIA', 'POLITICS']
                    ], 
                    ...
                ]

    ## Return Values
        - `mean_reciprocal_rank_score` (Float): Mean average reciprocal rank score among the predicted news category in the model
    '''
    rr_total = 0
    
    for item in eval_news_category_items:
        actual_label = item[0]
        pred_label_list = item[1]

        # Find the reciprocal rank (RR) for this row
        rr_at_k = _reciprocal_rank(actual_label, pred_label_list)

        # Add the row's RR to the accruing scores for the entire corpus
        rr_total = rr_total + rr_at_k

        # Update the Mean Reciprocal Rank (MRR) score with new row value
        mean_reciprocal_rank_score = rr_total / 1/float(len(eval_news_category_items))

    return mean_reciprocal_rank_score

def collect_preds(Y_test, Y_preds):
    '''
    ## Purpose
    Collect all predictions (predicted news genre labels) and ground truth (i.e., actual news genre label)
    '''
    pred_gold_list = [ [ [Y_test[index]], pred ] for index, pred in enumerate(Y_preds) ]
    return pred_gold_list
             
def compute_accuracy(eval_news_category_items:list):
    '''
    ## Purpose
    `compute_accuracy()`: Compute the overall accuracy score of the model across the training corpus

    ## Parameters
        - `eval_news_category_items` (List): List that contains 2 values
            1. String - Actual news genre category
            2. List of strings - Predicted news genre category

            Example: [
                [
                    ['HEALTHY LIVING'], ['POLITICS', 'ENTERTAINMENT', 'HEALTHY LIVING']
                ], 
                [
                    ['WORLDPOST'], ['WORLDPOST', 'MEDIA', 'POLITICS']
                ], 
                ...
            ]
    ## Return Values
        - `news_cat_prediction_accuracy` (Float): Percentage of accurately predicted news category in the model
    '''
    correct_news_cat = 0
    
    for news_genre_cat in eval_news_category_items:
        true_pred = news_genre_cat[0]
        machine_pred = set(news_genre_cat[1])
        
        for news_cat in true_pred:
            if news_cat in machine_pred:
                correct_news_cat += 1
                break
    
    news_cat_prediction_accuracy = correct_news_cat / float(len(eval_news_category_items))
    return news_cat_prediction_accuracy


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def extract_features(df, field, training_data, testing_data, type='binary'):
    '''
    # extract_features()
    
    **Purpose**: Extract features using different method types: binary, counts, or TF-IDF

    ### If BINARY Features
    Creates a new `CountVectorizer()` method object, which converts a collection of text documents to a matrix of binary token counts per document. In other words, 
    - `1` == the feature is represented in the document
    - `0` == the feature is not represented in the doc
    
    Logistic regression involves vectorizing weighted averages of these tokens.

    ### If COUNT Features
    Creates a new `CountVectorizer()` method object, which converts a collection of text documents to a matrix of `n` token counts per document.  In other words, 
    - `5` == the feature is represented 5 times in the document
    - `25` == the feature is represented 25 times in the document
    - `0` == the feature is not represented in the doc
    
    Logistic regression involves vectorizing weighted averages of these tokens.

    ### If TF-IDF Features
    Creates a new `CountVectorizer()` method object, which converts a collection of text documents to a matrix of `n` token counts per document.  In other words, 
    - `5` == the feature is represented 5 times in the document
    - `25` == the feature is represented 25 times in the document
    - `0` == the feature is not represented in the doc
    
    Logistic regression involves vectorizing weighted averages of these tokens.
    '''
    
    logging.info("Extracting features and creating vocabulary...")

    '''
        BINARY and COUNTS PROCESSES WILL DO THE FOLLOWING:

        sklearn's CountVectorizer() will convert text to numerical data.
    '''
    
    if 'binary' in type:
        
        # BINARY FEATURE REPRESENTATION
        # Creates a new CountVectorizer() method object, which can help us use built-in functions that convert a collection of text documents to a matrix of token counts. **REMEMBER** that logistic regression involves vectorizing weighted averages of these tokens.
        # NOTE: `max_df` == "Maximum Document Frequency. It enables us to programmatically ignore frequently occuring words, e.g., articles like 'a' or 'the'. `max_df` reviews how many documents contain the word, and if it exceeds the max_df threshold then it is eliminated from the sparse matrix. Below we set the threshold to 95%.
        cv = CountVectorizer(binary=True, max_df=0.95)
        # CountVectorizer()'s fit_transform() uses the training_data to learn the vocabulary dictionary and return document-term matrix.
        cv.fit_transform(training_data[field].values)
        # CountVectorizer()'s transform() 
        train_feature_set = cv.transform(training_data[field].values)
        test_feature_set = cv.transform(testing_data[field].values)
        
        return train_feature_set,test_feature_set,cv
  
    elif 'counts' in type:
        
        # COUNT BASED FEATURE REPRESENTATION
        cv = CountVectorizer(binary=False, max_df=0.95)
        cv.fit_transform(training_data[field].values)
        
        train_feature_set = cv.transform(training_data[field].values)
        test_feature_set = cv.transform(testing_data[field].values)
        
        return train_feature_set,test_feature_set,cv
    
    elif 'tfidf':    
        
        # TF-IDF BASED FEATURE REPRESENTATION
        tfidf_vectorizer=TfidfVectorizer(use_idf=True, max_df=0.95)
        tfidf_vectorizer.fit_transform(training_data[field].values)
        
        train_feature_set=tfidf_vectorizer.transform(training_data[field].values)
        test_feature_set=tfidf_vectorizer.transform(testing_data[field].values)
        
        return train_feature_set,test_feature_set,tfidf_vectorizer

def get_top_k_predictions(model, X_test, k, threshold=False):
    '''
    ## Purpose
    `get_top_k_predictions()`: Uses the input trained LogisticRegression model to return the news genre class/category with the top estimated probability score.
    ## Parameters
    - `model` (LogisticRegression()): Trained model scikit-learn object
    - `X_test` (pandas DataFrame): Sampled test data set returned by `training_test_split()` in the `training_model()` function
    - `k` (Integer): Number of top categories (news genres) to return based on the estimated probability to predict the news genre
    ## Return Value(s)
    - `preds` (List of list): A list within a list of the top k retruned news categories. For example:
        - `preds` is `[['SCIENCE', 'HEALTHY LIVING', 'GREEN']]` for an article with the headline of `"Exercise in space keeps astronauts from fainting when they return to Earth, study says"` and `k=3`
    '''
    if threshold == False:
        # get probabilities instead of predicted labels, since we want to collect top 3
        probs = model.predict_proba(X_test)

        # GET TOP K PREDICTIONS BY PROB - note these are just index
        best_n = np.argsort(probs, axis=1)[:,-k:]
        
        # GET CATEGORY OF PREDICTIONS
        preds = [[model.classes_[predicted_cat] for predicted_cat in prediction] for prediction in best_n]
        
        preds = [ item[::-1] for item in preds]
    
        return preds
    else:
        # get probabilities instead of predicted labels, since we want to collect top 3
        probs = (model.predict_proba(X_test)[:,1] >= thresh_val)

        # GET TOP K PREDICTIONS BY PROB - note these are just index
        best_n = np.argsort(probs, axis=1)[:,-k:]
        
        # GET CATEGORY OF PREDICTIONS
        preds = [[model.classes_[predicted_cat] for predicted_cat in prediction] for prediction in best_n]
        
        preds = [ item[::-1] for item in preds]
    
        return preds
   
def train_model(df, field="text_desc", feature_rep="binary", top_k=3):
    '''
    ## Purpose
    train_model() is the main controller function that conducts the following modeling procedure: 
        
    1. Create X data (List) by splitting the data to create two sampled sets: 1) for training, and 2) for testing.
    2. Create Y data (List) by assigning the actual (ground truth) labels
    3. Extract the features for the model to use, based on the chosen feature representation: binary vs. TF-IDF
    4. Fit, i.e., train, the logistic regression classifier model with scikit-learn's `LogisticRegression()` object
    5. Retrieve the evaluation items, e.g., the actual labels (ground truths) and predicted labels (list of top `k` number of estimated probable predicted categories)
    6. Use the evaluation iitems to compute the overall accuracy score and mean reciprocal rank score of the model

    ## Parameters
    - `df` (pandas DataFrame): the complete data set / corpus
    - `field` (String): the column name of the feature used to train the model
    - `feature_rep` (String): Type of LR analysis set as either "binary" or "count" or "tfidf"
    '''
    
    logging.info("Starting model training...")
    
    # 1. GET A TRAIN TEST SPLIT (set seed for consistent results)
    # train_test_split() from sklearn "splits arrays or matrices into random train and test subsets."
    # returns 2 new dataframes: one for training, another for testing the trained model
    y = df['category']
    x_training_data,x_testing_data = train_test_split(
        df,
        random_state=2000 #Controls the shuffling applied to the data before applying the split
    )

    # 2. GET LABELS FROM SPLIT DATA
    # Get the category values from each split data returned by #1
    Y_train = x_training_data['category'].values
    Y_test = x_testing_data['category'].values
     
    # 3. GET FEATURES
    X_train,X_test,feature_transformer = extract_features(
        df,
        field,
        x_training_data,
        x_testing_data,
        type=feature_rep
    )

    # INITIALIZE THE LOGISTIC REGRESSION CLASSIFIER OBJECT
    logging.info("Training a Logistic Regression Model. This may take a few minutes. ...")
    scikit_log_reg = LogisticRegression(
        verbose=0, #if you want the LR method to print out all the details, change this 0 to 1
        solver='liblinear',
        random_state=0,
        C=5,
        penalty='l2',
        max_iter=1000
    )
    # Create the model by providing the LR object the 
    model = scikit_log_reg.fit(X_train, Y_train)

    # GET TOP K PREDICTIONS
    preds = get_top_k_predictions(model, X_test, top_k)
    
    # GET PREDICTED VALUES AND GROUND TRUTH INTO A LIST OF LISTS - for ease of evaluation
    eval_items = collect_preds(Y_test, preds)
    
    # GET EVALUATION NUMBERS ON TEST SET -- HOW DID WE DO?
    logging.info("Starting evaluation...")
    simple_mean_avg_correct_prediction_accuracy = compute_accuracy(eval_items)
    mean_recip_rank_at_k = compute_mrr_at_k(eval_items)
    
    logging.info("Done training and evaluation.")

    # Return the herein computed model and other values for potential use and exploration
    return model,feature_transformer,simple_mean_avg_correct_prediction_accuracy,mean_recip_rank_at_k,X_train,X_test,Y_test,Y_train,preds,eval_items


'''
  # MODEL EVALUATION FUNCTIONS
'''

def roc_curve_per_category(list_classes, label_binarizer, y_proba, y_one_vs_all):
  '''
  ## Purpose
  `roc_curve_per_category`: Combine all of the ROC () values for each category in one DataFrame

  ## Parameters
  - `list_classes` (List): A list of the classes in the trained model
  - `y_proba` (Numpy array sparse matrix): A list of lists where each sublist contains estimated probability scores of each X input across all possible Y classes/categories
  - `y_one_vs_all` (Numpy array sparse matrix): A list of lists, where each sublist contains binary 0 or 1 values representing the predicted class/category as 1 (Int) and all other classes/categories as 0 (Int)

  ## Return Value(s)
  - `df_fpr_tpr` (pandas DataFrame): Dataframe with the following values per row:
    - `'Class'`: Class from the model for instance
    - `'FPR'`: False Positive Rate for instance
    - `'TPR'`: True Positive Rate for instance
    - `'Threshold_Value'`: Threshold Value for instance
    - `'Threshold_Optimal'`: Optimal threshold value for the class as a whole
    - `'FPR_Optimal'`: Optimal FPR value for the class as a whole
    - `'TPR_Optimal'`: Optimal TPR value for the class as a whole
  '''
  list_dicts_classes_roc = []

  for class_cat in list_classes:
    # get class category (news genre)
    class_id = np.flatnonzero(label_binarizer.classes_ == class_cat)[0]
    y_proba[:, class_id]

    fpr, tpr, thresholds = roc_curve(y_one_vs_all[:, class_id], y_proba[:, class_id])
    
    # Calculate the Geometric-Mean
    geometric_mean = np.sqrt(tpr * (1 - fpr))
    
    # Find the optimal threshold
    index = np.argmax(geometric_mean)
    threshold_optimal = round(thresholds[index], ndigits=4)
    fpr_optimal = round(fpr[index], ndigits=4)
    tpr_optimal = round(tpr[index], ndigits=4)

    for i in range(0, len(fpr)):
      list_dicts_classes_roc.append({
        'Class': class_cat,
        'FPR': fpr[i],
        'TPR': tpr[i],
        'FPR_Optimal': fpr_optimal,
        'TPR_Optimal': tpr_optimal,
      })

  df_fpr_tpr = pd.DataFrame(list_dicts_classes_roc)

  return df_fpr_tpr

def plot_class_roc_curve(class_of_interest, label_binarizer, Y_one_vs_all, Y_prob_a, df_class_row):
    '''
    ## Purpose
    `plot_class_roc_curve`: Automate the visualizing of the FPR vs TPR of a particular class/category using the `RocCurveDisplay` function

    ## Parameters
    - `class_of_interest` (String): Category/class name to isolate
    - `Y_one_vs_all` (numpy array): List (array) of binarized values produced by `LabelBinarizer`'s `.transform()` on the Y_test data set.
        - NOTE: Function assumes that `label_binarizer` as an object with that name has been initialized and assigned as such.
    - `Y_prob_a` (numpy array): Array of probability estimates for each class/category.
    - `df_class_row` (row from pandas Dataframe): This dataframe row includes the optimal

    ## Return Values
    - None. Instead, it "shows" the matplotlib plot object.
    '''
    class_id = np.flatnonzero(label_binarizer.classes_ == class_of_interest)[0]

    RocCurveDisplay.from_predictions(
        Y_one_vs_all[:, class_id],
        Y_prob_a[:, class_id],
        name=f"{class_of_interest} vs the rest",
        color="white",
        plot_chance_level=True,
    )

    # Plot best threshold
    # x = FPR_Optimal, y = TPR_Optimal
    plt.plot(
        df_class_row.FPR_Optimal, 
        df_class_row.TPR_Optimal,
        marker="o",
        markerfacecolor='white',
        markeredgecolor='white',
        markersize=10,
    )
    opt_x = df_class_row.FPR_Optimal
    opt_y = df_class_row.TPR_Optimal
    plt.annotate(
        f"Optimal threshold for {class_of_interest} ({str(opt_y.values.tolist()[0])})",
        (opt_x, opt_y), #x,y point to label
        xytext=(opt_x+0.03, opt_y-0.05)
    )

    plt.style.use('cyberpunk')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"One-vs-Rest ROC curves:\n{class_of_interest} vs Rest", color='white')
    plt.tight_layout()
    mplcyberpunk.add_glow_effects()
    plt.show()
