#!/usr/bin/env python
# coding: utf-8

import sqlite3
import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, recall_score, make_scorer

from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTENC
from imblearn.over_sampling import ADASYN
from imblearn.pipeline import make_pipeline as imb_make_pipeline


class DropBadData(BaseEstimator, TransformerMixin):
    """ Data cleaning by dropping bad features and duplicated data.

    Data from 3 factories are deleted as they produced very few cars and
    they all failed ones which is very suspicious. Those entries only make
    a small part of the data. It's safe to delete them.

    81 duplicated entries are deleted as well.

    These simple processes are written as a transform. So it can work 
    seamlessly with sklearn functionalities, especially pipeline. 
    
    Parameters
    ----------
    drop_duplicate : bool, default = True
        If "True", then drop duplicated entries, otherwise, keep them.
    drop_missing_membership : bool, default = False
        If "True", then drop data with missing membership. Otherwise, they
        will be kept and will be interpolated later.
        The data with missing membership are all failed ones. There are 327
        entries with missing membership which is about 30% of the total
        failured cars. It's highly recommended not to drop them. 
    drop_artificial_factory : bool, default = True
        If "True", then drop data from artificial_factory, 
        otherwise, keep them.
    artificial_factory_list : list, default = ['Seng Kang, China',
                                         'Newton, China', 'Bedok, Germany']
        A list of factories that should be deleted.

    Returns
    -------
    The DataFrame after cleaning. 
    Since sklearn v1.2, the output from transform can be DataFrame.
    """

    def __init__(self, drop_duplicate = True, drop_missing_membership = False,
                 drop_artificial_factory = True,
                 artificial_factory_list = ['Seng Kang, China', 
                                         'Newton, China', 'Bedok, Germany']
                 ):
        self.drop_duplicate = drop_duplicate
        self.drop_missing_membership = drop_missing_membership
        self.drop_artificial_factory = drop_artificial_factory
        self.artificial_factory_list = artificial_factory_list

    def fit(self, X):
        """ The fit method. Does nothing but return self.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Notes
        -----
        This is a preprocessing process. The whole data are processed as X.
        """
        return self

    def transform(self, X):
        """ Drop bad features and duplicated data. 

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Notes
        -----
        This is a preprocessing process. The whole data are processed as X.
        """
        if self.drop_duplicate:
            X = X.drop_duplicates()
        if self.drop_artificial_factory:
            X = X.drop(X[X.Factory.isin(self.artificial_factory_list)].index)
        if self.drop_missing_membership:
            X = X[X.Membership.notna()]
        return X


class FeatureSplitCombine(BaseEstimator, TransformerMixin):
    """ Split the "Model" into "M_class" and "M_year" and create the "Failure"
    attribute.

    The "Model" attribute in the data comprises two parts: model class and 
    year, such as "Model 3, 2019". It should be split based on the assumption
    that the difference between different years of the same model class should
    be quite small, i.e. brand new "Model 3, 2019" and "Model 3, 2020" should
    be similar. And the year in the "Model" should be highly related to the 
    manufacture year which indicates the age of the cars. It's very important
    information, but it's not available in the data. So the "M_year" split
    from the "Model" can be a good indicator of the age of the cars.

    There are 5 failure types in the data in a form of 5 attributes. It's good
    to combine them into one attribute to form a label. 

    These simple processes are written as a transform. So it can work
    seamlessly with sklearn functionalities, especially pipeline.

    Parameters
    ----------
    split_model_year : bool, default = True
        If "True", then split the "Model" into two attributes: model class 
        "M_class" and model year "M_year". 

    combine_failure : bool, default = True
        If "True", then combine all the failure information into one attribute.

    failure_type_list : list, default = ["Failure A", "Failure B", "Failure C",
                                        "Failure D", "Failure E"]
        A list of type of failure. There are 5 types based on the data.

    Returns
    -------
    The DataFrame after cleaning.
    Since sklearn v1.2, the output from transform can be DataFrame.
    
    Attributes added to the output data:
    
    "M_class": str,
    possible values: "Model 3", "Model 5", "Model 7"

    "M_year": int,
    possible values: 2009 - 2022

    "Failure": int,
    possible values: 0 - 5.
    0: no failure, 1: Failure A, 2: Failure B, 3: Failure C, 4: Failure D, 
    5: Failure E
    """

    def __init__(self, split_model_year = True, combine_failure = True, 
                failure_type_list = ["Failure A", "Failure B", "Failure C", 
                                     "Failure D", "Failure E"]):
        self.split_model_year = split_model_year
        self.combine_failure = combine_failure 
        self.failure_type_list = failure_type_list

    def fit(self, X):
        """ The fit method. Does nothing but return self.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Notes
        -----
        This is a preprocessing process. The whole data are processed as X.
        """
        return self

    def transform(self, X):
        """ Split the "Model" and combine the failure types. 

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Notes
        -----
        This is a preprocessing process. The whole data are processed as X.
        """
        if self.split_model_year:
            X["M_class"] = (X.Model.str.split(", ", expand=True)[0]).\
                                          astype('category').cat.as_ordered()
            X["M_year"] = X.Model.str.split(", ", expand=True)[1].astype("int")
        if self.combine_failure:
            X["Failure"] = pd.concat([1 - X[self.failure_type_list].
                         sum(axis=1).astype("int"), X[self.failure_type_list]], 
                         axis=1).values.nonzero()[1]
        return X


class TemperatureUnify(BaseEstimator, TransformerMixin):
    """ Unify the unit of the Temperature attribute to °C and correct 
    unrealistic high temperature. It's likely that the unit should be °F, but
    was recorded as °C accidentally. 
    
    The unit must be unified. So no need to provide a parameter to ask whether
    the unit should be unified or not.

    Parameters
    ----------
    extreme_high_tempertature : float, default = 220
        The temperature is in the range of 110°C to 138.7°C after unifying 
        the unit to °C except one entry which is 230.7°C. It's likely the 
        unit is actually °F for this one. If it's the case, the temperature 
        is 110.4°C, which is within the data range. To highlight and fix this
        unit error, the extreme_high_tempertature is set to 220 by default
        which is about 104.4°C. It means when the temperature is higher than
        the extreme_high_tempertature after the unit is unified to °C, then
        it's treated that the unit is wrong (i.e it's written as °C, but it's
        °F actually), so the unit conversion will be done for those entries.
        If it's decided not to do this correction, just set a really high
        extreme_high_tempertature.
    drop_ex_high_temp : bool, default = False
        If "True", then drop the data with temperature higher than 
        extreme_high_tempertature, no correction will be done.
        If "False", then assume the unit is mistaken as °C, but it's actually
        °F, then convert it to °C.

    Returns
    -------
    The DataFrame after the missing membership is interpolated.
    Since sklearn v1.2, the output from transform can be DataFrame.
    """

    def __init__(self, extreme_high_tempertature = 220, 
                 drop_ex_high_temp = False):
        self.extreme_high_tempertature = extreme_high_tempertature
        self.drop_ex_high_temp = drop_ex_high_temp

    def fit(self, X):
        """ The fit method. Does nothing but return self.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Notes
        -----
        This is a preprocessing process. The whole data are processed as X.
        """
        return self

    def transform(self, X):
        """ Unify the unit of the Temperature attribute to °C and correct 
        unrealistic high temperature. 

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Notes
        -----
        This is a preprocessing process. The whole data are processed as X.
        """
        X["Temperature"] = X["Temperature"].str.split().map(lambda x:
                    x[0] if (x[1] == "°C") else round((float(x[0])-32)/1.8,
                    1)).astype("float")
        if self.drop_ex_high_temp:
            X = X.loc[X["Temperature"] < self.extreme_high_tempertature]
        else:
            X.loc[X["Temperature"] > self.extreme_high_tempertature,
                "Temperature"] = round(
                (X.loc[X["Temperature"] > self.extreme_high_tempertature,
                    "Temperature"]-32)/1.8, 1)
        return X


class MissingMemberInterp(BaseEstimator, TransformerMixin):
    """ Interpolate the missing membership.
    
    To prevent leakage, the interpolation should be done after train test split.

    In the data discription document, it's clearly stated that car owners are 
    automatically subscribed with "Normal" membership for the first five years
    after car purchase. As all the cars with missing membership are within 5 
    years (2018 onwards), their membership should be either "Normal" or 
    "Premium". So the missing membership is randomly assigned to "Normal" or 
    "Premium" based on the ratio of "Normal" and "Premium" among the failed cars
    within 5 years as all the cars with missing membership are failed ones. 

    The ratio is computed during fitting stage according to the training data 
    and will be used for missing membership interpolation for test data to 
    avoid leakage.

    As whether dropping the missing membership or interpolating them is decided
    in class DropBadData, there is no input paramter for this class. If missing
    membership has been dropped in class DropBadData, this class will actually
    do nothing.

    Returns
    -------
    The DataFrame after the missing membership is interpolated.
    Since sklearn v1.2, the output from transform can be DataFrame.
    """

    def fit(self, X):
        """ The fit method. Compute the ratio of the "Normal" membership among
        failed cars that are within first 5 years. The ratio of "Premium" ones
        is 1 - the ratio of "Normal" ones.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector including the label "Failure", where `n_samples`
            is the number of samples and `n_features` is the number of features.

        Attributes
        ----------
        num_normal_2018onwards_ : float
            Number of cars with "Normal" membership among failed cars that are
            within first 5 years.
        num_premium_2018onwards_ : float
            Number of cars with "premium" membership among failed cars that are
            within first 5 years.
        ratio_normal_2018onwards_ : float
            The ratio of  cars with "Normal" membership among failed cars that
            are within first 5 years.

        Notes
        -----
        As the lable attribute "Failure" is still useful, it's not deleted from
        X. The interpoltion process doesn't mess up the data order,  y is not
        needed in this case. 
        """
        self.num_normal_2018onwards_ = X[(X.M_year>2017) & (X.Failure > 0) & 
                             (X.Membership == "Normal")]["Membership"].count()
        self.num_premium_2018onwards_ = X[(X.M_year>2017) & (X.Failure > 0) & 
                            (X.Membership == "Premium")]["Membership"].count()
        self.ratio_normal_2018onwards_ = self.num_normal_2018onwards_ / \
                (self.num_premium_2018onwards_ + self.num_normal_2018onwards_)
        return self

    def transform(self, X):
        """ The transform method. Interpolate the missing membership. 

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector including the label "Failure", where `n_samples`
            is the number of samples and `n_features` is the number of features.
        Notes
        -----
        As the lable attribute "Failure" is still useful, it's not deleted from
        X. The interpoltion process doesn't mess up the data order,  y is not
        needed in this case. 
        """
        _ = np.random.default_rng(seed=11)
        X.loc[X["Membership"].isnull(), "Membership"] = _.choice(
                  ["Normal", "Premium"], size=len(X[X["Membership"].isnull()]), 
                  p=[self.ratio_normal_2018onwards_, 
                  1 - self.ratio_normal_2018onwards_])
        return X


class NegativeRPMProcess(BaseEstimator, TransformerMixin):
    """ Correct the negative RPM by interpolation.

    During the EDA, it can be seen that there is certain level of correlation 
    between RPM and Failure (0-5) and M_class. The negative RPM can be replaced
    by the median or mean RPM of the same Failure and M_class group. It sounds
    there is risk of leakage. However, this is just to maintain the correlation
    between RPM and the label failure. On the other hand, the failure 
    distribution among the negative RPM is consistent with the whole data. 
    So it shouldn't matter too much if dropping the negative RPM.

    The median or mean RPM is computed during fitting stage according to the
    training data and will be used for missing membership interpolation for 
    test data to avoid leakage.

    Parameters
    ----------
    negative_rpm_process : str, default = "median"
        The possible values are "drop", "median" and "mean". 
        If "drop", drop the data with negative RPM.
        If "median", then replace the negative RPM with the group median RPM.
        If "mean", then replace the negative RPM with the group mean RPM.

    Returns
    -------
    The DataFrame after the missing membership is interpolated.
    Since sklearn v1.2, the output from transform can be DataFrame.
    """

    def __init__(self, negative_rpm_process = "median"):
        if negative_rpm_process in ["drop", "median", "mean"]:
            self.negative_rpm_process = negative_rpm_process
        else:
            raise ValueError(f'"negative_rpm_process" is the method fixing the' 
                             f'negative RPM. It can only be "drop", "median"'
                             f'or "mean".')

    def fit(self, X):
        """ The fit method. Generate a dict to record the median value of each
        (Failure, M_class) group.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector including the label "Failure", where `n_samples`
            is the number of samples and `n_features` is the number of features.

        Attributes
        ----------
        rpm_mean_dict_ : dict 
            A dict with (Failure, M_class) group pair as the keys and the
            median or mean RPM of the group as the values. It'll be an empty
            dictionary if the value of negative_rpm_process parameter is "drop".

        Notes
        -----
        As the lable attribute "Failure" is still useful, it's not deleted from
        X. The interpoltion process doesn't mess up the data order,  y is not
        needed in this case. 
        """
        if self.negative_rpm_process == "drop":
            self.rpm_interp_dict_ = {}
        if self.negative_rpm_process == "median":
            self.rpm_interp_dict_ = X.loc[X["RPM"] > 0, ["Failure", "M_class", 
                            "RPM"]].groupby(["Failure", "M_class"]).median().\
                            round().to_dict("index")
        if self.negative_rpm_process == "mean":
            self.rpm_interp_dict_ = X.loc[X["RPM"] > 0, ["Failure", "M_class", 
                            "RPM"]].groupby(["Failure", "M_class"]).mean().\
                            round().to_dict("index")
        return self

    def transform(self, X):
        """ The transform method. Interpolate the negative RPM. 

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector including the label "Failure", where `n_samples`
            is the number of samples and `n_features` is the number of features.
        Notes
        -----
        As the lable attribute "Failure" is still useful, it's not deleted from
        X. The interpoltion process doesn't mess up the data order,  y is not
        needed in this case. 
        """
        X.loc[X["RPM"] < 0, "RPM"] = np.nan
        if self.negative_rpm_process != "drop":
            rpm_corrected = X[["Failure", "M_class", "RPM"]].groupby(
                    ["Failure", "M_class"], group_keys=True).apply(lambda x: 
                    x.fillna(value=self.rpm_interp_dict_[x.name])).drop(
                    ["Failure", "M_class"], axis=1).reset_index().set_index(
                    "level_2").rename_axis("")
            X = pd.concat([X.drop("RPM", axis=1), rpm_corrected["RPM"]], axis=1)
        else:
            X = X[X.RPM.notna()]
        return X


def num_class(label, *, n=2):
    """ Define the label and decide to do a two-class classfication or 
    multiclass classfication.
  
    Parameters
    ----------
    label : 1D array (n_samples)
        label attribute,  where `n_samples` is the number of samples.
    n : int : 2, 6, default = 2
        number of classes of the classfication.
        2: binnary classification, fail or not fail.
        6: multiclass classification: not fail, failure A, B, C, D, E.

    Return
    ------
    updated_label : 1D array (n_samples)
        if it's a two-class classfication, then the return comprises 0 
        and 1 only.
    searching_score: dict
        evaluation metrics with both roc_auc and rec. The roc_auc and rec are
        designed for binary classification. For multiclass classification, 
        binarizing strategy or average method need to be defined here.
    """

    if n == 2:
        label = label.map(lambda x: 1 if (x>0) else 0)
    recall_multiclass = make_scorer(recall_score, average='macro', 
                                                      labels=[1, 2, 3, 4, 5])
    roc_multiclass = make_scorer(roc_auc_score, needs_proba=True, 
                                          multi_class='ovr', average='macro')
    searching_score = {"roc": "roc_auc" if (n == 2) else roc_multiclass,
                       "rec": "recall" if (n == 2) else recall_multiclass} 
    
    return label, searching_score


def refit_strategy(cv_results):
    """ Define the refit evaluation metric for hyperparameter tuning.
    We need to define the metric as we plan to do multiple metric evaluation
    using both roc_auc and recall.
    The metric is the average of roc_auc and recall.

    Parameters
    ----------
    cv_results : dict of numpy (masked) ndarrays
        The cv_results_ attribute outputted by searching module. 

    Return
    ------
    best_index : int
        The index of the cv_results_ which gives the best  average of roc_auc 
        and recall.
    """

    cv_results_ = pd.DataFrame(cv_results)
    cv_results_["avg_roc_rec"] = (cv_results_["mean_test_roc"] + 
                                             cv_results_["mean_test_rec"])/2
    best_index = cv_results_["avg_roc_rec"].idxmax()
    print('Top 5 sets of hyperparameters in terms of score of'
          ' ("roc_auc"+"recall")/2:') 
    print(cv_results_.sort_values(by='avg_roc_rec', ascending=False)[:5].loc[:,
                        ["params", "mean_test_roc", "std_test_roc",
                         "rank_test_roc", "mean_test_rec", "std_test_rec",
                         "rank_test_rec", "avg_roc_rec"]].to_string())
    return best_index


def data_reading(dbfilename = "data/failure.db", dbtname = "failure"):
    """ Data reading function

    Parameters
    ----------
    dbfilename : str, default =  "data/failure.db"
        database file name. 
    dbtname : str, default = "failure"
        database table name.
    Return
    ------
    data : pandas DataFrame
        The data in pandas DataFrame format.
    """

    con = sqlite3.connect(dbfilename)
    data = pd.read_sql_query(f"SELECT * from {dbtname}", con)
    con.close()
    return data


def main():
    """ Main function. Call all the objects and function to run models."""

    # Pipelines:
    # preprocessing_pipeline : drop bad data, split "Model" into "M_class" 
    #                          and "M_year" and unifying "Temperature" unit.
    # feature_interp_pipeline : handle missing membership and negative RPM.
    # column_tran_pipeline :  standardize numeric features and encode 
    #                         categorical features as a one-hot numeric array.
    # process_pipeline : combine feature_interp_pipeline and 
    #                    column_tran_pipeline
    preprocessing_pipeline = make_pipeline(DropBadData(), 
                             FeatureSplitCombine(), TemperatureUnify())
    
    feature_interp_pipeline = make_pipeline(MissingMemberInterp(), 
                              NegativeRPMProcess())
    
    numeric_features = ["Temperature", "RPM", "Fuel consumption", "M_year"]
    categorical_features = ["Usage", "Membership", "M_class"]
    categories_order = [["Low", "Medium", "High"], 
                         ['None', 'Normal', 'Premium'],
                         ['Model 3', 'Model 5', 'Model 7']]
    column_tran_pipeline = ColumnTransformer([
            ("num", preprocessing.StandardScaler(), numeric_features),
            ("cat", OrdinalEncoder(categories=categories_order), 
                    categorical_features),
            ])
    
    process_pipeline = Pipeline([
                            ("feature_interp", feature_interp_pipeline),
                            ("column_transformer", column_tran_pipeline),
                                ])
    
    ############################################################################
    # Data preparation:
    # Read in the data and run preprocessing_pipeline and then train test split.
    # The feature_interp_pipeline should be done after train test split to void
    # leakage. So fit is done on train data and then run transform on test data
    # based on the fit on train data. 
    df_cf = data_reading()

    print("*"*80)
    print("Data loading is done. Now start data preparation ........")
    
    df_cf_clean = preprocessing_pipeline.fit_transform(df_cf)

    # Binary or 6 classes classification: 
    #     Binary classification: fail or not.
    #     6 classes classification: not fail, failure A, B, C, D, E.
    # Can leave it to users to input. Commented it out to avoid interrupt.
    n_class = 2
    print("Number of classes (either 2 or 6, can be changed):", n_class)
    #n_class = int(input("Please specify how many classes to be classified:"))

    label, searching_score = num_class(df_cf_clean.Failure, n = n_class)
    
    X_train, X_test, y_train, y_test = train_test_split(df_cf_clean, label, 
                              stratify=label, test_size=0.30, random_state=123)
    
    X_train_processed = process_pipeline.fit_transform(X_train)
    
    X_test_processed = process_pipeline.transform(X_test)

    ############################################################################
    ############################   Modeling part   #############################
    # Three types of hyper parameters test are done:
    # 1. Randomized search for random forest classifer
    # 2. Bayes search for SVC
    # 3. Grid search for KNN, along with test on imbalanced classes handling.
    #    Grid search has been applied on random forest classifer and LightGBM
    #    classifiers as well.
    ############################################################################

    print("*"*80)
    print("Data preparation is done. Now start modeling ........")
    print("Four models are tested:\n"
          "1. Random Forest Classifier\n"
          "2. LightGBM Classifier\n"
          "3. C-Support Vector Classification (SVC) Classifier\n"
          "4. KK-Nearest Neighbors (KNN) Classifier")
    print("*"*80)
    print("This is a classfication problem. The random forest classifer is " 
          "probably one of\nthe best models.\n"
          "Other tree type ensemble methods, such as Gradient Boosting, "
          "XGBoost,\nLightGBM, might be good candidates as well.")
    print("*"*80)
    print("Random Forest Classifier...")
    print("-"*70)
    print("Hyperparameter Tuning:")
    print("-"*50)
    print("Run a whole complete hyperparameter tuning process:\n"
          "Firstly, run a randomized searching on a coarse parameter space to "
          "find\na set of reasonably good parameters as the starting point of "
          "the finer grid\nsearching. Then run grid searching with finer grid."
          " This is a computational\nintensive process. To save run time, for "
          "later models, only one round of\nparameters searching will be done.")
    print("-"*30)
    print("Run randomized searching with a coarse grid to find a set of "
          "reasonably good\nparameters as the starting point of the finer "
          "grid searching......\n" 
          "Both roc_auc and recall are used for evaluation.")

    ############################################################################
    ###### Random forest classifer hyper parameters test and model running #####
    # Some classifiers can handle imbalanced data with parameter such as 
    # class_weight or scale_pos_weight. No need to do the imbalance class 
    # processing before modeling. Some can't. The imbalance class processing 
    # should be done before modeling. 
    # "class_weight" parameter in Random forest classifer can handle the 
    # imbalance classes, so no need to run imbalanced classes handling.

    # Randomized search on best hyper parameters of random forest classifer
    # !!! Note: !!! to save runtime, the parameters search space is minimized.
    rfc = RandomForestClassifier(class_weight="balanced")

    rfc_coarse_grid = [
                  {"n_estimators": range(1, 401, 40), 
                   "max_depth": range(1, 11, 2),
                   "max_features": range(1, 11, 2), 
                   "min_samples_split": range(2, 10, 2)},
                 ]
    
    rfc_random_search = RandomizedSearchCV(rfc, rfc_coarse_grid, n_jobs = -1,
                 n_iter=20, cv=5, scoring=searching_score, error_score='raise',
                 refit=refit_strategy, random_state=111)
    rfc_random_search.fit(X_train_processed, y_train)
    cv_results = pd.DataFrame(rfc_random_search.cv_results_)
    cv_results["avg_roc_rec"] = (cv_results["mean_test_roc"] + 
                                 cv_results["mean_test_rec"])/2
    best_avg_roc_rec = cv_results["avg_roc_rec"][rfc_random_search.best_index_]

    print(f"The best parameter by randomized searching within coarse parameter"
          f" search space\nfor random forest classifier is:\n"
          f"{rfc_random_search.best_params_}\n"
          f'The corresponding best score ("roc_auc"+"recall")/2 among all '
          f"CVs is:\n{best_avg_roc_rec}")
    print("-"*30)
    print("Run grid searching with finer grid around the best parameters "
          "derived by\nrandomized searching.....\n"
          "Both roc_auc and recall are used for evaluation.")
    
    ##############
    ## Grid search around the best parameters derived from randomized searching.
    rfc_dense_grid = [
                  {"n_estimators": range(181, 221, 10), 
                   "max_depth": range(6, 8, 1),
                   "max_features": range(4, 6, 1), 
                   "min_samples_split": range(2, 6, 2)},
                 ]
    rfc_grid_search = GridSearchCV(rfc, rfc_dense_grid, n_jobs = -1,
                            cv=5, scoring=searching_score, error_score="raise",
                            refit=refit_strategy)
    rfc_grid_search.fit(X_train_processed, y_train)
    cv_results = pd.DataFrame(rfc_grid_search.cv_results_)
    cv_results["avg_roc_rec"] = (cv_results["mean_test_roc"] + 
                                 cv_results["mean_test_rec"])/2
    best_avg_roc_rec = cv_results["avg_roc_rec"][rfc_random_search.best_index_]

    print(f"The best parameter by grid searching within dense parameter "
          f"search space\nfor random forest classifier is:\n"
          f"forest\nclassifier by grid searching is:\n"
          f"{rfc_grid_search.best_params_}\n"
          f'The corresponding best score ("roc_auc"+"recall")/2 among all '
          f"CVs is:\n{best_avg_roc_rec}")

    ####################################
    # Best random forest model:
    rfc_best = RandomForestClassifier(n_estimators=221, max_features=9, 
                                   max_depth=6, class_weight="balanced")

    rfc_best.fit(X_train_processed, y_train)
    y_pred = rfc_best.predict(X_test_processed)
    y_proba = rfc_best.predict_proba(X_test_processed)
    rocauc_score = roc_auc_score(y_test, y_proba[:, 1]) if (n_class == 2) \
                        else \
            roc_auc_score(y_test, y_proba, multi_class='ovr', average="macro")
    rec_score = recall_score(y_test, y_pred) if (n_class == 2) else \
                recall_score(y_test, y_pred, average='macro',
                                     labels=[1, 2, 3, 4, 5])
    test_avg_roc_rec = (rocauc_score + rec_score) / 2
    
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    
    print("-"*70)
    print("Random Forest classifer modeling with best hyperparameters:\n"
          "Please Note: this best hyperparameter is not necessarily the one "
          "tested above.\nAs mentioned, the hyperparameter space tested above "
          "is limited to save runtime,\nso the assessor can have a quick "
          "evalutation.")
    print("-"*50)
    print("Confusion Matrix of prediction on test data:\n", cm)
    print("Classfication Report of prediction on test data:\n", cr)
    print("ROC AUC Score of prediction on test data (macro average, in the "
          "case of\n6 classes classification):", rocauc_score)
    print("Recall of the failure of prediction on test data (macro average, "
          "in the\ncase of 6 classes classification):", rec_score)
    print('The average of roc_auc and recall score ("roc_auc"+"recall")/2 of '
          'prediction\non test data:', test_avg_roc_rec)
    print("*"*80)

    print("Random forest classfier is done. Now start LightGBM modeling...")
    print("-"*70)
    print("Hyperparameter Tuning:")
    print("-"*50)
    print("To save run time, only one round of grid searching is done on a "
          "limited\nparameter space.......\n" 
          "Both roc_auc and recall are used for evaluation.")

    ############################################################################
    ###### LightGBM classifer hyper parameters test and model running #####
    # "class_weight" parameter in LightGBM classifer can handle the imbalance 
    # classes, so no need to run imbalanced classes handling.

    # Grid search on best hyper parameters of LightGBM classifer
    # !!! Note: !!! to save runtime, the parameters search space is minimized.
    lgbmc = LGBMClassifier(class_weight="balanced")

    lgbmc_param_grid = [
              {"n_estimators": range(100, 501, 100),
               "learning_rate": np.arange(0.04, 0.1, 0.04),
               "num_leaves": [11, 15, 21, 31, 41]},
              {"n_estimators": range(100, 501, 100),
               "learning_rate": np.arange(0.005, 0.021, 0.005),
               "num_leaves": [11, 15, 21, 31, 41]},
             ]
    
    lgbm_grid_search = GridSearchCV(lgbmc, lgbmc_param_grid, n_jobs = -1,
                            cv=5, scoring=searching_score, error_score="raise",
                            refit=refit_strategy)
    lgbm_grid_search.fit(X_train_processed, y_train)
    cv_results = pd.DataFrame(lgbm_grid_search.cv_results_)
    cv_results["avg_roc_rec"] = (cv_results["mean_test_roc"] + 
                                 cv_results["mean_test_rec"])/2
    best_avg_roc_rec = cv_results["avg_roc_rec"][lgbm_grid_search.best_index_]

    print(f"The best parameter within the parameter search space for LightGBM"
          f"\nclassifier is:\n"
          f"{lgbm_grid_search.best_params_}\n"
          f'The corresponding best score ("roc_auc"+"recall")/2 among all '
          f"CVs is:\n{best_avg_roc_rec}")

    ####################################
    # Best Lightgbm classifier model:
    lgbmc_best = LGBMClassifier(n_estimators=400, learning_rate=0.01, 
                                num_leaves=11, class_weight = "balanced")

    lgbmc_best.fit(X_train_processed, y_train)
    y_pred = lgbmc_best.predict(X_test_processed)
    y_proba = lgbmc_best.predict_proba(X_test_processed)
    y_pred4roc = y_proba[:, 1] if (n_class == 2) else y_proba
    rocauc_score = roc_auc_score(y_test, y_proba[:, 1]) if (n_class == 2) \
                        else \
            roc_auc_score(y_test, y_proba, multi_class='ovr', average="macro")
    rec_score = recall_score(y_test, y_pred) if (n_class == 2) else \
                        recall_score(y_test, y_pred, average='macro',
                                     labels=[1, 2, 3, 4, 5])
    test_avg_roc_rec = (rocauc_score + rec_score) / 2
    
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    
    print("-"*70)
    print("LightGBM classifer modeling with best hyperparameters:\n"
          "Please Note: this best hyperparameter is not necessarily the one "
          "tested above.\nAs mentioned, the hyperparameter space tested above "
          "is limited to save runtime,\nso the assessor can have a quick "
          "evalutation.")
    print("-"*50)
    print("Confusion Matrix of prediction on test data:\n", cm)
    print("Classfication Report of prediction on test data:\n", cr)
    print("ROC AUC Score of prediction on test data (macro average, in the "
          "case of\n6 classes classification):", rocauc_score)
    print("Recall of the failure of prediction on test data (macro average, "
          "in the\ncase of 6 classes classification):", rec_score)
    print('The average of roc_auc and recall score ("roc_auc"+"recall")/2 of '
          'prediction\non test data:', test_avg_roc_rec)
    print("*"*80)

    print("LightGBM classfier is done. Now start SVC modeling...\n"
          "SVC is a bit slow. Hang on, please!")
    print("-"*70)
    print("Hyperparameter Tuning:")
    print("-"*50)
    print("To save run time, only one round of Bayes searching is done on a "
          "limited\nparameter space........\n"
          "Only roc_auc is used for evaluation as multiple scores are NOT "
          "allowed by Bayes search yet.")

    ############################################################################
    ###### SVC classifer hyper parameters test and model running #####
    # "class_weight" parameter in SVC classifer can handle the imbalance
    # classes, so no need to run imbalanced classes handling outside SVC.

    # Bayes search on best hyper parameters of SVC classifer
    # !!! Note: !!! to save runtime, the parameters search space is minimized.
    svc_model = SVC(class_weight = "balanced", probability=True)

    svc_param_grid = {
        'C': Real(1e-1, 2e+2, prior='log-uniform'),
        'gamma': Real(1e-2, 1e+2, prior='log-uniform'),
        'kernel': Categorical(['linear', 'rbf']),
    }
    
    # !!! Note: !!! again, to save runtime, n_iter is probably not big enough.
    svc_bayes_search = BayesSearchCV(svc_model, search_spaces=svc_param_grid,
                         n_iter=10, cv=5, n_jobs=8, error_score='raise',
                         scoring=searching_score["roc"])
    svc_bayes_search.fit(X_train_processed, y_train)
    print(f"The best parameter within the parameter search space for SVC "
          f"classifier is:\n"
          f"{svc_bayes_search.best_params_}\n"
          f"The corresponding roc_auc score is:\n"
          f"{svc_bayes_search.best_score_}")

    ####################################
    # Best SVC model:
    svc_best = SVC(C=200, class_weight='balanced', gamma=0.012365889071507768, 
                   kernel='rbf', probability=True)

    svc_best.fit(X_train_processed, y_train)
    y_pred = svc_best.predict(X_test_processed)
    y_proba = svc_best.predict_proba(X_test_processed)
    rocauc_score = roc_auc_score(y_test, y_proba[:, 1]) if (n_class == 2) \
                        else \
            roc_auc_score(y_test, y_proba, multi_class='ovr', average="macro")
    rec_score = recall_score(y_test, y_pred) if (n_class == 2) else \
                        recall_score(y_test, y_pred, average='macro',
                                     labels=[1, 2, 3, 4, 5])
    test_avg_roc_rec = (rocauc_score + rec_score) / 2
    
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    
    print("-"*70)
    print("SVC classifer modeling with best hyperparameters:\n"
          "Please Note: this best hyperparameter is not necessarily the one "
          "tested above.\nAs mentioned, the hyperparameter space tested above "
          "is limited to save runtime,\nso the assessor can have a quick "
          "evalutation.")
    print("-"*50)
    print("Confusion Matrix of prediction on test data:\n", cm)
    print("Classfication Report of prediction on test data:\n", cr)
    print("ROC AUC Score of prediction on test data (macro average, in the "
          "case of\n6 classes classification):", rocauc_score)
    print("Recall of the failure of prediction on test data (macro average, "
          "in the\ncase of 6 classes classification):", rec_score)
    print('The average of roc_auc and recall score ("roc_auc"+"recall")/2 of '
          'prediction\non test data:', test_avg_roc_rec)
    print("*"*80)

    print("SVC classfier is done. Now start KNN classfier modeling...")
    print("-"*70)
    print("Hyperparameter Tuning:")
    print("-"*50)
    print("To save run time, only one round of grid searching is done on a "
          "limited\nparameter space.......\n"
          "Both roc_auc and recall are used for evaluation.")

    ############################################################################
    ###### KNN classifer hyper parameters test and model running #####
    # KNN may not be the best model for this case. But it's one of the those 
    # models that can't handle imbalance classes. So as an example running the
    # imbalance classes processing before modeling, KNN is chosen. 

    # Grid search on imbalance classes processing and hype parameter tunning
    # for KNN
    # !!! Note: !!! to save runtime, the parameters search space is minimized.
    imb_pipeline = imb_make_pipeline(RandomUnderSampler(), 
                                     KNeighborsClassifier())

    nimb_param_grid = [
            {
               "randomundersampler": [RandomUnderSampler(), 
                    SMOTENC(categorical_features=np.arange(4,6)), ADASYN()],
               "kneighborsclassifier__n_neighbors": [1,2,3,4,5,6,7,8], 
               "kneighborsclassifier__weights": ["uniform", "distance"], 
               "kneighborsclassifier__p": [1, 2, 3]}
    ]   

    nibm_grid_search = GridSearchCV(imb_pipeline, nimb_param_grid, cv=5, 
            scoring=searching_score, refit=refit_strategy)

    nibm_grid_search.fit(X_train_processed, y_train)
    cv_results = pd.DataFrame(nibm_grid_search.cv_results_)
    cv_results["avg_roc_rec"] = (cv_results["mean_test_roc"] + 
                                 cv_results["mean_test_rec"])/2
    best_avg_roc_rec = cv_results["avg_roc_rec"][nibm_grid_search.best_index_]

    print(f"The best parameter within the parameter search space for random "
          f"forest\nclassifier is:\n"
          f"{nibm_grid_search.best_params_}\n"
          f'The corresponding best score ("roc_auc"+"recall")/2 among all '
          f"CVs is:\n{best_avg_roc_rec}")

    ####################################
    # Best KNN model:
    X_train_processed_imb, y_train_imb = RandomUnderSampler().fit_resample(
                                         X_train_processed, y_train)

    knn_best = KNeighborsClassifier(n_neighbors=7, p=1)
    
    knn_best.fit(X_train_processed_imb, y_train_imb)
    y_pred = knn_best.predict(X_test_processed)
    y_proba = knn_best.predict_proba(X_test_processed)
    rocauc_score = roc_auc_score(y_test, y_proba[:, 1]) if (n_class == 2) \
                        else \
            roc_auc_score(y_test, y_proba, multi_class='ovr', average="macro")
    rec_score = recall_score(y_test, y_pred) if (n_class == 2) else \
                        recall_score(y_test, y_pred, average='macro',
                                     labels=[1, 2, 3, 4, 5])
    test_avg_roc_rec = (rocauc_score + rec_score) / 2
    
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    
    print("-"*70)
    print("KNN classifer modeling with best imbalance processing and best "
          "hyperparameters:")
    print("Please Note: this best hyperparameter is not necessarily the one "
          "tested above.\nAs mentioned, the hyperparameter space tested above "
          "is limited to save runtime,\nso the assessor can have a quick "
          "evalutation.")
    print("-"*50)
    print("Confusion Matrix of prediction on test data:\n", cm)
    print("Classfication Report of prediction on test data:\n", cr)
    print("ROC AUC Score of prediction on test data (macro average, in the "
          "case of\n6 classes classification):", rocauc_score)
    print("Recall of the failure of prediction on test data (macro average, "
          "in the\ncase of 6 classes classification):", rec_score)
    print('The average of roc_auc and recall score ("roc_auc"+"recall")/2 of '
          'prediction\non test data:', test_avg_roc_rec)
    print("*"*80)


if __name__ == "__main__":
    main()
