# Car_Failure_Predict
Email: ydxiao9807@gmail.com

The input is a dataset from a friend which is consist of real data and synthetic instances and features. The goal is to predict whether a car will fail or not.

As this data is not a public data, please don't make it public if you're given the permission to access to this repository.

The data is under the data/ directory. It's a database file.

The eda.ipynb is the Exploratory Data Analysis (EDA) part in Jupyter Notebook format. 

The src/ directory contains the End-to-end Machine Learning Pipeline. 

### Section 1: Summary of The End-to-end Pipeline:

Four models and three hyperparameters tuning methods are tested.

The four models are: random forest classifier, LightGBM classifier, SVC classifier and KNN classifier.

The three hyperparameters tuning methods are: GridSeachCV, RandomizedSearchCV and BayesSearchCV.

Besides not failed, there are 5 failure types: Failure A, Failure B, Failure C, Failure D and Failure E. So we have the choice to do a binary classification (failure or not) or 6 classes classification (No Failure and Failure A to E).

For binary classification case, among all the 4 models, LightGBM classifier gives the best roc_auc (0.8) and recall (0.66). However, random forest classifier gives better precision (0.89 vs. 0.84) with reasonably good roc_auc (0.78) and recall (0.58).

For 6 classes classification case, SVC classifier gives the best macro average roc_auc (0.75) and the best marco average recall of failure (0.303). The random forest classifier gives the same marco average recall of failure (0.303), but the macro avergae roc_auc is 0.67.

For multiclass classification, the macro average recall output from *classification_report* module is the macro average including the non failure (negative) class which is the dominated class. It's biased. We didn't use it. The recall_score after excluding the non failure class is more meaningful which is used in the pipeline.

### Section 2: The workflow of the pipeline

   For detailed workflow, please refer to the flow chart in later part. A few points to make here:

   - Binary classification or 6 classes classification

     There are 5 failure types: Failure A, Failure B, Failure C, Failure D and Failure E. So we have the choice to do a binary classification (failure or not) or 6 classes classification (No
 Failure and Failure A to E).

     Ideally, the user should be prompted to give the input. We have the code in the pipeline (line 621). But it was commented out to avoid interrupt. The number of classes variable ***n_class*** is hard coded as 2 (n_class = 2 at line 619) to do a binary classification. If you want to do a 6 classes classification, just change the n_class to 6. It has been tested that the whole pipeline works fine for 6 classes classification.

     Naturally, 6 classes classification is much slower than binary classification. Its performance is not as good as binary classification either.

   - Evaluation metrics and refit strategy

     As explained above, we decided to use both roc_auc and recall to form a multi-scores evaluation metrics. The roc_auc and recall are both designed for binary classification. For multiclass classification, the binarizing strategy and average method should be defined. They are defined in the ***num_class*** function along with label generation based on n_class variable.

     A refit strategy is needed when multiple metric evaluation is used in GridSearchCV and RandomizedSearchCV. It's defined as the average of roc_auc and recall, i.e. ("roc_auc"+"recall")/2, in ***refit_strategy*** function. By doing so, we can put more weight on recall to predict as many failures as possible.

     The evaluation metrics are data dependent and purpose dependent. For this case, we probably want to predict the failure, so we can avoid the failure. For multiclass classification, a more complicated evaluation metrics may be needed sometimes (of course, it's doable as well). For example, if certain failure (say failure B) is more dangerous, then we should put more weight on its recall when evaluating the model performance.

     Bayes search doesn't allow multi-scores evaluation metrics yet. Only roc_auc is used for Bayes search.

   - Hyperparameters tuning

     The hyperparameters tuning is a must to achieve the best performance. However, it's computational expensive and time consuming. I understand we should treat this task as a production work. But in reality, it's not really a production work. To avoid very long run time, this pipeline avoids big hyperparameter searching space. As mentioned above, two-step searching might be a good strategy. We did a rough search first to narrow down the possible searching space and then do a thorough searching. Unfortunately, this two-step searching is expensive too. So it's only done on random forest classifier only. One pass of searching is done on the rest of 3 models.

   Here is the flow chart of the workflow.

   ```mermaid
   flowchart TD;
       a("data_reading function\nread in the data")-->b("DropBadData transformer (customized transformer)\ndrop duplicated data and bad data, such as artificial factories");
       subgraph "Preprocessing"
           b-->c("FeatureSplitCombine transformer (customized transformer)\nSplit the Model into M_class and M_year and create the Failure attribute");
           c-->d("TemperatureUnify transformer (customized transformer)\nUnify the unit of the Temperature attribute to °C and correct unrealistic high temperature");
       end
       d-->e("num_class function\nDefine the label and decide to do a two-class classification or multiclass classification.");
       e-->f("train test split");
       f--"Run fit and transform of the later transformers on the training data first,\nand then run transform only on the test data\n to avoid leakage"--->g("MissingMemberInterp transformer (customized transformer)\ninterpolate the missing membership");
       subgraph "Imputation and feature engineering"
           g-->h("NegativeRPMProcess transformer (customized transformer)\nCorrect the negative RPM by interpolation");
           h-->i("StandardScaler transformer\nstandardize numerical features");
           h-->j("OrdinalEncoder transformer\nencode categorical features");
           i-->k("ColumnTransformer\nmerge numerical and categorical features");
           j-->k;
       end
           k-->l("Random Forest Classifier\nwith two-step hyperparameter tuning:\nrandomized search first\nand then grid search");
           k-->m("LightGBM classifer\n with grid search\nhyperparameters tuning");
           k-->n("SVC Classifier\nwith Bayes search\nhyperparameters tuning");
           k-->o("KNN classifier\nwith grid search on\nboth imbalanced classes\nhandling methods and hyperparameters");
   ```
