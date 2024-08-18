from scipy import stats

# Chi-Square Test and Cramer's v
def chi_sq_test(df, x,y):
    cross_tabs = pd.crosstab(df[x], df[y])
    chi2, p, dof, con_table = stats.chi2_contingency(cross_tabs)
    if p < 0.05:
        decision = 'Reject H0: there is significant association between ' + x + ' and ' + y
        # calculating cramer's v
        n = cross_tabs.sum().sum()
        minimum_dimension = min(cross_tabs.shape)-1
        v = np.sqrt(chi2/(n*dof))
        if v <= 0.2:
            strength = 'Weak Association between '  + x + ' and ' + y
        elif v > 0.2 and v <= 0.6:
            strength = 'Medium Association between '  + x + ' and ' + y
        else:
            strength = 'Strong Association between '  + x + ' and ' + y
    else: 
        decision = 'Do not reject H0: There is no relation between ' + x + ' and ' + y
        strength = 'No association between '  + x + ' and ' + y
        v = 0
    print(f'chi-squared = {chi2}\np value= {p}\ndegrees of freedom = {dof}')
    print(decision)
    print("Cramer's V: " + str(v))
    print(strength)
    return p


def cont_test(df, x, target):
    # Perform the two sample t-test with equal variances
    t = stats.ttest_ind(a=df[df[target]==1][x], b=df[df[target]==0][x], equal_var=True)
    print(df[df[target]==1][x].mean(), df[df[target]==0][x].mean())
    return t.pvalue

# cont Features selections
Row = 0
for i in num_cols:
    out = cont_test(train, i, 'Target')
    if out < 0.05:
        Row = Row + 1
        if Row == 1:
            significant = [i]
        else:
            significant = significant + [i]

# cat Features selections
Row = 0
for i in cat_cols:
    out = chi_sq_test(train, i, 'Target')
    if out < 0.05:
        Row = Row + 1
        if Row == 1:
            significant_cat = [i]
        else:
            significant_cat = significant_cat + [i]

s_features = significant_cat + significant


# Random Forest
model = RandomForestClassifier(bootstrap=False,
 max_depth= 7,
 max_features= 'sqrt',
 min_samples_leaf=55,
 min_samples_split= 10,
 n_estimators= 500, 
class_weight='balanced', 
random_state=42)

model.fit(X_train[s_features],y_train)


importance = model.feature_importances_
# summarize feature importance
feature_imp = pd.DataFrame({'Feature':s_features, 'Importance':importance}).sort_values('Importance', ascending=False)
# for i,v in enumerate(importance):
#  print(s_features[i],'Score: %.5f' % v)


import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import metrics


# Model Visualization script
class ModelViz:

    def __init__(self) -> None:
        pass

    @staticmethod
    def get_optimum_threshold(df, target='Target', score='Score'):
        '''
        Given probability scores and binary target, returns the optimum cut off point, where 
        `true positive rate` is high and `false positive rate` is low.

        Parameters
        ----------
        df: pandas.DataFrame, Dataframe that contains binary target values - 0 & 1 and prediction scores
        target: str, Name of the target column. Default = Target
        score: str, Name of the probability score column. Default = Score

        Returns
        -------
        float: returns ROC AUC
        pd.DataFrame: returns a new DataFrame that provides tpr, fpr and optimum threshold
        matplotlib.pyplot: returns a ROC curve with cut-off point
        matplotlib.pyplot: returns a Target Separability Plot with threshold

        '''
        fpr, tpr, thresholds = metrics.roc_curve(df[target], df['Score'])
        roc_auc = metrics.auc(fpr, tpr)

        ####################################
        # The optimal cut off would be where tpr is high and fpr is low
        # tpr - (1-fpr) is zero or near to zero is the optimal cut off point
        ####################################
        i = np.arange(len(tpr))  # index for df
        roc = pd.DataFrame({
            'fpr': pd.Series(fpr, index=i),
            'tpr': pd.Series(tpr, index=i),
            '1-fpr': pd.Series(1-fpr, index=i),
            'tf': pd.Series(tpr - (1-fpr), index=i),
            'thresholds': pd.Series(thresholds, index=i)
        })

        cutoff_df = roc.iloc[(roc.tf-0).abs().argsort()
                             [:1]].reset_index(drop=True)

        # Plot tpr vs 1-fpr
        fig, ax = plt.subplots()
        plt.plot(roc['tpr'], label='tpr')
        plt.plot(roc['1-fpr'], color='red', label='1-fpr')
        plt.xlabel('1-False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        ax.set_xticklabels([])
        plt.legend()

        # Plot tpr vs 1-fpr
        fig2, ax2 = plt.subplots()
        sns.kdeplot(x=df[df[target] == 0]['Score'], label='0')
        sns.kdeplot(x=df[df[target] == 1]['Score'], label='1')
        plt.axvline(x=cutoff_df['thresholds'].values[0],
                    label='thresh={:.2f}'.format(cutoff_df['thresholds'].values[0]), color='red', ls='--')
        plt.title('Target Separability')
        plt.legend()

        return roc_auc, cutoff_df, ax, ax2

    @staticmethod
    def get_decile_score(df, target='Target', score='Score', qcut_duplicates='drop', req_dig=True, category=None):
        '''
        Given probability scores and binary target, returns decile scores and cumulative gain plot

        Parameters
        ----------
        df: pandas.DataFrame, Dataframe that contains binary target values - 0 & 1 and prediction scores
        target: str, Name of the target column. Default = Target
        score: str, Name of the probability score column. Default = Score

        Returns
        -------
        pd.DataFrame: returns a new DataFrame that provides Decile Scores
        matplotlib.pyplot: returns a cumulative gain plot object

        '''

        df['inv_score'] = 1 - df[score]

        df.sort_values(by='inv_score', ascending=False,
                    #    ignore_index=True, 
                       inplace=True)
        if category is None:
            df['DecileRank'] = pd.qcut(
                df['inv_score'], q=10, labels=False, duplicates=qcut_duplicates) + 1
        else:
            df['DecileRank'] = df.groupby([category])['inv_score'].transform(
                     lambda x: pd.qcut(x, 10, labels=False, duplicates=qcut_duplicates)) + 1

        decile_performance = df.groupby('DecileRank')[target].agg(
            ['count', 'sum']).sort_values(by='DecileRank', ascending=True).reset_index()
        decile_performance['cumsum'] = np.cumsum(decile_performance['sum'])
        decile_performance['gain'] = decile_performance['cumsum'] / \
            decile_performance['sum'].sum() * 100

        if req_dig:
            ax = plt.figure(figsize=(12, 8))
            plt.title('Decile Score - Cumulative Gain Plot')
            sns.lineplot(
                x=decile_performance['DecileRank'], y=decile_performance['gain'], label='model')
            sns.lineplot(x=decile_performance['DecileRank'],
                         y=decile_performance['DecileRank']*10, label='avg')

            return decile_performance, ax
        else:
            return decile_performance

    @staticmethod
    def get_classification_report(clf, X, y, thres=0.5):
        '''
        Given model, X and y provides classification report for the model

        Parameters
        ----------
        clf: sklearn model, trained classification model that has predict_proba available
        X: pandas.DataFrame or numpy array, Dataframe/array that acts as independent variables for the model
        y: pandas.Series or numpy 1D-array, Series/1D-array that acts as the dependant/target variable for the model
        thres: float, optional. The probability threshold to determine 0 or 1. Default is 0.5

        Returns
        -------
        classification report: str, returns classification report

        '''

        x_train_proba = clf.predict_proba(X)[:, 1]
        x_train_pred = np.where(x_train_proba > thres, 1, 0)

        clf_report = metrics.classification_report(y, x_train_pred)

        return clf_report

    @staticmethod
    def get_d3_gain(y_actual, y_prob):
        '''
        Given ytrue and yproba returns the 3rd decile cumulative gain 

        Parameters
        ----------
        y_actual: pandas.Series or numpy 1D-array, Binary valued Series/1D-array that is actual values
        y_prob: pandas.Series or numpy 1D-array, Series/1D-array that is the probability score of y_actual being 1

        Returns
        -------
        classification report: str, returns classification report

        '''
        test_result = pd.DataFrame(y_actual.copy(), columns=['Target'])
        test_result['Score'] = y_prob

        ds = ModelViz.get_decile_score(test_result, req_dig=False)

        return ds[ds['DecileRank'] == 3]['gain'].values[0]


train_result = pd.DataFrame(y_train.copy()) 
y_pred_train =[x[1] for x in model.predict_proba(X_train[s_features])]
# y_pred_train =[x[1] for x in model.predict_proba(x_train[reduced_features])]
train_result['Score'] = y_pred_train
roc_auc, cutoff_df, ax, ax2 = ModelViz.get_optimum_threshold(train_result, target='Target')

print(roc_auc_score(y_train,y_pred_train))


# Function to impose training decile cut offs to test and validation
decile_prob = list(train_result.groupby('DecileRank')['Score'].min())

def decile_fun(score, prob_list):
  if score >= prob_list[0]:
    decile = 1
  elif score >= prob_list[1] < prob_list[0]:
    decile = 2
  elif score >= prob_list[2] < prob_list[1]:
    decile = 3
  elif score >= prob_list[3] < prob_list[2]:
    decile = 4
  elif score >= prob_list[4] < prob_list[3]:
    decile = 5
  elif score >= prob_list[5] < prob_list[4]:
    decile = 6
  elif score >= prob_list[6] < prob_list[5]:
    decile = 7
  elif score >= prob_list[7] < prob_list[6]:
    decile = 8
  elif score >= prob_list[8] < prob_list[7]:
    decile = 9
  else:
    decile = 10
  return decile

def decile_summary(prob, actual, prob_list):
  Decile = [decile_fun(col, prob_list) for col in prob]
  results = pd.DataFrame({'Sum': actual,'Count': actual,'Probability': prob}).reset_index(drop = True)
  results['Decile'] = Decile
  decile_sum = results.groupby('Decile')['Sum'].sum().reset_index()
  decile_cumsum = decile_sum['Sum'].cumsum().reset_index()
  decile_cumsum.columns = ['Decile', 'CumSum']
  decile_cumsum['Decile'] = decile_cumsum['Decile'] + 1
  decile_count = results.groupby('Decile')['Count'].count().reset_index()
  Decile_sum=decile_sum.join(decile_count.set_index('Decile'),on='Decile')
  Decile_sum=Decile_sum.join(decile_cumsum.set_index('Decile'),on='Decile')
  Decile_sum['gain'] = Decile_sum['CumSum']/decile_sum['Sum'].sum()
  Decile_sum['Event Rate']=Decile_sum['Sum']/Decile_sum['Count']
  return Decile, Decile_sum


training_decile, tr_decile_summary = decile_summary(prob = train_result['Score'], actual = train_result['TARGET'], prob_list= decile_prob)

oot_result['Decile'] = 10 - pd.qcut(oot_result['Score'], 10, labels=False)
