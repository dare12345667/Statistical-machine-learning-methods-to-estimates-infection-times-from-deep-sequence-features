#!/usr/bin/env python
# coding: utf-8

# In[57]:


#!pip install --upgrade numpy===1.22.4


# In[135]:


# ==============================================================================
# Packages 
# ==============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, cross_val_predict, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
import xgboost
from xgboost import XGBRegressor
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri

pandas2ri.activate()


# In[137]:



# =============================================================================
# Adding Mapped that corresponds to pt_letter for better data handling
# =============================================================================

df2_PHYLO1 = df2_PHYLO.drop(["tree.id"], axis = 1)
unique_values = df2_PHYLO1['pt_letter'].unique()
mapping = {value: index + 1 for index, value in enumerate(unique_values)}
df2_PHYLO1['Mapped'] = df2_PHYLO1['pt_letter'].map(mapping)
vector = list(df2_PHYLO1['Mapped'])
vector2 =np.array([ int(a) for a in vector])
df2_PHYLO1['Mapped'] = vector2

# =============================================================================
# Treating missing values 
# =============================================================================

df2_PHYLO1 = df2_PHYLO1.drop("pt_letter", axis = 1)
imputer = KNNImputer(n_neighbors=3)
imputed_data = imputer.fit_transform(df2_PHYLO1)
df2_PHYLO1= pd.DataFrame(imputed_data, columns=df2_PHYLO1.columns)


# =============================================================================
# Segmenting into different regions based on reference genomic cordinates 
# =============================================================================

Region_Gag_aggregate_Gag = df2_PHYLO1[(df2_PHYLO1["xcoord"] >= 800 ) & (df2_PHYLO1["xcoord"] <= 2000)]
Region_Gag_aggregate_Gag = Region_Gag_aggregate_Gag[["TSI_days", "tips","normalised.largest.rtt", "solo.dual.count", "Mapped" ]]
Region_Pol_aggregate_Pol = df2_PHYLO1[(df2_PHYLO1["xcoord"] >= 2001) & (df2_PHYLO1["xcoord"] <= 5402)]
Region_Pol_aggregate_Pol = Region_Pol_aggregate_Pol[["TSI_days", "tips","normalised.largest.rtt", "solo.dual.count", "Mapped" ]]
Region_Gp120_aggregate_Gp120 = df2_PHYLO1[(df2_PHYLO1["xcoord"] >= 6300) & (df2_PHYLO1["xcoord"] <= 8723)]
Region_Gp120_aggregate_Gp120 = Region_Gp120_aggregate_Gp120[["TSI_days", "tips","normalised.largest.rtt", "solo.dual.count", "Mapped" ]]
Region_Gp41_aggregate_Gp41 = df2_PHYLO1[(df2_PHYLO1["xcoord"] >= 8724) & (df2_PHYLO1["xcoord"] <= 9207)]
Region_Gp41_aggregate_Gp41 = Region_Gp41_aggregate_Gp41[["TSI_days", "tips","normalised.largest.rtt", "solo.dual.count", "Mapped" ]]
Genomic_regions = df2_PHYLO1[["TSI_days", "tips","normalised.largest.rtt", "solo.dual.count", "Mapped" ]]
Region_Gag_aggregate_Gag['Region'] = 'gag'
Region_Pol_aggregate_Pol['Region'] = 'pol'
Region_Gp120_aggregate_Gp120['Region'] = 'gp120'
Region_Gp41_aggregate_Gp41['Region'] = 'gp41'
Genomic_regions['Region'] ='genome'
merged_df = pd.concat([Region_Gag_aggregate_Gag, Region_Pol_aggregate_Pol, 
                       Region_Gp120_aggregate_Gp120, Region_Gp41_aggregate_Gp41, Genomic_regions],
                      ignore_index=True)



# ============================================================================
# The following code is adapted from Tanya's repository on GitHub. The original 
# script is available at https://github.com/BDI-pathogens/HIV-phyloTSI/blob/main/HIVPhyloTSI.py. 
# Specifically, I've utilized the section that extracts the Minor Allele Frequency (MAF) 
# for the first and second codon across the region.
# ============================================================================

def load_patstats(fpath):
    ''' Load phyloscanner output - PatStats.csv '''
    patstats = pd.read_csv(fpath)
    Xlrtt = patstats.groupby(['host.id', 'xcoord'])['normalised.largest.rtt'].mean().unstack()
    Xtips = patstats.groupby(['host.id', 'xcoord'])['tips'].mean().unstack()
    Xdual = patstats.groupby(['host.id', 'xcoord'])['solo.dual.count'].mean().unstack()
    try:
        assert (Xlrtt.index == Xtips.index).all()
        assert (Xlrtt.index == Xdual.index).all()
    except AssertionError:
        logerr('Index mismatch between phyloscanner outputs: {}, {}, {}').format(Xlrtt.shape, Xtips.shape, Xdual.shape)
    loginfo('Loaded phyloscanner data, shape={}'.format(Xlrtt.shape))
    return Xlrtt, Xtips, Xdual




# In[ ]:


# ======================================================================================
# To compare the algorithms based on training and testing MSE, 
# and to repeat the process 20 times to determine their average cross-validation R^2.
# ======================================================================================
# When window-level covariates aggregated by genes were considered, the metrics were
# derived using a threshold of 9 months. However, in the code, the threshold is set as sqrt(9), which equals 3.
# ======================================================================================

ols = LinearRegression()
knn = KNeighborsRegressor()
xgb_reg = xgb.XGBRegressor()
rf = RandomForestRegressor(n_estimators=1000, max_depth=7)
models = [('OLS', ols), ('KNN', knn), ('XGBoost', xgb_reg), ('Random Forest', rf)]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
for name, model in models:
    print(f"Model: {name}")
    for combination_name, features in feature_combinations.items():
        X_train_selected = X_train[features]
        X_test_selected = X_test[features]
        cv_scores = cross_val_score(model, X_train_selected, y_train, cv=10, scoring='r2')
        avg_cv_r2 = np.mean(cv_scores)
        model.fit(X_train_selected, y_train)
        y_train_pred = model.predict(X_train_selected)
        train_mse = mean_squared_error(y_train, y_train_pred)
        y_pred = model.predict(X_test_selected)
        test_r2 = r2_score(y_test, y_pred)
        test_mse = mean_squared_error(y_test, y_pred)
        print(f"Feature Combination: {combination_name}")
        print(f"Average CV R^2: {avg_cv_r2:.4f}")
        print(f"Training MSE: {train_mse:.4f}")
        print(f"Test MSE: {test_mse:.4f}")
        print(f"Test R^2: {test_r2:.4f}")
        print("---------------------------------------")
num_repeats = 20
kf = KFold(n_splits=10, shuffle=True, random_state=42)
for repeat in range(num_repeats):
    print(f"Repeat: {repeat + 1}")
    for name, model in models:
        print(f"Model: {name}")
        for combination_name, features in feature_combinations.items():
            cv_r2_scores = []
            for train_index, val_index in kf.split(X_train):
                X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
                y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
                model.fit(X_train_fold[features], y_train_fold)
                val_r2 = model.score(X_val_fold[features], y_val_fold)
                cv_r2_scores.append(val_r2)
            avg_cv_r2 = np.mean(cv_r2_scores)
            model.fit(X_train[features], y_train)
            test_r2 = model.score(X_test[features], y_test)
            print(f"Feature Combination: {combination_name}")
            print(f"Average CV R^2: {avg_cv_r2:.4f}")
            print(f"Test R^2: {test_r2:.4f}")
            print("---------------------------------------")
rf = RandomForestRegressor(n_estimators=1000, max_depth=10)
num_cv_iterations = 20
best_combination = None
best_metrics = {'r2': -float('inf'), 'precision': float('inf'), 'accuracy': -float('inf'), 'false_recency': float('inf')}
all_combinations_metrics = {}
for combination_name, features in feature_combinations.items():
    print(f"Evaluating combination: {combination_name}")
    r2_sum, precision_sum, mae_sum, accuracy_sum, false_recency_sum = 0, 0, 0, 0, 0
    for iteration in range(1, num_cv_iterations + 1):
        print(f"  Iteration {iteration}/{num_cv_iterations}")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=None)
        X_train_selected = X_train[features]
        X_test_selected = X_test[features]
        y_cv_pred = cross_val_predict(rf, X_train_selected, y_train)
        rf.fit(X_train_selected, y_train)
        y_pred = rf.predict(X_test_selected)
        r2 = r2_score(y_test, y_pred)
        precision = np.std([tree.predict(X_test_selected) for tree in rf.estimators_])
        mae = mean_absolute_error(y_test, y_pred)
        accuracy = np.mean(y_pred < 3) if np.mean(y_test < 3) > 0 else 0
        false_recency = np.mean((y_pred < 3) & (y_test >= 3)) if np.mean(y_test >= 3) > 0 else 0
        r2_sum += r2
        precision_sum += precision
        mae_sum += mae
        accuracy_sum += accuracy
        false_recency_sum += false_recency
    avg_r2 = r2_sum / num_cv_iterations
    avg_precision = precision_sum / num_cv_iterations
    avg_mae = mae_sum / num_cv_iterations
    avg_accuracy = accuracy_sum / num_cv_iterations
    avg_false_recency = false_recency_sum / num_cv_iterations
    all_combinations_metrics[combination_name] = {'r2': avg_r2, 'precision': avg_precision, 'mae': avg_mae, 'accuracy': avg_accuracy, 'false_recency': avg_false_recency}
    print(f"Metrics for combination {combination_name}:")
    print("  R^2:", avg_r2)
    print("  Precision:", avg_precision)
    print("  MAE:", avg_mae)
    print("  Accuracy:", avg_accuracy)
    print("  False Recency Rate:", avg_false_recency)
    print("---------------------------------------")
    if (avg_r2 > best_metrics['r2'] or
        (avg_r2 == best_metrics['r2'] and avg_precision < best_metrics['precision']) or
        (avg_r2 == best_metrics['r2'] and avg_precision == best_metrics['precision'] and avg_false_recency < best_metrics['false_recency'])):
        best_combination = combination_name
        best_metrics = {'r2': avg_r2, 'precision': avg_precision, 'mae': avg_mae, 'accuracy': avg_accuracy, 'false_recency': avg_false_recency}
print("\nBest Combination:", best_combination)
print("Metrics:")
print("  R^2:", best_metrics['r2'])
print("  Precision:", best_metrics['precision'])
print("  MAE:", best_metrics['mae'])
print("  Accuracy:", best_metrics['accuracy'])
print("  False Recency Rate:", best_metrics['false_recency'])


# In[ ]:



# ===============================================================================================
# When it comes to high-resolution features, aim for the combination that attains the highest R^2 score, while also 
# considering secondary criteria like lower precision and false recency rates, if necessary.
# ===============================================================================================

num_repeats = 20
kf = KFold(n_splits=10, shuffle=True, random_state=42)
rf = RandomForestRegressor(n_estimators=1000, max_depth=7)
num_cv_iterations = 20
best_combination = None
best_metrics = {'r2': -float('inf'), 'precision': float('inf'), 'accuracy': -float('inf'), 'false_recency': float('inf'), 'mae': float('inf')}
all_combinations_metrics = {}
r2_scores = {}
accuracy_scores = {}
mae_scores = {}
precision_scores = {}
false_recency_scores = {}
for combination_name, combination_df in feature_combinations.items():
    print(f"Evaluating combination: {combination_name}")
    # Initialize metrics
    r2_sum = 0
    precision_sum = 0
    mae_sum = 0
    accuracy_sum = 0
    false_recency_sum = 0
    r2_scores[combination_name] = []
    accuracy_scores[combination_name] = []
    mae_scores[combination_name] = []
    precision_scores[combination_name] = []
    false_recency_scores[combination_name] = []
    for iteration in range(1, num_cv_iterations + 1):
        print(f"  Iteration {iteration}/{num_cv_iterations}")
        # Split data into train and test sets for this iteration
        X = combination_df.drop(columns=['TSI_days', 'Region', 'Variable_name', 'Mapped'])
        y = np.sqrt(combination_df['TSI_days'] / 30.24)
        y_cv_pred = cross_val_predict(rf, X, y, cv=10)
        rf.fit(X, y)
        y_pred = rf.predict(X)
        r2 = r2_score(y, y_pred)
        precision = np.std([tree.predict(X) for tree in rf.estimators_])
        mae = mean_absolute_error(y, y_pred)
        accuracy = np.mean(y_pred < 3) if np.mean(y < 3) > 0 else 0
        false_recency = np.mean((y_pred < 3) & (y >= 3)) if np.mean(y >= 3) > 0 else 0
        r2_sum += r2
        precision_sum += precision
        mae_sum += mae
        accuracy_sum += accuracy
        false_recency_sum += false_recency
        r2_scores[combination_name].append(r2)
        accuracy_scores[combination_name].append(accuracy)
        mae_scores[combination_name].append(mae)
        precision_scores[combination_name].append(precision)
        false_recency_scores[combination_name].append(false_recency)
    avg_r2 = r2_sum / num_cv_iterations
    avg_precision = precision_sum / num_cv_iterations
    avg_mae = mae_sum / num_cv_iterations
    avg_accuracy = accuracy_sum / num_cv_iterations
    avg_false_recency = false_recency_sum / num_cv_iterations
    all_combinations_metrics[combination_name] = {'r2': avg_r2, 'precision': avg_precision, 'mae': avg_mae, 'accuracy': avg_accuracy, 'false_recency': avg_false_recency}
    print(f"Metrics for combination {combination_name}:")
    print("  R^2:", avg_r2)
    print("  Precision:", avg_precision)
    print("  MAE:", avg_mae)
    print("  Accuracy:", avg_accuracy)
    print("  False Recency Rate:", avg_false_recency)
    print("---------------------------------------")
    if (avg_r2 > best_metrics['r2'] or
        (avg_r2 == best_metrics['r2'] and avg_precision < best_metrics['precision']) or
        (avg_r2 == best_metrics['r2'] and avg_precision == best_metrics['precision'] and avg_false_recency < best_metrics['false_recency'])):
        best_combination = combination_name
        best_metrics = {'r2': avg_r2, 'precision': avg_precision, 'mae': avg_mae, 'accuracy': avg_accuracy, 'false_recency': avg_false_recency}
print(f"Best combination is: {best_combination}")
print(f"With metrics: {best_metrics}")

