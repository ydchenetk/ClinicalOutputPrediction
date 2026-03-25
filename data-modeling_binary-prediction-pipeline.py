import pandas as pd 
import numpy as np 
import re
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFECV, SelectFromModel
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import accuracy_score, confusion_matrix, brier_score_loss, classification_report, roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import joblib


def preprocess(data, y, drop=["USUBJID"], method="skf", thresh=5):
    """
    Prepare the dataset to drop unnecessary columns (i.e. identification column), 
        scale the numeric columns, and create dummies for categorical variables.
    Split the dataset to train and validation sets (stratified or simple split) for further steps.

    Args:
        data (Dataframe): The input dataframe. 
        y (str): variable name for the target variable.
        drop (list/str, optional): a list of variable name to be dropped from analysis. Defaults to ["USUBJID"].
        method (str, optional): the split method ('skf' or 'simple'). Defaults to "skf".
        thresh (int, optional): split ratio, should be (0,1) if simple split and >1 for skf. Defaults to 5.

    Returns:
        Dataframes (Series), dict: The analytical datasets for train and validation; 
            and the data series for train and validation target variable.
            A dictionary to store the original/cleaned column names.
    """
    
    df = data.copy()
    
    ## drop unnecessary variables (i.e. identification columns) from analytical dataset
    drop = [drop] if isinstance(drop, str) else list(drop)
    if y not in drop:
        drop.append(y)
    
    y = df[y]
    X_raw = df.drop(drop, axis=1)
    
    ## XGBoost doesn't accept bool, object, or logical values, map to 0/1
    mapping = {'Y': 1, 'N': 0, 'True': 1, 'False': 0, True: 1, False: 0}
    for col in X_raw.columns:
        if X_raw[col].dtype in ['bool', 'object']:
            unique_vals = set(X_raw[col].dropna().unique())
            if unique_vals.issubset(set(mapping.keys())):
                X_raw[col] = X_raw[col].map(mapping)
    y = y.map(mapping)

    ## train-validation split (simple/skf)
    if method == "skf":
        splitter = StratifiedKFold(n_splits=thresh, shuffle=True, random_state=42)
        train_idx, test_idx = next(splitter.split(X_raw, y))
        X_train, X_test = X_raw.iloc[train_idx].copy(), X_raw.iloc[test_idx].copy()
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    else:
        X_train, X_test, y_train, y_test = train_test_split(X_raw, y, test_size=thresh, stratify=y, random_state=42)

    ## OneHotEncoder(): create dummies for categorical variables
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns
    if not cat_cols.empty:
        encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        
        encoded_train = encoder.fit_transform(X_train[cat_cols])
        encoded_cols = encoder.get_feature_names_out(cat_cols)
        
        X_train_enc = pd.DataFrame(encoded_train, columns=encoded_cols, index=X_train.index)
        X_test_enc = pd.DataFrame(encoder.transform(X_test[cat_cols]), columns=encoded_cols, index=X_test.index)
        
        X_train = pd.concat([X_train.drop(columns=cat_cols), X_train_enc], axis=1)
        X_test = pd.concat([X_test.drop(columns=cat_cols), X_test_enc], axis=1)

    ## StandardScaler(): scale the numeric variables
    num_cols = [
        c for c in X_train.columns 
        if (X_train[c].nunique() > 2) and (c not in ["Time_to_Response", "Time_to_Response_Day"])
    ]
    
    if num_cols:
        scaler = StandardScaler()
        X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
        X_test[num_cols] = scaler.transform(X_test[num_cols])

    ## final clean: fill NA
    X_train = X_train.astype(float).fillna(0)
    X_test = X_test.astype(float).fillna(0)
    
    # if "Time_to_Response" in df.columns:
    #     X_train = pd.concat([X_train, X_train["Time_to_Response", "Time_to_Response_Day"]], axis=1)
    #     X_test = pd.concat([X_test, X_test["Time_to_Response", "Time_to_Response_Day"]], axis=1)
    
    ## Remove zero-variance variables based on train dataset
    valid_cols = X_train.columns[X_train.std() > 0]
    X_train, X_test = X_train[valid_cols], X_test[valid_cols]

    ## XGBoost doesn't accept special marks as variable names, clean and store the variables for further analysis
    column_mapping = {}
    new_cols = []
    for col in X_train.columns:
        clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', str(col))
        clean_name = re.sub(r'_+', '_', clean_name).strip('_')
        
        ## if duplicated clean name happens
        base_name = clean_name
        counter = 1
        while clean_name in column_mapping:
            clean_name = f"{base_name}_{counter}"
            counter += 1
            
        column_mapping[clean_name] = col
        new_cols.append(clean_name)
    
    X_train.columns = new_cols
    X_test.columns = new_cols

    return X_train, X_test, y_train, y_test, column_mapping


## initial feature selection for avoiding collinear features
def initial_feature_selection(data, method="cm", thresh=0.95, protect=[]):
    """
    Select features to avoid multilinearity and simplify the model building process. 
    
    Args:
        data (Dataframe): The training dataset.
        method (str, optional): the feature selection method (cm/vif/pca+vif). Defaults to "cm".
        thresh (int, optional): threshold for dropping the features. Defaults to 0.95.
        protect (list/str, optional): a list or string of variable that cannot be dropped . Defaults to [].

    Returns:
        Dataframe, dict: A dataframe with feature selected. 
            A dictionary for information about feature dropped for further analysis.
    """
    
    df = data.copy()
    protect = [protect] if isinstance(protect, str) else list(protect)
    
    dropped_features = []
    pca_models = {}
    
    ## Method 1: Correlation Mattrix
    if method == "cm":
        corr_matrix = df.corr().abs()
        mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        upper = corr_matrix.where(mask)
        dropped_features = [
            col for col in upper.columns 
            if (upper[col] > thresh).any() and col not in protect
        ]
        df_final = df.drop(columns=dropped_features)
    
    elif method in ["vif", "pca+vif"]:
        pca_dropped = []
        pca_comp = []
        
        ## Method 2: PCA + VIF (not recommended)
        if method == "pca+vif":
            corr_matrix = df.corr().abs()
            grouped = set()
            
            for col in corr_matrix.columns:
                if col in grouped: continue
                group = corr_matrix.index[corr_matrix[col] >= 0.95].tolist()
                
                if len(group) > 1:
                    pca = PCA(n_components=1)
                    pca_var = pca.fit_transform(df[group])
                    pca_varname = f"PCA_{group[0]}_cluster"
                    pca_models[pca_varname] = {'model': pca, 'group': group}
                    
                    pca_comp.append(pd.Series(pca_var.flatten(),
                                            name=pca_varname, index=data.index))
                    grouped.update(group)
                    group_protect = [col for col in group if col not in protect]
                    pca_dropped.extend(group_protect)
            
            df = df.drop(columns=pca_dropped, errors="ignore")
            if pca_comp:
                df = pd.concat([df]+pca_comp, axis=1)
            
            df = df.copy()

        ## Method 3: VIF
        vif_dropped = []
        while True:
            cols = df.columns
            if len(cols) <=1: break
            
            vif_val = {}
            for col in cols:
                y = df[col]
                X  = df.drop(columns=[col])
                
                r_sq = LinearRegression().fit(X,y).score(X,y)
                vif = 1. / (1. - r_sq) if r_sq < 1.0 else float('inf')
                vif_val[col] = vif
                
            vif_series = pd.Series(vif_val)
            vif_series = vif_series[~vif_series.index.isin(protect)]
            if vif_series.empty: break
            max_vif = vif_series.max()
            
            if max_vif > thresh or np.isinf(max_vif):
                feature_to_drop = vif_series.idxmax()
                print(f"Dropping {feature_to_drop} (VIF: {max_vif:.2f})")
                
                vif_dropped.append(feature_to_drop)
                df = df.drop(columns=[feature_to_drop])
        
            else: break
    
        df_final = df
        dropped_features = list(set(pca_dropped+vif_dropped))
    
    state = {
        'method': method,
        'pca_models': pca_models,
        'dropped_features': dropped_features,
        'final_columns': df.columns.tolist()
    }
    
    return df_final, state


def test_transform(data, state):
    """
    Transform the validation dataset based the initial_feature_selection().
    
    Args:
        data (Dataframe): the validation dataset.
        state (dict): the dictionary for information about dropped features.

    Returns:
        Dataframe: The transformed validation dataset
    """
    data = data.copy()
    
    if not state["pca_models"]: 
        feature_to_drop = state["dropped_features"]
        data = data.drop(columns=feature_to_drop)
    
    else:
        for var_name, info in state['pca_models'].items():
            group_cols = info["group"]
            pca_model = info["model"]
            
            pca_trans = pca_model.transform(data[group_cols])
            data[var_name] = pca_trans.flatten()
        
        data = data.drop(columns=state["dropped_features"], errors="ignore")
        
    return data


def modeling(X_train, y_train):
    """
    Build modeling system: test four types of models (Logistic, SVM, Random Forest, XGBoost);
        Tune the models by variation of feature selection methods and hyperparameters using RandomizedSearchCV;
        Find the best model (evaluated by AUC) for each type, and use GridSearchCV to further fine-tune them;
        Find the final, optimal model (evaluated by AUC) and saved as a '.pkl' file.

    Args:
        X_train (Dataframe): The training dataframe.
        y_train (Data Series): The data series for target variable.

    Returns:
        dict, Dataframe: a dictionary contain model information;
            A dataframe contained 4 types of models in comparison.
    """
    
    ## for Random Forest Classifier: dynamic control the steps and numbers of features by data shape
    n_total_features = len(X_train.columns)
    dynamic_step = max(1, min(10, n_total_features // 10))
    dynamic_min = max(1, n_total_features // 20)
    if n_total_features > 50:
        dynamic_max_features = ['sqrt', 'log2', 0.2, 0.4]
    else:
        dynamic_max_features = ['sqrt', 0.5, 0.8, None]
    
    ## model configuration for fine-tuning
    model_configs = {
        'Logistic': {
            'model': LogisticRegression(max_iter=5000),
            'rs_params': [
                {
                    'selector': [
                        RFECV(LogisticRegression(solver='liblinear', max_iter=1000), step=1, cv=3),
                        SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000)),
                        None
                    ],
                    'clf__solver': ['saga', 'liblinear'],
                    'clf__penalty': ['l1', 'l2'],
                    'clf__C': np.logspace(-3,1,10),
                    'clf__class_weight': [None, 'balanced']
                },
                {
                    ## have a separate list for 'lbfgs' solver because it doesn't accept L1 penalty
                    ## mathematical compatibility: doesn't include the SelectFromModel() feature selection for 'lbfgs'
                    'selector': [
                        RFECV(LogisticRegression(solver='lbfgs', max_iter=1000), step=1, cv=3),
                        None
                    ],
                    'clf__solver': ['lbfgs'],
                    'clf__penalty': ['l2'],
                    'clf__C': np.logspace(-3, 1, 10),
                    'clf__class_weight': [None, 'balanced']
                }
            ]
        },
        'SVM': {
            'model': SVC(probability=True, random_state=42),
            'rs_params': {
                ## doesn't introduce the RFECV() selector for its non-linear complicity might create problem with SCM
                'selector': [
                    SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000)),
                    None
                ],
                'clf__C': np.logspace(-2, 1.3, 10),
                ## do not include 'poly' as part of the kernel selection because it might introduce complicated interaction terms
                ## and degree is not tuned since 'poly' is omitted from the kernel selection
                'clf__kernel': ['rbf', 'sigmoid'],
                'clf__gamma': ['scale', 'auto'],
                'clf__class_weight': [None, 'balanced']
            }
        },
        'RandomForest': {
            'model': RandomForestClassifier(random_state=42),
            'rs_params': {
                ## introduce both linear and non-linear feature selection method
                'selector': [
                    SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000)),
                    RFECV(RandomForestClassifier(n_estimators=50, max_depth=5), step=dynamic_step, cv=3, min_features_to_select=dynamic_min),
                    None
                ],
                'clf__n_estimators': [100, 250, 500, 1000],
                'clf__max_depth': [3,5,8,10],
                'clf__max_features': dynamic_max_features,
                'clf__min_samples_split': [2, 5, 10],
                'clf__min_samples_leaf': [1, 5, 10],
                'clf__class_weight': [None, 'balanced', 'balanced_subsample']
            }
        },
        'XGBoost': {
            'model': xgb.XGBClassifier(eval_metric='auc', random_state=42),
            'rs_params': {
                'selector': [
                    SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear')),
                    RFECV(xgb.XGBClassifier(n_estimators=50, max_depth=3, eval_metric='auc'), step=dynamic_step, cv=3, min_features_to_select=dynamic_min),
                    None
                ],
                'clf__n_estimators': [100, 250, 500, 1000],
                'clf__learning_rate': np.logspace(-2, -0.301, 6),
                'clf__max_depth': [3, 5, 8, 10],
                'clf__subsample': np.linspace(0.6, 1.0, 5),
                'clf__colsample_bytree': [0.5, 0.8, 1.0],
                'clf__gamma': [0, 0.1, 0.5, 1.0],
                'clf__min_child_weight': [1, 5, 10],
                'clf__scale_pos_weight': [1, 3, 5, 10]
            }
        }
    }
    
    final_results = []
    ## process bar visualized
    pbar = tqdm(model_configs.items(), desc="Tournament Status")
    
    for model, config in pbar:
        ## create the pipeline for the model process
        pipe = Pipeline([
            ('selector', None),
            ('clf', config['model'])
        ])
        
        print(f"\n======Starting Tournament: {model}======")
        rs = RandomizedSearchCV(pipe, config['rs_params'], n_iter=20, cv=5, scoring='roc_auc', 
                                n_jobs=-1, random_state=42, verbose=1)
        rs.fit(X_train, y_train)
        
        best_rs_params = rs.best_params_
        print(f"Refining {model} around best parameters: {best_rs_params}")
        
        ## prepare fine-tune parameters based on the RandomizedSearchCV() result for GridSearchCV()
        grid_params = {}
        for param, value in rs.best_params_.items():
            if any(x in param for x in ['selector', 'class_weight', 'solver', 'kernel']):
                grid_params[param] = [value] if value is not None else [None]
            elif any(x in param for x in ['max_depth', 'min_samples']):
                val = int(value)
                grid_params[param] = sorted(list(set([max(1, val-1), val, val+1])))
            elif 'estimators' in param:
                val = int(value)
                grid_params[param] = [val] if val > 500 else [max(10, val-50), val, val+50]
            elif isinstance(value, (float, np.float64)):
                low, high = value * 0.9, value * 1.1
                if 'scale_pos_weight' in param:
                    grid_params[param] = [value, high]
                else:
                    grid_params[param] = [max(1e-5, low), value, min(1.0, high)]

        gs = GridSearchCV(pipe, grid_params, cv=5, scoring='roc_auc', n_jobs=-1, error_score='raise', verbose=2)
        gs.fit(X_train, y_train)
        
        calibrated_model = CalibratedClassifierCV(gs.best_estimator_, cv='prefit', method='sigmoid')
        calibrated_model.fit(X_train, y_train)
        
        y_probs = calibrated_model.predict_proba(X_train)[:, 1]
        brier = brier_score_loss(y_train, y_probs)
        
        sel = gs.best_estimator_.named_steps['selector']
        mask = sel.get_support() if sel is not None else np.ones(X_train.shape[1], dtype=bool)
        feats = X_train.columns[mask].tolist()
        
        ## get the information about the best model within each model type
        final_results.append({
            'Model': model,
            'AUC': gs.best_score_,
            'Brier Score': brier,
            'Selector': "None" if sel is None else type(sel).__name__,
            'Feature Count': len(feats),
            'Features': feats,
            'Estimator': gs.best_estimator_
        })
    
    comparison_df = pd.DataFrame(final_results).sort_values(by='AUC', ascending=False)
    print("\n" + "="*40)
    print("FINAL MODEL SELECTED")
    print("="*40)
    print(comparison_df[['Model', 'AUC', 'Brier Score', 'Selector']])

    ## save the optimal model
    winner_row = comparison_df.iloc[0]
    joblib.dump(winner_row['Estimator'], f"{winner_row['Model']}.pkl")
    print(f"\nMODEL SAVED: {winner_row['Model']} (AUC: {winner_row['AUC']:.4f}) as {winner_row['Model']}'.pkl'")
    
    return comparison_df.iloc[0]['Estimator'], comparison_df


def feature_importance(model, column_mapping):
    """
    Get the feature importance from model output

    Args:
        model (dict): Information about the optimal model.
        column_mapping (): original column name.

    Returns:
        Dataframe, fig: a dataframe with variable and feature importance score (high to low);
            A fig with Top 20 features with importance scores visualized.
    """
    
    clf = model.named_steps['clf']
    sel = model.named_steps['selector']

    try:
        feature_names_before_selection = np.array(model.feature_names_in_)
    except AttributeError:
        feature_names_before_selection = np.array(list(column_mapping.keys()))

    importance = None
    if hasattr(clf, 'feature_importances_'):
        importance = clf.feature_importances_
    elif hasattr(clf, 'coef_'):
        importance = np.abs(clf.coef_[0])
    
    if importance is not None:
        if sel is not None and hasattr(sel, 'get_support'):
            mask = sel.get_support()
        else:
            mask = np.ones(len(feature_names_before_selection), dtype=bool)
        final_names = feature_names_before_selection[mask]

        importance_df = pd.DataFrame({
            'Feature Name': final_names,
            'Importance Score': importance
        })
        
        importance_df['Original Feature Name'] = importance_df['Feature Name'].map(column_mapping)
        importance_df = importance_df.sort_values(by='Importance Score', ascending=False).reset_index(drop=True)
        
        fig_imp = plt.figure(figsize=(20,20))
        sns.barplot(
            x='Importance Score', 
            y='Original Feature Name', 
            data=importance_df.head(20), 
            palette='magma'
        )
        plt.title(f"Top 20 Most Influential Features ({type(clf).__name__})")
        plt.xlabel("Importance Weight")
        plt.ylabel("Feature Name")
    
    else:
        print("Feature importance data is not accessible for this model architecture.")
        importance_df = pd.DataFrame()
        fig_imp = None
    
    return importance_df, fig_imp


def evaluation(model, X, y, thresholds={50: 0.50, 75: 0.75, 90: 0.90}):
    """
    Evaluation methods include calibration plot, brier score, confusion matrix, summary data, and classification matrix.

    Args:
        model (dict): A dictionary with model information.
        X (Dataframe): analytical dataframe
        y (Data Series): data series for target variable
        thresholds (dict, optional): The probability threshold want to be showcased. Defaults to {50: 0.50, 75: 0.75, 90: 0.90}.

    Returns:
        Dataframes, figs: output dataframes for evaluation methods; 
        figures for calibration and confusion matrix
    """
    
    probs = model.predict_proba(X)[:, 1]
    preds = model.predict(X)
    brier = brier_score_loss(y, probs)
    
    ## Summary Statistics
    metrics_dict = {
        "Accuracy": accuracy_score(y, preds),
        "ROC_AUC": roc_auc_score(y, probs),
        "Average Precision": average_precision_score(y, probs)
    }
    
    report_dict = classification_report(y, preds, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose().reset_index()
    report_df.columns = ['Metric_Class', 'precision', 'recall', 'f1-score', 'support']
    report_df = report_df[report_df['Metric_Class'] != 'accuracy']
    
    summary_rows = pd.DataFrame([
        {"Metric_Class": "GLOBAL_Accuracy", "precision": metrics_dict["Accuracy"]},
        {"Metric_Class": "GLOBAL_ROC_AUC", "precision": metrics_dict["ROC_AUC"]},
        {"Metric_Class": "GLOBAL_Avg_Precision", "precision": metrics_dict["Average Precision"]},
        {"Metric_Class": "GLOBAL_Brier_Score", "precision": brier}
    ])
    
    metrics_df = pd.concat([summary_rows, report_df], axis=0).reset_index(drop=True)

    ## Calibration Plot with Brier Score
    prob_true, prob_pred = calibration_curve(y, probs, n_bins=10)
    fig_cab = plt.figure(figsize=(10, 10))
    plt.plot(prob_pred, prob_true, marker='o', label=f'Model (Brier: {brier:.4f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Ideal Calibration')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Actual Fraction of Positives')
    plt.title('Calibration Plot (Reliability)')
    plt.legend()
    plt.close()

    ## Prediction and probability
    prob_df = pd.DataFrame({
        'Actual Class': y,
        'Predicted Class': preds,
        'Predict Probability (Class 1)': probs
    })
    
    for label, t in thresholds.items():
        prob_df[f"Threshold_{label}%"] = np.where(prob_df["Predict Probability (Class 1)"] >= t, "Y", None)

    # Confusion Matrix
    cm = confusion_matrix(y, preds)
    fig_cm, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
    ax.set_title('Confusion Matrix: Prediction Breakdown')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('Actual Label')
    plt.close()

    return metrics_df, prob_df, fig_cab, fig_cm


def prediction_process(data, configs):
    pre_cfg = configs.get('preprocess', {})
    ifs_cfg = configs.get('initial_feature_selection', {})
    eval_cfg = configs.get('evaluation', {})
    
    X_train, X_test, y_train, y_test, column_mapping = preprocess(data, **pre_cfg)
    X_train_analytical, states = initial_feature_selection(X_train, **ifs_cfg)
    X_test_analytical = test_transform(X_test, states)
    best_model, model_comparison = modeling(X_train_analytical, y_train)
    df_importance, fig_importance = feature_importance(best_model, column_mapping)
    
    metrics_train, prob_train, fig_cab_train, fig_cm_train = evaluation(best_model, X_train_analytical, y_train, **eval_cfg)
    metrics_test, prob_test, fig_cab_test, fig_cm_test = evaluation(best_model, X_test_analytical, y_test, **eval_cfg)
    
    X_train_analytical = X_train_analytical.rename(columns=column_mapping)
    X_test_analytical = X_test_analytical.rename(columns=column_mapping)
    
    return (X_train_analytical, X_test_analytical, model_comparison, fig_importance, df_importance, 
            metrics_train, prob_train, fig_cab_train, fig_cm_train,
            metrics_test, prob_test, fig_cab_test, fig_cm_test)


## Example Run
if __name__ == "__main__":
    pasi = pd.read_csv("data/final_analytical_dataset.csv")
    configs = {
        'preprocess': {
            'y': "CRIT1FL", 
            'drop': ["USUBJID", "CRIT1FL", "CRIT2FL", "CRIT3FL"]
        }
    }

    (X_train_analytical, X_test_analytical, model_comparison, fig_importance, df_importance, 
                metrics_train, prob_train, fig_cab_train, fig_cm_train,
                metrics_test, prob_test, fig_cab_test, fig_cm_test) = prediction_process(pasi, configs)
