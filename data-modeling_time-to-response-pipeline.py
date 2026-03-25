import pandas as pd 
import numpy as np
import warnings
import itertools
import importlib
import joblib
from joblib import Parallel, delayed
from lifelines import LogNormalAFTFitter, CoxPHFitter, LogLogisticAFTFitter, WeibullAFTFitter
from lifelines.utils import concordance_index
from lifelines.exceptions import StatisticalWarning, ConvergenceError
import matplotlib.pyplot as plt


def time_to_response_dataset(raw_data, analytical_data, start_date, record_date, y, drop=["USUBJID"], VIF=False, analytical_cols=None):
    """
    Create the Survival Analysis dataset based on PASI (or other measures) or both PASI and analytical dataset (from data-cleaning_binary-model-pipeline.py) if available.
        Also split the data into train and test dataset.

    Args:
        raw_data (Dataframe): raw PASI (or other measures) dataset
        analytical_data (Dataframe): analytical dataset from previous cleaning, if available. Could be None.
        start_date (str): Column name for treatment start date.
        record_date (str): Column name for data record date.
        y (str): target variable for event occurance.
        drop (list, optional): A list of variables should be excluded from the analytical dataset. Defaults to ["USUBJID"].
        VIF (bool, optional): Whether to incorporate VIF in analytical dataset creation (feature selection). Defaults to False.
        analytical_cols (list, optional): A list of variable should be included in the final dataset from original PASI dataset if analytical dataset is not available. Defaults to None.

    Returns:
        Dataframe(s): Train and test datasets.
    """
    
    ## import functions from previous scripts
    spec = importlib.util.spec_from_file_location(
        "model_pipeline", 
        "data-modeling_binary-prediction-pipeline.py"
    )
    model_pipeline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_pipeline)
    preprocess = model_pipeline.preprocess
    initial_feature_selection = model_pipeline.initial_feature_selection
    
    spec2 = importlib.util.spec_from_file_location(
        "clean_pipeline", 
        "data-cleaning_analytical-dataset-pipeline.py"
    )
    clean_pipeline = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(clean_pipeline)
    analytical_data_pipeline = clean_pipeline.analytical_data_pipeline
    
    ## find responder and non-responder with first respond date (latest record date for non-responder)
    responder_pasi = (
        raw_data
        .loc[lambda df: (df[y] == "Y")]
        .loc[:, ["USUBJID", record_date, start_date]]
        .sort_values(["USUBJID", record_date], ascending=True)
        .groupby("USUBJID", as_index=False)
        .head(1)
    )

    nonresponder_pasi = (
        raw_data
        .loc[lambda df: ~df["USUBJID"].isin(responder_pasi["USUBJID"])]
        .loc[:, ["USUBJID", record_date, start_date]]
        .sort_values(["USUBJID", record_date], ascending=True)
        .groupby("USUBJID", as_index=False)
        .tail(1)
    )

    ## create the dur_col: by calculating the time for patients in the treatment (in days or weeks, for different measures)
    analytical_pasi = (
        pd.concat([responder_pasi, nonresponder_pasi])
        .assign(
            TRTSDT = lambda x: pd.to_datetime(x[start_date]),
            ADT = lambda x: pd.to_datetime(x[record_date]),
            Time_to_Response_Day = lambda x: np.ceil((x[record_date]-x[start_date]).dt.days),
            Time_to_Response = lambda x: np.ceil(x["Time_to_Response_Day"]/7)
        )
        .filter(items=["USUBJID", "Time_to_Response_Day", "Time_to_Response"])
    )
    
    ## create the final dataset
    if analytical_data is None:
        adpasi = analytical_data_pipeline(df=raw_data, idx_cols=analytical_cols, target_variable=y)
        analytical = (
            adpasi.merge(analytical_pasi, on="USUBJID", how="left")
            .dropna(subset=["Time_to_Response"])
        )
        
    else:
        analytical = (
            analytical_data.merge(analytical_pasi, on="USUBJID", how="left")
            .dropna(subset=["Time_to_Response"])
        )

    ## train-test split
    X_train, X_test, y_train, y_test, column_mapping = preprocess(
        data=analytical,
        y=y,
        drop=drop
    )

    X_train = X_train.rename(columns=column_mapping)
    X_test = X_test.rename(columns=column_mapping)
    
    ## if do initial feature selection with VIF (default to be False)
    if VIF:
        X_train, _ = initial_feature_selection(
            X_train, method="vif", thresh=10, protect=["Time_to_Response", "Time_to_Response_Day"]
        )
        
        X_test = X_test[X_train.columns]
    
        
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    
    return train_data, test_data


def evaluate_feature_worker(feature, selected, data, model_type, dur_col, event_col, mode):
    """
    Helper function for forward/backward feature selection with Parallel().

    Args:
        feature (str): a variable to be evaluated for addition/removal
        selected (list): a list of input variables for modeling
        data (Dataframe): The input dataframe
        model_type (str): The type of model be utilized (Cox or variation of AFT)
        dur_col (str): The duration column name (should be Time_of_Response or Time_of_Response_Day)
        event_col (str): The event column name
        mode (str): indicate whether forward or backward selection method is applied (or baseline for backward selection)

    Returns:
        tuple: feature (to remove or add) with the resulting AIC to be compared
    """
    
    ## different feature list (add or remove) for different selection method
    try:
        if mode == "forward":
            features = selected + [feature]
        elif mode == "backward":
            features = [f for f in selected if f not in [feature]]
        elif mode == "baseline":
            features = selected
        else:
            features = selected
        
        if not features: return (feature, np.inf)
        
        model_cols = list(set(features + [dur_col, event_col]))
        model = model_type(penalizer=0.001)

        ## fail-safe for model warnings/errors
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try: 
                model.fit(data[model_cols], dur_col, event_col)
                
            except ConvergenceError:
                return (feature, np.inf)

            if any(issubclass(warning.category, StatisticalWarning) for warning in w):
                return (feature, np.inf)

        ## get the AIC
        aic = getattr(model, "AIC_", getattr(model, "AIC_partial_", np.inf))

        return (feature, aic)

    except:
        return (feature, np.inf)


def forward_step_selection(data, model_type, dur_col, event_col, n_jobs=-1):
    """
    Forward feature selection function (Comparing to find the lowest AIC).

    Args:
        data (Dataframe): The input dataframe
        model_type (str): The type of model be utilized (Cox or variation of AFT)
        dur_col (str): The duration column name (should be Time_of_Response or Time_of_Response_Day)
        event_col (str): The event column name
        n_jobs (int, optional): Number of driver be used for Parallel(). Defaults to -1.

    Returns:
        List: The list of features for modeling with lowest AIC.
    """
    
    ## exclude the noise from the dataset
    if dur_col == "Time_to_Response":
        if "Time_to_Response_Day" in data.columns:
            data = data.drop(columns="Time_to_Response_Day")
    else:
        if "Time_to_Response" in data.columns:
            data = data.drop(columns="Time_to_Response")
        
    remaining = [c for c in data.columns if c not in [dur_col, event_col]]
    selected = []
    best_aic = np.inf
    
    while remaining:
        results = Parallel(n_jobs=n_jobs)(
            delayed(evaluate_feature_worker)(f, selected, data, model_type, dur_col, event_col, "forward")
            for f in remaining
        )
        
        valid_results = [r for r in results if r[1] < np.inf]
        
        if not valid_results:
            break
        
        best_feature, round_best_aic = min(valid_results, key=lambda x: x[1])
        
        if round_best_aic < best_aic:
            best_aic = round_best_aic
            selected.append(best_feature)
            remaining.remove(best_feature)
        
        else:
            break
    
    return selected


def backward_step_selection(data, model_type, dur_col, event_col, subset_features=None, n_jobs=-1):
    """
    Backward feature selection method (Comparing to find the lowest AIC).

    Args:
        data (Dataframe): The input dataframe
        model_type (str): The type of model be utilized (Cox or variation of AFT)
        dur_col (str): The duration column name (should be Time_of_Response or Time_of_Response_Day)
        event_col (str): The event column name
        subset_features (list, optional): A list of features (subset of the original columns) to be processed (usually as a result of forward or univariate feature selection). Defaults to None.
        n_jobs (int, optional): Number of driver be used for Parallel(). Defaults to -1.

    Returns:
        List: The list of features for modeling with lowest AIC.
    """
    
    if dur_col == "Time_to_Response":
        if "Time_to_Response_Day" in data.columns:
            data = data.drop(columns="Time_to_Response_Day")
    else:
        if "Time_to_Response" in data.columns:
            data = data.drop(columns="Time_to_Response")
        
    if subset_features is not None:
        remaining = list(set(subset_features) & set(data.columns))
        remaining = [c for c in remaining if c not in [dur_col, event_col]]
    else:
        remaining = [c for c in data.columns if c not in [dur_col, event_col]]

    if not remaining: return []

    ## find the baseline aic by including all features
    _, best_aic = evaluate_feature_worker(None, remaining, data, model_type, dur_col, event_col, "baseline")
    print(f"Parallel Backward Selection Started | Model: {model_type}, Starting AIC: {best_aic:0.4f}, Number of Features: {len(remaining)}")
    
    while len(remaining) > 1:
        current_remaining = list(remaining)
        
        results = Parallel(n_jobs=n_jobs)(
            delayed(evaluate_feature_worker)(
                f, current_remaining, data, model_type, dur_col, event_col, "backward"
            )
            for f in current_remaining
        )
        
        valid_results = [r for r in results if r[1] < np.inf]
        if not valid_results:
            break
        
        best_feature, round_best_aic = min(valid_results, key=lambda x: x[1])
        
        if round_best_aic < best_aic:
            remaining.remove(best_feature)
            best_aic = round_best_aic
        else:
            break
    
    return remaining


def univariate_aic_filter(data, model_type, dur_col, event_col, top_n):
    """
    Find the univariate AIC for each feature and use for feature selection (based on AIC).

    Args:
        data (Dataframe): The input dataframe
        model_type (str): The type of model be utilized (Cox or variation of AFT)
        dur_col (str): The duration column name (should be Time_of_Response or Time_of_Response_Day)
        event_col (str): The event column name
        top_n (int): number of variable should be selected

    Returns:
        List: The list of features with top N lowest univariate AIC.
    """
    
    aic_scores = {}
    
    for col in data.columns:
        if col in [dur_col, event_col]: continue
        if dur_col == "Time_to_Response":
            if col == "Time_to_Response_Day": continue
        else:
            if col == "Time_to_Response": continue
        
        try:
            model = model_type(penalizer=0.001).fit(data[[col, dur_col, event_col]], dur_col, event_col)
            aic = getattr(model, "AIC_", getattr(model, "AIC_partial_", np.inf))
            aic_scores[col] = aic
        
        except:
            continue
        
    top_n = min(top_n, len(aic_scores))
    selected = sorted(aic_scores, key=aic_scores.get)[:top_n]
    return selected + [dur_col, event_col]


def filter_to_backward(data, model_type, dur_col, event_col, top_n, n_jobs=-1):
    """
    Helper function for doing univariate-backward feature selection.

    Args:
        data (Dataframe): The input dataframe
        model_type (str): The type of model be utilized (Cox or variation of AFT)
        dur_col (str): The duration column name (should be Time_of_Response or Time_of_Response_Day)
        event_col (str): The event column name
        top_n (int): number of variable should be selected
        n_jobs (int, optional): Number of driver be used for Parallel(). Defaults to -1.

    Returns:
        List: The list of features for modeling with lowest AIC.
    """
    
    print(">>> Starting FILTER-TO-BACKWARD Pipeline <<<")
    selected_feature = univariate_aic_filter(data, model_type, dur_col, event_col, top_n)
    return backward_step_selection(data, model_type, dur_col, event_col, subset_features=selected_feature, n_jobs=n_jobs)


def forward_to_backward(data, model_type, dur_col, event_col, n_jobs=-1):
    """
    Helper function for doing forward-backward feature selection.

    Args:
        data (Dataframe): The input dataframe
        model_type (str): The type of model be utilized (Cox or variation of AFT)
        dur_col (str): The duration column name (should be Time_of_Response or Time_of_Response_Day)
        event_col (str): The event column name
        top_n (int): number of variable should be selected
        n_jobs (int, optional): Number of driver be used for Parallel(). Defaults to -1.

    Returns:
        List: The list of features for modeling with lowest AIC.
    """
    
    print(">>> Starting HYBRID FORWARD-BACKWARD Pipeline <<<")
    forward_features = forward_step_selection(data, model_type, dur_col, event_col, n_jobs=n_jobs)
    required_cols = list(forward_features) + [dur_col, event_col]
    return backward_step_selection(data, model_type, dur_col, event_col, subset_features=required_cols, n_jobs=n_jobs)


def modeling(data, dur_col, event_col, top_n=30,  fast=True):
    """
    Train multiple models (Cox, variations of AFT), find the best model (with lowest AIC) and saved for further use.

    Args:
        data (Dataframe): The input dataframe
        dur_col (str): The duration column name (should be Time_of_Response or Time_of_Response_Day)
        event_col (str): The event column name
        top_n (int): number of variable should be selected. Default to 30.
        fast (bool, optional): Whether to apply a quick features selection (filter-backward) or a comprehensive features selection (forward-backward). Defaults to True.

    Returns:
        Dict, Dataframe(s): dict with model information;
            Dataframe with model comparison (across models and for different hyperparameter)
    """
    
    penalizers = [1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
    
    model_configs = [
        {'name': 'CoxPH', 'class': CoxPHFitter, 'params': {'penalizer': penalizers, 'l1_ratio': np.linspace(0,1,5)}},
        {'name': 'LogNormalAFT', 'class': LogNormalAFTFitter, 'params': {'penalizer': penalizers}},
        {'name': 'LogLogisticAFT', 'class': LogLogisticAFTFitter, 'params':{'penalizer': penalizers}},
        {'name': 'WeibullAFT', 'class': WeibullAFTFitter, 'params':{'penalizer': penalizers}}
    ]
    
    model_results = []
    
    for cfg in model_configs:
        model_type = cfg['class']
        print(f">>> Testing Model: {cfg['name']} <<<")
        
        if fast:
            selected_features = filter_to_backward(data, model_type, dur_col, event_col, top_n)
        else:
            selected_features = forward_to_backward(data, model_type, dur_col, event_col)
            
        features = list(set(selected_features + [dur_col, event_col]))
        
        keys, values = zip(*cfg['params'].items())
        all_combo = [dict(zip(keys,v)) for v in itertools.product(*values)]
        search_space = np.random.choice(all_combo, size=min(10, len(all_combo)), replace=False)

        best_rand_aic = np.inf
        best_rand_params = None
        
        ## similar to a RandomizedSearchCV
        for params in search_space:
            try:
                model = model_type(**params).fit(data[features], dur_col, event_col)
                aic = getattr(model, "AIC_", getattr(model, "AIC_partial_", np.inf))
                if aic < best_rand_aic:
                    best_rand_aic = aic
                    best_rand_params = params
                
                if hasattr(model, "predict_expectation"):
                    preds = model.predict_expectation(data)
                else:
                    preds = -model.predict_partial_hazard(data)
                
                c_idx = concordance_index(data[dur_col], preds, data[event_col])
                
            except: continue
        
        if best_rand_params:
            model_results.append({
                'Name': cfg['name'],
                'Class': model_type,
                'AIC': best_rand_aic,
                'C-Index': c_idx,
                'Params': best_rand_params,
                'Features': features
            })
    
    df_model_result = pd.DataFrame(model_results)[['Name', 'AIC', 'C-Index']].sort_values('AIC')
    print(df_model_result)
    top_model = min(model_results, key=lambda x: x['AIC'])
    print(f">>> Winner for Tournament: {top_model['Name']} (AIC: {top_model['AIC']:.4f}, with tentative parameter: {top_model['Params']}) <<<")
    
    
    best_p = top_model['Params']['penalizer']
    best_l1 = top_model['Params'].get('l1_ratio', None)
    
    grid_p = list(set([best_p*0.9, best_p*0.95, best_p*0.99, best_p, best_p*1.01, best_p*1.05, best_p*1.1]))
    
    if top_model['Name'] == "CoxPH" and best_l1 is not None:
        grid_l1 = [best_l1*0.5, best_l1*0.75, best_l1, best_l1*1.25, best_l1*1.5]
        grid_params = [dict(penalizer=p, l1_ratio=l) for p in grid_p for l in grid_l1]
    else:
        grid_params = [dict(penalizer=p) for p in grid_p]
    
    final_results = []
    ## similar to a GridSearchCV
    for params in grid_params:
        try:
            model = top_model['Class'](**params).fit(data[top_model['Features']], dur_col, event_col)
            aic = getattr(model, "AIC_", getattr(model, "AIC_partial_", np.inf))
            if hasattr(model, "predict_expectation"):
                preds = model.predict_expectation(data)
            else:
                preds = model.predict_partial_hazard(data)
            
            c_idx = concordance_index(data[dur_col], preds, data[event_col])
            final_results.append({
                "Name": top_model['Name'],
                'Model': model, 
                'Penalizer': params.get('penalizer'),
                'L1 Ratio': params.get('l1_ratio'), 
                'AIC': aic, 
                'C-Index': c_idx,
                'Features': top_model['Features']
            })

        except: continue
    
    df_final_result = pd.DataFrame(final_results)[['Penalizer', 'L1 Ratio', 'AIC', 'C-Index']].sort_values('AIC')
    print(df_final_result)
    best_refined = min(final_results, key=lambda x: x['AIC'])
    joblib.dump(best_refined['Model'], f"{top_model['Name']}.pkl")
    print(f">>> Final Model Saved: {top_model['Name']} | AIC: {best_refined['AIC']: .4f} | C-index: {best_refined['C-Index']: .4f} | Penalizer: {best_refined['Penalizer']: .6f} | L1 Ratio: {best_refined['L1 Ratio']} <<<")
    
    return best_refined, df_model_result, df_final_result
        

def evaluation(model, data, dur_col, event_col, weeks=[1,2,4,8,12,16], thresh=[0.5, 0.75, 0.9]):
    """
    Evaluation function with c-index, prediction output, and forest plot (feature importance).

    Args:
        data (Dataframe): The input dataframe
        model_type (str): The type of model be utilized (Cox or variation of AFT)
        dur_col (str): The duration column name (should be Time_of_Response or Time_of_Response_Day)
        event_col (str): The event column name
        weeks (list, optional): A list of anchor weeks to be evaluated. Defaults to [1,2,4,8,12,16].
        thresh (list, optional): A list of probability to be evaluated (0<=thresh<=1). Defaults to [0.5, 0.75, 0.9].

    Returns:
        _type_: _description_
    """
    
    data = data.copy().reset_index(drop=True)
    data = data[model['Features']]
    
    prob_df = 1 - model['Model'].predict_survival_function(data, times=weeks).T
    prob_df.columns = [f"Week {w}" for w in weeks]
    prob_df.index = data.index
    
    # find the week of achieving certain probability of event happening
    for t in thresh:
        col_name = f"First Week >={int(t*100)}%"
        milestone_cols = [f"Week {w}" for w in weeks]
        
        prob_df[col_name] = (prob_df[milestone_cols]>=t).idxmax(axis=1)
        
        no_reach_mask = prob_df[milestone_cols].max(axis=1) < t
        prob_df.loc[no_reach_mask, col_name] = ""
        
    result_df = pd.DataFrame({
        "Actual Time": data[dur_col],
        "Event Observed": data[event_col],
        "Predicted Median Time": model['Model'].predict_median(data)
    }, index=data.index).join(prob_df)
    
    if hasattr(model["Model"], "predict_expectation"):
        pred = model["Model"].predict_expectation(data)
    else:
        pred = -model["Model"].predict_partial_hazard(data)
        
    c_index = concordance_index(data[dur_col], pred, data[event_col])
    
    ## forest plot
    plt.figure(figsize=(10,10))
    model["Model"].plot()
    plt.title("Model Coefficients (Forest Plot)")
    plt.axvline(0, color='black', linestyle='--', alpha=0.5)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()
    
    return result_df, c_index


def survival_pipeline(pasi, analytical, configs):
    """
    Helper function for connecting everything.

    Args:
        pasi (Dataframe): The raw dataset for measured event(s)
        analytical (Dataframe): The analytical dataset (if available)
        configs (dict): The input dictionary for function with inputs

    Returns:
        Dataframe, float: Dataframes for train and test predicted dataframe;
            floats for train and test c-index values
    """
    
    pre_config = configs.get('preprocess', {})
    model_config = configs.get('modeling', {})
    eval_config = configs.get('eval', {})
    
    train, test = time_to_response_dataset(raw_data=pasi, analytical_data=analytical, **pre_config)
    best_res, model_results, final_results = modeling(train, **model_config)
    train_result, train_c_index = evaluation(best_res, train, **eval_config)
    test_result, test_c_index = evaluation(best_res, test, **eval_config)
    
    return train_result, train_c_index, test_result, test_c_index

## example run
if __name__ == "__main__":
    pasi = pd.read_csv("data/adpasi.csv")
    analytical = pd.read_csv("data/final_analytical_dataset.csv")
    configs = {
        'preprocess': {
            'y': "CRIT1FL", 
            'start_date': "TRTSDT",
            'record_date': "ADT",
            'drop': ["USUBJID", "CRIT1FL", "CRIT2FL", "CRIT3FL"]
        },
        'modeling': {
            'dur_col': "Time_to_Response", 
            'event_col': "CRIT1FL"
        },
        'eval': {
            'dur_col': "Time_to_Response",
            'event_col': "CRIT1FL"
        }
    }
    
    train_result, train_c_index, test_result, test_c_index = survival_pipeline(pasi, analytical, configs)
