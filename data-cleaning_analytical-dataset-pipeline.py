import pandas as pd
import numpy as np
import warnings
from functools import reduce 

adpasi = pd.read_csv("data/adpasi.csv")
addlqi = pd.read_csv("data/addlqi.csv")
adnrs = pd.read_csv("data/adnrs.csv")
adpga = pd.read_csv("data/adpga.csv")
adphq8 = pd.read_csv("data/adphq8.csv")
adpssd = pd.read_csv("data/adpssd.csv")
adsl = pd.read_csv("data/adsl.csv")
adae = pd.read_csv("data/adae.csv")


def multiple_record(df, testcd, visit_col, visit_num, idx_col):
    """
    Address dataframes (i.e. ADLB) having multiple records per visit.

    Args:
        df (Dataframe): Raw dataframe
        testcd (str): Name of the test code column (i.e. TESTCD, PARAMCD)
        visit_col (str): Name of the visit column (i.e. AVISIT)
        visit_num (str): Name of the visit number column (i.e. AVISITN)
        idx_col (_type_): column that should exist in the output

    Returns:
        Dataframe: A dataframe addressed multiple records in the same visit.
    """
    
    group_cols = {"USUBJID", visit_col, visit_num}
    
    if testcd in df.columns:
        group_cols.add(testcd)
        
    if idx_col:
        if isinstance(idx_col, list):
            group_cols.update(idx_col)
        else:
            group_cols.add(idx_col)
            
    group_cols = list(group_cols)
    group_cols = [c for c in group_cols if c in df.columns]

    ## have different way of handling multiple records per visit: take advantage for numeric; use mode for categorical
    agg_func = {}
    for col in df.columns:
        if col not in group_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                agg_func[col] = 'mean'
            else:
                agg_func[col] = lambda x: x.mode().iloc[0] if not x.mode().empty else None
    
    result = df.groupby(group_cols, as_index=False).agg(agg_func)
    
    return result


def fill_NA(df, backfill=True, negative=False):
    """
    Function to fll NA values for analytical datasets

    Args:
        df (dataframe): Raw dataframe
        backfill (bool, optional): Whether to use backfill for all values or project for later weeks. Defaults to True.
        negative (bool, optional): Whether accept negative value for projection. Defaults to False.

    Returns:
        Dataframe: A dataframe with NA filled.
    """
    
    df = df.copy()
    
    ## Get the columns with need to fill NA
    value_cols = [c for c in df.columns if c.startswith("AVAL_")]
    params = set([c.split('_')[1] for c in value_cols])
    flags = {}
    
    for p in params:
        val_cols = [c for c in df.columns if f'AVAL_{p}_' in c and not c.endswith('_FL')]
        ## sort the column from earlier to later weeks
        val_cols_sorted = sorted(val_cols, key=lambda x: float(x.split('_')[-1]))
        base_col = val_cols_sorted[0]
        
        for i in val_cols:
            v = i.split('_')[-1]
            chg_col = f'CHG_{p}_{v}'
            
            ## add the flag columns to indicate whether the value is original or filled
            flags[f"{i}_FL"] = df[i].isna()
            if chg_col in df.columns:
                flags[f"{chg_col}_FL"] = df[chg_col].isna()
                df[i] = df[i].fillna(df[base_col] + df[chg_col])
        
        ## use backfill for all columns
        if backfill:
            if len(val_cols_sorted) > 1:
                df[val_cols_sorted] = df[val_cols_sorted].ffill(axis=1)
        
        ## use backfill for the first three weeks, and project the future weeks using the first three weeks
        else:
            if len(val_cols_sorted) > 3:
                stable_zone = val_cols_sorted[:3]
                projection_zone = val_cols_sorted[3:]
                
                df[stable_zone] = df[stable_zone].ffill(axis=1)
                
                x_stable = np.array([float(c.split('_')[-1]) for c in stable_zone])
                y_stable = df[stable_zone].values
                
                x_mean = np.mean(x_stable)
                y_mean = np.nanmean(y_stable, axis=1, keepdims=True)
                
                num = np.nansum((x_stable - x_mean) * (y_stable - y_mean), axis=1)
                den = np.sum((x_stable - x_mean)**2)
                
                slopes = np.divide(num, den, out=np.zeros_like(num), where=den!=0)
                
                last_v = x_stable[-1]
                last_y = df[stable_zone[-1]]
                
                for col in projection_zone:
                    curr_v = float(col.split('_')[-1])
                    dist = curr_v - last_v
                    projection = last_y + (slopes * dist)
                    
                    ## whether accept negative projection
                    if not negative:
                        projection = projection.clip(lower=0)
                    
                    df[col] = df[col].fillna(projection)
            
            else: 
                df[val_cols_sorted] = df[val_cols_sorted].ffill(axis=1)
        
        ## calculate the CHG value using baseline and AVAL values
        for i in val_cols:
            v = i.split('_')[-1]
            value_v, chg_v = f"AVAL_{p}_{v}", f"CHG_{p}_{v}"
            if all(c in df.columns for c in [value_v, chg_v, base_col]):
                df[chg_v] = df[chg_v].fillna(df[value_v]-df[base_col])
        
        ## drop patients with no baseline values
        df = df.dropna(subset=base_col)
                
    ## add the flag columns
    flag_df = pd.DataFrame(flags, index=df.index)
    df = pd.concat([df, flag_df], axis=1)

    ## adjust the flag columns to be right after the column
    final_cols = []
    for col in df.columns:
        if not col.endswith("_FL"):
            final_cols.append(col)
            if f"{col}_FL" in df.columns:
                final_cols.append(f"{col}_FL")
    
    df = df[final_cols]
    
    return df
        

def analytical_data_pipeline(df, idx_cols=None, filter_var_value = None,
                            testcd="PARAMCD" ,visit_col="AVISIT", visit_num="AVISITN", 
                            base_visit=0, target_visit=16, target_variable=None, 
                            multiple_record=False, backfill=True, negative=False):
    
    """
    Analytical pipeline that pivots the raw dataframe to subject-level, fill the NA values, and drop unnecessary columns.
    
    Args:
        df (dataframe): Raw dataframe
        idx_cols (list, optional): columns from the raw dataframe should be included in the analysis. Default to None.
        fliter_var_value (dict, optional): dict.keys(): column name; dict.values(): a list of values should be included. The value can be str, num, or tuple. Default to None.
        testcd (str, optional): the column name for test column (i.e. TESTCD, PARAMCD). Default to "PARAMCD".
        visit_col (str, optional): the column name for the visit (i.e. AVISIT). Default to "AVISIT".
        visit_num (str, optional): the column name for the visit number column (i.e. AVISITN). Default to "AVISITN".
        base_visit (num, optional): the visit number for the base visit. Default to 0.
        target_visit (num, optional): the visit number for the visit that want to be analyzed (usually a study endpoint). Default to 16.
        target_variable (str/list, optional): the target variable column(s) for the analytical dataset. Default to None.
        multiple_record (bool, optional): whether to use the multiple_record() function to clean the dataset. Default to False.
        backfill (bool, optional): Whether to use backfill for all values or project for later weeks. Defaults to True.
        negative (bool, optional): Whether accept negative value for projection. Defaults to False.

    Returns:
        Dataframe: A cleaned, transformed analytical dataset.
    """
    
    df = df.copy()
    target_variable = target_variable if target_variable is not None else []
    
    ## filter to select desired value
    if filter_var_value is not None:
        for key, value in filter_var_value.items():
            if isinstance(value, list):
                df = df[df[key].isin(value)]
            elif isinstance(value, tuple) and len(value) == 2:
                low, high = value
                df = df[df[key].between(low, high)]
    
    ## address multiple records if dataset has multiple records for each visit (i.e. ADLB dataset)
    if not multiple_record:
        df = df
    else:
        df = multiple_record(df, testcd, visit_col, visit_num, idx_cols)
    
    val_cols = ["AVAL", "CHG", "PCHG"]
    val_cols = [col for col in df.columns if col in set(val_cols)]
    
    if idx_cols is None:
        idx_cols = [col for col in df.columns if col not in val_cols and col not in [testcd, "BASE"] and col not in target_variable]
    else:
        idx_cols = [col for col in idx_cols if col not in val_cols and col not in [testcd, "BASE"] and col not in target_variable and col in df.columns]
    
    ## pivot the dataset to have individual columns for each test
    wide_params = (
        df
        .loc[lambda x: (x[visit_num] >= base_visit) & (x[visit_num] <= target_visit)]
        .pivot_table(index=idx_cols, columns=testcd, values=val_cols)
        .reset_index()
        .pipe(lambda d: d.set_axis(
            ['_'.join(col).strip('_') for col in d.columns], axis=1)
        )
    )
    
    ## if the idx_cols have similar structure with the pivot and value column, the pivot table might return zero output
    if len(wide_params) == 0:
        warnings.warn(
            "\n[PIPELINE WARNING]: The resulting DataFrame is empty. \n"
            "This usually happens because 'idx_cols' contains variables with "
            "all NaN values or incompatible combinations. Please reselect your index columns.",
            UserWarning
        )
        
        return wide_params # Return the empty df so the script doesn't crash here
        
    
    ## pivot the dataset to have individual columns for each visit
    val_cols_final = [c for c in wide_params.columns if c.startswith(('AVAL', 'PCHG', 'CHG'))]
    idx_cols_final = wide_params.columns.difference([visit_num, visit_col, *val_cols_final]).tolist()[::-1]
    
    analytical = (
        wide_params.pivot_table(index=idx_cols_final, columns=visit_num, values=val_cols_final)
        .reset_index()
        .pipe(lambda d: d.set_axis(
            ['_'.join(map(str, col)).strip('_') for col in d.columns], axis=1)
        )
    )
    
    ## get rid of variable that relate to the target visit to ensure modeling quality
    if target_variable == None:
        analytical = (
            analytical
            .pipe(lambda d: d.drop(columns=[c for c in d.columns if c.endswith(f"_{float(target_visit)}")]))
            .pipe(lambda d: d.drop(columns=[c for c in d.columns if c.startswith("PCHG_")]))
        )
    
    else:
        ## use CRIT (the target variable column(s)) and visit number to create target columns
        if isinstance(target_variable, str):
            target_variable = [target_variable]
            
        cols_to_keep = ['USUBJID'] + target_variable
        crit_params = (
            df
            .loc[df[visit_num] == target_visit, cols_to_keep] 
            .dropna(subset=target_variable)                  
            .drop_duplicates()
        )
        
        analytical = (
            analytical
            .merge(crit_params, on=['USUBJID'], how="left")
            .dropna(subset=target_variable)
            .pipe(lambda d: d.drop(columns=[c for c in d.columns if c.endswith(f"_{float(target_visit)}")]))
            .pipe(lambda d: d.drop(columns=[c for c in d.columns if c.startswith("PCHG_")]))
        )
    
    analytical = analytical.pipe(fill_NA, backfill=backfill, negative=negative)
    
    counts = analytical.nunique(dropna=False)
    cols_to_drop = counts[counts <= 1].index.tolist()
    analytical = analytical.drop(columns=cols_to_drop)

    return analytical


def general_fill_NA(df, cols=None, adae=None):
    """
    Handle missing value with a generalized approach

    Args:
        df (_type_): Raw dataframe
        cols (list, optional): A list of column names that need to be handled missing value. Defaults to None.
        adae (dataframe, optional): The dataframe that only need to fill NA with zero. Defaults to None.

    Returns:
        dataframe: The analytical dataset with no NA
    """
    if cols is None:
        cols = list(df.columns)
    else:
        cols = [cols] if isinstance(cols, str) else list(cols)
    
    ## make sure that input columns exist in the dataframe
    cols = [c for c in cols if c in df.columns]

    for col in cols:
        target_series = df[col]
        if isinstance(target_series, pd.DataFrame):
            target_series = target_series.iloc[:, 0]
        
        ## special handling method: fill NA with zero
        if adae is not None and col in adae.columns:
            df[col] = target_series.fillna(0)
        
        ## Numeric column NA: fill the mean
        elif target_series.dtype.kind in 'bifc': 
            mean_val = target_series.mean()
            fill_val = round(mean_val, 2) if pd.notna(mean_val) else 0
            df[col] = target_series.fillna(fill_val)
        
        ## Categorical column NA: fill with "Missing"
        else:
            df[col] = target_series.fillna('Missing')
                
    return df[cols]


def ADAE_analytical_pipeline(df, cols=None, target_visit=16, additional_func=None):
    """
    Create subject-level analytical dataset at subject-level using ADAE.

    Args:
        df (dataframe): Raw dataframe
        cols (list, optional): A list of column names should be included. Defaults to None.
        target_visit (int, optional): the visit number for the visit that want to be analyzed (usually a study endpoint). Defaults to 16.
        additional_func (dict, optional): A dictionary for additional expression should be included other than the default ones. Defaults to None.

    Returns:
        dataset: The analytical ADAE dataset at subject-level
    """
    valid_cols = df.columns if cols is None else [c for c in (cols if isinstance(cols, list) else [cols]) if c in df.columns]
    
    ## columns related to adverse events that I think should be included in the analytical output
    agg_func = {
        "unique_AE": ("AEDECOD", "nunique"),
        "AE_Starts_before_W16": ("ASTDY", lambda x: (x <= target_visit*7).sum()),
        "AE_Ends_before_W16": ("AENDY", lambda x: (x <= target_visit*7).sum()),
        "Relative_AE": ("AREL", lambda x: (x == "Related").sum()),
        "Serious_AE": ("AESER", lambda x: (x == "Y").sum()),
        "Other_Serious_Event": ("AESMIE", lambda x: (x == "Y").sum()),
        "High_Toxic_Grade": ("AETOXGR", lambda x: (x.astype(float) > 3).sum())
    }
    
    ## update more columns if needed
    if additional_func and isinstance(additional_func, dict):
        agg_func.update(additional_func)
    
    final_agg = {k: v for k, v in agg_func.items() if v[0] in valid_cols}
    
    return df.groupby("USUBJID").agg(**final_agg).reset_index()


## pipeline to combine all the steps in one
def create_analytical_dataframe(config, first=None):
    """
    A function to combine all steps into one

    Args:
        config (dict): a dictionary with dict.keys() as analytical dataset, and dict.values() as params should be applied
        first (str, optional): The dataset name for anchor dataframe in deciding number of subjects. Defaults to None.

    Raises:
        KeyError: make sure the anchor dataset is available

    Returns:
        dataframe: The final analytical dataset
    """
    
    analytical_dfs = {}
    for domain, configs in config.items():
        df = configs['df'].copy()
        params = configs['params']
        
        if domain == "adsl":
            analytical_dfs[domain] = df.pipe(general_fill_NA, **params)
        elif domain == "adae":
            analytical_dfs[domain] = df.pipe(ADAE_analytical_pipeline, **params)
        else:
            analytical_dfs[domain] = df.pipe(analytical_data_pipeline, **params)
        
    
    
    if first == None:
        merge_list = []
    else:
        if first in analytical_dfs:
            merge_list = [analytical_dfs[first]]
        
        else:
            raise KeyError(f"Anchor domain '{first}' not found in processed datasets.")
    
    
    for domain, df in analytical_dfs.items():
        if domain != first:
            merge_list.append(df)
    
    merged_data = reduce(
            lambda left, right: pd.merge(
                left, 
                right, 
                on=["USUBJID"], 
                how="left", 
                suffixes=(None, '_drop')  
            ), 
            merge_list
        )
    
    # drop duplicated columns
    merged_data = merged_data.drop(columns=[c for c in merged_data.columns if c.endswith('_drop')])
    
    ## handle NA value emerged because of the merge
    if 'adae' in config.keys():
        merged_data = merged_data.pipe(general_fill_NA,adae=analytical_dfs['adae']).pipe(general_fill_NA)
    else:
        merged_data = merged_data.pipe(general_fill_NA)
    
    return merged_data
    
## create a sample analytical dataset
adpasi_col = ["USUBJID", "AVISIT", "AVISITN", "DISADURY", "PARAMCD", "AVAL", "BASE", "CHG", "PCHG", "CRIT1FL", "CRIT2FL", "CRIT3FL"]
analytical_col = ["USUBJID", "AVISIT", "AVISITN", "PARAMCD", "BASE", "AVAL", "CHG", "PCHG"]
adsl_col = ["USUBJID", "AGE", "SEX", "RACE", "ETHNIC", "TRTSEQP", "TRTSEQA",
            "SPGAGR1", "PBSAGR1", "PBSAGR2", "STRAT1R", "STRAT4R", 
            "HEIGHTBL", "WEIGHTBL", "BMIBL", 
            "PRPSOFL", "PRBIOFL", "TRGTNF", "TRGIL17", "TRGIL23", "TRGTBC", "TRGOTH", "PRJAKFL", "PRAPRFL", "PROTHFL", "PRONOFL", "PRMSFL",
            "NRSJDBL", "NRSJPBL", "PASESBL", "PASEFBL", "PASETBL", "PBSABL", "SF36PBL", "SF36MBL"
            ]
adae_col = ["USUBJID", "AEDECOD", "ASTDY", "AENDY", "AREL", "AESER", "AESMIE", "AETOXGR"]

configs = {
    'adpasi':{'df': adpasi, 'params':{'idx_cols': adpasi_col, 'target_variable': ["CRIT1FL", "CRIT2FL", "CRIT3FL"]}},
    'addlqi':{'df': addlqi, 'params':{'idx_cols': analytical_col}},
    'adpga': {'df': adpga, 'params':{'idx_cols': analytical_col}},
    'adphq8': {'df': adphq8, 'params':{'idx_cols': analytical_col, 'target_visit': 21600}},
    'adpssd': {'df': adpssd, 'params':{'idx_cols': analytical_col}},
    'adnrs': {'df': adnrs, 'params':{'idx_cols': analytical_col, 'filter_var_value':{'PARAMCD':["PNRS101", "PNRS102"]}}},
    'adsl': {'df': adsl, 'params':{'cols': adsl_col}},
    'adae': {'df': adae, 'params':{'cols': adae_col}}
}

dfs = create_analytical_dataframe(configs, first='adpasi')
dfs.to_csv("data/final_analytical_dataset.csv", index=False)