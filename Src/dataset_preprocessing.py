"""
This script analyzes a CSV file to find which columns contain non-integer values.
"""
#%%
import pandas as pd
import numpy as np

#%%
def is_integer_column(series):
    s = series.dropna()
    if s.empty:
        return True  # 沒有值時視為沒有非 integer 的證據
    if pd.api.types.is_integer_dtype(s):
        return True
    for val in s.iloc[:]:  # 遍歷值（避免一次轉整列型別問題）
        # 跳過已為整數型別的值
        if isinstance(val, (int, np.integer)):
            continue
        # 浮點數：檢查是否為整數值（例如 3.0）
        if isinstance(val, (float, np.floating)):
            if np.isnan(val):
                continue
            if val.is_integer():
                continue
            return False
        # 其他類型（字串等）：嘗試轉 float，再檢查是否為整數
        try:
            fv = float(str(val).strip())
            if np.isnan(fv):
                continue
            if fv.is_integer():
                continue
            return False
        except Exception:
            return False
    return True

#%%
# --- 1. Set file path ---
# Set the CSV file path to analyze here
path = "/Users/peggy/Documents/uu_master_data_science/uu_data_intensive_systems_group_project/Data/marketing_campaign_converted.csv"

#%%
# --- 2. Read and preview data ---
try:
    df = pd.read_csv(path)
    print(f"Successfully read file: {path}")
    print("Data preview:")
    print(df.head())
except FileNotFoundError:
    print(f"Error: file '{path}' not found. Please check the path.")
except Exception as e:
    print(f"讀取 CSV 時發生錯誤: {e}")

#%%
# --- 3. Analyze columns and find non-integer values ---
nonint_cols = []
samples = {} # 儲存範例值
unique_nonint_info = {} # 儲存唯一非整數值的資訊

for col in df.columns:
    if not is_integer_column(df[col]):
        nonint_cols.append(col)
        
        # 找出所有非整數值，並計算其唯一數量
        all_nonints = set()
        for v in df[col].dropna():
            try:
                fv = float(str(v).strip())
                if not fv.is_integer():
                    all_nonints.add(v)
            except (ValueError, TypeError):
                all_nonints.add(v)
        
        unique_nonint_info[col] = len(all_nonints)
        # 從集合中取出最多 5 個作為範例
        samples[col] = list(all_nonints)[:10]

#%%
# --- 4. Output results ---
if not nonint_cols:
    print("\nAnalysis complete: all columns can be considered integer (or are empty).")
else:
    print("\nAnalysis complete: found the following columns that contain non-integer values:")
    for c in nonint_cols:
        print(f"- {c} (has {unique_nonint_info.get(c, 0)} distinct non-integer values)")
        if samples.get(c):
            print(f"  Example values: {samples[c]}")
# %%
# --- 5. 資料清理與轉換 (範例：處理 Education 欄位) ---
print("\n" + "="*50)
print("Step 5: Data cleaning and transformation - Education column")
print("="*50)

# Based on observation, the 'Education' column contains categorical values; we
# perform ordinal encoding for this example.
# Define mapping of education levels (low to high)
education_mapping = {
    'Basic': 1,
    'Graduation': 2,  # 通常指大學畢業
    'Master': 3,
    '2n Cycle': 3,    # 在歐洲學制中，2nd Cycle 通常等同於碩士
    'PhD': 4
}

# 使用 .map() 方法進行轉換，並建立一個新欄位
df['Education_encoded'] = df['Education'].map(education_mapping)

print("\nMapped 'Education' to numeric values and stored in 'Education_encoded'.")
print("Value distribution after mapping:")
print(df['Education_encoded'].value_counts().sort_index())
print("\nPreview before/after mapping:")
print(df[['ID', 'Education', 'Education_encoded']].head(10))

# %%
# --- 6. 資料清理與轉換 (範例：處理 Marital_Status 欄位) ---
print("\n" + "="*50)
print("Step 6: Data cleaning and transformation - Marital_Status column")
print("="*50)

# Based on observation, 'Marital_Status' is categorical; we perform ordinal
# encoding using a specified order.
marital_status_mapping = {
    'Married': 1,
    'Together': 2,
    'Divorced': 3,
    'Widow': 4,
    'Single': 5,
    'Alone': 6,
    'YOLO': 7,
    'Absurd': 8
}

# 使用 .map() 方法進行轉換，並建立一個新欄位
df['Marital_Status_encoded'] = df['Marital_Status'].map(marital_status_mapping)

print("\nMapped 'Marital_Status' to numeric values and stored in 'Marital_Status_encoded'.")
print("Value distribution after mapping:")
print(df['Marital_Status_encoded'].value_counts().sort_index())
print("\nPreview before/after mapping:")
print(df[['ID', 'Marital_Status', 'Marital_Status_encoded']].head(10))
# %%
# --- 7. 資料清理與轉換 (處理 Dt_Customer 日期欄位) ---
print("\n" + "="*50)
print("Step 7: Data cleaning and transformation - Dt_Customer column")
print("="*50)

# The 'Dt_Customer' column is a string in 'DD-MM-YYYY' format. Convert it to
# standard datetime objects for further processing. Specifying
# format='%d-%m-%Y' ensures correct parsing.
dt_series = pd.to_datetime(df['Dt_Customer'], format='%d-%m-%Y')

# 將 datetime 物件格式化為 'YYYYMMDD' 字串，再轉換為整數
df['Dt_Customer_int'] = dt_series.dt.strftime('%Y%m%d').astype(int)

print("\nConverted 'Dt_Customer' to YYYYMMDD integer format and stored in 'Dt_Customer_int'.")
print("Preview before/after conversion:")
print(df[['ID', 'Dt_Customer', 'Dt_Customer_int']].head(10))

# %%
df.drop(columns=['Education', 'Marital_Status', 'Dt_Customer'], inplace=True)
print("\nRemoved original columns: 'Education', 'Marital_Status', 'Dt_Customer'")
# %%
# --- 8. 儲存清理後的資料 ---
print("\n" + "="*50)
print("Step 8: Save cleaned data")
print("="*50)

# Set output file path
output_path = "/Users/peggy/Documents/uu_master_data_science/uu_data_intensive_systems_group_project/Data/marketing_campaign_cleaned.csv"

# 將清理後的 DataFrame 儲存為新的 CSV 檔案
# index=False 表示不將 DataFrame 的索引寫入檔案中
df.to_csv(output_path, index=False)

print(f"\nCleaned data saved to:\n{output_path}")
print("\nFinal column preview:")
print(df.columns.tolist())
# %%
