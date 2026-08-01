# -*- coding: utf-8 -*-
"""
Created on Sat May  9 13:00:34 2026

@author: nicol
"""

import pyodbc
import pandas as pd


#התחברות למסד נתונים הנמצא בתוך השרת במחשב דרך SQL
server = 'NICOLE'
database = 'Collage_Project'

drivers = [x for x in pyodbc.drivers() if 'SQL Server' in x]
if not drivers:
    print("Error")
else:
    selected_driver = drivers[0]
    conn_str = f'DRIVER={selected_driver};SERVER={server};DATABASE={database};Trusted_Connection=yes;'

    try:
        conn = pyodbc.connect(conn_str)
        print("Successfully connected")

        tables_query = "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE'"
        tables_list = pd.read_sql(tables_query, conn)['TABLE_NAME'].tolist()

        all_data = {}

        for table in tables_list:
            query = f"SELECT * FROM [{table}]"
            all_data[table] = pd.read_sql(query, conn)
            print(f"הטבלה '{table}' נטענה בהצלחה.")

        print("Successfully")

    except Exception as e:
        print(f"אירעה שגיאה: {e}")
    finally:
        if 'conn' in locals():
            conn.close()
            
###############################################################################   



#ניקוי הערכים החסרים (מחיקתם) - הסבר במייל שנשלח
# רשימת הטבלאות לניקוי
tables_to_fix = ['master_dataset', 'fact_prices_daily']

for table_name in tables_to_fix:
    # שמירת כמות השורות לפני הניקוי
    rows_before = len(all_data[table_name])
    
    # מחיקת כל שורה שיש בה ערך חסר (בעמודות התשואה)
    all_data[table_name].dropna(inplace=True)
    
    # בדיקה כמה שורות נמחקו
    rows_after = len(all_data[table_name])
    print(f"בטבלה {table_name} נמחקו {rows_before - rows_after} שורות עם ערכים חסרים.")

print("\n--- בדיקה סופית ---")
print(all_data['master_dataset'].isnull().sum())

# שמירת הנתונים הנקיים לצורך ה-Dashboard
columns_for_dashboard = [
    'date',
    'symbol',
    'open_price',
    'adj_close',
    'volume',
    'FED_RATE',
    'INFLATION_CPI',
    'CONSUMER_CONFIDENCE',
    'SP500',
    'VIX',
    'US10Y',
    'stock_return'
]

dashboard_data = all_data['master_dataset'][columns_for_dashboard].copy()

dashboard_data.to_csv(
    'master_dataset.csv',
    index=False,
    encoding='utf-8-sig'
)

print("הקובץ master_dataset.csv נשמר בהצלחה")



###############################################################################
#גרף של מחיר המניה לאורך זמן:
    
import matplotlib.pyplot as plt

# 1. ודוא שהתאריך בפורמט הנכון (חשוב לגרף)
df = all_data['master_dataset']
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# 2. יצירת הגרף
plt.figure(figsize=(12, 6))
plt.plot(df['date'], df['adj_close'], label='Adjusted Close Price', color='blue')

# הוצאת כותרות
plt.title('Stock Price Over Time - Cleaned Data')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

#3 המניות על אותו הגרף
plt.figure(figsize=(14, 7))

# לולאה שעוברת על כל מניה ייחודית שקיימת בעמודת ה-symbol
for ticker in df['symbol'].unique():
    subset = df[df['symbol'] == ticker]
    plt.plot(subset['date'], subset['adj_close'], label=ticker)

plt.title('Comparison of Stock Prices')
plt.xlabel('Date')
plt.ylabel('Adjusted Close Price')
plt.legend() # זה יוסיף ריבוע עם השמות והצבעים בצד
plt.grid(True, alpha=0.3)
plt.show()

###############################################################################

#חלוקת נתונים 

from sklearn.model_selection import train_test_split

# 1. שלב הכנת הנתונים - מושכת את הטבלה מהמילון
# אנו עובדות על עותק של הנתונים כדי לא לדרוס את המקור
df = all_data['master_dataset'].copy()

# תיקון שמות העמודות: אני מוודאת שהתאריך בפורמט הנכון
# השתמשתי ב-'date' באותיות קטנות כי ככה זה מופיע ב-SQL שלך
df['date'] = pd.to_datetime(df['date'])

# מיון לפי תאריך - קריטי כדי שהחלוקה ל-Train/Test תהיה לפי ציר הזמן
df = df.sort_values('date')

# 2. הפרדה ל-X (פיצ'רים) ו-y (משתנה מטרה)
# הורדנו את 'stock_return' כי זה מה שאנחנו מנסים לחזות (התשובה)
# הורדנו את 'date' ו-'symbol' כי המודל לא יכול לעשות חישובים על טקסט/תאריך
#(דוח 5-הורדנו פיצ'רים של DAILY לצורך EVALUATION)
X = df.drop(columns=[
    'stock_return',
    'date',
    'symbol',
    'daily_return_high',
    'daily_return_low'
], errors='ignore')

y = df['stock_return']

# 3. חלוקה ראשונה: אימון (70%) מול השאר (30%)
# שמתי shuffle=False כי בשוק ההון אסור לערבב ימים, אנחנו חייבים לשמור על הסדר
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, shuffle=False)

# 4. חלוקה שנייה: פיצול ה-30% הנותרים לולידציה וטסט (חצי-חצי)
# כך יוצא ש-15% מהנתונים הם לבדיקות תוך כדי עבודה ו-15% לבדיקה הסופית
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)

# הדפסת התוצאות כדי לוודא שהכל חולק כמו שצריך
print(f"שורות לאימון (Train): {len(X_train)}")
print(f"שורות לוולידציה (Validation): {len(X_val)}")
print(f"שורות לבדיקה סופית (Test): {len(X_test)}")

###############################################################################

# מודל מספר 1 Random Forest

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# --- התיקון הקטן שקורה כאן במקום בחלוקת הנתונים ---
# המודל חייב לקבל 0 ו-1 במקום מספרים עשרוניים.
# אנחנו הופכים את y_train ו-y_val לקטגוריות (עלה=1, לא עלה=0)
y_train_class = (y_train > 0).astype(int)
y_val_class = (y_val > 0).astype(int)

# 1. הגדרת המודל
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)

# 2. אימון המודל (שימי לב שאנחנו משתמשים ב-y_train_class החדש)
model_rf.fit(X_train, y_train_class)

# 3. ביצוע תחזית על נתוני ה-Validation
y_pred_val = model_rf.predict(X_val)

# 4. הדפסת התוצאות (משווים מול y_val_class)
print(" תוצאות הרצה ראשונית (Validation Set)")
print(f"Accuracy (דיוק כללי): {accuracy_score(y_val_class, y_pred_val):.2f}")

print("\nClassification Report - Random Forest:")
print(classification_report(y_val_class, y_pred_val))

# 5. המשתנים שהכי השפיעו
importances = pd.Series(model_rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print("\nהמשתנים שהכי השפיעו על החיזוי:")
print(importances.head(5))

#Confusion Matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# יצירת המטריצה
cm_rf = confusion_matrix(y_val_class, y_pred_val)

# תצוגה גרפית
plt.figure(figsize=(8, 6))
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=['Down (0)', 'Up (1)'])
disp_rf.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix - Random Forest')
plt.show()


# Random Forest - Fine Tuning (דוח מס'5)

models_to_test = {
    "RF_Default": RandomForestClassifier(n_estimators=100, random_state=42),

    "RF_200_depth10": RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    ),

    "RF_300_depth15": RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        random_state=42
    )
}

for name, model in models_to_test.items():

    model.fit(X_train, y_train_class)

    y_pred = model.predict(X_val)

    print("\n", name)

    print(f"Accuracy: {accuracy_score(y_val_class, y_pred):.2f}")

    print(classification_report(y_val_class, y_pred))



# מודל מספר 2 XGBoost

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# 1. שלב התיקון: הופכים את הנתונים למספרים שהמודל מסוגל לקרוא
# אנחנו יוצרים משתנים חדשים בשם X_train_xgb ו-X_val_xgb
X_train_xgb = X_train.apply(pd.to_numeric, errors='coerce')
X_val_xgb = X_val.apply(pd.to_numeric, errors='coerce')

# 2. הפיכת התשובות ל-0 ו-1 (ירידה/עליה)
y_train_class = (y_train > 0).astype(int)
y_val_class = (y_val > 0).astype(int)

# 3. הגדרת המודל
model_xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

# 4. אימון המודל - שימי לב! משתמשים ב-X_train_xgb (המתוקן)
model_xgb.fit(X_train_xgb, y_train_class)

# 5. תחזית על נתוני התיקוף - שימי לב! משתמשים ב-X_val_xgb (המתוקן)
y_pred_xgb = model_xgb.predict(X_val_xgb)

# 6. הדפסת התוצאות למסך
print("--- תוצאות מודל XGBoost (Validation Set) ---")
print(f"Accuracy (דיוק כללי): {accuracy_score(y_val_class, y_pred_xgb):.2f}")

print("\nClassification Report - XGBoost:")
print(classification_report(y_val_class, y_pred_xgb))

# 7. המשתנים שהכי השפיעו ב-XGBoost
xgb_importances = pd.Series(model_xgb.feature_importances_, index=X_train_xgb.columns).sort_values(ascending=False)
print("\n5 המשתנים שהכי השפיעו ב-XGBoost:")
print(xgb_importances.head(5))

# יצירת המטריצה
cm_xgb = confusion_matrix(y_val_class, y_pred_xgb)

#Confusion Matrix
cm_xgb = confusion_matrix(y_val_class, y_pred_xgb)

# תצוגה גרפית
plt.figure(figsize=(8, 6))
disp_xgb = ConfusionMatrixDisplay(confusion_matrix=cm_xgb, display_labels=['Down (0)', 'Up (1)'])
disp_xgb.plot(cmap=plt.cm.Greens) # צבע ירוק כדי להבדיל מהמודל הקודם
plt.title('Confusion Matrix - XGBoost')
plt.show()



#הפרמטרים שהשתמשנו בכל מודל
# הצגת הפרמטרים המשפיעים ביותר ב-Random Forest
print("--- Features Importance: Random Forest ---")
rf_features = pd.Series(model_rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print(rf_features)

print("\n" + "-"*40 + "\n")

# הצגת הפרמטרים המשפיעים ביותר ב-XGBoost
print("--- Features Importance: XGBoost ---")
xgb_features = pd.Series(model_xgb.feature_importances_, index=X_train_xgb.columns).sort_values(ascending=False)
print(xgb_features)


#(דוח 5)
# Final Test Evaluation- RANDOM FOREST

y_test_class = (y_test > 0).astype(int)

final_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    random_state=42
)

final_model.fit(X_train, y_train_class)

y_pred_test = final_model.predict(X_test)
print()
print("Final Test Results-RANDOM FOREST")

print(f"Accuracy: {accuracy_score(y_test_class, y_pred_test):.2f}")

print(classification_report(y_test_class, y_pred_test))

#(דוח 5- שמירת קובץ PKL)
import joblib

joblib.dump(final_model, 'best_random_forest_model.pkl')
print()
print("Model saved successfully")



#דוח 5- ROC/AUC
# מודל סופי - Random Forest אחרי Fine Tuning
final_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    random_state=42
)

final_model.fit(X_train, y_train_class)

y_test_class = (y_test > 0).astype(int)

from sklearn.metrics import roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt

y_pred_proba = final_model.predict_proba(X_test)[:, 1]

auc_score = roc_auc_score(y_test_class, y_pred_proba)

print(f"ROC-AUC: {auc_score:.2f}")

RocCurveDisplay.from_estimator(final_model, X_test, y_test_class)
plt.title("ROC Curve - Final Random Forest Model")
plt.show()