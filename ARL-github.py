import pandas as pd
from datashape import datetime_
from future.backports.datetime import datetime

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules

# Determining Outlier Thresholds

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

# Replace Outliers with Thresholds

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

#Data Preparation

df_ = pd.read_csv(r"C:\Users\Yagmu\OneDrive\Masaüstü\DATA SCIENCE BOOTCAMP\4-Recommendation Systems\ARMUT PROJE\armut_data.csv")
df = df_.copy()

def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[dataframe["CategoryId"] > 0]
    dataframe = dataframe[dataframe["ServiceId"] > 0]
    replace_with_thresholds(dataframe, "CategoryId")
    replace_with_thresholds(dataframe, "ServiceId")
    return dataframe

retail_data_prep(df)

# Data Analysis and First Insights

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

# ServiceID represents a different service for each CategoryID. We will create a new variable to represent
# the services by combining ServiceID and CategoryID.

df["Service"] = df["ServiceId"].astype(str) + "_" + df["CategoryId"].astype(str)

#We will create a new date variable that contains only the year and month. Then, we will combine the UserID and the
# newly created date variable with "_" and assign it to a new variable called BasketID.

df["CreateDate"] = pd.to_datetime(df["CreateDate"], errors='coerce')

df["New_Date"] = df["CreateDate"].dt.to_period("M")

df["BasketId"] = df["UserId"].astype(str) + "_" + df["New_Date"].astype(str)

df.head()

# We will create a pivot table consisting of Basketıd and Service.

pivot_table = df.groupby(['BasketId', 'Service']).size().unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)
pivot_table.head()


# Support Values Using Apriori Algorithm

frequent_itemsets = apriori(pivot_table,
                            min_support=0.01,
                            use_colnames=True)

# Association Rule Learning

rules = association_rules(frequent_itemsets,
                          metric="support",
                          min_threshold=0.01)

rules.head()


# Service recommendation to a user who last received the 2_0 service using the arl_recommender function

def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in sorted_rules["antecedents"].items():
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"]))
    recommendation_list = list({item for item_list in recommendation_list for item in item_list})
    return recommendation_list[:rec_count]

arl_recommender(rules, "2_0", 1)


