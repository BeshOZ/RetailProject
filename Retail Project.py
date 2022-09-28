###############################################################
# Customer Segmentation with RFM and CLTV Prediction
###############################################################

############################
# Functions
############################
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
###############################################################
# 1. Data Understanding and importing
###############################################################

import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from sklearn.preprocessing import MinMaxScaler
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df_ = pd.read_excel("crm_analytics/datasets/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()

## Get an idea about the data:
df.head()
df.shape
df.isnull().sum()
df.dtypes
df.describe().T # We can see that there might be an outliers problem as well as the negative values in both
# Quantity and Price columns.

df["Description"].nunique() #Number of unique products: 4223

df["Description"].value_counts().head() #How many each product was in an invoice

df.groupby("Description").agg({"Quantity": "sum"}).sort_values("Quantity", ascending=False).head() #How many each product was sold

df["Invoice"].nunique() # We have 25900 unique invoice

df["TotalPrice"] = df["Quantity"] * df["Price"] # New feature that represents each product's total price in each invoice.

df.groupby("Invoice").agg({"TotalPrice": "sum"}).head() # Total price for each invoice


###############################################################
# 2. Data Preparation
###############################################################

df = df[~df["Invoice"].str.contains("C", na=False)]

# Get all invoices that doesn't have negative values for the quantity
df = df[(df['Quantity'] > 0)]

# As we are working on customers analysis we will not need rows with no customer ID
df.dropna(inplace=True)


for col in ["Quantity","Price","TotalPrice"]:
    replace_with_thresholds(df,col)


###############################################################
# 3. Calculating RFM Metrics
###############################################################
# We will take the last date and add 2 days to it and consider that our current day.
df["InvoiceDate"].max()
today_date = dt.datetime(2011, 12, 11)

rfm = df.groupby("Customer ID").agg({'InvoiceDate' : lambda x: (today_date - x.max()).days,
                                     'Invoice' : lambda x: x.nunique(),
                                     'TotalPrice': lambda x: x.sum()})

rfm.columns = ['recency', 'frequency', 'monetary']

rfm.head()

rfm.describe().T

# Filter monetary values with 0 value
rfm = rfm[rfm["monetary"] > 0]

###############################################################
# 4. Calculating RFM Scores
###############################################################

rfm["recency_score"] = pd.qcut(rfm["recency"],5,labels=[5,4,3,2,1])

rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"),5,labels=[5,4,3,2,1])

rfm["monetary_score"] = pd.qcut(rfm["monetary"],5,labels=[5,4,3,2,1])

rfm["RF_SCORE"] = (rfm['recency_score'].astype(str) +
                    rfm['frequency_score'].astype(str))

rfm.describe().T

###############################################################
# 6. Creating & Analysing RFM Segments
###############################################################
# regex

# RFM isimlendirmesi
seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm['segment'] = rfm['RF_SCORE'].replace(seg_map, regex=True)

# General look into the last segmentations
rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"])

# Here we got the customer IDs of our loyal customers to start actions for them
Loayas = rfm[rfm["segment"] == "loyal_customers"].index

# Here we got the customer IDs of our new customers to start actions to get them
#into buying more from us.
New = rfm[rfm["segment"] == "new_customers"].index


###############################################################
# 6. Creating Customer LifeTime Values
###############################################################

df.head()
cltv_c = df.groupby("Customer ID").agg({"Invoice": lambda  x : x.nunique(),
                                        "Quantity": lambda  x : x.sum(),
                                        "TotalPrice": lambda x : x.sum()})

cltv_c.columns = ['total_transaction', 'total_unit', 'total_price']

##################################################
# 2. Average Order Value (average_order_value = total_price / total_transaction)
##################################################

cltv_c["average_order_value"] = cltv_c["total_price"] / cltv_c["total_transaction"]

##################################################
# 3. Purchase Frequency (total_transaction / total_number_of_customers)
##################################################

cltv_c["purchase_frequency"] = cltv_c["total_transaction"] / cltv_c.shape[0]

##################################################
# 4. Repeat Rate & Churn Rate (frequency more than one / All clients)
##################################################

repeat_rate = cltv_c[cltv_c["total_transaction"] > 1].shape[0] / cltv_c.shape[0]

churn_rate = 1 - repeat_rate

##################################################
# 5. Profit Margin (profit_margin =  total_price * 0.10)
##################################################

cltv_c["profit_margin"] = cltv_c["total_price"] * 0.10

##################################################
# 6. Customer Value (customer_value = average_order_value * purchase_frequency)
##################################################

cltv_c['customer_value'] = cltv_c["average_order_value"] * cltv_c["purchase_frequency"]

##################################################
# 7. Customer Lifetime Value (CLTV = (customer_value / churn_rate) x profit_margin)
##################################################

cltv_c["cltv"] = (cltv_c['customer_value'] /churn_rate) * cltv_c["profit_margin"]

cltv_c.sort_values(by="cltv", ascending=False).head()

##################################################
# 8. Make Segments
##################################################

cltv_c["segment"] = pd.qcut(cltv_c["cltv"], 4, labels=["D", "C", "B", "A"])

cltv_c.groupby("segment").agg({"count", "mean", "sum"})

##################################################
# CLTV Predictions
##################################################
# Calculate recency, T, frequency and monetary
cltv_df = df.groupby('Customer ID').agg(
    {'InvoiceDate': [lambda InvoiceDate: (InvoiceDate.max() - InvoiceDate.min()).days,
                     lambda InvoiceDate: (today_date - InvoiceDate.min()).days],
     'Invoice': lambda Invoice: Invoice.nunique(),
     'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

cltv_df.columns = cltv_df.columns.droplevel(0)

cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']

# Get the average of monetary for each client
cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]

# Get only clients who bought more than once
cltv_df = cltv_df[(cltv_df['frequency'] > 1)]

# Transform to weeks
cltv_df["recency"] = cltv_df["recency"] / 7

# Transform to weeks
cltv_df["T"] = cltv_df["T"] / 7

##############################################################
# 2. BG-NBD
##############################################################

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df['frequency'],
        cltv_df["recency"],
        cltv_df["T"])

################################################################
# Expected purchases for 1 week
################################################################

cltv_df["expected_purc_1_week"] = bgf.predict(1,
                                              cltv_df['frequency'],
                                              cltv_df['recency'],
                                              cltv_df['T'])

################################################################
# Expected purchases for 1 month
################################################################

cltv_df["expected_purc_1_month"] = bgf.predict(4,
                                              cltv_df['frequency'],
                                              cltv_df['recency'],
                                              cltv_df['T'])

# Top 10 customers for purchases during the next month
cltv_df["expected_purc_1_month"].sort_values(ascending=False).head(10)

bgf.predict(4,
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T']).sum()

################################################################
# Expected purchases for 3 months
################################################################

cltv_df["expected_purc_3_month"] = bgf.predict(4 * 3,
                                              cltv_df['frequency'],
                                              cltv_df['recency'],
                                              cltv_df['T'])

# Top 10 customers for purchases during the next 3 months
cltv_df["expected_purc_3_month"].sort_values(ascending=False).head(10)

################################################################
# Prediction results
################################################################

plot_period_transactions(bgf)
plt.show(block = True)

##############################################################
# 3. GAMMA-GAMMA
##############################################################

ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(cltv_df['frequency'] , cltv_df["monetary"])


cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                             cltv_df['monetary'])
cltv_df.sort_values("expected_average_profit", ascending=False).head(10)

##############################################################
# 4. CLTV by BG-NBD and GG models
##############################################################

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=3,  # 3 months
                                   freq="W",  # T frequency type.
                                   discount_rate=0.01)

cltv = cltv.reset_index()

cltv.head()

cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")
cltv_final.sort_values(by="clv", ascending=False).head(10)

##############################################################
# 5. CLTV Segmentation
##############################################################

cltv_final["segment"] = pd.qcut(cltv_final["clv"],4,labels=["D","C","B","A"])

cltv_final.sort_values(by="clv", ascending=False).head(50)

cltv_final.groupby("segment").agg(
    {"count", "mean", "sum"})