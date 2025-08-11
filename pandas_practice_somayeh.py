# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 20:23:20 2025

@author: somayeh 
"""

# Creating a simple DataFrame of students

import pandas as pd

data={'Name':['Ali','Sara','Mehdi','Niloofar'],
      'Age':[17,18,16,17],
      'Grade':[18.5,19.0,17.25,18.75]
}
df=pd.DataFrame(data)
print("data is :\n",df)

# 1. Print only the Name column

import pandas as pd

data={'Name':['Ali','Sara','Mehdi','Niloofar'],
      'Age':[17,18,16,17],
      'Grade':[18.5,19.0,17.25,18.75]
}
df=pd.DataFrame(data)
print("data is :\n",df)

print("the Name coulmns is:\n",df['Name'])
print("the Grade of Sara is:\n",df.loc[ df['Name']=='Sara' , 'Grade'])
print(" info of students whose age is 17:\n",df.loc[df['Age']==17, :])
print("name and grade of students with grade >= 18:\n",df[df['Grade']>=18][['Name','Grade']])

# Edit DataFrame: Add column, Modify values, Drop rows

import pandas as pd

data={'Name':['Ali','Sara','Mehdi','Niloofar'],
      'Age':[17,18,16,17],
      'Grade':[18.5,19.0,17.25,18.75]
}
df=pd.DataFrame(data)
print("data is :\n",df)

#new column
df['Passed'] = df['Grade']>=18
print("new column of passing student:\n",df)
#change a grade
df.loc[df['Name']=='Mehdi','Grade']=18
#delet a student info
df.drop(0,axis=0,inplace=True)
#reset index
df.reset_index(drop=True,inplace=True)
print("final data frame is:\n",df)

# Data Summary and Sorting: describe, mean, sort, top record

import pandas as pd

data={'Name':['Ali','Sara','Mehdi','Niloofar'],
      'Age':[17,18,16,17],
      'Grade':[18.5,19.0,17.25,18.75]
}
df=pd.DataFrame(data)
print("data is :\n",df)

#describe data frame 
print("descriptoin of data frame is :\n ", df.describe())
#mean of grade and mean of age 
print("mean of age is:\n", df['Age'].mean(),"\n mean of grade is:\n",df['Grade'].mean())
#sorting  grade of data frame
print("\nsorting by grade is :\n ",df.sort_values(by=['Grade'],ascending=[False]))
# Get the student with the highest grade
top_student = df[df['Grade'] == df['Grade'].max()][['Name', 'Grade']]
print("Top student is:\n", top_student)

# Handling Missing Data: check, drop, fill

import pandas as pd
import numpy as np

data={'Name':['Ali','Sara','Mehdi','Niloofar','Reza'],
      'Age':[17,np.nan,16,17,np.nan],
      'Grade':[18.5,19.0,np.nan,18.75,np.nan]
}
df=pd.DataFrame(data)
print("data is :\n",df)
# number of Nan in each coulumn
print("number of null values in data frame:\n",df.isnull().sum())
# number of not Nan in each coulmn
print("number of notnull values in data frame:\n",df.notnull().sum())
#fillna and replace
df = df.dropna(subset=['Grade'])
print("grade coulmn by deliting null value is:\n", df)
df['Age'] = df['Age'].fillna(df['Age'].mean())
print("Age coulmn by filling null value whith mean is:\n",df)
df['Grade']=17
print("filling grade whith 17:\n",df['Grade'])

# Grouping by Class: mean, count, max, aggregation

import pandas as pd

data = {
    'Name': ['Ali', 'Sara', 'Mehdi', 'Niloofar', 'Reza', 'Maryam', 'Kian'],
    'Class': ['A', 'B', 'A', 'B', 'A', 'C', 'C'],
    'Grade': [18.5, 19.0, 17.25, 18.75, 16.0, 20.0, 18.0]
}

df = pd.DataFrame(data)

print("data is :\n",df)
print(df.columns)

#mean of grade each class
mean_grade=df.groupby('Class')['Grade'].mean()
print("mean of grade each class is:\n",mean_grade)
#count students each class
count_student=df.groupby('Class')['Name'].count()
print("\nNumber of students per class:\n",count_student)
#top grade in each class
top_grade=df.groupby('Class')['Grade'].max()
print("top grade in each class is:\n",top_grade)
#groupby_analysis_on_student_grades
collect=df.groupby(['Class','Name'])['Grade'].agg(['mean','sum','max'])
print("collection of data:\n",collect)

# Convert dtype and apply custom function in DataFrame

import pandas as pd

data = {
    'Name': ['Ali', 'Sara', 'Mehdi', 'Niloofar', 'Reza', 'Maryam', 'Kian'],
    'Class': ['A', 'B', 'A', 'B', 'A', 'C', 'C'],
    'Grade': [18.5, 19.0, 17.25, 18.75, 16.0, 20.0, 19.0]
}

df = pd.DataFrame(data)
df['Grade']=df['Grade'].astype(int)
def Grade_level(Grade):
    if Grade>18:
        return"_Excellent"
    elif (Grade>=15) and (Grade<=18):
        return "Good"
    else :
        return"failed"
df['Result']=df['Grade'].apply(Grade_level)    
print(df)


# 📌 Basic Pandas Series – Creating, Indexing, Filtering

import pandas as pd

s=pd.Series([45000, 52000, 61000, 49000, 58000],
index=["Ali", "Sara", "Reza", "Niloofar", "Mehdi"])
print(s)
print("\n saras salary is :",s["Sara"])
print("\n means of salary is :",s.mean())

high_salary=s[s>=50000]
print("\nsalarise >=50000\n",high_salary)

# 📌 Pandas DataFrame – Creating, Selecting, Filtering, Aggregation

import pandas as pd

Data={
      'Name':['Ali','Sara','Reza','Niloofar','Mahdi'],
      'Age':[21,22,20,23,21],
      'Major':['Computer Eng','Math','Physics','Math','Computer Eng'],
      'Grade':[17.5,18.2,15.7,19.1,14.3]
       }
df=pd.DataFrame(Data)
high_grade=df['Grade']>=17
math_students = df[df['Major'] == 'Math'][['Name', 'Grade']]
mean_grade=df['Grade'].mean()
print("complete data is:")
print(df)
print("\nname columnis:\n",df['Name'])
print("student that their grade is >17:\n")
print(df[high_grade])
print("math student Grade and Name:\n",math_students)
print("mean grade is:",mean_grade)


# 📌 Reading CSV with Pandas – Head, Info, Describe, Filtering, Value Counts


import pandas as pd

df=pd.read_csv('my_data.csv')
count=df.value_counts("Major")
low_grade_count = (df['Grade'] <= 16).sum()
print(df)
print("5 first row in table is:\n", df.head())
print("data information is:")
df.info()
print("data description is:\n", df.describe())
print("majors  students count is:\n", count)
print("count of low grade student is :",low_grade_count)

# 📌 Pandas Data Cleaning – NaN, Duplicates, Type Conversion, Rename


import pandas as pd

df=pd.read_csv('my_data.csv')
print("orginal data is:\n",df)
print("data information is :")
df.info()
print("nan values:\n",df.isnull().sum())
df=df.drop_duplicates()
print("df after delete duplication alues is:\n",df)
df=df.dropna(subset=['Grade'])
print("delete column that have null value of grade:\n",df)

if df['Age'].dtype == object:
    df['Age'] = df['Age'].astype(int)
    print("Converted Age column to int.")
else:
    print("Age column is already int.")

    
df=df.rename(columns={'Major':'field'})
print("cleaning data is :\n",df)

 # 📌 Pandas GroupBy and Aggregation
 
import pandas as pd

data = {
    'Product': ['Laptop', 'Phone', 'Laptop', 'Tablet', 'Phone'],
    'Sales': [1200, 2300, 1800, 2500, 1500],
    'Region': ['North', 'West', 'North', 'West', 'East'],
    'Seller': ['Ali', 'Sara', 'Reza', 'Niloofar', 'Mahdi']
}

df = pd.DataFrame(data)

df.to_csv("sales_data.csv", index=False)

sales_by_region = df.groupby('Region')['Sales'].sum()
print("Total Sales by Region:\n", sales_by_region)

sales_by_seller = df.groupby('Seller')['Sales'].sum()
print("\nTotal Sales by Seller:\n", sales_by_seller)

sales_by_product = df.groupby('Product')['Sales'].sum()
top_product = sales_by_product.idxmax()
print("\n Total Sales by Product:\n", sales_by_product)
print("\n Best-selling product is:", top_product)

 # 📌 Pandas MultiColumn GroupBy, Sorting, Pivot Table

import pandas as pd

data = {
    'OrderID': [101, 102, 103, 104, 105, 106],
    'Product': ['Laptop', 'Phone', 'Headphone', 'Phone', 'Mouse', 'Laptop'],
    'Category': ['Electronics', 'Electronics', 'Accessories', 'Electronics', 'Accessories', 'Electronics'],
    'Seller': ['Ali', 'Sara', 'Ali', 'Reza', 'Sara', 'Niloofar'],
    'Quantity': [1, 2, 3, 1, 2, 1],
    'Price': [1200, 800, 150, 900, 50, 1300]
}
df=pd.DataFrame(data)
price=df['Price']
df['Total']=df['Quantity']*price
print("total price for each product is :\n",df)
df_sum_sell_category=df.groupby('Category')['Total'].sum()
print("sum of sells for each category is :\n",df_sum_sell_category)
sells_by_sellers = df.groupby('Seller')['Total'].sum().sort_values(ascending=False)

print("sels of each seller is :\n",sells_by_sellers)
pivot = pd.pivot_table(df, values='Total', index='Seller', columns='Category', aggfunc='sum', fill_value=0)
print("\n Pivot Table (Sales by Seller and Category):\n", pivot)


#️⃣ #pandas_practice #data_cleaning #missing_values #grade_analysis #student_performance #top_students
import pandas as pd
import numpy as np


data = {
    'Name': ['Ali', 'Sara', 'Mehdi', 'Niloofar', 'Reza', 'Maryam', 'Kian', 'Fatemeh', 'Hamed', 'Zahra'],
    'Class': ['A', 'B', 'A', 'B', 'A', 'C', 'C', 'B', 'C', 'A'],
    'Math': [18.5, 19.0, np.nan, 18.75, 16.0, 20.0, 14.5, np.nan, 17.25, 18.0],
    'Physics': [17.0, 18.5, 15.5, np.nan, 14.0, 19.0, 16.5, 15.0, np.nan, 17.75],
    'Chemistry': [18.0, 17.25, 16.5, 19.0, 15.0, np.nan, 14.75, 16.0, 18.25, 19.0]
}

df = pd.DataFrame(data)
print("Our dataframe is :\n",df)

df_null_count=df.isnull().sum()
print("\n  count datas that are null: \n",df_null_count)

df['Math']=df['Math'].fillna(df['Math'].mean())
df['Physics']=df['Physics'].fillna(df['Physics'].mean())
df['Chemistry']=df['Chemistry'].fillna(df['Chemistry'].mean())
print("\n clean up data is: \n",df)

df['Grade_average'] = df[['Math', 'Physics', 'Chemistry']].mean(axis=1)


def result(Grade_average):
    if Grade_average>=17:
        return"Accepted"
    else:
        return"rejected"
df['Result']=df['Grade_average'].apply(result)
print("\nresult data frame is:\n",df)

top_student=df[df['Grade_average']>=18][['Name','Grade_average']]
print("\ntop student list is :\n",top_student)


#health_data_analysis #BMI_calculation #risk_classification #pandas_exercise #medical_data #high_risk_patients

import pandas as pd
import numpy as np

data = {
    'PatientID': [101, 102, 103, 104, 105, 106, 107, 108],
    'Name': ['Ali', 'Sara', 'Reza', 'Maryam', 'Kian', 'Niloofar', 'Hamed', 'Zahra'],
    'Weight_kg': [72, 55, 92, 68, 110, 48, 84, 65],
    'Height_cm': [175, 160, 180, 165, 172, 158, 178, 162],
    'Blood_Pressure': [120, 110, 140, 135, 160, 115, 130, 125],
    'Blood_Sugar': [85, 90, 150, 100, 180, 95, 110, 88]
}

df = pd.DataFrame(data)
print("Patient data:\n", df)

df['BMI']=df['Weight_kg']/((df['Height_cm']/100)**2)
def Risk_level(BMI):
    if BMI<18.5:
        return "under weight"
    elif 18.5<=BMI<25:
        return "Normal"
    elif 25<=BMI<30:
        return "over weight"
    else:
        return "obese"
df['Risk_level']=df['BMI'].apply(Risk_level)
print("\nresult data is :\n",df)
high_risk=df[(df.BMI >=30)|(df.Blood_Pressure >=140)|(df.Blood_Sugar >=140)]
print("\nhigh risk patient are:\n",high_risk)

#transport_analysis #pandas_groupby #delay_analysis #travel_time #weekday_stats #public_transit

import pandas as pd

data = {
    'Line': ['A', 'B', 'A', 'C', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C'],
    'Date': ['2025-07-21', '2025-07-21', '2025-07-22', '2025-07-22', '2025-07-22',
             '2025-07-22', '2025-07-23', '2025-07-23', '2025-07-23', '2025-07-24',
             '2025-07-24', '2025-07-24', '2025-07-25', '2025-07-25', '2025-07-25'],
    'Travel_Time_Min': [45, 50, 60, 55, 70, 65, 40, 52, 66, 43, 58, 61, 48, 55, 60],
    'Delay_Min': [5, 10, 15, 12, 20, 10, 2, 5, 15, 0, 8, 5, 6, 9, 11]
}

df = pd.DataFrame(data)
df['Date'] = pd.to_datetime(df['Date'])  # تبدیل به نوع datetime
df['Weekday'] = df['Date'].dt.day_name()  # افزودن نام روز هفته
print("Data Frame is:\n",df)

mean_of_trave_time=df.groupby('Line')['Travel_Time_Min'].mean()
print("\n mean of travel time in each line:\n",mean_of_trave_time)

Delay_line=df.groupby('Line')['Delay_Min'].mean()
Delay_line=Delay_line[Delay_line>=10]
print(Delay_line)
mean_of_delay_time_in_weekdays=df.groupby('Weekday')['Delay_Min'].mean()
print("mean of delay time in weekdays:\n",mean_of_delay_time_in_weekdays)

#RealEstateAnalysis #PandasFiltering #GroupByMean #PricePerSquareMeter

import pandas as pd

data = {
    'City': ['Tehran', 'Tehran', 'Shiraz', 'Shiraz', 'Tabriz', 'Tabriz', 'Isfahan', 'Isfahan'],
    'Area_m2': [85, 120, 95, 70, 110, 60, 90, 150],
    'Rooms': [2, 3, 2, 1, 3, 1, 2, 4],
    'Price': [6800000000, 10000000000, 5400000000, 3200000000, 8000000000, 3900000000, 7500000000, 12000000000]
}

df = pd.DataFrame(data)
print("Our data frame is:\n",df)

df['price_each_m']=df['Price']/df['Area_m2']
print("\ndata frame whit price of each meter is:\n",df)

expensive_houses=df[df['Price']>=9000000000]
print("\nexpensive price list is:\n",expensive_houses)

city_price=df.groupby('City')['price_each_m'].mean()
print("\ncity prices is :\n",city_price)

#MovieRatingAnalysis #GenreRanking #TopMovie #PandasGroupBy #DataAnalysis #FilmData



import pandas as pd


data = {
    'Title': [
        'The Matrix', 'Inception', 'Titanic', 'Interstellar', 'The Godfather',
        'Avengers: Endgame', 'Parasite', 'Joker', 'La La Land', 'Whiplash'
    ],
    'Genre': [
        'Sci-Fi', 'Sci-Fi', 'Romance', 'Sci-Fi', 'Crime',
        'Action', 'Drama', 'Crime', 'Musical', 'Drama'
    ],
    'Rating': [
        8.7, 8.8, 7.8, 8.6, 9.2,
        8.4, 8.6, 8.5, 8.0, 8.5
    ],
    'Year': [
        1999, 2010, 1997, 2014, 1972,
        2019, 2019, 2019, 2016, 2014
    ]
}

df = pd.DataFrame(data)
print(df)
mean_of_rating=df['Rating'].mean()
print("mean of rating is:\n",mean_of_rating)
mean_rate_by_year=df.groupby('Year')['Rating'].mean()
print("mean of rating in diffrent years is:\n",mean_rate_by_year)
mean_rate_by_genre=df.groupby('Genre')['Rating'].mean()
print("mean of rating by genre is:\n",mean_rate_by_genre)
max_rate=mean_rate_by_genre.idxmax()
print("max rating in Genre is:\n ",max_rate)
top_movie = df.loc[df['Rating'].idxmax()]
print("top movie is:\n", top_movie)

print("top movie is:\n",top_movie)


#  Customer Service Satisfaction Analysis using Pandas


import pandas as pd

data = {
    'CustomerID': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
    'ServiceType': ['Internet', 'Internet', 'Mobile', 'Mobile', 'Banking',
                    'Banking', 'Mobile', 'Internet', 'Banking', 'Mobile'],
    'Rating': [5, 4, 3, 4, 5, 2, 5, 3, 4, 5]
}

df = pd.DataFrame(data)
print("our data frame is:\n",df)

mean_rating=df['Rating'].mean()
print("rating mean is:\n",mean_rating)

sort_rates=df[df['Rating']>=4]
print(sort_rates)
percent=(len(sort_rates)/len(df.index))*100
print(percent)
avg_rating_by_service = df.groupby('ServiceType')['Rating'].mean()
print("mean of rating by service :\n", avg_rating_by_service)


#online_sales_analysis

import pandas as pd
#read data &data information 
def Read_Data():
    data={'OrderID': [1, 2, 3, 4, 5, 6, 7],
    'OrderDate': ['2023-01-15', '2023-01-16', '2023-02-01', '2023-02-02', '2023-02-02', '2023-03-05', None],
    'Product': ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Laptop', 'Mouse', 'Monitor'],
    'Category': ['Electronics', 'Accessories', 'Accessories', 'Electronics', 'Electronics', 'Accessories', 'Electronics'],
    'Quantity': [1, 2, 1, 2, 1, 3, None],
    'UnitPrice': [1000, 25, 45, 200, 950, 30, 210],
    'Seller': ['Ali', 'Reza', 'Sara', 'Ali', 'Reza', 'Sara', 'Ali']
}
    df = pd.DataFrame(data)
    df.to_csv('online_sales.csv', index=False)
    df=pd.read_csv('online_sales.csv')

    print("our data frame as first is:\n")
    print(df)
    print("5 first data is:\n")
    print(df.head())
    print("data information is:\n")
    print(df.info())
    print("data description is:\n")
    print(df.describe())
    return df

#clean Duplicate data 
def Clean_data(df):
    print("clean duplicate data is:\n")
    clean_dup_data=df.drop_duplicates()
    print(clean_dup_data)
    return df.drop_duplicates()
#fill nan data

def Fillna(df):
    print("fill nan data is:\n")
    df['Quantity'].fillna(2.0, inplace=True)
    df['OrderDate'].fillna('2023-03-03', inplace=True)

    print(df)
    return df
#total price each order 

def Total_Price(df):
    print(" total price each order is:\n")
    df['Total_Price']=df['Quantity']*df['UnitPrice']
    print(df[['Total_Price','OrderID','Product']])
    return df
#checking day and month sales
def check_Sales(df):
    
    df['OrderDate']=pd.to_datetime(df['OrderDate'])
    df['Weekday'] = df['OrderDate'].dt.day_name() 
    df['Month']=df['OrderDate'].dt.month_name()
    sales_per_mo=df.groupby('Month')['Total_Price'].sum()
    daily_sales=df.groupby('Weekday')['Total_Price'].sum()
    print("monthly sales is :\n")
    print(sales_per_mo)
    print("daily sales is:\n")
    print(daily_sales)
    return df

#max Sales by categories,product
def Max_sales(df):
    
    top_product = df.groupby('Product')['Total_Price'].sum().sort_values(ascending=False).head(1)
    print("Top selling product:\n", top_product)

    top_category = df.groupby('Category')['Total_Price'].sum().sort_values(ascending=False).head(1)
    print("Top category:\n", top_category)

#sort data 
def Sort_data(df):
    df_sorted=df.sort_values(by=['Quantity','Total_Price'],ascending=[True,True])
    print("sort data is:\n")
    print(df_sorted)
    return df
 

#seller group
def Seller(df):
    df_seller=df.groupby('Seller').agg({'Quantity':'count','Total_Price':'sum'})
    print(df_seller)
    return df


df=Read_Data()
Clean_data(df)
Fillna(df)
Total_Price(df)
check_Sales(df)
Max_sales(df)
Sort_data(df)
Seller(df)


