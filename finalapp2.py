# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 18:50:54 2021

@author: helen
"""

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from sklearn.linear_model import LinearRegression

st.title("Customer Information Analysis :wine_glass:")
st.markdown("[Helen Timchenko] (https://github.com/htimchen)")

st.write("We will be exploring how income affects spending habits on different products. The idea for this app was inspired by [Aman Kharwal](https://thecleverprogrammer.com/2021/02/08/customer-personality-analysis-with-python/) who analyzed this dataset using different methods.")
#st.write("To begin, please upload the Customer Personality Analysis [dataset](https://www.kaggle.com/imakash3011/customer-personality-analysis).") 

#uploaded_file = st.file_uploader("Input marketing_campaign.csv", type= "csv")
uploaded_file = "marketing_campaign.csv"

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, sep='\t')
    
    #removing unnecessary columns
    df2 = df.drop(["ID", "Recency", "Response","Dt_Customer","Z_Revenue","Z_CostContact","Complain","AcceptedCmp1","AcceptedCmp2","AcceptedCmp3","AcceptedCmp4","AcceptedCmp5"], axis=1)
    df2 = df2[~df2.isna().any(axis=1)] # removing NaN
    
    df2["Children"] = df2["Kidhome"] + df2["Teenhome"]
    df2["Marital_Status"] = df2["Marital_Status"].replace({"Divorced":"Single", "Married":"In couple", "Together":"In couple", "Absurd":"Single","Widow":"Single","YOLO":"Single","Alone":"Single"})
    df2["Education"] = df2["Education"].replace({'Basic':'Undergraduate','2n Cycle':'Undergraduate','Graduation':'Postgraduate'})
    #I didn't use some of the columns that I cleaned up, but I decided to leave in the code even though I think the app will still work without the cleaning.
    
    df3 = df2.rename(columns={'Marital_Status':'Marital Status','Year_Birth':'Birth Year','MntWines': "Wines",'MntFruits':'Fruits','MntMeatProducts':'Meat','MntFishProducts':'Fish','MntSweetProducts':'Sweets','MntGoldProds':'Gold'})
        
    df4 = df3[(df3["Income"] < 100000) & (df3["Birth Year"] > 1939)]
    
    st.write("Using the drop down menu below, choose a product.")
    
    y_axis = st.selectbox("Choose a product for the y-value.",["Sweets","Fruits","Meat","Fish","Wines","Gold"])
    
    st.write("In the chart below, you can see how the product you picked (" + y_axis + ") is purchased based on annual income of customers.")
    st.write("*Hint: click + drag over the chart to see the distribution of Education levels amongst the customers.*")
    st.write(" ")
    
    brush = alt.selection_interval(empty='none')
    chart1 = alt.Chart(df4).mark_circle().encode(
        alt.X("Income",
            axis = alt.Axis(title = "Income"),
            scale=alt.Scale(zero=False),
        ),
        alt.Y(y_axis,
            axis = alt.Axis(title = y_axis),
            scale=alt.Scale(zero=False)),
            color=alt.condition(brush, 'Education', alt.value('rosybrown')),
    ).add_selection(brush
    ).properties(
        width = 500,
        height = 350,
        title = "Amount of " + y_axis + " purchased based on Income"
    )
    st.altair_chart(chart1)
    
    st.write("We can see that all of these graphs follow a smiliar trend, regardless of the product.")
    st.write("Let's explore this more with the *Wine* purchased data. *Hint: scroll over the data with the tooltip to see details*.")
    
    reg = LinearRegression()
    X = np.array(df4['Income']).reshape(-1,1)
    y = np.array(df4['Wines']).reshape(-1,1)
    reg.fit(X,y)
    reg_coef = float(reg.coef_)
    reg_int = float(reg.intercept_) #this part of the code is from the class website https://christopherdavisuci.github.io/UCI-Math-10/Week5/Week5-Wednesday.html
    
    wine = alt.Chart(df4).mark_circle().encode(
        alt.X('Income',
            axis = alt.Axis(title = "Annual Income"),
            scale=alt.Scale(zero=False)
        ),
        alt.Y('Wines',
            axis = alt.Axis(title = "Amount Spent on Wine"),
            scale=alt.Scale(0,1600)),
        color = alt.value('firebrick'),
        tooltip = ["Income","Wines"]
    )
    
    x = np.arange(100000)
    source = pd.DataFrame({
        'x': x,
        'f(x)': (reg_coef)*(x)+(reg_int)
    })

    line = alt.Chart(source).mark_line().encode(
        x='x',
        y=alt.Y('f(x)',scale=alt.Scale(domain=(0,1600),clamp=True)),
        color = alt.value('black'),
    ).properties(
        title = "Wine"
    ) #linear regression chart from https://altair-viz.github.io/gallery/simple_line_chart.html
    
    st.write("The equation for the linear regression in the chart below is:")
    st.write(f"$Amount of $ spent on wine = {reg_coef} * (annual income) + {reg_int}$") 
       
    wineline = wine + line
    st.altair_chart(wineline)
    
    st.write("But since you cannot have a negative income and you technically cannot spend negative money on a product, I resized the equation to begin at (0,0) using `clamp=True`.")
    st.write("On the graph, we see the amount spent on wine increases as income increases, shown by the positive slope of the linear regression.") 
    st.write("The range of incomes $0 - $30,000 has a small positive slope, with the amount spent on wine being about $150. Then, past roughly $30,000 annual income the slope increases faster.")
    
    st.subheader("So what does this mean?")
    st.write("These results make sense.  \n Income and spending are directly related, and when you have a lower income, you have less money to spend. When you have a higher income, you have more money to spend.")
    st.write("I would guess that the sudden increase in spending around the $30,000 mark has to do with fact that once basic needs and necessities are met, then people more comfortable with spending money on items that seem more like luxuries, such as Wine, Sweets and Gold.")
    st.write("I would guess that if we had more data that included higher incomes, the trend would *not* continue upward in the direction of the linear regression. I predict that as the income increased (and we moved to the right on the graph) the slope would flatten. Just because you have more money, does not mean you can consume more food. In other words, even with a higher income, you wouldn't physically be able to consume more products than the average person. This would cause the flattening of the slope of the line over time.")

with st.expander("Open to see references!"):
    st.write("**References :sunglasses:**")
    st.write("The part in the code where I added columns, renamed some columns and renamed some elements was inspired by [Aman Kharwal](https://thecleverprogrammer.com/2021/02/08/customer-personality-analysis-with-python/).")
    st.write("The code for the linear regression is from [Chris Davis](https://christopherdavisuci.github.io/UCI-Math-10/Week5/Week5-Wednesday.html) and from [Altair](https://altair-viz.github.io/gallery/simple_line_chart.html).")
    st.write("The source for this expander is from [Streamlit](https://docs.streamlit.io/).")