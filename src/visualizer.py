import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

data = pd.read_csv("../database.csv")

def year_wise(data):
    result = data.groupby("Year").sum()["Victim Count"]
    result.plot()
    plt.show()

def race_wise(data):
    result = data[["Record ID", "Victim Race"]].groupby("Victim Race").count()
    result.plot(kind="bar")
    plt.show()

def column_wise(data, column):
    result = data[["Record ID", column]].groupby(column).count()
    print(result)
    if result.shape[0] < 100:
        result.plot(kind='bar')
        plt.show()


# column_wise(data, "Weapon")





"""
Record ID,Agency Code,Agency Name,Agency Type,City,State,Year,Month,Incident,Crime Type,
Crime Solved,Victim Sex,Victim Age,Victim Race,Victim Ethnicity,Perpetrator Sex,Perpetrator Age,
Perpetrator Race,Perpetrator Ethnicity,Relationship,Weapon,Victim Count,Perpetrator Count,Record Source
"""