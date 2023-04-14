import pandas as pd
import bs4
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression


def get_dataset(file):
    """
    This function takes an HTML file as an argument, parses it using BeautifulSoup
    and then extracts and cleans the relevant data from the file before returning a pandas DataFrame object.

    :param
        file (string): file name

    :return:
        pd.DataFrame: dataframe of cleaned data
    """
    # Open the HTML file and read its contents into a string
    data = open(file, encoding = 'utf-8').read()
    # Parse the HTML string using BeautifulSoup
    soup = bs4.BeautifulSoup(data, 'html.parser')
    
    # Find all the <tr> elements (HTML table rows) in the parsed HTML
    raw_data = soup.find_all('tr')
    # Convert the ResultSet object to a list
    raw_data = list(raw_data)
    # Initialize an empty list to hold the cleaned data
    cleaned_data = []
    
    # Iterate over each table row in the list
    for element in raw_data:
        # If the row doesn't contain a link (<a> tag), skip it
        if element.find('a') is None:
            continue
        # Extract relevant data from the row and store it in a list
        result = [element.find('a').text,
                  element.find('img')['alt'],
                  element.find_all('img')[1].get('title'),
                  int(element.find_all('td', {'class': 'statsDetail'})[0].text),
                  int(element.find_all('td', {'class': 'statsDetail'})[1].text),
                  element.find('td', {'class': 'kdDiffCol'}).text,
                  float(element.find_all('td', {'class': 'statsDetail'})[2].text),
                  float(element.find('td', {'class': 'ratingCol'}).text)]
        # Append the list of data to the cleaned_data list
        cleaned_data.append(result)
    
    # Create a DataFrame object from the cleaned data
    df = pd.DataFrame(cleaned_data,
                      columns = ['Player', 'Nationality', 'Teams', 'Maps', 'Rounds', 'K-D Diff', 'K/D', "Rating 2.0"])
    
    # Return the DataFrame object
    return df

    
def sample_300(df):
    """
    Sample 300 professionals from the DataFrame df and return as a new DataFrame.
    :param df: DataFrame that contains all professionals
    """
    return df.sample(300)


def get_pre_columns(df):
    """
    This function gets the pre-existing columns from the DataFrame and creates a Pipeline object with a LinearRegression model
    to predict the 'Rating 2.0' column.
    The function then fits the model, creates the prediction and returns the model and the prediction.
    :param df: A DataFrame
    :return: trained model and prediction
    """
    model = Pipeline([
        ("SelectColumns", ColumnTransformer([("keep", "passthrough", ['Kill/Death Ratio', 'Damage Per Round', 'KAST'])])),
        ("LinearModel", LinearRegression())
    ])
    
    model.fit(df, df['Rating 2.0'])
    
    prediction = model.predict(df)
    
    return model, prediction
    

def rmse(y, yhat):
    return np.sqrt(np.mean((y - yhat)**2))
