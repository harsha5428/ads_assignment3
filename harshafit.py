# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
import errors as err

def read_data_file(file_name):
    """
    Read data from a csv file and preprocess it for further analysis.
    
    Args:
        file_name (str): The path of the csv file to read.
        
    Returns:
        tuple: A tuple containing two pandas DataFrames. The first DataFrame contains the original data with 
        dropped columns, renamed columns, and transposed rows and columns. The second DataFrame contains the 
        preprocessed data with cleaned NaN values, parsed year values, and converted data types.
    """
    df = pd.read_csv(file_name)
    df_changed = df.drop(columns=["Country Code", "Indicator Name", "Indicator Code"])
    df_changed = df_changed.replace(np.nan, 0)

    # Header
    df_changed = df_changed.rename(columns={'Country Name': 'Year'})
    df_t = np.transpose(df_changed)

    # Header setting
    header = df_t.iloc[0].values.tolist()
    df_t.columns = header
    df_t = df_t.reset_index()
    df_t = df_t.rename(columns={"index": "year"})
    df_t = df_t.iloc[1:]
    df_t = df_t.dropna()
    df_t["year"] = df_t["year"].str[:4]
    df_t["year"] = pd.to_numeric(df_t["year"])
    df_t["United States"] = pd.to_numeric(df_t["United States"])
    df_t["Ireland"] = pd.to_numeric(df_t["Ireland"])
    df_t["World"] = pd.to_numeric(df_t["World"])
    df_t["year"] = df_t["year"].astype(int)
    df_t.to_csv("test.csv")
    return df, df_t


df_meth, df_elec = read_data_file("electricity1.csv")
df_meth, df_pop = read_data_file("population1.csv")


def curve_f(t, scale, growth):
  """
  Parameters
  ----------
  t : TYPE
  List of values
  scale : TYPE
  Scale of curve.
  growth : TYPE
  Growth of the curve.
  Returns
  -------
  c : TYPE
  Result
  """
  a = scale * np.exp(growth * (t-1960))
  return a

# Fitting the curve to the data and getting the covariance matrix and its diagonal
par2, cov = opt.curve_fit(curve_f, df_elec["year"], df_elec["India"], p0=[4e8, 0.1])
sigma = np.sqrt(np.diag(cov))

# Getting the upper and lower error ranges using the sigma values and the covariance matrix
low, up = err.err_ranges(df_elec["year"], curve_f, par2, sigma)

# Creating a new column in the dataframe with the fit values and plotting the data and fit
df_elec["fit_value_year"] = curve_f(df_elec["year"], *par2)
plt.figure()
plt.title("Access of Electricity - India", fontweight='bold')
plt.plot(df_elec["year"], df_elec["India"], label="data")
plt.plot(df_elec["year"], df_elec["fit_value_year"], c="grey", label="fit")
plt.fill_between(df_elec["year"], low, up, alpha=0.2)
plt.legend()
plt.xlim(1990, 2020)
plt.xlabel("Year", fontweight='bold')
plt.ylabel("Access to electricity", fontweight='bold')
plt.savefig("elec_india.png", dpi=300)
plt.show()
plt.figure()

# Set title for the plot
plt.title("Access of Electricity Prediction for 2050 - India", fontweight='bold')

# Generate predicted values for 1980-2050 using the fitted parameters
pred_year = np.arange(1980, 2050)
pred_ind_acc = curve_f(pred_year, *par2)

# Plot the actual data and the predicted values
plt.plot(df_elec["year"], df_elec["India"], label="data")
plt.plot(pred_year, pred_ind_acc, label="predicted access of electricity")
plt.legend()
plt.xlim(1980, 2050)
plt.xlabel("Year", fontweight='bold')
plt.ylabel("Access of electricity", fontweight='bold')
plt.savefig("acc_prediction.png", dpi=300)
plt.show()

# Perform curve fitting
par3, cov = opt.curve_fit(curve_f, df_pop["year"], df_pop["India"], p0=[4e8, 0.1])
sigma = np.sqrt(np.diag(cov))

# Calculate error ranges
low, up = err.err_ranges(df_pop["year"], curve_f, par3, sigma)

# Add fitted values to the DataFrame
df_pop["fit_value_pop"] = curve_f(df_pop["year"], *par3)

# Plot the data and the fitted curve
plt.figure()
plt.title("Population - India", fontweight='bold')
plt.plot(df_pop["year"], df_pop["India"], label="data")
plt.plot(df_pop["year"], df_pop["fit_value_pop"], c="gray", label="fit")
plt.fill_between(df_pop["year"], low, up, alpha=0.3)
plt.legend()
plt.xlim(1980, 2019)
plt.xlabel("Year", fontweight='bold')
plt.ylabel("Population", fontweight='bold')
plt.savefig("Pop_india.png", dpi=300)
plt.show()

# Plot the predicted data and the fitted curve
plt.figure()
plt.title("population Prediction for 2060-India",fontweight='bold')
pred_year = np.arange(1980,2050)
pred_ind_pop = curve_f(pred_year,*par3)
plt.plot(df_pop["year"],df_pop["India"],label="data")
plt.plot(pred_year,pred_ind_pop,label="predicted population")
plt.legend()
plt.xlim(1980,2050)
plt.xlabel("Year",fontweight='bold')
plt.ylabel("Population",fontweight='bold')
plt.savefig("pop_predicition.png")
plt.show()
