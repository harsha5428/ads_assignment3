import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn import metrics as skmet
from sklearn.preprocessing import StandardScaler

# Read electricity and population data from CSV files
data_read = pd.read_csv("electricity.csv", skiprows=3)
data_read1 = pd.read_csv("population.csv", skiprows=3)

# Assign data to data frames
df_electricity = data_read
df_population = data_read1

# Drop unnecessary columns from data frames
df1 = df_electricity.drop(columns=["Country Code", "Indicator Code"], axis=1)
df2 = df_population.drop(columns=["Country Code", "Indicator Code"], axis=1)

# Export data frames to CSV files
df1.to_csv('df1.csv')
df2.to_csv('df2.csv')


df_electricty_2019 = df_electricity[["Country Name", "2019"]].copy()
df_population_2019 = df_population[["Country Name", "2019"]].copy()

# Merge renewable energy and Co2 emission columns on Country name
df_2019_elec_pop = pd.merge(df_electricty_2019, df_population_2019, on="Country Name", how="outer")
df_2019_elec_pop.to_csv("elec_pop.csv")

# Rename axis
df_2019_elec_pop = df_2019_elec_pop.dropna()
df_2019_elec_pop = df_2019_elec_pop.rename(columns={"2019_x": "Access to  Electricity", "2019_y": "Population"})

pd.plotting.scatter_matrix(df_2019_elec_pop, figsize=(9.0, 9.0))
plt.tight_layout()
plt.show()
for ncluster in range(2, 10):
    # set up the clusterer with the number of expected clusters
    kmeans = cluster.KMeans(n_clusters=ncluster)
    # Fit the data, results are stored in the kmeans object
    kmeans.fit(df_2019_elec_pop[['Access to  Electricity', 'Population']])
    # fit done on x,y pairs
    labels = kmeans.labels_
    # extract the estimated cluster centres
    cen = kmeans. cluster_centers_
    # calculate the silhoutte score
    print(ncluster,skmet.silhouette_score(df_2019_elec_pop[[ 'Access to  Electricity', 'Population']], labels))
    
selected_columns = ['Access to  Electricity', 'Population']

# Copy the selected columns to a new DataFrame
df_2019_elec_pop_normalized = df_2019_elec_pop[selected_columns].copy()

# Perform feature scaling using StandardScaler
scaler = StandardScaler()
df_2019_elec_pop_normalized[selected_columns] = scaler.fit_transform(df_2019_elec_pop[selected_columns])

#Number of clusters are = 4
ncluster = 4
kmeans = cluster.KMeans(n_clusters=ncluster)
kmeans.fit(df_2019_elec_pop_normalized)
df_2019_elec_pop["elec_agri_cluster"] = kmeans.labels_
cen = kmeans.cluster_centers_
xcen = cen[:, 0]
ycen = cen[:, 1]

# cluster by cluster
plt.figure(figsize=(8.0, 8.0))
cm = plt.cm.get_cmap('tab10')
plt.scatter(df_2019_elec_pop_normalized["Access to  Electricity"], df_2019_elec_pop_normalized["Population"], 10, labels, marker="o",cmap=cm)
plt.scatter(xcen, ycen, 45,"k", marker="d")
plt.xlabel ("Access to electricity")
plt.ylabel("population")
plt.title("Access to electricity  vS Population ")
plt.show()
