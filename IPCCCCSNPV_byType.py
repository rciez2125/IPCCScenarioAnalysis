import pandas as pd
import numpy as np
import warnings
import io
import itertools
import yaml
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os 

# read csv data
df = pd.read_excel('./assessment/output/fig2.9_data_table.xlsx')
# drop the negative negatives
df = df[df.Variable != 'Emissions|CO2|Net-negative-negative']

# add category data
cf = pd.read_excel('./assessment/output/spm_sr15_figure3a_data_table.xlsx')
print(cf.columns)
#t = cf.groupby(['model', 'scenario']).size().reset_index().rename(columns={0: "count"})

yrs = np.linspace(2000,2100,21)

dfcost = df # make a copy 
afolu_cost = 50 #$/ton
ccs_biomass = 80 #$/ton
ccs_dac = 100 #$/ton 
ccs_ew = 100 #$/ton I have no idea on this one 
ccs_other = 100 # need to figure out if net negative is just unspecified get rid of it or what the deal is 

print(len(dfcost))
# calculate costs for different types of ccs
for n in range(len(dfcost)):
	if dfcost.Variable.iloc[n] == 'AFOLU CDR':
		c = afolu_cost
	elif dfcost.Variable.iloc[n] == 'Carbon Sequestration|CCS|Biomass':
		c = ccs_biomass
	elif dfcost.Variable.iloc[n] == 'Carbon Sequestration|Direct Air Capture':
		c = ccs_dac
	elif dfcost.Variable.iloc[n] == 'Carbon Sequestration|Enhanced Weathering':
		c = ccs_ew
	else:
		c = ccs_other 
	for r in range(21):
		if math.isnan(dfcost[str(int(yrs[r]))].iloc[n])==False:
			dfcost[str(int(yrs[r]))].iloc[n] = dfcost[str(int(yrs[r]))].iloc[n]*c*1000*1000 #convert from Mt to tons

dfcost.to_csv('costouttest.csv')

# calculate one annual total per model & scenario combo
g = dfcost.groupby(['Model', 'Scenario']).agg({'2020': 'sum', '2025': 'sum', '2030': 'sum', '2035': 'sum', '2040': 'sum', '2045': 'sum', '2050': 'sum', '2055': 'sum', '2060': 'sum', '2065': 'sum', '2070': 'sum', '2075': 'sum', '2080': 'sum', '2085': 'sum', '2090': 'sum', '2095': 'sum', '2100':'sum'})
g = g.reset_index()

# add a category column 
for n in range(len(g)):
	# find the matching row and column in cf
	y = cf.index[cf.model == g.Model[n]].tolist()
	y2 = cf.index[cf.scenario == g.Scenario[n]].tolist()
	lst3 = [value for value in y if value in y2] 
	print('lst3', lst3)


# reformat 
# add a bunch of empty columns
for r in range(16):
	n = r+4
	lb = yrs[n]
	ub = yrs[n+1]
	for m in range(4):
		g[str(int(lb+m+1))] = np.zeros(len(g))

for r in range(16):
	n = r+4
	for m in range(len(g)):
		lb = yrs[n]
		ub = yrs[n+1]
		if math.isnan(df[str(int(lb))].iloc[m])==True:
			lb = yrs[n-1]
		if math.isnan(df[str(int(ub))].iloc[m])==True:
			ub = yrs[n+2]
		z = int(ub - lb - 1)
		for q in range(z): 
			g[str(int(lb+q+1))].iloc[m] = (g[str(int(lb))].iloc[m]*(z-q)+g[str(int(ub))].iloc[m]*(q+1))/(z+1)
print(g[:5])
a = np.linspace(2000, 2100, 101).astype(int) #tolist()
a = a.tolist()
#print(a)
#print(df.columns)
#print(g.columns)
g = g.reindex(columns = ['Model', 'Scenario', '2000', '2005', '2010', '2015', '2020', '2021', '2022', '2023', '2024', '2025', '2026', '2027', '2028', '2029', '2030', '2031', '2032', '2033', '2034', '2035', '2036', '2037', '2038', '2039', '2040', '2041', '2042', '2043', '2044', '2045', '2046', '2047', '2048', '2049', '2050', '2051', '2052', '2053', '2054', '2055', '2056', '2057', '2058', '2059', '2060', '2061', '2062', '2063', '2064', '2065', '2066', '2067', '2068', '2069', '2070', '2071', '2072', '2073', '2074', '2075', '2076', '2077', '2078', '2079', '2080', '2081', '2082', '2083', '2084', '2085', '2086', '2087', '2088', '2089', '2090', '2091', '2092', '2093', '2094', '2095', '2096', '2097', '2098', '2099', '2100'])
#print(df.Variable.unique())

g_discounted_stern = g.copy()
g_discounted_nordhaus = g.copy()
g_discounted_avg = g.copy()
# do some discounting

stern_delta = 0.001
nordhaus_delta = 0.015
avg_delta = (stern_delta+nordhaus_delta)/2
global_growth = 0.03 # this is a guesstimate, would need to review actual historical data 

r = 0.03 # dummy discount rate

#n = 81
#g_discounted_stern['2100'] = g_discounted_stern['2100']*(1/((1+(r+stern_delta))**(n+1)))

for n in range(2100-2020+1):
	g_discounted_stern[str(int(n+2020))] = g[str(int(n+2020))]*(1/((1+(r+stern_delta))**(n+1)))
	g_discounted_nordhaus[str(int(n+2020))] = g[str(int(n+2020))]*(1/((1+(r+nordhaus_delta))**(n+1)))
	g_discounted_avg[str(int(n+2020))] = g[str(int(n+2020))]*(1/((1+(r+avg_delta))**(n+1)))

# calculate NPVs
def calcNPV(df):
	df['NPV'] = np.zeros(len(df))
	for n in range(2100-2020+1):
		df['NPV'] = df['NPV'] + df[str(int(n+2020))]
	return df 

g_discounted_stern = calcNPV(g_discounted_stern)
g_discounted_nordhaus = calcNPV(g_discounted_nordhaus)
g_discounted_avg = calcNPV(g_discounted_avg)


def makePlot(df, figName):
	df1 = df.drop(columns = ['Model', 'Scenario', 'NPV', '2000', '2005', '2010', '2015'])
	d = df1.values
	plt.figure(figsize=(5,3.5))
	ax1 = plt.subplot(position=[0.15, 0.13, 0.6, 0.7])
	for m in range(len(d)):
		plt.plot(np.linspace(2020, 2100, 81), d[m,:]/1000000000)
	plt.ylim(0,1100)
	plt.xlabel('Year')
	plt.ylabel('Billions of dollars')
	plt.title('Annual Costs (discounted)')

	npv = np.sum(d, axis = 1)
	print(npv.shape)
	ax2 = plt.subplot(position = [0.85, 0.13, 0.1, 0.7])
	plt.boxplot(npv/1000000000000) #trillions of dollars 
	plt.ylim(0, 50)
	plt.title('NPV')
	plt.ylabel('Trillions of dollars')
	plt.savefig(figName, dpi=300)

makePlot(g_discounted_stern, 'SternOut.png')
makePlot(g_discounted_nordhaus, 'NordhausOut.png')
makePlot(g_discounted_avg, 'AvgOut.png')
#print(g_discounted_avg[:5])