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
#df = pd.read_excel('./assessment/output/fig2.9_data_table.xlsx')
df = pd.read_csv('./assessment/output/CombinedCsvsWithOutcome.csv')
# drop the negative negatives
#df = df[df.Variable != 'Emissions|CO2|Net-negative-negative']
yrs = np.linspace(2020,2100,17) # every 5 years included
print(df.Variable.unique())
print(df.shape)
x = df.groupby(['model', 'scenario']).size().reset_index().rename(columns={0:'count'})
print(type(x))
print(x.iloc[0])
print(x.iloc[0].model)
print(len(x))


#newRows = make a new dataframe/new rows for dac 
for n in range(len(x)):
	y = df[df.model == x.iloc[n].model]
	y = y[y.scenario == x.iloc[n].scenario].drop(columns = ['model', 'scenario', 'Unnamed: 0', 'marker', 'category'])
	t = y[y.Variable == 'Total CDR'].drop(columns=['Variable']).values
	a = y[y.Variable == 'AFOLU CDR'].drop(columns=['Variable']).values
	b = y[y.Variable == 'BECCS'].drop(columns=['Variable']).values
	nn = y[y.Variable == 'Net negative CO2'].drop(columns=['Variable']).values
	c = y[y.Variable == 'Compensate CDR'].drop(columns=['Variable']).values
	if a.size == 0:
		d = np.round(t-b,4)
	elif b.size == 0:
		d = np.round(t-a, 4)
	else:
		d = np.round((t-(a+b)),4)
	print(d)
	print(np.sum(d))
	#if np.sum(d)>0:




#print(df.groupby(['model', 'scenario']).count())

# check to see how much DAC is included



dfcost = df # make a copy 
afolu_cost = 50 #$/ton
ccs_biomass = 80 #$/ton
ccs_dac = 100 #$/ton 
ccs_ew = 100 #$/ton I have no idea on this one 
ccs_other = 100 # need to figure out if net negative is just unspecified get rid of it or what the deal is 


# calculate costs for different types of ccs
for n in range(len(dfcost)):
	if dfcost.Variable.iloc[n] == 'AFOLU CDR':
		c = afolu_cost
	elif dfcost.Variable.iloc[n] == 'BECCS':
		c = ccs_biomass
	elif dfcost.Variable.iloc[n] == 'Net negative CO2':
		c = 0
	elif dfcost.Variable.iloc[n] == 'Compensate CDR':
		c = 0
	else:
		c = 0
	for r in range(17):
		if math.isnan(dfcost[str(int(yrs[r]))][n])==False:
			dfcost[str(int(yrs[r]))].iloc[n] = dfcost[str(int(yrs[r]))].iloc[n]*c*1000*1000*1000 #convert from Gt to tons

dfcost.to_csv('costouttest.csv') # still has nans

# calculate one annual total per model & scenario combo
g = dfcost.groupby(['category', 'model', 'scenario']).agg({'2020': 'sum', '2025': 'sum', '2030': 'sum', '2035': 'sum', '2040': 'sum', '2045': 'sum', '2050': 'sum', '2055': 'sum', '2060': 'sum', '2065': 'sum', '2070': 'sum', '2075': 'sum', '2080': 'sum', '2085': 'sum', '2090': 'sum', '2095': 'sum', '2100':'sum'})
g = g.reset_index()
print(len(g))

print(g.columns)

# add a category column 
g.to_csv('midpointout.csv')

# interpolate where necessary 
for n in range(len(g)):
	for r in range(15):
		# check previous and next values
		if g.iloc[n,r+4]==0:
			if g.iloc[n, r+3] != 0:
				if g.iloc[n,r+5] != 0:
					g.iat[n, r+4] = (g.iloc[n, r+3] + g.iloc[n, r+5])/2
					


# reformat 
for r in range(17-1):
	lb = yrs[r]
	ub = yrs[r+1]
	for m in range(4):
		g[str(int(lb+m+1))] = np.zeros(len(g))



for r in range(16):
	lb = yrs[r]
	ub = yrs[r+1]
	for m in range(len(g)):	
		if math.isnan(df[str(int(lb))].iloc[m])==True:
			lb = yrs[r-1]
		if math.isnan(df[str(int(ub))].iloc[m])==True:
			ub = yrs[r+2]
		z = int(ub - lb - 1)
		for q in range(z): 
			g[str(int(lb+q+1))].iloc[m] = (g[str(int(lb))].iloc[m]*(z-q)+g[str(int(ub))].iloc[m]*(q+1))/(z+1)

a = np.linspace(2000, 2100, 101).astype(int) #tolist()
a = a.tolist()
#print(a)
#print(df.columns)
#print(g.columns)
g = g.reindex(columns = ['model', 'scenario', 'category', '2020', '2021', '2022', '2023', '2024', '2025', '2026', '2027', '2028', '2029', '2030', '2031', '2032', '2033', '2034', '2035', '2036', '2037', '2038', '2039', '2040', '2041', '2042', '2043', '2044', '2045', '2046', '2047', '2048', '2049', '2050', '2051', '2052', '2053', '2054', '2055', '2056', '2057', '2058', '2059', '2060', '2061', '2062', '2063', '2064', '2065', '2066', '2067', '2068', '2069', '2070', '2071', '2072', '2073', '2074', '2075', '2076', '2077', '2078', '2079', '2080', '2081', '2082', '2083', '2084', '2085', '2086', '2087', '2088', '2089', '2090', '2091', '2092', '2093', '2094', '2095', '2096', '2097', '2098', '2099', '2100'])
print(df.Variable.unique())

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
	df1 = df.drop(columns = ['category', 'model', 'scenario', 'NPV'])
	d = df1.values
	plt.figure(figsize=(5,3.5))
	ax1 = plt.subplot(position=[0.15, 0.13, 0.6, 0.7])
	for m in range(len(d)):
		plt.plot(np.linspace(2020, 2100, 81), d[m,:]/1000000000) #billions of dollars
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

makePlot(g_discounted_stern, 'Figures/SternOut.png')
makePlot(g_discounted_nordhaus, 'Figures/NordhausOut.png')
makePlot(g_discounted_avg, 'Figures/AvgOut.png')
#print(g_discounted_avg[:5])


def makePlotbyCategory(df, figName):
	d1 = df[df.category == 'Below 1.5C'].drop(columns = ['category', 'model', 'scenario', 'NPV'])
	d2 = df[df.category == '1.5C low overshoot'].drop(columns = ['category', 'model', 'scenario', 'NPV'])
	d3 = df[df.category == '1.5C high overshoot'].drop(columns = ['category', 'model', 'scenario', 'NPV'])
	d4 = df[df.category == 'Lower 2C'].drop(columns = ['category', 'model', 'scenario', 'NPV'])
	d5 = df[df.category == 'Higher 2C'].drop(columns = ['category', 'model', 'scenario', 'NPV'])
	
	c = {'d1':d1.values, 'd2':d2.values, 'd3':d3.values, 'd4':d4.values, 'd5':d5.values}
	
	labs = ('Below 1.5C', '1.5C low\novershoot', '1.5C high\novershoot', 'Lower 2C', 'Higher 2C')

	plt.figure(figsize=(7, 4))

	for n in range(5):
		p = c['d'+str(n+1)]
		plt.subplot(position = [0.08+0.135*n, 0.14, 0.12, 0.78])
		#plt.subplot(position = [0.08+0.17*n, 0.14, 0.15, 0.78])
		for m in range(len(p)):
			plt.plot(np.linspace(2020, 2100, 81), p[m,:]/1000000000) #billions of dollars
		plt.xlabel('Year', fontsize = 8)
		plt.ylim(-100, 1000)
		plt.title(labs[n], fontsize = 8)
		plt.yticks([0, 200, 400, 600, 800, 1000], labels = (' ', ' ', ' ', ' ', ' ', ' '), fontsize = 8)
		plt.xticks([2020, 2030, 2040, 2050, 2060, 2070, 2080, 2090, 2100], labels = ('2020', ' ', ' ', ' ', '2060', ' ', ' ', ' ', '2100'), fontsize =6, rotation = 90)
		if n == 0:
			plt.ylabel('Billions of dollars', fontsize = 8)
			plt.yticks([0, 200, 400, 600, 800, 1000], labels = ('0', '200', '400', '600', '800', '1000'), fontsize = 6)

	plt.subplot(position = [0.81, 0.14, 0.17, 0.78])
	for n in range(5):
		p = c['d'+str(n+1)]
		npv = np.sum(p, axis = 1)/1000000000000 # trillions of dollars
		plt.boxplot(npv, positions = [n]) #trillions of dollars 
		
	plt.ylim(-10, 50)
	plt.xlim(-0.5, 4.5)
	plt.ylabel('Trillions of dollars', fontsize = 8)
	plt.yticks([0, 10, 20, 30, 40, 50], labels = ('0', '10', '20', '30', '40', '50'), fontsize = 6)
	plt.xticks([0, 1,2,3,4], labels = labs, fontsize = 6, rotation = 90)
	#plt.xticks([0, 1,2], labels = labs, fontsize = 6, rotation = 90)

	
	plt.savefig(figName, dpi=300)
	

makePlotbyCategory(g_discounted_stern, 'Figures/SternByScenario.png')
makePlotbyCategory(g_discounted_nordhaus, 'Figures/NordhausByScenario.png')
makePlotbyCategory(g_discounted_avg, 'Figures/AvgByScenario.png')
