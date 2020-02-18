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
#plt.style.use('style_sr15.mplstyle') # figure out what this is, this seems like a good thing to do
#import pyam 

# read csv file with scenarios 

df = pd.read_excel('./assessment/output/spm_sr15_figure3a_data_table.xlsx')
yrs = np.linspace(2000,2100,21)
#print(yrs)
#print(df.shape)
#print(df.columns)
#print(df[2035])
# interpolate emissions between years

# create a bunch of empty columns
for r in range(18):
	n = r+2
	lb = yrs[n]
	ub = yrs[n+1]
	for m in range(4):
		df[int(lb+m+1)] = np.zeros(len(df))

for r in range(18):
	n = r+2
	for m in range(len(df)):
		lb = yrs[n]
		if math.isnan(df[int(lb)].iloc[m])==True:
			lb = yrs[n-1]
		ub = yrs[n+1]
		if math.isnan(df[int(ub)].iloc[m])==True:
			ub = yrs[n+2]
		z = int(ub - lb - 1)
		for q in range(z): 
			df[int(lb+q+1)].iloc[m] = (df[int(lb)].iloc[m]*(z-q)+df[int(ub)].iloc[m]*(q+1))/(z+1)
a = np.linspace(2000, 2100, 101).astype(int) #tolist()
a = a.tolist()
#print(a)

df = df.reindex(columns = ['model', 'scenario', 'region', 'variable', 'unit', 'marker', 'category', 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035, 2036, 2037, 2038, 2039, 2040, 2041, 2042, 2043, 2044, 2045, 2046, 2047, 2048, 2049, 2050, 2051, 2052, 2053, 2054, 2055, 2056, 2057, 2058, 2059, 2060, 2061, 2062, 2063, 2064, 2065, 2066, 2067, 2068, 2069, 2070, 2071, 2072, 2073, 2074, 2075, 2076, 2077, 2078, 2079, 2080, 2081, 2082, 2083, 2084, 2085, 2086, 2087, 2088, 2089, 2090, 2091, 2092, 2093, 2094, 2095, 2096, 2097, 2098, 2099, 2100])

#print(df.shape)
#print(below15.shape)
#print(lowOS.shape)
#print(highOS.shape)

stern_delta = 0.001
nordhaus_delta = 0.015
global_growth = 0.03 # this is a guesstimate, would need to review actual historical data 

r = 0.03 # dummy discount rate
rs = np.ones(2100-2000+1)

for n in range(2100-2020):
	rs[n+21] = 1/((1+r)**(n+1))

#print(rs)
ccs_current = 600 #$/ton
ccs_est = 100 # $/ton
ccs_min = 80 # $/ton 

df2 = df 
df2['NPV'] = np.zeros(len(df2))
for n in range(2100-2020+1):
	for m in range(len(df2)):
		df2[n+2020].iloc[m] = min(0, df2[n+2020].iloc[m])*ccs_est*1000*1000*1000*rs[n]/(1000*1000*1000) # convert to billions of dollars
	df2.NPV = df2.NPV + df2[n+2020]

def makePlot(df, figName):
	df1 = df.drop(columns = ['model', 'scenario', 'region', 'variable', 'unit', 'marker', 'category', 'NPV'])
	d = df1.values
	d1 = np.delete(d, np.s_[0:20], axis = 1)
	plt.figure(figsize=(5,3.5))
	ax1 = plt.subplot(position=[0.15, 0.13, 0.6, 0.7])
	for m in range(len(d1)):
		plt.plot(np.linspace(2020, 2100, 81), -1*d1[m,:])
	plt.ylim(0,1000)
	plt.xlabel('Year')
	plt.ylabel('Billions of dollars')
	plt.title('Annual Costs (discounted)')

	npv = np.sum(d1, axis = 1)
	print(npv.shape)
	ax2 = plt.subplot(position = [0.85, 0.13, 0.1, 0.7])
	plt.boxplot(-1*npv/1000) #trillions of dollars 
	plt.ylim(0, 50)
	plt.title('NPV')
	plt.ylabel('Trillions of dollars')
	plt.savefig(figName, dpi=300)

makePlot(df2, 'testOut.png')


below15 = df2[df2.category == 'Below 1.5C']
lowOS = df2[df2.category == '1.5C low overshoot']
highOS = df2[df2.category == '1.5C high overshoot']

makePlot(below15, 'Below15.png')
makePlot(lowOS, 'LowOverShoot.png')
makePlot(highOS, 'HighOverShoot.png')
makePlot(df2, 'AllScenarios.png')
