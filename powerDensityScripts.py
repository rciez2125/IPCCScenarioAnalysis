import pandas as pd
import numpy as np
import csv
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import math

# specific capacities
c_gasoline = 12992/1000 # kWh/kg
c_ammonia = 6300/1000 #kWh/kg
c_hydrogen = 33600/1000 # kWh/kg

# liquid densities
rL_gasoline = 780 # kg/m3
rL_ammonia = 681.9 #kg/m3

tankerTruckVol = 11600 * 0.00378541 # m3 #liquid tanker trucks
tankerTrainVol = 30110 * 0.00378541 # m3 #liquid tank for a train car
numTrainTanks = 100 #keep this the same

trainSpeed = 40
roadSpeed = 65
shipSpeed = 19

truckShippingMaximum = 80000*0.453592 #kg
cryoH2truck = 600 # kg

gasolineColors 	= [0,0,0]
ammoniaColors 	= [1, 0.5, 0.3]
metalColors 	= [0.25, 0.4, 0.88]
hydrogenColors 	= [0.18, 0.55, 0.34]

a_g = 0.1
a_l = 0.3
a_s = 0.5

def fixedRate_Volume(textLabel, color, phase, distances, flowRate, r, c, ax, fs):
	energy_carried = flowRate*r*c
	if len(flowRate)>1:
		ax.fill_between(distances, np.ones(len(distances))*energy_carried[0], np.ones(len(distances))*energy_carried[1], color = color, alpha = phase, linewidth = 0)
		plt.text(distances[-1]/1000, (energy_carried[0] + energy_carried[1])/2, textLabel, fontsize = fs, verticalalignment = 'center')
	else:
		plt.plot(distances, np.ones(len(distances))*energy_carried, '-', color = color)
		plt.text(distances[-1], energy_carried, textLabel, fontsize = fs, verticalalignment = 'center')

def fixedRate_Mass(textLabel, color, phase, distances, flowRate, c, ax, fs):
	energy_carried = flowRate*c
	if len(flowRate)>1:
		ax.fill_between(distances, np.ones(len(distances))*energy_carried[0], np.ones(len(distances))*energy_carried[1], color = color, alpha = phase, linewidth = 0)
		plt.text(distances[-1]/1000, (energy_carried[0] + energy_carried[1])/2, textLabel, fontsize = fs, verticalalignment = 'center')
	else:
		plt.plot(distances, np.ones(len(distances))*energy_carried, '-', color = color)
		plt.text(distances[-1]/1000, energy_carried, textLabel, fontsize = fs, verticalalignment = 'top')

def fixedVolume(textLabel, color, phase, speed, distances, v, r, c, ax, fs):
	time = distances/speed
	energy_carried = v*r*c
	ax.fill_between(distances, energy_carried/time, energy_carried/(2*time), color = color, alpha = phase, linewidth = 0)
	plt.text(distances[-1], (energy_carried/time[-1] + energy_carried/(2*time)[-1])/2, textLabel, fontsize = fs, verticalalignment = 'center')

def fixedMass(textLabel, color, phase, speed, distances, m, c, ax, fs):
	time = distances/speed
	energy_carried = m*c
	resistance = m*9.81*0.01 # this will need to be fixed, the 0.01 is the rolling resistance of trucks
	work = distances*1609.34*resistance/3600000 #distances from miles to m, joules to kWh
	print(work, work/energy_carried)
	ax.fill_between(distances, energy_carried/time, energy_carried/(2*time), color = color, alpha = phase, linewidth = 0)
	plt.text(distances[-1], (energy_carried/time[-1] + energy_carried/(2*time)[-1])/2, textLabel, fontsize = fs, verticalalignment = 'center')

def fixedMassMetalHydrogen(textLabel, color, phase, speed, distances, m, mm_metal, moleRatio, c, ax, fs):
	molarMass_h2 = 2.01 #g/mol
	massH2 = m*molarMass_h2*moleRatio/mm_metal
	energy_carried = c*massH2
	time = distances/speed
	ax.fill_between(distances, energy_carried/time, energy_carried/(2*time), color = color, alpha = phase, linewidth = 0)
	plt.text(distances[-1], (energy_carried/time[-1] + energy_carried/(2*time)[-1])/2, textLabel, fontsize = fs, verticalalignment = 'center')

def plotPowerLines(textLabel, color, distances, powerCapacity, ax, fs):
	plt.plot(distances, np.ones(len(distances))*powerCapacity, '-', color = color)
	#plt.text(distances[-1]*1.1, powerCapacity, textLabel, fontsize = fs, verticalalignment = 'center')
	plt.text(np.mean(distances), powerCapacity*.9, textLabel, fontsize = fs, verticalalignment = 'top', horizontalAlignment = 'right', color = color)

def makePowerDensityPlot():
	plt.figure(figsize=(7,4))
	fs = 5
	ax = plt.subplot(1,1,1)
	distances = np.linspace(5,3000, 600)
	fixedVolume('Gasoline via Truck', gasolineColors, a_l, roadSpeed, distances, tankerTruckVol, rL_gasoline, c_gasoline, ax, fs)
	fixedVolume('Gasoline via Train', gasolineColors, a_l, trainSpeed, distances, tankerTrainVol*numTrainTanks, rL_gasoline, c_gasoline, ax, fs)

	fixedVolume('Liquid Ammonia via Train', ammoniaColors, a_l, trainSpeed, distances, tankerTrainVol*numTrainTanks, rL_ammonia, c_ammonia, ax, fs)
	fixedVolume('Liquid Ammonia via Truck', ammoniaColors, a_l, roadSpeed, distances, tankerTruckVol, rL_ammonia, c_ammonia, ax, fs)

	#fixedMassMetalHydrogen('Zn via Truck', metalColors, a_s, roadSpeed, distances, truckShippingMaximum, 65.38, 1, c_hydrogen, ax, fs)
	#fixedMassMetalHydrogen('Al via Truck', metalColors, a_s, roadSpeed, distances, truckShippingMaximum, 26.98, 1.5, c_hydrogen, ax, fs)
	fixedMass('Cryogenic H$_2$ via Truck', hydrogenColors, a_l, roadSpeed, distances, cryoH2truck, c_hydrogen, ax, fs)


	flow_rate = 360*60 #ft/min --> ft/hr
	area_big = math.pi*(4/12)**2
	area_small = math.pi*(3/12)**2
	flowRate = np.array([flow_rate*area_big, flow_rate*area_small])*0.0283168
	fixedRate_Volume('Liquid Ammonia Pipeline', ammoniaColors, a_l, [0.1,3000], flowRate, rL_ammonia, c_ammonia, ax, fs)

	flowRate = np.array([210]) #kg/hr
	fixedRate_Mass('Pressurized H$_2$ Pipeline$^*$', hydrogenColors, a_g, [0.1, 1500], flowRate, c_hydrogen, ax, fs)
	plt.text(2000, 5, '$^*$Cumulative Installed US H$_2$ Pipeline\n  Infrastructure = 1500 miles', fontsize = fs)

	

	# refueling, work on this later
	gas_pump = 10*3785.41*0.78*46.4*0.28*60/1000
	ax.plot([0.05,0.051], [gas_pump, gas_pump], '-', color = [0.5, 0.5, 0.5])
	plt.text(0.055, gas_pump, 'Gas Pump', fontsize = fs)

	powerLineColor = [0, 0, 0.7]

	plotPowerLines('500 kVAC-Double', powerLineColor, [700, 1180], 3000000, ax, fs)

	ax.plot([0.05, 0.05], [50, 350], '-', color = powerLineColor)
	plt.text(0.052, 100, 'DC Fast Charge', fontsize = fs, color = powerLineColor)

	#plotPowerLines('Typical Home Circuit Breaker', [0,0,0], [0.05, 0.05], 240*20/2000, ax, fs)
	
	ax.plot([0.05, 0.05], [240*20/2000, 120*20/2000], color = powerLineColor)
	plt.text(0.052, 150*20/2000, 'Typical Home Circuit Breaker', fontsize = fs, color = powerLineColor)

	plotPowerLines('City Electricity Distribution Network', powerLineColor, [0.1, 25], 3000, ax, fs)
	plotPowerLines('Rural Electricity Distribution Network', powerLineColor, [0.1, 25], 400, ax, fs)
	
	plotPowerLines('Low Voltage Distribution Grid', powerLineColor, [10, 90], 150000, ax, fs)
	plotPowerLines('Transmission Line', powerLineColor, [110, 1000], 150000, ax, fs)

	# working on supertanker
	distances = np.linspace(100,13000, 1000)
	energy_carried_low = 503410*870*44/3.6 # kWh
	energy_carried_high = 503410*920*44/3.6 #kWh
	speed = 19 # miles per hour
	time = distances/speed

	ax.fill_between(distances, energy_carried_low/(2*time), energy_carried_high/(time), color = 'k', alpha = 0.5, linewidth = 0)
	plt.text(distances[-1], (energy_carried_low/(2*time)[-1] + energy_carried_high/(time)[-1])/2, 'Crude Oil\nSupertanker', fontsize = fs, verticalalignment = 'center')

	# liquid ammonia
	#https://www.gbrx.com/manufacturing/north-america-rail/tank-cars/343k-anhydrous-ammonia-tank-car/

	ax.set_yscale('log')
	ax.set_xscale('log')
	ax.set_ylim(1,10000000000)
	ax.set_xlim(0.02,50000)
	ax.set_position([0.1, 0.15, 0.87, 0.8])
	plt.ylabel('Power [kW]')
	plt.xlabel('Distance Traveled [miles]')
	plt.savefig('Power Density.png', dpi = 300)
	#plt.show()
	plt.clf()
	plt.close()

def makeCarPlot():
	kgCO2_kgGasoline = 0.24
	vr = 300 # miles
	mpg = np.linspace(20, 100, 17)
	gpm = 1/mpg
	gals = vr/mpg
	mm_co2 = 12.01 + 2*15.99
	kg_fuel = gals*rL_gasoline*0.00378541
	kg_CO2 = kg_fuel*c_gasoline*0.24
	kg_psa = kg_CO2/0.1 + kg_CO2 + kg_fuel
	kg_li2co3 = (kg_CO2/mm_co2)*73.862 + kg_CO2 + kg_fuel
	kg_mgco3 = (kg_CO2/mm_co2)*84.285 + kg_CO2 + kg_fuel
	kg_caco3 = (kg_CO2/mm_co2)*100.058 + kg_CO2 + kg_fuel
	vol_fuel = gals*0.00378541
	vol_psa = (kg_CO2/0.1)/805 + gals*0.00378541
	vol_li2co3 = (kg_CO2/mm_co2*73.862)/2110 + gals*0.00378541
	vol_mgco3 = (kg_CO2/(12.01+2*15.99)*84.285)/2960 + gals*0.00378541
	vol_caco3 = (kg_CO2/(12.01+2*15.99)*100.058)/2710 + gals*0.00378541
	
	plt.figure(figsize=(4,6))
	fs = 5
	ax = plt.subplot(1,1,1)
	plt.plot(kg_psa, vol_psa, 'bx')
	plt.text(kg_psa[0]*1.02, vol_psa[0]*1.02, 'PSA', fontsize = fs, color = 'b')
	#plt.text(kg_psa[0]*1.03, vol_psa[0]/1.02, '20 mpg', fontsize = fs, verticalalignment = 'top')
	#plt.text(kg_psa[1]*1.03, vol_psa[1]/1.02, '25 mpg', fontsize = fs, verticalalignment = 'top')
	#plt.text(kg_psa[2]*1.03, vol_psa[2]/1.02, '30 mpg', fontsize = fs, verticalalignment = 'top')
	#ax.annotate(' ', xy = (kg_psa[1]*1.01, vol_psa[1]*1.01), xycoords = 'data', xytext=(kg_psa[0]*0.99, vol_psa[0]*0.99),
	#	arrowprops=dict(arrowstyle = '->'), fontsize = fs)
	#plt.text((kg_psa[0]+kg_psa[1])/2, (vol_psa[0]+vol_psa[1])/2.05, '+5 mpg', fontsize = fs, verticalalignment = 'center')
	#ax.annotate(' ', xy = (kg_psa[2]*1.01, vol_psa[2]*1.01), xycoords = 'data', xytext=(kg_psa[1]*0.99, vol_psa[1]*0.99),
	#	arrowprops=dict(arrowstyle = '->'), fontsize = fs)
	#plt.text((kg_psa[2]+kg_psa[1])/2, (vol_psa[2]+vol_psa[1])/2.05, '+5 mpg', fontsize = fs, verticalalignment = 'center')
	#plt.plot(kg_li2co3, vol_li2co3, 'ro')
	#plt.text(kg_li2co3[0]*1.02, vol_li2co3[0]*1.15, 'Li$_2$CO$_3$ (Ideal)', fontsize = fs, color = 'r', verticalalignment = 'bottom')
	#plt.plot(kg_mgco3, vol_mgco3, 'g*')
	#plt.text(kg_mgco3[0]*1.02, vol_mgco3[0]*0.8, 'MgCO$_3$ (Ideal)', fontsize = fs, color = 'g',verticalalignment = 'top')
	#plt.plot(kg_caco3, vol_caco3, 'y+')
	#plt.text(kg_caco3[0]*1.1, vol_caco3[0]*1, 'CaCO$_3$ (Ideal)', fontsize = fs, color = 'y', verticalalignment = 'center')
	#plt.plot([0,2000], [2.405, 2.405], '-k')
	#plt.text(1000, 2.41, 'Full volume of compact car', fontsize = fs)
	#plt.plot(540, 0.4, 'm*')
	#plt.text(550, 0.42, 'Tesla Battery Pack 2016', fontsize = fs, color = 'm')
	#plt.plot(kg_fuel, vol_fuel, 'c.')
	#plt.text(kg_fuel[-1]*1.05, vol_fuel[-1]*0.9, 'Gas Alone', fontsize = fs, color = 'c', verticalalignment = 'top')
	#plt.ylabel('Total Volume: Fuel + CO$_2$ Capture [m$^3$]')
	#plt.xlabel('Total Mass: Fuel + CO$_2$ Capture [kg]')
	#plt.show()
	plt.savefig('tryingagain.png', dpi=300)
	plt.clf()
	plt.close()


makePowerDensityPlot()