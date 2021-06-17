'''
exploratory  data analysis: deficit between ecological footprint and biocapacity in the world.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#World_pc
ef_bc_pc = pd.read_csv('./../data/NAME')

bc_pc = ef_bc_pc[ef_bc_pc['record'] == 'BiocapPerCap']
ef_pc = ef_bc_pc[ef_bc_pc['record'] == 'EFConsPerCap']

ef_pc['val-carbon'] = ef_pc['value'] - ef_pc['carbon']

#Plot total:
x_val = bc_pc['year']
leg = ['ecological footprint', 'biocapacity']
x_ticks = np.arange(1961,2020,2)

x_label = [1961, '', 1965, '', 1969, '', 1973, '', 1977, '', 1981,
       '', 1985, '', 1989, '', 1993, '', 1997, '', 2001, '',
       2005, '', 2009, '', 2013, '', 2017, '']

y_ticks = np.arange(1.4,3.6,0.2)

plt.figure(figsize=(10,8))
plt.plot(x_val, ef_pc['value'], c='red')
plt.plot(x_val, bc_pc['value'], c='green')
plt.axvline(x=1970, c='darkred')
plt.text(1971, 1.8, '1970', fontdict={'fontsize': 15, 'color': 'darkred', 'weight':'bold'})
plt.fill_between(x_val[0:10], ef_pc['value'][0:10], bc_pc['value'][0:10], color='green', alpha =0.3)
plt.fill_between(x_val[9::], ef_pc['value'][9::], bc_pc['value'][9::], color='red', alpha =0.3)
plt.legend(leg, fontsize=12, loc='best')
plt.xticks(rotation = 45, fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('years', fontdict={'fontsize': 15})
plt.ylabel('global hectares per person', fontdict={'fontsize': 15})
ax = plt.subplot()
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_label)
ax.set_yticks(y_ticks)
plt.title('World: 1961 - 2017', fontdict={'fontsize': 20})

plt.savefig('world_pc.jpg')


#World_total
ef_bc_tot = pd.read_csv('./../data/NAME')

#divid df in bc and ef
bc_tot = ef_bc_tot[ef_bc_tot['record'] == 'BiocapTotGHA']
ef_tot = ef_bc_tot[ef_bc_tot['record'] == 'EFConsTotGHA']

ef_tot['val-carbon'] = ef_tot['value'] - ef_tot['carbon']

x_val = bc_tot['year']
leg = ['ecological footprint', 'biocapacity']

x_ticks = np.arange(1961,2020,2)

x_label = [1961, '', 1965, '', 1969, '', 1973, '', 1977, '', 1981,
       '', 1985, '', 1989, '', 1993, '', 1997, '', 2001, '',
       2005, '', 2009, '', 2013, '', 2017, '']

y_ticks = np.arange(0.6,2.5,0.2)

plt.figure(figsize=(10,8))
plt.plot(x_val, ef_tot['value']/1e+10, c='red')
plt.plot(x_val, bc_tot['value']/1e+10, c='green')
plt.axvline(1970, c='darkred')
plt.text(1971, 1.4, '1970', fontdict={'fontsize': 15, 'color': 'darkred', 'weight':'bold'})
plt.fill_between(x_val[0:10], ef_tot['value'][0:10]/1e+10, bc_tot['value'][0:10]/1e+10, color='green', alpha =0.3)
plt.fill_between(x_val[9::], ef_tot['value'][9::]/1e+10, bc_tot['value'][9::]/1e+10, color='red', alpha =0.3)
plt.legend(leg, fontsize=12, loc='lower right')
plt.xticks(rotation = 45, fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('years', fontdict={'fontsize': 15})
plt.ylabel('*10¹⁰ global hectares*', fontdict={'fontsize': 15})
ax = plt.subplot()
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_label)
ax.set_yticks(y_ticks)
plt.title('World: 1961 - 2017', fontdict={'fontsize': 20})
plt.savefig('world_tot.jpg')


#Plot Carbon:
leg = ['ef of carbon','ef without carbon']
x_ticks = np.arange(1961,2020,2)
y_ticks = np.arange(0.5,4.4,0.4)

plt.figure(figsize=(10,8))
plt.plot(x_val, ef_tot['carbon']/1e+09, c='firebrick', linewidth=2 )
plt.plot(x_val, ef_tot['val-carbon']/1e+09, c='black', linewidth=2 )
plt.plot(x_val, ef_tot['value']/1e+09, c='grey', linewidth=1 )
plt.plot(x_val, bc_tot['value']/1e+09, c='grey', linewidth=1 )
plt.fill_between(x_val, ef_tot['value']/1e+09, bc_tot['value']/1e+09, color='grey', alpha =0.3)
plt.legend(leg, fontsize=12)
plt.xticks(rotation = 45, fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('years', fontdict={'fontsize': 15})
plt.ylabel('*10⁹ global hectares', fontdict={'fontsize': 15})
ax = plt.subplot()
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_label)
ax.set_yticks(y_ticks)
plt.title('Europe: 1961 - 2017', fontdict={'fontsize': 20})
plt.savefig('world_tot_carbon.jpg')


#Plot all features:
leg = ['cropLand', 'grazingLand', 'forestLand','fishingGround', 'builtupLand', 'carbon', 'ecological footprint total']
x_ticks = np.arange(1961,2020,2)
y_ticks = np.arange(0.5,4.4,0.4)

plt.figure(figsize=(10,8))
plt.plot(x_val, ef_tot['cropLand']/1e+09, c='black', linewidth=1.5 )
plt.plot(x_val, ef_tot['grazingLand']/1e+09, c='grey', linewidth=1.5 )
plt.plot(x_val, ef_tot['forestLand']/1e+09, c='firebrick', linewidth=1.5 )
plt.plot(x_val, ef_tot['fishingGround']/1e+09, c='darkorange', linewidth=1.5)
plt.plot(x_val, ef_tot['builtupLand']/1e+09, c='green', linewidth=1.5 )
plt.plot(x_val, ef_tot['carbon']/1e+09, c='blue', linewidth=1.5 )
#plt.plot(x_val, ef_tot['value']/1e+09, c='grey', linewidth=2 )
plt.legend(leg, fontsize=12)
plt.xticks(rotation = 45, fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('years', fontdict={'fontsize': 15})
plt.ylabel('*10⁹ global hectares', fontdict={'fontsize': 15})
ax = plt.subplot()
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_label)
plt.title('World: 1961 - 2017', fontdict={'fontsize': 20})
plt.savefig('../world_tot_features.jpg')



