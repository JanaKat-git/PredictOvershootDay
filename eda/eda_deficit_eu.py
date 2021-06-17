'''
exploratory  data analysis: deficit between ecological footprint and biocapacity in europe.
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#EU_per person
ef_bc_pc = pd.read_csv('./../data/NAME')

#divid df in bc and ef
bc_pc = ef_bc_pc[ef_bc_pc['record'] == 'BiocapPerCap']
ef_pc = ef_bc_pc[ef_bc_pc['record'] == 'EFConsPerCap']

ef_pc['val-carbon'] = ef_pc['value'] - ef_pc['carbon']

#EU_pc:
x_val = bc_pc['year']
leg = ['ecological footprint', 'biocapacity']
x_ticks = np.arange(1961,2020,2)

x_label = [1961, '', 1965, '', 1969, '', 1973, '', 1977, '', 1981,
       '', 1985, '', 1989, '', 1993, '', 1997, '', 2001, '',
       2005, '', 2009, '', 2013, '', 2017, '']

y_ticks = np.arange(1.6,6.0,0.4)

plt.figure(figsize=(10,8))
plt.plot(x_val, ef_pc['value'], c='red')
plt.plot(x_val, bc_pc['value'], c='green')
plt.fill_between(x_val, ef_pc['value'], bc_pc['value'], color='red', alpha =0.3)
plt.legend(leg, fontsize=12, loc='lower right')
plt.xticks(rotation = 45, fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('years', fontdict={'fontsize': 15})
plt.ylabel('global hectars per person', fontdict={'fontsize': 15})
ax = plt.subplot()
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_label)
ax.set_yticks(y_ticks)
plt.title('Europe: 1961 - 2017', fontdict={'fontsize': 20})

plt.savefig('eu_pc.jpg')


#Plot Carbon:
ef_pc['val-carbon'] = ef_pc['value'] - ef_pc['carbon']

leg = ['ef of carbon','ef without carbon']
x_ticks = np.arange(1961,2020,2)
y_ticks = np.arange(1.5,7.0,0.4)

plt.figure(figsize=(10,8))
plt.plot(x_val, ef_pc['carbon'], c='firebrick', linewidth=2 )
plt.plot(x_val, ef_pc['val-carbon'], c='black', linewidth=2 )
plt.plot(x_val, ef_pc['value'], c='grey', linewidth=1 )
plt.plot(x_val, bc_pc['value'], c='grey', linewidth=1 )
plt.fill_between(x_val, ef_pc['value'], bc_pc['value'], color='grey', alpha =0.3)
plt.legend(leg, fontsize=12)
plt.xticks(rotation = 45, fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('years', fontdict={'fontsize': 15})
plt.ylabel('global hectars per person', fontdict={'fontsize': 15})
ax = plt.subplot()
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_label)
ax.set_yticks(y_ticks)
plt.title('Europe: 1961 - 2017', fontdict={'fontsize': 20})

plt.savefig('eu_pc_carbon.jpg')


#EU_total
ef_bc_tot = pd.read_csv('./../data/NAME')

#divid df in bc and ef
bc_tot = ef_bc_tot[ef_bc_tot['record'] == 'BiocapTotGHA']
ef_tot = ef_bc_tot[ef_bc_tot['record'] == 'EFConsTotGHA']

ef_tot['val-carbon'] = ef_tot['value'] - ef_tot['carbon']

#Plot total:
x_val = bc_tot['year']
leg = ['ecological footprint', 'biocapacity']
x_ticks = np.arange(1961,2020,2)

x_label = [1961, '', 1965, '', 1969, '', 1973, '', 1977, '', 1981,
       '', 1985, '', 1989, '', 1993, '', 1997, '', 2001, '',
       2005, '', 2009, '', 2013, '', 2017, '']

y_ticks = np.arange(0.6,3.8,0.4)

plt.figure(figsize=(10,8))
plt.plot(x_val, ef_tot['value']/1e+09, c='red')
plt.plot(x_val, bc_tot['value']/1e+09, c='green')
plt.fill_between(x_val, ef_tot['value']/1e+09, bc_tot['value']/1e+09, color='red', alpha =0.3)
plt.legend(leg, fontsize=12, loc='lower right')
plt.xticks(rotation = 45, fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('years', fontdict={'fontsize': 15})
plt.ylabel('*10⁹ global hectars', fontdict={'fontsize': 15})
ax = plt.subplot()
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_label)
ax.set_yticks(y_ticks)
plt.title('Europe: 1961 - 2017', fontdict={'fontsize': 20})
plt.savefig('eu_tot.jpg')


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
plt.ylabel('*10⁹ global hectars', fontdict={'fontsize': 15})
ax = plt.subplot()
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_label)
ax.set_yticks(y_ticks)
plt.title('Europe: 1961 - 2017', fontdict={'fontsize': 20})
plt.savefig('eu_tot_carbon.jpg')








