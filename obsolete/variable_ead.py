import numpy as np
import pandas as pd


def calculate_dynamic_return_periods(return_period_dict, num_years, increase_factor):
    years = np.linspace(0, num_years, num_years + 1)
    return_periods = {}
    for category, rp in return_period_dict.items():
        rp_new = rp / increase_factor[category]
        rps = np.interp(years, [0, num_years], [rp, rp_new])
        return_periods[category] = rps.tolist()

    return return_periods

return_period_dict = {
    '_H_': 10,
    '_M_': 100,
    '_L_': 200
}

increase_factor = {
    '_H_': 1.5,
    '_M_': 1.8,
    '_L_': 2
}

num_years = 100

return_periods = calculate_dynamic_return_periods(return_period_dict, num_years, increase_factor)

aggregated_output = {
    '_H_': (10, 20),
    '_M_': (30, 40),
    '_L_': (50, 60)
}

aggregated_df = pd.DataFrame.from_dict(aggregated_output, orient='index', columns=['Total Damage Lower Bound', 'Total Damage Upper Bound'])


return_period_dict = {}
return_period_dict['DERP'] = return_periods #{ #TODO: make generic
#     '_H_': [10, 5],
#     '_M_': [100, 50],
#     '_L_': [200, 100]
# }

# add the return period column to aggregated_df
aggregated_df['Return Period'] = [return_period_dict['DERP'][index] for index in aggregated_df.index]
print(aggregated_df)

# sort the DataFrame by return period
aggregated_df = aggregated_df.sort_values('Return Period', ascending=True)

# Calculate the probability of each return period
aggregated_df['Probability'] = [[1 / x for x in i] for i in aggregated_df['Return Period']]

# probabilities = aggregated_df['Probability'].tolist()
probabilities = aggregated_df['Probability']
dmgs = []

for ts in range(len(probabilities.iloc[0])):    
    dmgs_l = []
    dmgs_u = []

    for rp in range(len(probabilities)-1):
        d_rp= probabilities.iloc[rp][ts] - probabilities.iloc[rp + 1][ts]
        mean_damage_l = 0.5 * (aggregated_df['Total Damage Lower Bound'].iloc[rp] + aggregated_df['Total Damage Lower Bound'].iloc[rp + 1])
        mean_damage_u = 0.5 * (aggregated_df['Total Damage Upper Bound'].iloc[rp] + aggregated_df['Total Damage Upper Bound'].iloc[rp + 1])
        dmgs_l.append(d_rp * mean_damage_l)
        dmgs_u.append(d_rp * mean_damage_u)
    
    # adding the portion of damages corresponding to p=0 to p=1/highest return period
    # This calculation considers the damage for return periods higher than the highest return period the same as the highest return period
    d0_rp = probabilities.iloc[-1][ts]
    mean_damage_l0 = max(aggregated_df['Total Damage Lower Bound'])
    mean_damage_u0 = max(aggregated_df['Total Damage Upper Bound'])
    dmgs_l.append(d0_rp * mean_damage_l0)
    dmgs_u.append(d0_rp * mean_damage_u0)

    # This calculation considers that no assets are damaged at a return period of 4 years
    d_end_rp = (1/4)-probabilities.iloc[0][ts]
    mean_damage_l_end = 0.5 * min(aggregated_df['Total Damage Lower Bound'])
    mean_damage_u_end = 0.5 * min(aggregated_df['Total Damage Upper Bound'])
    dmgs_l.append(d_end_rp * mean_damage_l_end)
    dmgs_u.append(d_end_rp * mean_damage_u_end)

    dmgs.append((sum(dmgs_l), sum(dmgs_u)))
    

ead_by_ts = pd.DataFrame(dmgs, columns=['Total Damage Lower Bound', 'Total Damage Upper Bound'])

import matplotlib.pyplot as plt
plt.fill_between(ead_by_ts.index, ead_by_ts['Total Damage Lower Bound'], ead_by_ts['Total Damage Upper Bound'], alpha=0.3, color='red')
plt.xlabel('Years from baseline')
plt.ylabel('EAD (euros)')
plt.legend()
plt.ylim(0)  # Set y-axis lower limit to 0
plt.show()


probabilities_df = pd.DataFrame(probabilities.tolist())

fig, ax = plt.subplots()

for ts in range(len(probabilities_df.columns)):
    if ts % 100 == 0:  # Only plot for years ending in 0
        # Extend the data
        x = [1,0.25] + list(probabilities_df.iloc[:, ts]) + [0.001]
        y_lower = [0,0] + list(aggregated_df['Total Damage Lower Bound']) + [aggregated_df['Total Damage Lower Bound'].max()]
        y_upper = [0,0] + list(aggregated_df['Total Damage Upper Bound']) + [aggregated_df['Total Damage Upper Bound'].max()]

        # Plot the data
        ax.plot(x, y_lower, label=f'TS {ts+1} Lower Bound', color='grey')
        ax.fill_between(x, y_lower, y_upper, alpha=0.3)
        ax.plot(x, y_upper, label=f'TS {ts+1} Upper Bound', color='grey')

ax.set_xscale('log')
ax.set_xlim(0.001, 1)
ax.xaxis.set_major_formatter(plt.ScalarFormatter())
ax.xaxis.set_minor_formatter(plt.NullFormatter())
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'1/{int(1/x)}'))
ax.set_xlabel('Probability')
ax.set_ylabel('Total Damage')
plt.show()
