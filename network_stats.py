import geopandas as gpd
import matplotlib.pyplot as plt

def create_histogram(dataframe, variable, cut_offs):
    counts=[]
    str_mid=[]

    for i in range(len(cut_offs)):
        co=cut_offs[i]
        if i==0:
            filt_fraction=(dataframe[variable]<co)
            str_start='0 - '+str(co-0.001)
        elif i==len(cut_offs)-1:
            filt_fraction=(dataframe[variable]>=cut_offs[i-1]) & (dataframe[variable]<cut_offs[i])
            counts.append(len(dataframe.loc[filt_fraction]))
            str_mid.append(str(cut_offs[i-1])+' - '+str(co-0.001))
            filt_fraction=(dataframe[variable]>=co)
            str_end=str(co)+'+'
        else:
            filt_fraction=(dataframe[variable]>=cut_offs[i-1]) & (dataframe[variable]<cut_offs[i])
            str_mid.append(str(cut_offs[i-1])+' - '+str(co-0.001))

        counts.append(len(dataframe.loc[filt_fraction]))

    str_cut_offs=[str_start]+(str_mid)+[str_end]
    
    return [str_cut_offs, [(x/sum(counts)) for x in counts]]

def find_adjacents(dataframe,attribute):
    filt_att=(dataframe[attribute]=='yes')
    adjacents=dataframe.loc[filt_att]['u'].to_list()+dataframe.loc[filt_att]['v'].to_list()
    filt_hits=(dataframe['u'].isin(adjacents) | dataframe['v'].isin(adjacents)) & (dataframe[attribute]!='yes')
    df_out=dataframe.loc[filt_hits]
    return df_out #Boolean series indicating if they are adjacent to features with the indicated attribute


# Open dataframe of flooded graph
# filename=r'C:\Users\peregrin\osm\RA2CE_inputs\static\output_graph\jrc_base_graph_hazard_edges.gpkg'
# var_name='RPEFAS'
filename=r'C:\Users\peregrin\osm\RA2CE_inputs\static\output_graph\base_graph_hazard_edges.gpkg'
var_name='RPRW'
rail_track_raw = gpd.read_file(filename)
# rail_track_raw.rename(columns={"length": "length_original"})
rail_track_gdf=rail_track_raw.to_crs(crs=3857)




rail_track_gdf['length']=rail_track_gdf.length
flooded_extent=rail_track_gdf.length * rail_track_gdf.RPRW_fr
rail_track_gdf['length_flooded']=flooded_extent.where(rail_track_gdf.RPRW_fr>0, other=0)

bridge_adjacent_sections=find_adjacents(rail_track_gdf,'bridge')

# filt_fraction=(rail_track_gdf[var_name+'_fr']>=0.05)
filt_not_flooded=(rail_track_gdf[var_name+'_fr']==0)|(rail_track_gdf['bridge']=='yes')

filt_flooded=(rail_track_gdf[var_name+'_fr']>0)
filt_bridge=(rail_track_gdf['bridge']=='yes')
filt_tunnel=(rail_track_gdf['tunnel']=='yes')

# Initial stats for data
railways_nr=len(rail_track_gdf['length'])
railways_nr_flooded=len(rail_track_gdf.loc[filt_flooded])
railways_length=sum(rail_track_gdf['length'])
railways_length_flooded=sum(rail_track_gdf.loc[filt_flooded]['length_flooded'])
railways_length_flooded_fullsec=sum(rail_track_gdf.loc[filt_flooded]['length'])

bridges_nr=len(rail_track_gdf.loc[filt_bridge])
bridges_nr_flooded=len(rail_track_gdf.loc[filt_bridge & filt_flooded])
bridges_length=sum(rail_track_gdf.loc[filt_bridge]['length'])
bridges_length_flooded=sum(rail_track_gdf.loc[filt_bridge & filt_flooded]['length_flooded'])
tunnels_nr=len(rail_track_gdf.loc[filt_tunnel])
tunnels_nr_flooded=len(rail_track_gdf.loc[filt_tunnel & filt_flooded])
tunnels_length=sum(rail_track_gdf.loc[filt_tunnel]['length'])
tunnels_length_flooded=sum(rail_track_gdf.loc[filt_tunnel & filt_flooded]['length_flooded'])
track_sections_nr=len(rail_track_gdf.loc[~(filt_bridge | filt_tunnel)])
track_sections_nr_flooded=len(rail_track_gdf.loc[~(filt_bridge | filt_tunnel) & filt_flooded])
track_sections_length=sum(rail_track_gdf.loc[~(filt_bridge | filt_tunnel)]['length'])
track_sections_length_flooded=sum(rail_track_gdf.loc[~(filt_bridge | filt_tunnel) & filt_flooded]['length_flooded'])

# Reporting
print('-- OVERALL --')
print(f"Railways total length [km]: {railways_length/1000:.2f},", '[', railways_nr,' sections ]')
print(f"Railways flooded length [km]: {railways_length_flooded/1000:.2f},", '[', railways_nr_flooded,' sections ]')
print(f'Fraction length flooded: {railways_length_flooded/railways_length:.3f}, [',railways_nr_flooded,'/',railways_nr,' sections ]')


print("\n- Bridges -")
print(f"Bridges total length [km]: {bridges_length/1000:.2f},", '[', bridges_nr,' sections ]')
print(f"Bridges flooded length [km]: {bridges_length_flooded/1000:.2f},", '[', bridges_nr_flooded,' sections ]')
print(f'Fraction length flooded: {bridges_length_flooded/bridges_length:.3f}, [',bridges_nr_flooded,'/',bridges_nr,' sections ]')

print("\n- Tunnels -")
print(f"Tunnels total length [km]: {tunnels_length/1000:.2f},", '[', tunnels_nr,' sections ]')
print(f"Tunnels flooded length [km]: {tunnels_length_flooded/1000:.2f},", '[', tunnels_nr_flooded,' sections ]')
print(f'Fraction length flooded: {tunnels_length_flooded/tunnels_length:.3f}, [',tunnels_nr_flooded,'/',tunnels_nr,' sections ]')

print("\n- Track sections -")
print(f"Track sections total length [km]: {track_sections_length/1000:.2f},", '[', track_sections_nr,' sections ]')
print(f"Track sections flooded length [km]: {track_sections_length_flooded/1000:.2f},", '[', track_sections_nr_flooded,' sections ]')
print(f'Fraction length flooded: {track_sections_length_flooded/track_sections_length:.3f}, [',track_sections_nr_flooded,'/',track_sections_nr,' sections ]')
print('\n')
# filt_fraction=(rail_track_flooded_gdf[var_name+'_fr']>=0.05) 
# filt_bridge=(rail_track_flooded_gdf['bridge']=='yes')
# filt_tunnel=(rail_track_flooded_gdf['tunnel']=='yes')


rail_track_flooded_gdf=rail_track_gdf.loc[~filt_not_flooded]
cut_offs=[0.05, 0.2, 0.4, 0.6, 0.8, 0.95]
var_x=var_name+'_fr'


arguments=create_histogram(rail_track_gdf.loc[~(filt_bridge | filt_tunnel) & filt_flooded],var_x,cut_offs)

plt.rcParams.update({'font.size': 8})
plt.bar(arguments[0], arguments[1], width=0.7)
plt.title('Fraction flooded of affected sections \n (excluding bridges and tunnels')
plt.show()



# cut_offs=[0.5, 1, 2, 3, 4]
# var_x='RPRW_me'

# arguments=create_histogram(rail_track_flooded_gdf,var_x,cut_offs)
# plt.bar(arguments [0], arguments[1])
# plt.show()