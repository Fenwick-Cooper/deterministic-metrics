# Load diagnostic data and make plots

import numpy as np
import cartopy.io.shapereader as shpreader
from cartopy.feature import ShapelyFeature
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from datetime import datetime

import load_daily_mean_data as ldmd
import data_utils as du


shapefile_name = "/Users/cooperf/Documents/WFP/Operational/operational-cGAN/show-forecasts/show_forecasts/shapes/Kenya_shapes/Kenya_region.shp"

# Prints some basic statistics
def print_basic_stats(dm):

    print(f"{dm["truth_source"]}")
    print(f"Mean               : {dm['mean_rgn_truth'][0]} mm/day")
    print(f"Standard deviation : {dm['std_rgn_truth'][0]} mm/day")
    print()

    print(f"{dm["forecast_source"]}")
    print(f"Mean               : {dm['mean_rgn_forecast'][0]} mm/day")
    print(f"Standard deviation : {dm['std_rgn_forecast'][0]} mm/day")
    print()

    print(f"{dm["forecast_source"]} with respect to {dm["truth_source"]}")
    print(f"Bias                : {dm['bias_rgn'][0]} mm/day")
    print(f"RMSE                : {dm['RMSE_rgn'][0]} mm/day")
    print(f"MAE                 : {dm['MAE_rgn'][0]} mm/day")
    print(f"R2                  : {dm['R2_rgn'][0]}")
    print(f"Anomaly correlation : {dm['anomaly_corr_rgn'][0]}")
    print()

    print(f"Probability of detection (POD) threshold : {dm['POD_threshold'][0]}")
    print(f"Probability of detection (POD)           : {dm['POD_rgn'][0]}")
    print(f"probability of false alarm (POFA)        : {dm['POFA_rgn'][0]}")


# XXX Plots as a function of lead time. To be implemented here.
# # Plot the mean rainfall as a function of lead time
# for idx in range(len(dm_list)):
#     plt.plot(lead_times, dm_list[idx]['mean_rgn_forecast'],'o-',label=dm_list[idx]["forecast_source"])
# # Plot the truth
# # XXX Assumes that all the truths are the same but they are not (eg. CHIRPS vs IMERG)
# plt.plot([lead_times[0],lead_times[-1]],[dm_list[0]['mean_rgn_truth'],dm_list[0]['mean_rgn_truth']],
#          'k--',label=dm_list[idx]["truth_source"])
# plt.plot([lead_times[0],lead_times[-1]],
#          [np.mean(precip_clim_ICPAC[month-1,:,:]),np.mean(precip_clim_ICPAC[month-1,:,:])],
#          'r--',label='ICPAC climatology')
# plt.grid()
# plt.xlabel('Forecast lead time (days)')
# plt.ylabel('Mean (mm/day)')
# plt.title(f'{d.strftime("%B %Y")} {region}')
# plt.legend()
# plt.show()

# # Plot the standard deviation of rainfall
# plt.plot(lead_times, dm_IMERG_IFS['std_rgn_forecast'],'o-',label='IFS')
# plt.plot(lead_times, dm_IMERG_cGAN['std_rgn_forecast'],'o-',label='cGAN')
# plt.plot(lead_times, dm_IMERG_KMD_WRF['std_rgn_forecast'],'o-',label='KMD_WRF')
# plt.plot([lead_times[0],lead_times[-1]],[dm_IMERG_IFS['std_rgn_truth'],dm_IMERG_IFS['std_rgn_truth']],'k--',label='IMERG')
# plt.grid()
# plt.xlabel('Forecast lead time (days)')
# plt.ylabel('Standard deviation (mm/day)')
# plt.title(f'{d.strftime("%B %Y")} {region}')
# plt.legend()
# plt.show()

# # Plot the rainfall bias
# plt.plot(lead_times, dm_IMERG_IFS['bias_rgn'],'o-',label='IFS')
# plt.plot(lead_times, dm_IMERG_cGAN['bias_rgn'],'o-',label='cGAN')
# plt.plot(lead_times, dm_IMERG_KMD_WRF['bias_rgn'],'o-',label='KMD_WRF')
# plt.grid()
# plt.xlabel('Forecast lead time (days)')
# plt.ylabel('Bias (mm/day)')
# plt.title(f'{d.strftime("%B %Y")} {region}')
# plt.legend()
# plt.show()

# # Plot the rainfall RMSE
# plt.plot(lead_times, np.sqrt(dm_IMERG_IFS['MSE_rgn']),'o-',label='IFS')
# plt.plot(lead_times, np.sqrt(dm_IMERG_cGAN['MSE_rgn']),'o-',label='cGAN')
# plt.plot(lead_times, dm_IMERG_KMD_WRF['MSE_rgn'],'o-',label='KMD_WRF')
# plt.grid()
# plt.xlabel('Forecast lead time (days)')
# plt.ylabel('RMSE (mm/day)')
# plt.title(f'{d.strftime("%B %Y")} {region}')
# plt.legend()
# plt.show()

# # Plot the rainfall MAE
# plt.plot(lead_times, dm_IMERG_IFS['MAE_rgn'],'o-',label='IFS')
# plt.plot(lead_times, dm_IMERG_cGAN['MAE_rgn'],'o-',label='cGAN')
# plt.plot(lead_times, dm_IMERG_KMD_WRF['MAE_rgn'],'o-',label='KMD_WRF')
# plt.grid()
# plt.xlabel('Forecast lead time (days)')
# plt.ylabel('MAE (mm/day)')
# plt.title(f'{d.strftime("%B %Y")} {region}')
# plt.legend()
# plt.show()

# # Plot the rainfall coefficient of determination (R2)
# plt.plot(lead_times, dm_IMERG_IFS['R2_rgn'],'o-',label='IFS')
# plt.plot(lead_times, dm_IMERG_cGAN['R2_rgn'],'o-',label='cGAN')
# plt.plot(lead_times, dm_IMERG_KMD_WRF['R2_rgn'],'o-',label='KMD_WRF')
# plt.grid()
# plt.xlabel('Forecast lead time (days)')
# plt.ylabel('R2')
# plt.title(f'{d.strftime("%B %Y")} {region}')
# plt.legend()
# plt.show()

# # Plot the rainfall anomaly correlation
# plt.plot(lead_times, dm_IMERG_IFS['anomaly_corr_rgn'],'o-',label='IFS')
# plt.plot(lead_times, dm_IMERG_cGAN['anomaly_corr_rgn'],'o-',label='cGAN')
# plt.plot(lead_times, dm_IMERG_KMD_WRF['anomaly_corr_rgn'],'o-',label='KMD_WRF')
# plt.grid()
# plt.xlabel('Forecast lead time (days)')
# plt.ylabel('Anomaly correlation')
# plt.title(f'{d.strftime("%B %Y")} {region}')
# plt.legend()
# plt.show()

# # Plot the probability of detection
# plt.plot(lead_times, dm_IMERG_IFS['POD_rgn'],'o-',label='IFS')
# plt.plot(lead_times, dm_IMERG_cGAN['POD_rgn'],'o-',label='cGAN')
# plt.plot(lead_times, dm_IMERG_KMD_WRF['POD_rgn'],'o-',label='KMD_WRF')
# plt.grid()
# plt.xlabel('Forecast lead time (days)')
# plt.ylabel(f'Probability of detection at {dm_IMERG_IFS['POD_threshold'][0]} mm/day')
# plt.title(f'{d.strftime("%B %Y")} {region}')
# plt.legend()
# plt.show()

# # Plot the probability of false alarm
# plt.plot(lead_times, dm_IMERG_IFS['POFA_rgn'],'o-',label='IFS')
# plt.plot(lead_times, dm_IMERG_cGAN['POFA_rgn'],'o-',label='cGAN')
# plt.plot(lead_times, dm_IMERG_KMD_WRF['POFA_rgn'],'o-',label='KMD_WRF')
# plt.grid()
# plt.xlabel('Forecast lead time (days)')
# plt.ylabel(f'Probability of false alarm at {dm_IMERG_IFS['POD_threshold'][0]} mm/day')
# plt.title(f'{d.strftime("%B %Y")} {region}')
# plt.legend()
# plt.show()


# Maps with the statistic taken in the time direction over the month
# XXX Specify the style (eg 'KMD') or allow specification of vmin and vmax
def plot_truth_and_forecast_maps(dm_list,           # A list of the deterministic metric dictionaries
                                 region,            # XXX get from data
                                 d,                 # XXX get from data
                                 lead_times=[1.25],   # XXX get from data
                                 statistic='mean',  # Can be "mean" or "std"
                                 lead_time_idx=0,   # Which lead time to plot
                                 file_name=None):
    
    # Check the lead time
    key = f"{statistic}_forecast"
    if (lead_time_idx > dm_list[0][key].shape[0] - 1):
        print("ERROR: Lead time idx {lead_time_idx} does not exist.")
        return

    # Which country are we looking at. Can be:
    # "South Sudan","Rwanda","Burundi","Djibouti","Eritrea",
    # "Ethiopia","Sudan","Somalia","Tanzania","Kenya","Uganda"
    latitude, longitude, mask = ldmd.load_country_mask(region)

    # Set the contour plot style
    plot_levels, plot_colours = du.get_contour_levels('KMD')
    plot_norm = 24  # We are plotting mm/day and plot_norm == 1 corresponds to mm/h

    # Load the border shapefile
    reader = shpreader.Reader(shapefile_name)
    borders_feature = ShapelyFeature(reader.geometries(), ccrs.PlateCarree(), facecolor='none')

    # Define the figure and each axes for the rows and columns
    fig, axs = plt.subplots(nrows=1, ncols=3, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(16,6.5))

    # axs is a 2 dimensional array of `GeoAxes`. Flatten it into a 1-D array
    axs=axs.flatten()

    vmin = np.min(dm_list[0][f"{statistic}_truth"])
    vmax = np.max(dm_list[0][f"{statistic}_truth"])

    ax=axs[0]  # First plot (left)
    ax.add_feature(borders_feature, linewidth=0.5, edgecolor='black')  # The borders
    key = f"{statistic}_truth"
    ax.pcolormesh(longitude,latitude,dm_list[0][key],
                #colors=plot_colours, levels=plot_levels*plot_norm,
                vmin=vmin, vmax=vmax, 
                transform=ccrs.PlateCarree())
    ax.set_title(dm_list[0]["truth_source"])
    ax.set_frame_on(False)

    for idx in range(len(dm_list)):
        ax=axs[idx+1]
        ax.add_feature(borders_feature, linewidth=0.5, edgecolor='black')  # The borders
        key = f"{statistic}_forecast"
        c = ax.pcolormesh(longitude,latitude,dm_list[idx][key][lead_time_idx,:,:],
                        #colors=plot_colours, levels=plot_levels*plot_norm,
                        vmin=vmin, vmax=vmax, 
                        transform=ccrs.PlateCarree())
        ax.set_title(dm_list[idx]["forecast_source"])
        ax.set_frame_on(False)

    # # Don't show axes without plots
    # num_plots = 4
    # num_rows = 2
    # num_cols = 3
    # for ax_idx in range(num_plots, num_rows*num_cols):
    #     axs[ax_idx].set_axis_off()

    # Adjust the location of the subplots on the page to make room for the colorbar
    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                        wspace=0.02, hspace=0.02)

    # Add a colorbar axis at the bottom of the graph
    cbar_ax = fig.add_axes([0.81, 0.215, 0.015, 0.57])

    # Draw the colorbar
    cb=fig.colorbar(c, cax=cbar_ax,orientation='vertical')
    #cb.ax.tick_params(labelsize=10)
    #cb_labels = np.round(plot_levels*plot_norm,1).astype(str).tolist()
    #cb_labels[-1] = ''  # Remove the final label
    #cb.set_ticks(ticks=plot_levels*plot_norm, labels=cb_labels)
    cb.set_label('mm/day')

    if (statistic == "mean"):
        stats_string = "mean"
    elif (statistic == "std"):
        stats_string = "standard deviation"
    else:
        print("WARNING: Unknown statistic")
        stats_string = statistic

    title_string = f"""Ensemble mean, daily {stats_string}
    {region}, lead time: {lead_times[lead_time_idx] * 24:.0f}h, valid {d.strftime("%B %Y")}"""
    fig.suptitle(title_string)

    # Save the plot
    if (file_name != None):
        if file_name[-4:] in ['.png','.jpg','.pdf']:
            plt.savefig(file_name, format=file_name[-3:], bbox_inches='tight')
        else:
            print("ERROR: File type must be specified by '.png', '.jpg' or '.pdf'")

    plt.show()


# Maps with the statistic taken in the time direction over the month
# XXX Specify the style (eg 'KMD') or allow specification of vmin and vmax
def plot_forecast_maps(dm_list,          # A list of the deterministic metric dictionaries
                       region,           # XXX get from data
                       d,                # XXX get from data
                       lead_times=[1.25],  # XXX get from data
                       statistic='MAE',  # Statistic to plot. Choose from 'bias', 'MSE', 'MAE', 'R2', 'anomaly_corr'
                       lead_time_idx=0,  # Which lead time to plot
                       vmin=None,        # Minimum value in the colour scale (None implies automatic)
                       vmax=None,        # Maximum values in the colour scale (None implies automatic)
                       file_name=None):

    # Which country are we looking at. Can be:
    # "South Sudan","Rwanda","Burundi","Djibouti","Eritrea",
    # "Ethiopia","Sudan","Somalia","Tanzania","Kenya","Uganda"
    latitude, longitude, mask = ldmd.load_country_mask(region)

    # Set the contour plot style
    plot_levels, plot_colours = du.get_contour_levels('KMD')
    plot_norm = 24  # We are plotting mm/day and plot_norm == 1 corresponds to mm/h

    # Load the border shapefile
    reader = shpreader.Reader(shapefile_name)
    borders_feature = ShapelyFeature(reader.geometries(), ccrs.PlateCarree(), facecolor='none')

    # Define the figure and each axes for the rows and columns
    fig, axs = plt.subplots(nrows=1, ncols=3, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(16,6.5))

    # axs is a 2 dimensional array of `GeoAxes`. Flatten it into a 1-D array
    axs=axs.flatten()

    for idx in range(len(dm_list)):
        ax=axs[idx]
        ax.add_feature(borders_feature, linewidth=0.5, edgecolor='black')  # The borders

        # If the statistic has units of mm/day use the chosen colour scheme
        # XXX if (statistic == 'bias') or (statistic =='RMSE') or (statistic == 'MAE'):
        if (False):
            c = ax.contourf(longitude,latitude,dm_list[idx][statistic][lead_time_idx,:,:],
                            colors=plot_colours, levels=plot_levels*plot_norm,
                            transform=ccrs.PlateCarree())
            
        else:  # The statistic has some other units
            c = ax.pcolormesh(longitude,latitude,dm_list[idx][statistic][lead_time_idx,:,:],
                              vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
            
        ax.set_title(dm_list[idx]['forecast_source'])
        ax.set_frame_on(False)

    # XXX Difference
    vmin = np.min(dm_list[1][statistic][lead_time_idx,:,:] - dm_list[0][statistic][lead_time_idx,:,:])
    vmax = np.max(dm_list[1][statistic][lead_time_idx,:,:] - dm_list[0][statistic][lead_time_idx,:,:])
    minmax = np.max([np.abs(vmin),np.abs(vmax)]) / 4
    ax=axs[2]
    ax.add_feature(borders_feature, linewidth=0.5, edgecolor='black')  # The borders
    c = ax.pcolormesh(longitude,latitude,dm_list[1][statistic][lead_time_idx,:,:] - dm_list[0][statistic][lead_time_idx,:,:],
                      cmap='coolwarm', vmin=-minmax, vmax=minmax, transform=ccrs.PlateCarree())        
    ax.set_title(f"{dm_list[1]['forecast_source']} - {dm_list[0]['forecast_source']}")
    ax.set_frame_on(False)

    # Adjust the location of the subplots on the page to make room for the colorbar
    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                        wspace=0.02, hspace=0.02)

    # Add a colorbar axis at the bottom of the graph
    cbar_ax = fig.add_axes([0.81, 0.215, 0.015, 0.57])

    # Draw the colorbar
    cb=fig.colorbar(c, cax=cbar_ax,orientation='vertical')
    #cb.ax.tick_params(labelsize=10)

    # If the statistic has units of mm/day use the chosen colour scheme
    if (statistic == 'bias') or (statistic =='RMSE') or (statistic == 'MAE'):
        # cb_labels = np.round(plot_levels*plot_norm,1).astype(str).tolist()
        # cb_labels[-1] = ''  # Remove the final label
        # cb.set_ticks(ticks=plot_levels*plot_norm, labels=cb_labels)
        cb.set_label('mm/day')

    # else:  # The statistic has some other units

    title_string = f"""Ensemble mean, daily {statistic}
    {region}, lead time: {lead_times[lead_time_idx] * 24:.0f}h, valid {d.strftime("%B %Y")}"""
    fig.suptitle(title_string)

    # Save the plot
    if (file_name != None):
        if file_name[-4:] in ['.png','.jpg','.pdf']:
            plt.savefig(file_name, format=file_name[-3:], bbox_inches='tight')
        else:
            print("ERROR: File type must be specified by '.png', '.jpg' or '.pdf'")

    plt.show()


# Plots in time with the statistic taken in the spatial dimension over the domain
def plot_truth_and_forecast_days(dm_list,                # A list of the deterministic metric dictionaries
                                 year,                   # XXX get from data
                                 month,                  # XXX get from data
                                 region,                 # XXX get from data
                                 lead_times=[1.25],        # XXX get from data
                                 statistic="mean",       # Can be "mean" or "std"
                                 lead_time_idx=0,        # Which lead time to plot
                                 precip_clim_ICPAC=None, # The climatology from ICPAC (optional)
                                 file_name=None):

    # Plot the mean
    days_this_month = len(dm_list[0][f'{statistic}_rgn_daily_truth'])
    x = np.arange(1,days_this_month+1) + 0.25  # XXX including the 6h offset

    # Plot the truth
    # XXX Assumes that all the truths are the same but they are not (eg. CHIRPS vs IMERG)
    plt.plot(x, dm_list[0][f'{statistic}_rgn_daily_truth'],
            label=f"{dm_list[0]['truth_source']} (all days: {dm_list[0][f'{statistic}_rgn_truth'][0]:.1f} mm/day)")

    # Plot the forecast
    for idx in range(len(dm_list)):
        plt.plot(x, dm_list[idx][f'{statistic}_rgn_daily_forecast'][:,lead_time_idx],
                label=f"{dm_list[idx]['forecast_source']} (all_days: {dm_list[idx][f'{statistic}_rgn_forecast'][lead_time_idx]:.1f} mm/day)")

    # The ICPAC WRF climatology
    if (statistic == "mean") and (type(precip_clim_ICPAC) != type(None)):
        y = np.mean(precip_clim_ICPAC[month-1,:,:])
        plt.plot([np.min(x), np.max(x)],[y,y],'k--', label=f"ICPAC WRF climatology, {float(y):.1f} mm/day")

    plt.grid()
    plt.xlim(0, days_this_month+1)
    plt.xlabel(f'Day of {datetime(year,month,1).strftime("%B")}')

    if (statistic == "mean"):
        plt.ylabel(f"Daily mean Rainfall (mm/day)")
    elif (statistic == "std"):
        plt.ylabel(f"Daily standard deviation of Rainfall (mm/day)")
    else:
        print("WARNING: Unknown statistic")
        plt.ylabel({statistic})

    plt.title(f"{region} {year}, Lead time: {lead_times[lead_time_idx]*24:.0f}h")
    plt.legend()

    # Save the plot
    if (file_name != None):
        if file_name[-4:] in ['.png','.jpg','.pdf']:
            plt.savefig(file_name, format=file_name[-3:], bbox_inches='tight')
        else:
            print("ERROR: File type must be specified by '.png', '.jpg' or '.pdf'")

    plt.show()


# Plots in time with the statistic taken in the spatial dimension over the domain
def plot_forecast_days(dm_list,                # A list of the deterministic metric dictionaries
                       year,                   # XXX get from data
                       month,                  # XXX get from data
                       region,                 # XXX get from data
                       lead_times=[1.25],        # XXX get from data
                       statistic="MSE",        # Statistic to plot. Choose from 'bias', 'MSE', 'MAE', 'R2', 'anomaly_corr'
                       lead_time_idx=0,        # Which lead time to plot
                       file_name=None):

    if (statistic == "bias") or (statistic == "RMSE") or (statistic == "MAE"):
        units = "mm/day"
    else:
        units = None

    # Plot the mean
    days_this_month = len(dm_list[0][f'{statistic}_rgn_daily'])
    x = np.arange(1,days_this_month+1) + 0.25  # XXX including the 6h offset

    # Plot the forecast
    for idx in range(len(dm_list)):
        plt.plot(x, dm_list[idx][f'{statistic}_rgn_daily'][:,lead_time_idx],
                label=f"{dm_list[idx]['forecast_source']} (all_days: {dm_list[idx][f'{statistic}_rgn'][lead_time_idx]:.1f} {units})")
        
    plt.grid()
    plt.xlim(0, days_this_month+1)
    plt.xlabel(f'Day of {datetime(year,month,1).strftime("%B")}')

    if (units == None):
        plt.ylabel(statistic)
    else:
        plt.ylabel(f"{statistic} ({units})")
    plt.title(f"{region} {year}, Lead time: {lead_times[lead_time_idx]*24:.0f}h")
    plt.legend()

    # Save the plot
    if (file_name != None):
        if file_name[-4:] in ['.png','.jpg','.pdf']:
            plt.savefig(file_name, format=file_name[-3:], bbox_inches='tight')
        else:
            print("ERROR: File type must be specified by '.png', '.jpg' or '.pdf'")

    plt.show()
