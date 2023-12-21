import plotly.express as px 
from dash import Dash, dcc, html, Input, Output

import pandas as pd
import numpy as np
import math



import dash
import dash_core_components as dcc
import dash_html_components as html
from sqlalchemy import create_engine

def plot_top_locations(df, top_n=10, isPickup=True, same_neighborhood=True):
    """
    Return a plotly figure of the top pickup and dropoff locations within the same or different neighborhoods.
    """
    if same_neighborhood:
        title = f'Top {top_n} Pickup and Dropoff Locations within the Same Neighborhood'
        locations = df[(df['pu_location'] == df['do_location']) & (
            df['pu_location'] != 'unknown')].groupby('pu_location').size().nlargest(top_n)
    elif not same_neighborhood and isPickup:
        title = f'Top {top_n} Pickup Locations from Different Neighborhoods'
        locations = df[df['pu_location'] != df['do_location']
                       ].groupby('pu_location').size().nlargest(top_n)
    elif not same_neighborhood and not isPickup:
        title = f'Top {top_n} Dropoff Locations from Different Neighborhoods'
        locations = df[df['pu_location'] != df['do_location']
                       ].groupby('do_location').size().nlargest(top_n)

    fig = px.bar(locations, x=locations.values, y=locations.index,
                 labels={'x': 'Frequency', 'y': 'Location'}, orientation='h',
                 title=title )

    return fig



def plot_mean_median_count_by_period(df: pd.DataFrame, column_name: str, period_series: pd.Series, time_column: str, label: str, figsize=(9, 5)):
    """
    Return Plotly figures for the mean, median, and count values of a numeric column by hour and print a tabulated summary.
    """
    # Check if the time_column is of datetime data type
    if not pd.api.types.is_datetime64_any_dtype(df[time_column]):
        print(f'Convert first the {time_column} to datetime')
        return None, None

    # Prepare a pivot table to aggregate the specified column by hour
    table = df.pivot_table(index=period_series, values=column_name, aggfunc=(
        'mean', 'median', 'count')).reset_index()

    # Rename columns for clarity
    table.columns = [label, f'Count_{column_name}',
                     f'Mean_{column_name}', f'Median_{column_name}']

    # Plot the mean, median, and count values using Plotly Express
    fig_mean_median = px.line(table, x=label, y=[f'Mean_{column_name}', f'Median_{column_name}'],
                              title=f'Distribution of {column_name} per {label}',
                              labels={'value': f'{column_name} (units)', 'variable': label})

    fig_count = px.line(table, x=label, y=[f'Count_{column_name}'],
                        title=f'Count of {column_name} per {label}',
                        labels={'value': f'Count of {column_name}', 'variable': label})

    # Print tabulated summary
    print(f'----- Distribution of {column_name} per {label} -----\n')

    # Maximum mean value at which period
    max_mean = table[f"Mean_{column_name}"].max()
    max_period_mean = table[table[f"Mean_{column_name}"]
                            == max_mean].iloc[0][label]
    print(f'Maximum mean value ({max_mean}) at {label}: {max_period_mean}\n')

    # Maximum median value at which period
    max_median = table[f"Median_{column_name}"].max()
    max_period_median = table[table[f"Median_{column_name}"]
                              == max_median].iloc[0][label]
    print(
        f'Maximum median value ({max_median}) at {label}: {max_period_median}\n')
def plot_mean_median_count_by_period(df: pd.DataFrame, column_name: str, period_series: pd.Series, time_column: str, label: str, figsize=(9, 5)):
    """
    Return Plotly figures for the mean, median, and count values of a numeric column by hour and print a tabulated summary.
    """
    # Check if the time_column is of datetime data type
    if not pd.api.types.is_datetime64_any_dtype(df[time_column]):
        print(f'Convert first the {time_column} to datetime')
        return None, None

    # Prepare a pivot table to aggregate the specified column by hour
    table = df.pivot_table(index=period_series, values=column_name, aggfunc=(
        'mean', 'median', 'count')).reset_index()

    # Rename columns for clarity
    table.columns = [label, f'Count_{column_name}',
                     f'Mean_{column_name}', f'Median_{column_name}']

    # Plot the mean, median, and count values using Plotly Express
    fig_mean_median = px.line(table, x=label, y=[f'Mean_{column_name}', f'Median_{column_name}'],
                              title=f'Distribution of {column_name} per {label}',
                              labels={'value': f'{column_name} (units)', 'variable': label})

    fig_count = px.line(table, x=label, y=[f'Count_{column_name}'],
                        title=f'Count of {column_name} per {label}',
                        labels={'value': f'Count of {column_name}', 'variable': label})

    # Print tabulated summary
    print(f'----- Distribution of {column_name} per {label} -----\n')

    # Maximum mean value at which period
    max_mean = table[f"Mean_{column_name}"].max()
    max_period_mean = table[table[f"Mean_{column_name}"]
                            == max_mean].iloc[0][label]
    print(f'Maximum mean value ({max_mean}) at {label}: {max_period_mean}\n')

    # Maximum median value at which period
    max_median = table[f"Median_{column_name}"].max()
    max_period_median = table[table[f"Median_{column_name}"]
                              == max_median].iloc[0][label]
    print(
        f'Maximum median value ({max_median}) at {label}: {max_period_median}\n')

    # Maximum count value at which period
    max_count = table[f"Count_{column_name}"].max()
    max_period_count = table[table[f"Count_{column_name}"]
                             == max_count].iloc[0][label]
    print(
        f'Maximum count value ({max_count}) at {label}: {max_period_count}\n')


    return fig_mean_median, fig_count

    # Maximum count value at which period
    max_count = table[f"Count_{column_name}"].max()
    max_period_count = table[table[f"Count_{column_name}"]
                             == max_count].iloc[0][label]
    print(
        f'Maximum count value ({max_count}) at {label}: {max_period_count}\n')



    return fig_mean_median, fig_count


def create_dashboard_ui(filename):
    df = pd.read_csv(filename)
    app = Dash()

    # Convert columns to datetime type
    df['lpep_pickup_datetime'] = pd.to_datetime(df['lpep_pickup_datetime'])
    df['lpep_dropoff_datetime'] = pd.to_datetime(df['lpep_dropoff_datetime'])

    # Generate figures for mean, median, and count by period
    fig_mean_median_hr, fig_count_hr = plot_mean_median_count_by_period(
        df, 'trip_distance', df['lpep_pickup_datetime'].dt.hour, 'lpep_pickup_datetime', 'Hour'
    )

    fig_mean_median_day, fig_count_day =    plot_mean_median_count_by_period(df, 'trip_distance', df['lpep_pickup_datetime'].dt.day, 'lpep_pickup_datetime',
                                     'Day')
    # Define CSS styles
    app.css.append_css({
        'external_url': 'https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css'
    })

    app.layout = html.Div([
        html.H1("Omar Sherif Ali Hassan , 49-3324 , MET", className='main-title text-center' , style={'text-align': 'center'}),
        html.Div([
            html.H2("NYC Green Taxi 10/2016 dataset", className='dataset-title text-center' , style={'text-align': 'center'}),
        ], className='section container'),

        html.Div([
            html.H2("Top 10 pickup and dropoff locations within the same neighborhood based on trip frequency", className='section-title'),
            dcc.Graph(figure=plot_top_locations(df, top_n=10, same_neighborhood=True)),
        ], className='section container'),

        html.Div([
            html.H2("Top 10 pickup locations from different neighborhoods based on trip distance", className='section-title'),
            dcc.Graph(figure=plot_top_locations(df, top_n=10, isPickup=True, same_neighborhood=False)),
        ], className='section container'),

        html.Div([
            html.H2("Top 10 dropoff locations from different neighborhoods based on trip distance", className='section-title'),
            dcc.Graph(figure=plot_top_locations(df, top_n=10, isPickup=False, same_neighborhood=False)),
        ], className='section container'),

        html.Div([
            html.H2("Trip distance vs Trip amount", className='section-title'),
            dcc.Graph(figure=px.scatter(df, x='trip_distance', y='total_amount', title='Trip distance with Trip amount relationship', labels={'x': 'Trip Distance', 'y': 'Total Amount'})),
        ], className='section container'),

        html.Div([
            html.H2("Distribution of trip distance per pickup hour", className='section-title'),
            dcc.Graph(figure=fig_mean_median_hr),
        ], className='section container'),

        html.Div([
            html.H2("Count of trips per pickup hour", className='section-title'),
            dcc.Graph(figure=fig_count_hr),
        ], className='section container'),
        html.Div([
            html.H2("Distribution of trip distance per pickup day", className='section-title'),
            dcc.Graph(figure=fig_mean_median_day),
        ], className='section container'),

        html.Div([
            html.H2("Count of trips per pickup day", className='section-title'),
            dcc.Graph(figure=fig_count_day),
        ], className='section container'),

    ], className='dashboard')



    app.run_server(host='0.0.0.0',debug=False)


   