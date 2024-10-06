import pandas as pd
import numpy as np
import requests
import os
from itertools import zip_longest
import functools
import pickle
from typing import List, Tuple
from itertools import zip_longest
from betfair_data import PriceSize
from betfair_data import bflw
import betfair_data
import math
from scipy.stats import poisson

import betfairlightweight
from betfairlightweight import StreamListener
from betfairlightweight.resources.bettingresources import (
    PriceSize,
    MarketBook
)

from sklearn import ensemble

import scrape

def main_script(games,pot): 

    def get_avg_stats(df_fixtures, avg_over=5):

        def apply_stats(row, df_season):

            def filter_fixtures(df_season, team, i):
                # grab 5 latest fixtures
                subdf = df_season[(df_season[['home_team','away_team']] == team).any(axis=1) & (df_season.index < i)].iloc[-avg_over:]
                subdf_home = pd.DataFrame()
                subdf_away = pd.DataFrame()
                subdf_home[cols] = subdf.loc[subdf.home_team == team,['home_' + c for c in cols]].reset_index(drop=True)
                subdf_away[cols] = subdf.loc[subdf.away_team == team,['away_' + c for c in cols]].reset_index(drop=True)
                return pd.concat([subdf_home,subdf_away],ignore_index=True)
            
            # Columns to add
            cols = ['GF','GA','ST','SF','SA','CF','CA','Pts']
            for side in ['home','away']:
                cols_to_edit = [side + '_' + col for col in cols]
                subdf = filter_fixtures(df_season,row[side + '_team'],row.name)
                row[cols_to_edit] = subdf.mean().round(3)
                # Add STPrc
                row.rename({'ST':'STPrc'},inplace=True)
                row[side + '_STPrc'] = round((subdf.ST / subdf.SF).mean(),3)
            # Get corners from that game (aka "y_true")
            row['C'] = df_season.loc[(df_season.home_team == row.home_team) & (df_season.away_team == row.away_team), ['home_CF','away_CF']].sum(axis=1).iloc[0]

            return row
        
        df_fixtures.drop(columns=['home_GD','away_GD'],inplace=True)
        df_averaged = df_fixtures.apply(apply_stats, args=(df_fixtures,), axis=1).loc[avg_over*10:]
        return df_averaged

    def update_df_avg():
        # Read previous files
        f = open('df_avg.pckl', 'rb') # previous seasons stats (from 2017)
        df_avg = pickle.load(f)
        f.close()

        f = open('df_fixtures.pckl', 'rb') # current season stats
        df_fixtures = pickle.load(f)
        f.close()

        df_fixtures = scrape.update(df_fixtures)

        df_fixtures = pd.concat([df_fixtures,
                                 scrape.get_next_10()],
                                 ignore_index=True)
        df_avg = pd.concat([df_avg,
                            get_avg_stats(df_fixtures)],
                            ignore_index=True)
        df_avg.loc[df_avg.C == 0, 'C'] = None
        
        # Overwrite files
        f = open('df_fixtures.pckl', 'wb')
        pickle.dump(df_fixtures,f)
        f.close()

        f = open('df_avg.pckl', 'wb')
        pickle.dump(df_avg,f)
        f.close()

        return df_avg

    def train(df):
        # drop matches to be predicted
        df = df.dropna(subset='C')
        X = df[df.columns[2:-1]] # remove team names and y
        y = df['C']
        rf = ensemble.RandomForestRegressor(n_estimators=200, criterion='friedman_mse')
        rf.fit(X,y)
        return rf

    def calc_stake(rf, df, games, pot):
        pd.options.mode.chained_assignment = None
        # Reduce DF to test set
        df = df.loc[df.C.isna()]
        # Predict number of corners
        df['C_pred'] = rf.predict(df[df.columns[2:-1]])
        df_odds = pd.DataFrame(data=games,columns=['home_team','away_team','line','odds_under','odds_over'])
        df = df.merge(df_odds,how='left',on=['home_team','away_team'])
        
        #score = rf.score(df[df.columns[2:-5]],df.C)

        # Apply poisson distribution to predictions
        df['P_under'] = poisson.cdf(np.floor(df['line']-.5),df['C_pred'])
        df['P_over'] = 1 - poisson.cdf(np.floor(df['line']),df['C_pred']) # allow for whole and half lines

        # Calculate Kelly Criterion for each line
        for line in ['under','over']:
            df['kc_'+line] = df['P_'+line] - (1 - df['P_'+line]) / (df['odds_'+line]-1)

        # Find which line to bet on and set-up for printing
        df['best_line'] = df[['kc_under','kc_over']].idxmax(axis=1)
        df['best_kc'] = df[['kc_under','kc_over']].max(axis=1)
        df.loc[df.best_kc < 0, 'best_kc'] = None
        df.loc[df.best_line == 'kc_under', 'best_odds'] = df.odds_under
        df.loc[df.best_line == 'kc_over', 'best_odds'] = df.odds_over
        df['to_stake'] = df.best_kc * pot / df.best_kc.sum()

        # Print
        for row in df.itertuples():
            if row.best_kc > 0:
                print(f'{row.home_team} vs {row.away_team}: Stake £{row.to_stake:.2f} on {row.best_line[3:]} {row.line} corners at {row.best_odds}')

        return
        
    df_avg = update_df_avg()     
    rf = train(df_avg)
    calc_stake(rf, df_avg, games, pot)
    return

# manually input odds (home_team, away_team, line, odds_under, odds_over)
games = [
    ['Crystal Palace', 'Liverpool', 10.5, 1.875, 1.925],
    ['Arsenal', 'Southampton', 11, 1.8, 2],
    ['Brentford', 'Wolves', 10.5, 1.8, 2],
    ['Leicester City', 'Bournemouth', 10.5, 1.8, 2],
    ['Manchester City', 'Fulham', 10.5, 2, 1.8],
    ["West Ham", 'Ipswich Town', 10.5, 1.9, 1.9],
    ['Everton', 'Newcastle Utd', 11, 1.8, 2],
    ['Aston Villa', 'Manchester Utd', 10.5, 1.8, 2],
    ['Chelsea', "Nott'ham Forest", 10.5, 1.9, 1.9],
    ['Brighton', 'Tottenham', 10.5, 2, 1.8],
]

main_script(games, pot=100) # home_team, away_team, line, under, over
