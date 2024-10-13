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

    def get_odds(year):
        # Parse downloaded odds data for backtesting strategies
        
        # returning smaller of two numbers where min not 0
        def min_gr0(a: float, b: float) -> float:
            if a <= 0:
                return b
            if b <= 0:
                return a

            return min(a, b)

        # filtering markets to those that fit the following criteria
        def filter_market(market: bflw.MarketBook) -> bool: 
            d = market.market_definition
            return (d != None and d.market_type == 'CORNER_ODDS')

        # parsing price data and pulling out weighted avg price, matched, min price and max price
        def parse_traded(traded: List[PriceSize]) -> Tuple[float, float, float, float]:
            if len(traded) == 0: 
                return (None, None, None, None)

            (wavg_sum, matched, min_price, max_price) = functools.reduce(
                lambda total, ps: (
                    total[0] + (ps.price * ps.size), # wavg_sum before we divide by total matched
                    total[1] + ps.size, # total matched
                    min(total[2], ps.price), # min price matched
                    max(total[3], ps.price), # max price matched
                ),
                traded,
                (0, 0, 1001, 0) # starting default values
            )

            wavg_sum = (wavg_sum / matched) if matched > 0 else None # dividing sum of wavg by total matched
            matched = matched if matched > 0 else None 
            min_price = min_price if min_price != 1001 else None
            max_price = max_price if max_price != 0 else None

            return (wavg_sum, matched, min_price, max_price)

        # Return teams from event_name
        def parse_teams(event_name):
            i = event_name.find(' v ')
            return event_name[:i], event_name[i+3:]

        # Get list of EPL teams from year
        def get_teams(year):
            misspelled = { # some commonly misspelled team names, from web scraped fixture table to betfair odds
                'Cardiff City': 'Cardiff',
                "Nott'ham Forest":'Nottm Forest', 
                'Newcastle Utd':'Newcastle', 
                'Manchester Utd':'Man Utd', 
                'Leeds United':'Leeds', 
                'Leicester City':'Leicester', 
                'Manchester City':'Man City',
                'Norwich City':'Norwich',
                'Swansea City':'Swansea',
                'Stoke City':'Stoke',
                }
            url = "https://fbref.com/en/comps/9/" + str(year) + '-' + str(year+1) + "/" + str(year) + "-" + str(year+1) + "-Premier-League-Stats"
            data = requests.get(url)
            df = pd.read_html(data.text, match="Premier League")[0]
            teams = [misspelled[team] if team in misspelled.keys() else team for team in df.Squad]
            if "Crystal Palace" in teams:
                teams.append("C Palace")
            return teams

        def process_odds(df_odds):
                df_odds = df_odds[['home_team','away_team','selection_name','preplay_ltp']]
                df_odds.drop_duplicates(inplace=True)
                df_odds.dropna(inplace=True)
                df_odds = df_odds[~df_odds.preplay_ltp.str.contains('None')]
                df_odds.preplay_ltp = df_odds.preplay_ltp.astype('float')
                df_odds = df_odds.pivot_table(index=['home_team','away_team'], columns='selection_name', values='preplay_ltp').round(3)
                df_odds.columns.name = None
                df_odds.rename(columns={'9 or less':'odds_under','10 - 12':'odds_1012','13 or more':'odds_over'}, inplace=True)
                df_odds.reset_index(inplace=True)
                df_odds = df_odds[['home_team','away_team','odds_under', 'odds_1012', 'odds_over']]

                misspelled = {
                    "Nott'ham Forest":'Nottm Forest', 
                    'Newcastle Utd':'Newcastle', 
                    'Manchester Utd':'Man Utd', 
                    'Leeds United':'Leeds', 
                    'Leicester City':'Leicester', 
                    'Manchester City':'Man City'
                    }
                misspelled = {v:k for k,v in misspelled.items()}
                df_odds.home_team = [misspelled[team] if team in misspelled.keys() else team for team in df_odds.home_team]
                df_odds.away_team = [misspelled[team] if team in misspelled.keys() else team for team in df_odds.away_team]
                return df_odds

        df = pd.DataFrame()
        folder = 'data_corners/' + str(year) + '/'
        market_paths = [folder + f for f in os.listdir(folder)]
        epl_teams = get_teams(year)

        for i, g in enumerate(bflw.Files(market_paths)):
            #print("Market {}".format(i), end='\r')

            home_team, away_team = parse_teams(next(g)[0].market_definition.event_name)
            if home_team in epl_teams and away_team in epl_teams:

                def get_pre_post_final():
                    eval_market = None
                    prev_market = None
                    preplay_market = None
                    postplay_market = None       

                    for market_books in g:
                        for market_book in market_books:
                            # if market doesn't meet filter return out
                            if eval_market is None and ((eval_market := filter_market(market_book)) == False):
                                return (None, None, None)

                            # final market view before market goes in play
                            if prev_market is not None and prev_market.inplay != market_book.inplay:
                                preplay_market = prev_market

                            # final market view at the conclusion of the market
                            if prev_market is not None and prev_market.status == "OPEN" and market_book.status != prev_market.status:
                                postplay_market = market_book

                            # update reference to previous market
                            prev_market = market_book

                    return (preplay_market, postplay_market, prev_market) # prev is now final

                (preplay_market, postplay_market, final_market) = get_pre_post_final()

                # no price data for market
                if postplay_market is None:
                    continue; 

                preplay_traded = [ (r.last_price_traded, r.ex.traded_volume) for r in preplay_market.runners ] if preplay_market is not None else None
                postplay_traded = [ (
                    r.last_price_traded,
                    r.ex.traded_volume,
                    # calculating SP traded vol as smaller of back_stake_taken or (lay_liability_taken / (BSP - 1))        
                    min_gr0(
                        next((pv.size for pv in r.sp.back_stake_taken if pv.size > 0), 0),
                        next((pv.size for pv in r.sp.lay_liability_taken if pv.size > 0), 0)  / ((r.sp.actual_sp if (type(r.sp.actual_sp) is float) or (type(r.sp.actual_sp) is int) else 0) - 1)
                    ) if r.sp.actual_sp is not None else 0,
                ) for r in postplay_market.runners ]

                # generic runner data
                runner_data = [
                    {
                        'selection_id': r.selection_id,
                        'selection_name': next((rd.name for rd in final_market.market_definition.runners if rd.selection_id == r.selection_id), None),
                        'selection_status': r.status,
                        'sp': str(r.sp.actual_sp),
                    }
                    for r in final_market.runners 
                ]

                # runner price data for markets that go in play
                if preplay_traded is not None:
                    def runner_vals(r):
                        try:
                            (pre_ltp, pre_traded), (post_ltp, post_traded, sp_traded) = r
                        except:
                            flag = 1

                        inplay_only = list(filter(lambda ps: ps.size > 0, [
                            PriceSize(
                                price=post_ps.price, 
                                size=post_ps.size - next((pre_ps.size for pre_ps in pre_traded if pre_ps.price == post_ps.price), 0)
                            )
                            for post_ps in post_traded 
                        ]))

                        (ip_wavg, ip_matched, ip_min, ip_max) = parse_traded(inplay_only)
                        (pre_wavg, pre_matched, pre_min, pre_max) = parse_traded(pre_traded)

                        return {
                            'preplay_min': str(pre_min),
                            'preplay_max': str(pre_max),
                            'preplay_wavg': str(pre_wavg),
                            'preplay_ltp': str(pre_ltp),
                            'preplay_matched': str((pre_matched or 0) + (sp_traded or 0)),
                            'inplay_min': str(ip_min),
                            'inplay_max': str(ip_max),
                            'inplay_wavg': str(ip_wavg),
                            'inplay_ltp': str(post_ltp),
                            'inplay_matched': str(ip_matched),
                        }

                    runner_traded = [ runner_vals(r) for r in zip_longest(preplay_traded, postplay_traded, fillvalue=PriceSize(0, 0)) ]

                # runner price data for markets that don't go in play
                else:
                    def runner_vals(r):
                        (ltp, traded, sp_traded) = r
                        (wavg, matched, min_price, max_price) = parse_traded(traded)

                        return {
                            'preplay_min': str(min_price),
                            'preplay_max': str(max_price),
                            'preplay_wavg': str(wavg),
                            'preplay_ltp': str(ltp),
                            'preplay_matched': str((matched or 0) + (sp_traded or 0)),
                            'inplay_min': '',
                            'inplay_max': '',
                            'inplay_wavg': '',
                            'inplay_ltp': '',
                            'inplay_matched': '',
                    }

                    runner_traded = [ runner_vals(r) for r in postplay_traded ]

                # Add to dataframe
                market_data = {'market_id': postplay_market.market_id,
                            'market_time': postplay_market.market_definition.market_time,
                            'home_team': home_team,
                            'away_team': away_team,
                            'name': postplay_market.market_definition.name}
                
                df = pd.concat([df,
                                pd.concat([pd.DataFrame(data, index=range(len(runner_data))) 
                                        for data in [market_data, runner_data, runner_traded]], axis=1, join='outer')],
                                axis=0,
                                ignore_index=True)
        df = process_odds(df)
        return df
    
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
