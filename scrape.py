import pandas as pd
import time
import pickle
import requests
from bs4 import BeautifulSoup

def get_stats(link):
    match_report = requests.get(link)
    soup = BeautifulSoup(match_report.text,features="lxml")
    score = [int(s.string) for s in soup.find_all('div', {'class':'score'})]
    stats = soup.find('div', id='team_stats').find_all('div')
    stats_extra = soup.find('div', id='team_stats_extra').find_all('div')

    # [on target, shots] for each team
    shotsA = [int(i) for i in stats[23].text.split() if i.isdigit()]
    shotsB = [int(i) for i in stats[28].text.split() if i.isdigit()]

    if score[0] > score[1]:
        pts = [3,0]
    elif score[0] < score[1]:
        pts = [0,3]
    else:
        pts = [1,1]

    return pd.Series({
        'home_team': stats_extra[1].text,
        'away_team': stats_extra[3].text,
        'home_GF': score[0], # Goals For
        'home_GA': score[1], # Goals Against
        'home_GD': score[0] - score[1], # Goal Diff
        'home_ST': shotsA[0], # Shots on Target
        'home_SF': shotsA[1], # Total Shots
        'home_SA': shotsB[1], # Total Shots Against
        'home_CF': int(stats_extra[7].text), # Corners For
        'home_CA': int(stats_extra[9].text), # Corners Against
        'home_Pts': pts[0],
        'away_GF': score[1],
        'away_GA': score[0],
        'away_GD': score[1] - score[0],
        'away_ST': shotsB[0],
        'away_SF': shotsB[1],
        'away_SA': shotsA[1],
        'away_CF': int(stats_extra[9].text),
        'away_CA': int(stats_extra[7].text),
        'away_Pts': pts[1],
    })

def update_df(match_stats, df):
    # Apply Result
    if match_stats[0]['GF'] > match_stats[1]['GF']:
        df.loc[df.Squad == match_stats[0]['name'], 'W'] += 1
        df.loc[df.Squad == match_stats[0]['name'], 'Pts'] += 3
        df.loc[df.Squad == match_stats[1]['name'], 'L'] += 1

    elif match_stats[0]['GF'] < match_stats[1]['GF']:
        df.loc[df.Squad == match_stats[0]['name'], 'L'] += 1
        df.loc[df.Squad == match_stats[1]['name'], 'W'] += 1
        df.loc[df.Squad == match_stats[1]['name'], 'Pts'] += 3

    else:
        df.loc[df.Squad == match_stats[0]['name'], 'D'] += 1
        df.loc[df.Squad == match_stats[0]['name'], 'Pts'] += 1
        df.loc[df.Squad == match_stats[1]['name'], 'D'] += 1
        df.loc[df.Squad == match_stats[1]['name'], 'Pts'] += 1

    # Apply Stats
    for team_stats in match_stats:
        team = team_stats.pop('name')
        df.loc[df.Squad == team, 'MP'] += 1
        for stat, value in team_stats.items():
            df.loc[df.Squad == team, stat] += value        

    return df

def init_fixtures(year):
    # Init fixtures
    #scores_fixtures = "https://fbref.com/en/comps/9/2023-2024/schedule/2023-2024-Premier-League-Scores-and-Fixtures"
    scores_fixtures = "https://fbref.com/en/comps/9/" + str(year) + "-" + str(year+1) + "/schedule/" + str(year) + "-" + str(year+1) + "-Premier-League-Scores-and-Fixtures"
    data = requests.get(scores_fixtures)
    soup = BeautifulSoup(data.text,features="lxml")
    #fixtures = soup.find('table', {"id":"sched_2023-2024_9_1"})
    tableid = "sched_" + str(year) + "-" + str(year+1) + "_9_1"
    fixtures = soup.find('table', {"id":tableid})
    return fixtures

def scrape_fixtures(year):
    df_fixtures = pd.DataFrame()#index=range(380))
    fixtures = init_fixtures(year)
    time.sleep(5)

    for j, link in enumerate(fixtures.find_all("a", string="Match Report")):

        # Get match stats
        link = "https://fbref.com" + link['href']
        match_stats = get_stats(link)
        df_fixtures.loc[j,match_stats.index] = match_stats
        time.sleep(5)

    return df_fixtures

def from_pickle(year):
    f = open('dfs.pckl', 'rb')
    dfs = pickle.load(f)
    f.close()

    return dfs[year - 2015]

def update(df_fixtures):
    fixtures = init_fixtures(2024)
    time.sleep(5)
    links = fixtures.find_all("a", string="Match Report")
    for i in range(df_fixtures.shape[0]+1,len(links)):
        link = "https://fbref.com" + links[i]['href']
        match_stats = get_stats(link)
        df_fixtures.loc[i,match_stats.index] = match_stats
        time.sleep(5)
    
    return df_fixtures


def get_next_10():
    # Init fixtures
    scores_fixtures = "https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures"
    data = requests.get(scores_fixtures)
    soup = BeautifulSoup(data.text, features="lxml")
    fixtures = soup.find('table', {"id":"sched_2024-2025_9_1"})
    df = pd.read_html(str(fixtures))[0]
    df = df.loc[df['Match Report'] == 'Head-to-Head',['Home','Away']] \
        .reset_index(drop=True).loc[:9] \
        .rename(columns={'Home':'home_team','Away':'away_team'})
    df.C = None
    #tableid = "sched_" + str(year) + "-" + str(year+1) + "_9_1"
    #fixtures = soup.find('table', {"id":tableid})
    return df
