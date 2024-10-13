# prem_corners
### Analyze Premier League Corners Odds and Provide Stakes

Manually input odds to the model as such:

```
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
```
and an amount to bet `pot`, and outputs optimal stakes based on the Kelly Criterion as such:

```
Aston Villa vs Manchester Utd: Stake £19.98 on over 10.5 corners at 2.0
Chelsea vs Nott'ham Forest: Stake £7.54 on under 10.5 corners at 1.9
Brighton vs Tottenham: Stake £72.48 on under 10.5 corners at 2.0
```

Various parameters, metrics, and models were tested before converging on the averaged stats from the last 5 games, fed into a friedman-mse Random Forest Regressor, which when backtested provided an average yearly return of 15% based on odds data from the last 7 Premier League seasons.

Excerpt of final "testing" function:

```
def calc_stake(rf, df, games, pot):
    pd.options.mode.chained_assignment = None
    # Reduce DF to test set
    df = df.loc[df.C.isna()]
    # Predict number of corners
    df['C_pred'] = rf.predict(df[df.columns[2:-1]])
    df_odds = pd.DataFrame(data=games,columns=['home_team','away_team','line','odds_under','odds_over'])
    df = df.merge(df_odds,how='left',on=['home_team','away_team'])

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
    sum_kc = df.best_kc.sum()
    for row in df.itertuples():
        if row.best_kc > 0:
            print(f'{row.home_team} vs {row.away_team}: Stake £{row.to_stake:.2f} on {row.best_line[3:]} {row.line} corners at {row.best_odds}')

    return
```

Future features include:
- Automatic odds retrieval
- Goals Over/Under for various lines
- Additional features
- Comparison with a Deep Neural Network
