# prem_corners
Analyze Premier League Corners Odds and Provide Stakes

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
