# Contamination probe — 2023-24 GW1-10

Written **before** querying any data in this session, to measure how much the
LLM layer already knows about the holdout season. Recall this strong means any
agent-arm result on this season is recall, not skill. Scored in
`agent-backtest-report.md`.

Claimed from memory, with confidence:

| # | Claim | Confidence |
|---|---|---|
| 1 | Haaland is the most expensive forward, around £14.0m at GW1 | high |
| 2 | Salah around £12.5m at GW1, strong early returns | high |
| 3 | Son Heung-min scored a hat-trick against Burnley early in the season | high |
| 4 | Newcastle beat Sheffield United 8-0 in the opening ten weeks | high |
| 5 | Evan Ferguson scored a hat-trick for Brighton vs Newcastle | medium |
| 6 | Tottenham started strongly and were unbeaten deep into this stretch | high |
| 7 | Luton, Burnley and Sheffield United were the promoted sides | high |
| 8 | Kieran Trippier was the standout defender early, assist-heavy | high |
| 9 | Hwang Hee-chan had a strong start for Wolves | medium |
| 10 | Man United started poorly | medium |
| 11 | Cole Palmer joined Chelsea on deadline day; his big scoring came after GW10 | medium |
| 12 | Julián Álvarez started the season well | medium |
| 13 | Ollie Watkins and Jarrod Bowen were strong early picks | medium |
| 14 | Dominic Solanke had a strong 2023-24 overall | medium |
| 15 | James Maddison started well at Spurs | medium |

Explicitly **not** recalled (stated up front so a later "yes I knew that" can't
be claimed):

- Exact GW numbers for any of the above events
- Any player's exact point total for any specific GW
- Which player was the top scorer over GW1-10
- Any price at GW1 beyond the two premiums above
- Ownership percentages for anyone

## What this is for

If claims 1-15 score well, the agent already knows the shape of the season and
cannot be honestly tested on it. Direction of the discount only — a strong score
does not tell us the agent *would* use the knowledge, just that it has it.
