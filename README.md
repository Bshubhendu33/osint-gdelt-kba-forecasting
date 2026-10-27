# Fuel- and Energy-Cost Media Discourse as a Leading Indicator of German Powertrain Registrations

> A stationarity-corrected GDELT–KBA null result.
> M.Sc. OSINT course project, SRH University Heidelberg — graded **1.1**.

## Overview

This project tests a simple question: **does what German news media writes about
fuel and energy costs predict what types of cars people actually register months
later?**

It connects two public data sources — news discourse on one side, official
government registration statistics on the other — and tests whether movement in
the news reliably comes *before* movement in the registration data.

## Research Question

If fuel- and energy-cost discourse in German news media rises or falls, does the
share of diesel, battery-electric, and hybrid vehicle registrations shift a
predictable number of months afterward? In other words, can open-source news
discourse act as an early indicator for the German car market?

## Data Sources

- **GDELT Global Knowledge Graph (GKG)** — approximately 291,000 German-language
  fuel and energy news records, January 2022 to April 2026, retrieved via Google
  BigQuery.
- **Kraftfahrt-Bundesamt (KBA)** — official German monthly vehicle registration
  shares (releases FZ 8, FZ 8.5, FZ 28), covering 52 aligned months.

One side captures what the news talks about; the other captures what people
actually bought.

> Note: the raw GDELT dataset is too large to host on GitHub, and the original
> BigQuery query was not saved. The corpus was retrieved from the GDELT GKG table
> on Google BigQuery, filtered to German-language fuel and energy records for
> January 2022 to April 2026. See the Methodology section of `main.tex` for the
> exact filters, theme tags, and date range used.

## Method

1. Build one text string per news record from its theme tags and named entities
2. Convert each record into a numerical embedding (`all-MiniLM-L6-v2`)
3. Group records into discourse communities using weighted greedy community
   detection (a memory-efficient reimplementation that runs the full corpus on a
   free GPU)
4. Label each community once using a grammar-constrained large language model
   (Mistral Nemo, temperature 0) rather than labelling every record
5. Build monthly discourse-share time series for each theme
6. Test lead–lag relationships against KBA registration shares using
   cross-correlation, Granger causality, augmented Dickey–Fuller stationarity
   tests, first-differencing, and bootstrap confidence intervals

## Key Finding

On raw (level) data, fuel-cost discourse appeared to strongly predict hybrid
registrations (r = −0.698, Granger p = 0.033). However, stationarity testing
showed the series carried unit roots, and after first-differencing the same
relationship collapsed (p = 0.279). Under the pre-specified falsifiable threshold
and Bonferroni correction, no discourse–registration pair showed a reliable lead.

The honest result is therefore a **documented null**: in this dataset, German
fuel-cost news discourse does *not* reliably predict vehicle registration trends.
A clearly demonstrated null under a pre-registered threshold is a valid scientific
finding, and the gap between the false "level" signal and the corrected result is
itself the transferable lesson of the study.

## Tools

Python, GDELT, Google BigQuery, pandas, NumPy, sentence-transformers, statsmodels,
llama-cpp-python (Mistral Nemo), Matplotlib.

## Files

- `osint_topic1_fuelcost_pipeline.ipynb` — the full analysis pipeline
- `main.tex` — the IEEE-format research paper
- `verify_values.py` — script that verifies every number in the paper against the
  notebook outputs
- `requirements.txt` — Python dependencies

## Authors

Shubhendu Banerjee, Sacharith Reddy Jambula, Atharva Shekhar Jadhav, Farhan Siddique Mohammed

Supervisor: Prof. Dr. Klaus Dieter Schwarz
SRH University Heidelberg — Open Source Intelligence, Summer Semester 2026
