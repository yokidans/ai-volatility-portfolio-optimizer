import json

# Read the existing notebook
with open(r'C:\Users\tefer\ai-volatility-portfolio-optimizer\03_dashboard_story_fixed.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Find the allocation simulator cell and add a check for analysis_data
for i, cell in enumerate(notebook['cells']):
    if cell.get('cell_type') == 'code':
        source_code = ''.join(cell.get('source', []))
        if 'create_allocation_simulator(analysis_data)' in source_code:
            # Add a check to ensure analysis_data exists
            new_source = [
                "# Check if analysis_data exists, if not recreate it\n",
                "try:\n",
                "    analysis_data.shape\n",
                "except NameError:\n",
                "    print('analysis_data not found. Recreating...')\n",
                "    market_data = fetch_market_data('TSLA', '2017-01-01')\n",
                "    gjr_forecasts = load_or_simulate_forecasts(market_data, 'gjr')\n",
                "    quantile_forecasts = load_or_simulate_forecasts(market_data, 'quantile')\n",
                "    analysis_data = market_data.join(quantile_forecasts, how='inner')\n",
                "    analysis_data = analysis_data.join(gjr_forecasts[['sigma_gjr']], how='left')\n",
                "    analysis_data = analysis_data.dropna(subset=['q50', 'realized_vol_21d'])\n",
                "    print(f'✓ Analysis dataset prepared: {len(analysis_data)} rows')\n",
                "\n"
            ] + cell['source']
            notebook['cells'][i]['source'] = new_source
            print("✓ Added analysis_data check to allocation simulator cell")
            break

# Save the fixed notebook
with open(r'C:\Users\tefer\ai-volatility-portfolio-optimizer\03_dashboard_story_fixed.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print("✓ Notebook saved successfully")
