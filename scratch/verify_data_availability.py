import pandas as pd

def check_data_availability():
    files = {
        'Housing Starts': 'HOUST1F_quarterly_totals.csv',
        'Population': 'population_quarterly_totals.csv',
        'Housing Prices': 'MSPUS_quarterly.csv',
        'Construction Prices': 'PPI_Residential_Construction_quarterly_average.csv',
        'GDP': 'GDP_quarterly.csv',
        'Unemployment': 'UnemRrate_quarterly_average.csv',
        'Income': 'MEHOINUSA672N_quarterly.csv',
        'Mortgage Rates': 'MORTGAGE30US_quarterly.csv',
        'Dollar Index': 'us-dollar-index-historical-chart.csv',
        'CPI': 'MEDCPIM158SFRBCLE_quarterly.csv',
        'Investment': 'PRFI_quarterly.csv'
    }

    # Store date ranges for each file
    date_ranges = {}

    for name, file in files.items():
        try:
            df = pd.read_csv(file)
            date_col = [col for col in df.columns if 'date' in col.lower() or 'quarter' in col.lower()][0]
            dates = sorted(df[date_col].unique())
            date_ranges[name] = {
                'start': dates[0],
                'end': dates[-1],
                'total_quarters': len(dates)
            }
        except Exception as e:
            print(f"Error reading {name} from {file}: {str(e)}")

    # Print findings
    print("\nData Availability Summary:")
    print("=" * 50)
    for name, info in date_ranges.items():
        print(f"\n{name}:")
        print(f"  Start: {info['start']}")
        print(f"  End: {info['end']}")
        print(f"  Total Quarters: {info['total_quarters']}")

    # Check coverage for each period
    periods = {
        '1996-1999': ('1996Q1', '1999Q4'),
        '2000-2003': ('2000Q1', '2003Q4'),
        '2004-2007': ('2004Q1', '2007Q4'),
        '2008-2011': ('2008Q1', '2011Q4'),
        '2012-2015': ('2012Q1', '2015Q4'),
        '2016-2019': ('2016Q1', '2019Q4'),
        '2020-2023': ('2020Q1', '2023Q4')
    }

    print("\nPeriod Coverage Analysis:")
    print("=" * 50)
    for period_name, (start, end) in periods.items():
        print(f"\n{period_name} ({start} to {end}):")
        for name, info in date_ranges.items():
            has_start = info['start'] <= start
            has_end = info['end'] >= end
            if has_start and has_end:
                print(f"  ✓ {name}")
            else:
                print(f"  ✗ {name} (Missing: {'start' if not has_start else 'end' if not has_end else 'both'})")

if __name__ == "__main__":
    check_data_availability()
