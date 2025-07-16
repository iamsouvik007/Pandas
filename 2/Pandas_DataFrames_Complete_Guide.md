# Pandas DataFrames: Complete Guide and Documentation

> **A Comprehensive Deep-Dive into Pandas DataFrames for Data Analysis**

## Table of Contents
1. [Introduction](#introduction)
2. [Setting Up Environment](#setting-up-environment)
3. [Creating DataFrames](#creating-dataframes)
4. [DataFrame Attributes and Methods](#dataframe-attributes-and-methods)
5. [Mathematical Operations](#mathematical-operations)
6. [Data Selection and Indexing](#data-selection-and-indexing)
7. [Data Filtering](#data-filtering)
8. [Advanced Filtering Techniques](#advanced-filtering-techniques)
9. [Adding and Modifying Columns](#adding-and-modifying-columns)
10. [Data Type Optimization](#data-type-optimization)
11. [Data Analysis Functions](#data-analysis-functions)
12. [Performance Optimization](#performance-optimization)
13. [Error Handling and Debugging](#error-handling-and-debugging)
14. [Real-World Case Studies](#real-world-case-studies)
15. [Best Practices and Tips](#best-practices-and-tips)
16. [Common Pitfalls and Solutions](#common-pitfalls-and-solutions)

## Introduction

### What are Pandas DataFrames?

Pandas DataFrames are **two-dimensional, size-mutable, and heterogeneous tabular data structures** with labeled axes (rows and columns). Think of them as:

- **Excel spreadsheets in Python** - but much more powerful
- **SQL tables** - but with more flexible operations
- **2D arrays** - but with labeled indices and mixed data types

### Why Use DataFrames?

1. **Intuitive Data Manipulation**: Easy-to-understand syntax for complex operations
2. **Performance**: Built on NumPy for fast operations on large datasets
3. **Flexibility**: Handle missing data, different data types, and irregular data
4. **Integration**: Works seamlessly with other Python libraries (matplotlib, seaborn, scikit-learn)
5. **Real-world Ready**: Designed for messy, real-world data

### What You'll Learn

This comprehensive guide covers everything from basic DataFrame creation to advanced filtering and analysis techniques, using **real-world datasets** including:
- **Movie ratings database** (298 movies with IMDB ratings, genres, cast)
- **IPL cricket match data** (950+ matches with teams, venues, results)
- **Student performance data** (academic records with IQ, marks, packages)

### Prerequisites

- Basic Python knowledge (variables, functions, loops)
- Understanding of lists and dictionaries
- Familiarity with basic programming concepts

## Setting Up Environment

### Essential Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # For basic plotting
import seaborn as sns           # For advanced visualization
from collections import Counter # For frequency analysis
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output
```

### Library Breakdown

| Library | Purpose | Key Features |
|---------|---------|--------------|
| **pandas** | Data manipulation and analysis | DataFrames, Series, data I/O |
| **numpy** | Numerical computing foundation | Arrays, mathematical functions |
| **matplotlib** | Basic plotting and visualization | Line plots, histograms, scatter plots |
| **seaborn** | Statistical data visualization | Beautiful statistical plots |
| **collections.Counter** | Frequency counting | Efficient counting of elements |

### Pandas Configuration

```python
# Display options for better DataFrame viewing
pd.set_option('display.max_columns', None)     # Show all columns
pd.set_option('display.max_rows', 100)        # Show up to 100 rows
pd.set_option('display.width', 1000)          # Wider display
pd.set_option('display.precision', 2)         # 2 decimal places
pd.set_option('display.float_format', '{:.2f}'.format)  # Format floats

# Check pandas version
print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")
```

### Memory and Performance Settings

```python
# For large datasets - memory optimization
pd.set_option('mode.copy_on_write', True)     # Enable copy-on-write mode
pd.set_option('compute.use_bottleneck', True) # Use bottleneck for faster operations
pd.set_option('compute.use_numexpr', True)    # Use numexpr for faster evaluation
```

## Creating DataFrames

DataFrames can be created from various data sources. Understanding these methods is crucial for data analysis workflows.

### Method 1: From Lists (Manual Data Entry)

#### Basic List Creation

```python
# Creating DataFrame from 2D list structure
student_data = [
    [100, 80, 10],
    [90, 70, 7],
    [120, 100, 14],
    [80, 50, 2]
]

df = pd.DataFrame(student_data, columns=['iq', 'marks', 'package'])
print("DataFrame from lists:")
print(df)
print(f"\nShape: {df.shape}")
print(f"Data types:\n{df.dtypes}")
```

**Output:**
```
   iq  marks  package
0  100     80       10
1   90     70        7
2  120    100       14
3   80     50        2

Shape: (4, 3)
Data types:
iq         int64
marks      int64
package    int64
```

#### Advanced List Creation with Custom Index

```python
# With custom row indices
df_custom = pd.DataFrame(
    student_data, 
    columns=['iq', 'marks', 'package'],
    index=['Alice', 'Bob', 'Charlie', 'Diana']
)
print("DataFrame with custom index:")
print(df_custom)

# Mixed data types in lists
mixed_data = [
    ['Alice', 25, 85.5, True],
    ['Bob', 30, 92.3, False],
    ['Charlie', 22, 78.9, True]
]

df_mixed = pd.DataFrame(
    mixed_data, 
    columns=['name', 'age', 'score', 'active']
)
print("\nMixed data types:")
print(df_mixed)
print(f"\nData types:\n{df_mixed.dtypes}")
```

**When to Use List Method:**
- ✅ Small datasets (< 100 rows)
- ✅ Manual data entry
- ✅ Quick prototyping and testing
- ✅ Creating sample data for examples

### Method 2: From Dictionaries (Structured Data)

#### Basic Dictionary Creation

```python
# Creating DataFrame from dictionary
student_dict = {
    'name': ['nitish', 'ankit', 'rupesh', 'rishabh', 'amit', 'ankita'],
    'iq': [100, 90, 120, 80, 95, 105],
    'marks': [80, 70, 100, 50, 85, 90],
    'package': [10, 7, 14, 2, 8, 12]
}

students = pd.DataFrame(student_dict)
print("DataFrame from dictionary:")
print(students)

# Setting custom index
students.set_index('name', inplace=True)
print("\nWith 'name' as index:")
print(students)
```

#### Advanced Dictionary Techniques

```python
# Nested dictionaries for complex data
nested_dict = {
    'personal': {
        'name': ['Alice', 'Bob', 'Charlie'],
        'age': [25, 30, 22],
        'city': ['NYC', 'LA', 'Chicago']
    },
    'professional': {
        'job': ['Engineer', 'Designer', 'Analyst'],
        'salary': [90000, 75000, 65000],
        'experience': [3, 5, 1]
    }
}

# Flatten nested dictionary
flat_dict = {}
for category, data in nested_dict.items():
    for key, values in data.items():
        flat_dict[f"{category}_{key}"] = values

df_nested = pd.DataFrame(flat_dict)
print("Flattened nested dictionary:")
print(df_nested)

# Dictionary with missing values
sparse_dict = {
    'A': [1, 2, 3],
    'B': [4, 5],        # Shorter list
    'C': [7, 8, 9, 10]  # Longer list
}

# Pandas handles different lengths automatically
df_sparse = pd.DataFrame.from_dict(sparse_dict, orient='index').T
print("\nDictionary with different lengths:")
print(df_sparse)
```

**When to Use Dictionary Method:**
- ✅ Column-oriented thinking
- ✅ Different data types per column
- ✅ Data from APIs (JSON-like structure)
- ✅ When column names are known upfront

### Method 3: From CSV Files (Real-World Data)

#### Basic CSV Reading

```python
# Reading data from CSV files
movies = pd.read_csv('movies.csv')
ipl = pd.read_csv('ipl-matches.csv')

print("Movies dataset info:")
print(f"Shape: {movies.shape}")
print(f"Columns: {list(movies.columns)}")
print("\nFirst 3 rows:")
print(movies.head(3))
```

#### Advanced CSV Reading Options

```python
# Advanced CSV reading with options
movies_advanced = pd.read_csv(
    'movies.csv',
    sep=',',                    # Separator (default is comma)
    header=0,                   # Row to use as column names
    index_col=None,             # Column to use as index
    usecols=['title_x', 'imdb_rating', 'genres'],  # Specific columns
    dtype={'imdb_rating': 'float32'},  # Specify data types
    na_values=['N/A', 'NULL', ''],     # Additional NA values
    parse_dates=['release_date'],       # Parse date columns
    encoding='utf-8',           # Character encoding
    nrows=100                   # Read only first 100 rows
)

# Reading CSV with custom settings
custom_csv_options = {
    'sep': ',',
    'header': 0,
    'na_values': ['', 'NA', 'N/A', 'null', 'NULL'],
    'keep_default_na': True,
    'low_memory': False,        # Read entire file for consistent dtypes
    'skipinitialspace': True,   # Skip spaces after delimiter
}

df_custom = pd.read_csv('movies.csv', **custom_csv_options)
```

#### Error Handling in CSV Reading

```python
import os

def safe_read_csv(filename, **kwargs):
    """
    Safely read CSV with error handling
    """
    try:
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} not found")
        
        df = pd.read_csv(filename, **kwargs)
        print(f"✅ Successfully loaded {filename}")
        print(f"   Shape: {df.shape}")
        print(f"   Memory usage: {df.memory_usage().sum() / 1024**2:.2f} MB")
        return df
    
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return None
    except pd.errors.EmptyDataError:
        print(f"❌ Error: {filename} is empty")
        return None
    except pd.errors.ParserError as e:
        print(f"❌ Parsing error in {filename}: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error reading {filename}: {e}")
        return None

# Usage
movies = safe_read_csv('movies.csv')
ipl = safe_read_csv('ipl-matches.csv')
```

**When to Use CSV Method:**
- ✅ Real-world data analysis (90% of cases)
- ✅ Large datasets (> 1000 rows)
- ✅ Data from databases, APIs, web scraping
- ✅ Sharing data between different tools

### Method 4: From Other Data Sources

#### From Excel Files

```python
# Reading Excel files
try:
    excel_df = pd.read_excel(
        'data.xlsx',
        sheet_name='Sheet1',     # Specific sheet
        header=0,                # Header row
        skiprows=0,              # Skip initial rows
        usecols='A:E',          # Specific columns
        engine='openpyxl'       # Excel engine
    )
except ImportError:
    print("Install openpyxl: pip install openpyxl")
```

#### From JSON Data

```python
# JSON data (common from APIs)
json_data = '''
[
    {"name": "Alice", "age": 25, "city": "NYC"},
    {"name": "Bob", "age": 30, "city": "LA"},
    {"name": "Charlie", "age": 22, "city": "Chicago"}
]
'''

df_json = pd.read_json(json_data)
print("DataFrame from JSON:")
print(df_json)
```

#### From Database Connections

```python
# Example of reading from database (requires SQLAlchemy)
"""
import sqlalchemy as db

# Create connection
engine = db.create_engine('sqlite:///database.db')

# Read from database
df_db = pd.read_sql_query('SELECT * FROM table_name', engine)
"""
```

### DataFrame Creation Comparison

| Method | Best For | Pros | Cons |
|--------|----------|------|------|
| **Lists** | Small data, prototyping | Simple, direct | Manual typing, limited scalability |
| **Dictionaries** | Structured data, APIs | Column-oriented, flexible | Memory usage for large data |
| **CSV** | Real-world analysis | Fast, handles large files | Requires file management |
| **Excel** | Business data | Multiple sheets, formatting | Slower than CSV |
| **JSON** | Web APIs, nested data | Hierarchical data | Complex parsing needed |
| **Database** | Production systems | Live data, large datasets | Requires connection setup |

## DataFrame Attributes and Methods

Understanding DataFrame attributes and methods is fundamental to effective data analysis. These tools help you inspect, understand, and summarize your data.

### Essential Attributes Deep Dive

#### Shape and Size Information

```python
# Dataset dimensions
print(f"Movies dataset shape: {movies.shape}")  # (rows, columns)
print(f"Number of rows: {movies.shape[0]}")
print(f"Number of columns: {movies.shape[1]}")
print(f"Total elements: {movies.size}")
print(f"Number of dimensions: {movies.ndim}")

# Memory usage
print(f"\nMemory usage:")
print(f"Deep memory usage: {movies.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(f"Shallow memory usage: {movies.memory_usage().sum() / 1024**2:.2f} MB")
```

#### Data Types and Structure

```python
# Comprehensive data type information
print("Data Types Analysis:")
print("=" * 50)

# Basic data types
print(f"Data types overview:\n{movies.dtypes}\n")

# Detailed type information
type_summary = movies.dtypes.value_counts()
print(f"Type distribution:\n{type_summary}\n")

# Memory usage per column
memory_usage = movies.memory_usage(deep=True)
print("Memory usage per column:")
for col, mem in memory_usage.items():
    if col != 'Index':
        print(f"  {col}: {mem / 1024:.1f} KB")
```

#### Index and Column Information

```python
# Index analysis
print("Index Information:")
print(f"Index type: {type(movies.index)}")
print(f"Index name: {movies.index.name}")
print(f"Index values: {movies.index.values[:5]}...")  # First 5
print(f"Is unique: {movies.index.is_unique}")
print(f"Is monotonic: {movies.index.is_monotonic}")

# Column analysis
print("\nColumn Information:")
print(f"Column names: {list(movies.columns)}")
print(f"Number of columns: {len(movies.columns)}")
print(f"Column data types: {dict(movies.dtypes)}")
```

### Essential Attributes Reference Table

| Attribute | Returns | Description | Example Usage |
|-----------|---------|-------------|---------------|
| `shape` | tuple | (rows, columns) dimensions | `df.shape` → (1000, 5) |
| `size` | int | Total number of elements | `df.size` → 5000 |
| `ndim` | int | Number of dimensions (always 2) | `df.ndim` → 2 |
| `dtypes` | Series | Data type of each column | Column type checking |
| `index` | Index | Row labels/index | Custom indexing operations |
| `columns` | Index | Column names | Column manipulation |
| `values` | ndarray | Underlying NumPy array | NumPy operations |
| `empty` | bool | True if DataFrame is empty | Data validation |
| `T` | DataFrame | Transposed DataFrame | Matrix operations |

### Data Inspection Methods

#### Quick Data Preview

```python
# Data preview with detailed analysis
def analyze_dataframe(df, name="DataFrame"):
    """Comprehensive DataFrame analysis"""
    print(f"\n{'='*60}")
    print(f"ANALYSIS: {name}")
    print(f"{'='*60}")
    
    # Basic info
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Data preview
    print(f"\nFirst 3 rows:")
    print(df.head(3))
    
    print(f"\nLast 3 rows:")
    print(df.tail(3))
    
    # Random sample
    print(f"\nRandom sample (3 rows):")
    print(df.sample(3))
    
    # Data types
    print(f"\nData types:")
    print(df.dtypes.value_counts())

# Usage
analyze_dataframe(movies, "Movies Dataset")
analyze_dataframe(ipl, "IPL Dataset")
```

#### Comprehensive Data Information

```python
# info() method - most important for data understanding
print("MOVIES DATASET INFO:")
print("=" * 40)
movies.info()

print("\nCustom info analysis:")
print(f"Non-null counts by column:")
non_null_counts = movies.count()
total_rows = len(movies)

for col in movies.columns:
    non_null = non_null_counts[col]
    null_count = total_rows - non_null
    null_percentage = (null_count / total_rows) * 100
    print(f"  {col}: {non_null}/{total_rows} ({null_percentage:.1f}% missing)")
```

#### Statistical Summary

```python
# describe() for numerical columns
print("NUMERICAL COLUMNS STATISTICS:")
print("=" * 40)
numerical_desc = movies.describe()
print(numerical_desc)

# Custom statistical analysis
def detailed_describe(df):
    """Enhanced describe function"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        print("No numerical columns found")
        return
    
    stats = []
    for col in numeric_cols:
        col_stats = {
            'column': col,
            'count': df[col].count(),
            'missing': df[col].isnull().sum(),
            'missing_pct': (df[col].isnull().sum() / len(df)) * 100,
            'mean': df[col].mean(),
            'median': df[col].median(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max(),
            'range': df[col].max() - df[col].min(),
            'skewness': df[col].skew(),
            'kurtosis': df[col].kurtosis()
        }
        stats.append(col_stats)
    
    stats_df = pd.DataFrame(stats)
    return stats_df

detailed_stats = detailed_describe(movies)
print("\nDetailed Statistics:")
print(detailed_stats)
```

#### Non-numerical Column Analysis

```python
# Analysis for categorical/text columns
def analyze_categorical_columns(df):
    """Analyze non-numerical columns"""
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    print("CATEGORICAL COLUMNS ANALYSIS:")
    print("=" * 40)
    
    for col in categorical_cols:
        print(f"\nColumn: {col}")
        print(f"  Unique values: {df[col].nunique()}")
        print(f"  Missing values: {df[col].isnull().sum()}")
        print(f"  Most common values:")
        
        # Show top 5 most common values
        value_counts = df[col].value_counts().head(5)
        for value, count in value_counts.items():
            percentage = (count / len(df)) * 100
            print(f"    '{value}': {count} ({percentage:.1f}%)")

analyze_categorical_columns(movies)
```

### Data Quality Assessment

#### Missing Data Analysis

```python
# Comprehensive missing data analysis
def missing_data_analysis(df):
    """Detailed missing data analysis"""
    print("MISSING DATA ANALYSIS:")
    print("=" * 40)
    
    # Overall missing data
    total_cells = df.size
    missing_cells = df.isnull().sum().sum()
    missing_percentage = (missing_cells / total_cells) * 100
    
    print(f"Total cells: {total_cells:,}")
    print(f"Missing cells: {missing_cells:,}")
    print(f"Missing percentage: {missing_percentage:.2f}%")
    
    # Missing data per column
    missing_per_column = df.isnull().sum()
    missing_per_column = missing_per_column[missing_per_column > 0].sort_values(ascending=False)
    
    if len(missing_per_column) > 0:
        print(f"\nColumns with missing data:")
        for col, count in missing_per_column.items():
            percentage = (count / len(df)) * 100
            print(f"  {col}: {count} ({percentage:.1f}%)")
    else:
        print("\n✅ No missing data found!")
    
    # Missing data patterns
    if missing_cells > 0:
        print(f"\nMissing data patterns:")
        missing_pattern = df.isnull().sum(axis=1).value_counts().sort_index()
        for missing_count, frequency in missing_pattern.items():
            if missing_count > 0:
                print(f"  {frequency} rows with {missing_count} missing values")

missing_data_analysis(movies)
```

#### Duplicate Data Analysis

```python
# Comprehensive duplicate analysis
def duplicate_analysis(df):
    """Analyze duplicate data"""
    print("DUPLICATE DATA ANALYSIS:")
    print("=" * 40)
    
    # Overall duplicates
    total_duplicates = df.duplicated().sum()
    duplicate_percentage = (total_duplicates / len(df)) * 100
    
    print(f"Total rows: {len(df):,}")
    print(f"Duplicate rows: {total_duplicates:,}")
    print(f"Duplicate percentage: {duplicate_percentage:.2f}%")
    
    if total_duplicates > 0:
        print(f"\nDuplicate rows preview:")
        duplicate_rows = df[df.duplicated(keep=False)].sort_values(list(df.columns))
        print(duplicate_rows.head())
        
        # Subset duplicates
        print(f"\nChecking for duplicates in key columns...")
        for col in df.columns:
            if df[col].dtype == 'object' and col in ['title', 'name', 'id']:
                col_duplicates = df[col].duplicated().sum()
                if col_duplicates > 0:
                    print(f"  {col}: {col_duplicates} duplicates")
    else:
        print("✅ No duplicate rows found!")

duplicate_analysis(movies)
```

### Column Management Operations

#### Renaming Columns

```python
# Comprehensive column renaming strategies
def demonstrate_column_renaming():
    """Show various column renaming techniques"""
    
    # Create sample DataFrame
    sample_df = students.copy()
    
    print("COLUMN RENAMING TECHNIQUES:")
    print("=" * 40)
    
    # Method 1: Dictionary mapping
    sample_df.rename(columns={'iq': 'intelligence_quotient', 'marks': 'percentage'}, inplace=True)
    print("After dictionary renaming:")
    print(sample_df.columns.tolist())
    
    # Method 2: Function-based renaming
    sample_df.columns = [col.upper() for col in sample_df.columns]
    print("\nAfter uppercase transformation:")
    print(sample_df.columns.tolist())
    
    # Method 3: String operations
    sample_df.columns = sample_df.columns.str.replace('_', ' ').str.title()
    print("\nAfter string formatting:")
    print(sample_df.columns.tolist())
    
    # Method 4: Complete column replacement
    sample_df.columns = ['IQ_Score', 'Grade_Percentage', 'Salary_Package']
    print("\nAfter complete replacement:")
    print(sample_df.columns.tolist())

demonstrate_column_renaming()
```

This enhanced section provides comprehensive coverage of DataFrame attributes and methods with practical examples, detailed analysis functions, and real-world applications.

## Mathematical Operations

Mathematical operations in pandas are vectorized and highly efficient. Understanding these operations is crucial for data analysis and statistical computations.

### Aggregation Functions Deep Dive

#### Basic Aggregation Operations

```python
# Comprehensive aggregation example
def demonstrate_aggregations(df):
    """Show all major aggregation functions"""
    
    print("AGGREGATION FUNCTIONS DEMONSTRATION:")
    print("=" * 50)
    
    # Numerical columns only
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        print("No numerical columns found")
        return
    
    print(f"Working with columns: {list(numeric_df.columns)}")
    
    # Basic aggregations
    aggregations = {
        'Count': numeric_df.count(),
        'Sum': numeric_df.sum(),
        'Mean': numeric_df.mean(),
        'Median': numeric_df.median(),
        'Mode': numeric_df.mode().iloc[0] if not numeric_df.empty else None,
        'Standard Deviation': numeric_df.std(),
        'Variance': numeric_df.var(),
        'Minimum': numeric_df.min(),
        'Maximum': numeric_df.max(),
        'Range': numeric_df.max() - numeric_df.min(),
        'Quantile 25%': numeric_df.quantile(0.25),
        'Quantile 75%': numeric_df.quantile(0.75),
        'Skewness': numeric_df.skew(),
        'Kurtosis': numeric_df.kurtosis()
    }
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(aggregations).T
    print("\nComprehensive Aggregation Summary:")
    print(summary_df.round(2))
    
    return summary_df

# Demonstrate with students dataset
students_summary = demonstrate_aggregations(students)
```

#### Axis Parameter Detailed Explanation

```python
# Understanding axis parameter with visual examples
def explain_axis_operations():
    """Detailed explanation of axis parameter"""
    
    print("AXIS PARAMETER EXPLANATION:")
    print("=" * 40)
    
    # Create sample data for clear demonstration
    sample_data = pd.DataFrame({
        'Math': [85, 92, 78, 96],
        'Science': [88, 85, 95, 89],
        'English': [79, 88, 82, 94]
    }, index=['Alice', 'Bob', 'Charlie', 'Diana'])
    
    print("Sample DataFrame:")
    print(sample_data)
    print()
    
    # Axis=0: Operations along rows (column-wise results)
    print("AXIS=0 (along rows → column-wise results):")
    print("=" * 40)
    print(f"Sum along rows (total per subject):")
    print(sample_data.sum(axis=0))
    print(f"\nMean along rows (average per subject):")
    print(sample_data.mean(axis=0))
    print()
    
    # Axis=1: Operations along columns (row-wise results)
    print("AXIS=1 (along columns → row-wise results):")
    print("=" * 40)
    print(f"Sum along columns (total per student):")
    print(sample_data.sum(axis=1))
    print(f"\nMean along columns (average per student):")
    print(sample_data.mean(axis=1))
    print()
    
    # Visual representation
    print("VISUAL REPRESENTATION:")
    print("=" * 40)
    print("""
    Original DataFrame:
    
           Math  Science  English
    Alice    85       88       79  → axis=1 (row-wise)
    Bob      92       85       88  → operations
    Charlie  78       95       82
    Diana    96       89       94
     ↓        ↓        ↓        ↓
    axis=0  axis=0  axis=0  axis=0
    (column-wise operations)
    """)

explain_axis_operations()
```

#### Advanced Mathematical Operations

```python
# Advanced mathematical operations
def advanced_math_operations(df):
    """Demonstrate advanced mathematical operations"""
    
    print("ADVANCED MATHEMATICAL OPERATIONS:")
    print("=" * 45)
    
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        print("No numerical columns available")
        return
    
    # Cumulative operations
    print("1. CUMULATIVE OPERATIONS:")
    print("-" * 25)
    print("Cumulative sum (first 5 rows):")
    print(numeric_df.cumsum().head())
    print("\nCumulative product (first 3 rows):")
    print(numeric_df.cumprod().head(3))
    print("\nCumulative maximum:")
    print(numeric_df.cummax().head(3))
    print("\nCumulative minimum:")
    print(numeric_df.cummin().head(3))
    
    # Rolling operations (moving averages)
    print("\n2. ROLLING OPERATIONS (Moving Averages):")
    print("-" * 40)
    window_size = 3
    print(f"Rolling mean (window={window_size}):")
    rolling_mean = numeric_df.rolling(window=window_size).mean()
    print(rolling_mean.head(5))
    
    print(f"\nRolling standard deviation (window={window_size}):")
    rolling_std = numeric_df.rolling(window=window_size).std()
    print(rolling_std.head(5))
    
    # Expanding operations
    print("\n3. EXPANDING OPERATIONS:")
    print("-" * 25)
    print("Expanding mean (cumulative mean):")
    expanding_mean = numeric_df.expanding().mean()
    print(expanding_mean.head(5))
    
    # Pct_change (percentage change)
    print("\n4. PERCENTAGE CHANGE:")
    print("-" * 25)
    print("Percentage change between consecutive rows:")
    pct_change = numeric_df.pct_change()
    print(pct_change.head(5))
    
    # Correlation and covariance
    print("\n5. CORRELATION AND COVARIANCE:")
    print("-" * 35)
    if len(numeric_df.columns) > 1:
        print("Correlation matrix:")
        correlation_matrix = numeric_df.corr()
        print(correlation_matrix)
        
        print("\nCovariance matrix:")
        covariance_matrix = numeric_df.cov()
        print(covariance_matrix)
    else:
        print("Need at least 2 numerical columns for correlation")

# Apply to movies dataset (if it has numerical columns)
advanced_math_operations(movies)
```

### Custom Aggregation Functions

```python
# Creating custom aggregation functions
def custom_aggregations():
    """Demonstrate custom aggregation functions"""
    
    print("CUSTOM AGGREGATION FUNCTIONS:")
    print("=" * 35)
    
    # Custom functions
    def coefficient_of_variation(series):
        """Calculate coefficient of variation (std/mean)"""
        return series.std() / series.mean() if series.mean() != 0 else np.nan
    
    def outlier_count(series):
        """Count outliers using IQR method"""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        return len(outliers)
    
    def range_ratio(series):
        """Calculate ratio of max to min"""
        return series.max() / series.min() if series.min() != 0 else np.nan
    
    # Apply custom functions
    numeric_students = students.select_dtypes(include=[np.number])
    
    if not numeric_students.empty:
        print("Applying custom functions to students dataset:")
        print("\nCoefficient of Variation (std/mean):")
        cv_results = numeric_students.apply(coefficient_of_variation)
        print(cv_results)
        
        print("\nOutlier count (IQR method):")
        outlier_results = numeric_students.apply(outlier_count)
        print(outlier_results)
        
        print("\nRange ratio (max/min):")
        range_results = numeric_students.apply(range_ratio)
        print(range_results)
        
        # Multiple aggregations at once
        print("\nMultiple aggregations:")
        multi_agg = numeric_students.agg([
            'mean', 'std', 'min', 'max',
            coefficient_of_variation,
            outlier_count,
            range_ratio
        ])
        print(multi_agg.round(2))

custom_aggregations()
```

### GroupBy Operations (Advanced Aggregation)

```python
# GroupBy operations for conditional aggregations
def demonstrate_groupby_math():
    """Demonstrate mathematical operations with groupby"""
    
    print("GROUPBY MATHEMATICAL OPERATIONS:")
    print("=" * 40)
    
    # Create sample data with groups
    sample_data = pd.DataFrame({
        'Department': ['Math', 'Math', 'Science', 'Science', 'English', 'English'],
        'Teacher': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank'],
        'Students': [25, 30, 28, 22, 35, 27],
        'AvgScore': [85, 88, 92, 87, 79, 83],
        'Budget': [50000, 45000, 60000, 55000, 40000, 42000]
    })
    
    print("Sample data:")
    print(sample_data)
    print()
    
    # Group by department and calculate statistics
    dept_stats = sample_data.groupby('Department').agg({
        'Students': ['sum', 'mean', 'std'],
        'AvgScore': ['mean', 'min', 'max'],
        'Budget': ['sum', 'mean']
    }).round(2)
    
    print("Department-wise statistics:")
    print(dept_stats)
    
    # Custom groupby aggregations
    print("\nCustom aggregations by department:")
    custom_stats = sample_data.groupby('Department').agg({
        'Students': [np.sum, np.mean, lambda x: x.max() - x.min()],
        'AvgScore': [np.mean, lambda x: x.std() / x.mean()],  # CV
        'Budget': [np.sum, np.median]
    }).round(2)
    
    # Rename columns for clarity
    custom_stats.columns = ['Total_Students', 'Avg_Students', 'Student_Range',
                           'Avg_Score', 'Score_CV', 'Total_Budget', 'Median_Budget']
    print(custom_stats)

demonstrate_groupby_math()
```

### Mathematical Operations Performance

```python
# Performance comparison of different mathematical operations
def math_performance_comparison():
    """Compare performance of different mathematical operations"""
    
    import time
    
    print("MATHEMATICAL OPERATIONS PERFORMANCE:")
    print("=" * 45)
    
    # Create large dataset for performance testing
    large_df = pd.DataFrame({
        'A': np.random.randn(100000),
        'B': np.random.randn(100000),
        'C': np.random.randn(100000)
    })
    
    operations = {
        'Sum': lambda df: df.sum(),
        'Mean': lambda df: df.mean(),
        'Std': lambda df: df.std(),
        'Var': lambda df: df.var(),
        'Min': lambda df: df.min(),
        'Max': lambda df: df.max(),
        'Median': lambda df: df.median(),
        'Quantile': lambda df: df.quantile([0.25, 0.75]),
        'Cumsum': lambda df: df.cumsum(),
        'Rolling': lambda df: df.rolling(100).mean()
    }
    
    print(f"Testing with DataFrame shape: {large_df.shape}")
    print("\nOperation performance (seconds):")
    print("-" * 30)
    
    for name, operation in operations.items():
        start_time = time.time()
        try:
            result = operation(large_df)
            end_time = time.time()
            duration = end_time - start_time
            print(f"{name:12}: {duration:.4f}s")
        except Exception as e:
            print(f"{name:12}: Error - {e}")

# Note: Uncomment to run performance test
# math_performance_comparison()
```

### Practical Mathematical Analysis Examples

```python
# Real-world mathematical analysis examples
def practical_math_examples():
    """Practical examples using mathematical operations"""
    
    print("PRACTICAL MATHEMATICAL ANALYSIS:")
    print("=" * 40)
    
    # Example 1: Grade Analysis
    if 'students' in globals():
        print("1. STUDENT PERFORMANCE ANALYSIS:")
        print("-" * 30)
        
        numeric_students = students.select_dtypes(include=[np.number])
        
        # Performance metrics
        print("Performance metrics:")
        print(f"Class average IQ: {numeric_students['iq'].mean():.1f}")
        print(f"Class average marks: {numeric_students['marks'].mean():.1f}")
        print(f"Average package: {numeric_students['package'].mean():.1f} LPA")
        
        # Performance categories
        students_copy = students.copy()
        students_copy['performance_score'] = (
            students_copy['iq'] * 0.3 + 
            students_copy['marks'] * 0.5 + 
            students_copy['package'] * 0.2
        )
        
        print(f"\nPerformance score statistics:")
        print(students_copy['performance_score'].describe())
        
        # Correlation analysis
        print(f"\nCorrelation between metrics:")
        correlation = numeric_students.corr()
        print(correlation)
    
    # Example 2: Movie Rating Analysis (if movies dataset available)
    if 'movies' in globals():
        print("\n2. MOVIE RATING ANALYSIS:")
        print("-" * 25)
        
        # Check if rating column exists
        rating_cols = [col for col in movies.columns if 'rating' in col.lower()]
        if rating_cols:
            rating_col = rating_cols[0]
            print(f"Analyzing column: {rating_col}")
            
            ratings = movies[rating_col].dropna()
            print(f"Rating statistics:")
            print(f"  Mean rating: {ratings.mean():.2f}")
            print(f"  Median rating: {ratings.median():.2f}")
            print(f"  Standard deviation: {ratings.std():.2f}")
            print(f"  Rating range: {ratings.min():.1f} - {ratings.max():.1f}")
            
            # Rating distribution
            print(f"\nRating distribution:")
            print(f"  Excellent (>8.0): {(ratings > 8.0).sum()} movies")
            print(f"  Good (7.0-8.0): {((ratings >= 7.0) & (ratings <= 8.0)).sum()} movies")
            print(f"  Average (6.0-7.0): {((ratings >= 6.0) & (ratings < 7.0)).sum()} movies")
            print(f"  Below average (<6.0): {(ratings < 6.0).sum()} movies")

practical_math_examples()
```

### Mathematical Operations Summary

| Operation Type | Function | Use Case | Performance |
|----------------|----------|----------|-------------|
| **Basic Aggregation** | `sum()`, `mean()`, `std()` | Descriptive statistics | Fast |
| **Cumulative** | `cumsum()`, `cumprod()` | Running totals | Medium |
| **Rolling** | `rolling().mean()` | Moving averages | Medium |
| **Expanding** | `expanding().mean()` | Progressive averages | Medium |
| **Correlation** | `corr()`, `cov()` | Relationship analysis | Slow for large data |
| **Quantiles** | `quantile()`, `median()` | Distribution analysis | Medium |
| **Custom** | `apply()`, `agg()` | Complex calculations | Varies |

**Key Takeaways:**
1. **Vectorization**: Pandas operations are vectorized and much faster than Python loops
2. **Axis Understanding**: axis=0 for column-wise, axis=1 for row-wise operations
3. **Memory Efficiency**: Use appropriate data types for better performance
4. **Missing Data**: Most operations handle NaN values gracefully
5. **Custom Functions**: Use `apply()` and `agg()` for complex mathematical operations

## Data Selection and Indexing

### Column Selection

```python
# Single column (returns Series)
movies['title_x']

# Multiple columns (returns DataFrame)
movies[['title_x', 'year_of_release', 'imdb_rating']]
```

### Row Selection

#### Using iloc (Position-based)

```python
# Single row by position
movies.iloc[5]              # 6th row (0-indexed)

# Multiple rows by position
movies.iloc[:5]             # First 5 rows
movies.iloc[[0, 4, 5]]      # Specific positions

# Rows and columns by position
movies.iloc[0:3, 0:3]       # First 3 rows and columns
```

#### Using loc (Label-based)

```python
# Single row by label
students.loc['nitish']

# Multiple rows by label
students.loc['nitish':'rishabh':2]    # With step
students.loc[['nitish', 'ankita']]    # Specific labels

# Rows and columns by label
movies.loc[0:2, 'title_x':'poster_path']
```

**Key Differences:**
- `iloc`: Uses integer positions (0-indexed)
- `loc`: Uses labels/index names (includes both endpoints in slicing)

## Data Filtering

### Basic Boolean Filtering

```python
# Simple condition
finals = ipl[ipl['MatchNumber'] == 'Final']

# Multiple conditions with AND (&)
csk_kolkata = ipl[(ipl['City'] == 'Kolkata') & 
                  (ipl['WinningTeam'] == 'Chennai Super Kings')]

# Counting filtered results
super_overs = ipl[ipl['SuperOver'] == 'Y'].shape[0]
```

**Important Notes:**
- Use parentheses around each condition when combining with `&` or `|`
- `&` for AND operations
- `|` for OR operations
- `~` for NOT operations

### Percentage Calculations

```python
# Calculate percentage of matches where toss winner wins
toss_wins = ipl[ipl['TossWinner'] == ipl['WinningTeam']].shape[0]
total_matches = ipl.shape[0]
percentage = (toss_wins / total_matches) * 100

print(f"Toss winners win {percentage:.1f}% of matches")
```

**Step-by-step Process:**
1. Create boolean mask: `ipl['TossWinner'] == ipl['WinningTeam']`
2. Filter DataFrame: `ipl[boolean_mask]`
3. Count rows: `.shape[0]`
4. Calculate percentage: `(count / total) * 100`

## Advanced Filtering Techniques

### String-based Filtering

#### Method 1: Using str.contains() (Less Precise)

```python
# May match partial strings
action_movies = movies[movies['genres'].str.contains('Action', na=False)]
```

⚠️ **Warning:** Can match 'Action' within other words like 'Interaction'

#### Method 2: Using str.split() + apply() (Recommended)

```python
# Precise genre matching
mask1 = movies['genres'].str.split('|').apply(lambda x: 'Action' in x)
mask2 = movies['imdb_rating'] > 7.5
action_high_rated = movies[mask1 & mask2]
```

**How it works:**
1. `str.split('|')`: "Action|Drama|Thriller" → ['Action', 'Drama', 'Thriller']
2. `apply(lambda x: 'Action' in x)`: Checks if 'Action' exists in the list
3. Returns boolean Series for filtering

### Complex Multi-condition Filtering

```python
# Movies that are BOTH Action AND Drama
action_drama_mask = movies['genres'].str.split('|').apply(
    lambda x: 'Action' in x and 'Drama' in x
)

# Action movies that are NOT Horror
action_not_horror = movies['genres'].str.split('|').apply(
    lambda x: 'Action' in x and 'Horror' not in x
)
```

### Reusable Filtering Function

```python
def filter_movies_by_genre_and_rating(df, target_genre, min_rating=7.0, exclude_genres=None):
    """
    Filter movies by genre and rating with optional exclusions
    
    Parameters:
    -----------
    df : DataFrame
        Movies DataFrame
    target_genre : str
        Genre to include (e.g., 'Action')
    min_rating : float
        Minimum IMDB rating (default: 7.0)
    exclude_genres : list
        List of genres to exclude (optional)
    
    Returns:
    --------
    DataFrame
        Filtered movies
    """
    # Include target genre
    mask = df['genres'].str.split('|').apply(lambda x: target_genre in x)
    
    # Rating filter
    mask = mask & (df['imdb_rating'] >= min_rating)
    
    # Exclude genres if specified
    if exclude_genres:
        for exclude_genre in exclude_genres:
            mask = mask & ~df['genres'].str.split('|').apply(lambda x: exclude_genre in x)
    
    return df[mask]

# Usage
comedy_movies = filter_movies_by_genre_and_rating(
    movies, 
    target_genre='Comedy', 
    min_rating=7.5, 
    exclude_genres=['Horror']
)
```

### Alternative Filtering Methods

```python
# Method 1: Using boolean arithmetic
percentage_alt1 = (ipl['TossWinner'] == ipl['WinningTeam']).mean() * 100

# Method 2: Using value_counts()
toss_comparison = (ipl['TossWinner'] == ipl['WinningTeam']).value_counts()
percentage_alt2 = (toss_comparison[True] / toss_comparison.sum()) * 100

# Method 3: Using query() method
matches_won = ipl.query('TossWinner == WinningTeam').shape[0]
percentage_alt3 = (matches_won / len(ipl)) * 100
```

## Adding and Modifying Columns

### Adding Constant Values

```python
# Add same value to all rows
movies['Country'] = 'India'
```

### Creating Columns from Existing Data

```python
# Extract lead actor from actors string
movies['lead_actor'] = movies['actors'].str.split('|').apply(lambda x: x[0])
```

### Data Cleaning

```python
# Remove rows with missing values
movies.dropna(inplace=True)
```

## Performance Optimization

Performance optimization is crucial when working with large datasets. Understanding these techniques can dramatically improve your data analysis speed.

### Memory Optimization Strategies

#### Data Type Optimization

```python
# Comprehensive data type optimization
def optimize_datatypes(df, report=True):
    """
    Optimize DataFrame data types for memory efficiency
    """
    if report:
        print("MEMORY OPTIMIZATION REPORT:")
        print("=" * 40)
        print(f"Original memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    df_optimized = df.copy()
    
    # Optimize integer columns
    for col in df_optimized.select_dtypes(include=['int64']).columns:
        min_val = df_optimized[col].min()
        max_val = df_optimized[col].max()
        
        if min_val >= -128 and max_val <= 127:
            df_optimized[col] = df_optimized[col].astype('int8')
        elif min_val >= -32768 and max_val <= 32767:
            df_optimized[col] = df_optimized[col].astype('int16')
        elif min_val >= -2147483648 and max_val <= 2147483647:
            df_optimized[col] = df_optimized[col].astype('int32')
    
    # Optimize float columns
    for col in df_optimized.select_dtypes(include=['float64']).columns:
        df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='float')
    
    # Optimize object columns to category if beneficial
    for col in df_optimized.select_dtypes(include=['object']).columns:
        unique_ratio = df_optimized[col].nunique() / len(df_optimized)
        if unique_ratio < 0.5:  # Convert to category if less than 50% unique values
            df_optimized[col] = df_optimized[col].astype('category')
    
    if report:
        print(f"Optimized memory usage: {df_optimized.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        memory_reduction = ((df.memory_usage(deep=True).sum() - df_optimized.memory_usage(deep=True).sum()) 
                          / df.memory_usage(deep=True).sum()) * 100
        print(f"Memory reduction: {memory_reduction:.1f}%")
        
        print("\nData type changes:")
        for col in df.columns:
            if df[col].dtype != df_optimized[col].dtype:
                print(f"  {col}: {df[col].dtype} → {df_optimized[col].dtype}")
    
    return df_optimized

# Apply optimization
if 'ipl' in globals():
    ipl_optimized = optimize_datatypes(ipl)
```

#### Category Data Type Deep Dive

```python
# Advanced category optimization
def category_optimization_demo():
    """Demonstrate category data type benefits"""
    
    print("CATEGORY DATA TYPE OPTIMIZATION:")
    print("=" * 45)
    
    # Create sample data with repeated values
    teams = ['Mumbai Indians', 'Chennai Super Kings', 'Royal Challengers Bangalore'] * 1000
    cities = ['Mumbai', 'Chennai', 'Bangalore'] * 1000
    
    df_object = pd.DataFrame({
        'team': teams,
        'city': cities,
        'score': np.random.randint(100, 200, 3000)
    })
    
    df_category = df_object.copy()
    df_category['team'] = df_category['team'].astype('category')
    df_category['city'] = df_category['city'].astype('category')
    
    print(f"Object dtype memory: {df_object.memory_usage(deep=True).sum() / 1024:.1f} KB")
    print(f"Category dtype memory: {df_category.memory_usage(deep=True).sum() / 1024:.1f} KB")
    
    memory_savings = ((df_object.memory_usage(deep=True).sum() - 
                      df_category.memory_usage(deep=True).sum()) / 
                     df_object.memory_usage(deep=True).sum()) * 100
    print(f"Memory savings: {memory_savings:.1f}%")
    
    # Performance comparison
    import time
    
    # Object dtype groupby
    start = time.time()
    obj_result = df_object.groupby('team')['score'].mean()
    obj_time = time.time() - start
    
    # Category dtype groupby
    start = time.time()
    cat_result = df_category.groupby('team')['score'].mean()
    cat_time = time.time() - start
    
    print(f"\nGroupby performance:")
    print(f"Object dtype: {obj_time:.4f} seconds")
    print(f"Category dtype: {cat_time:.4f} seconds")
    print(f"Speed improvement: {((obj_time - cat_time) / obj_time) * 100:.1f}%")

category_optimization_demo()
```

### Efficient Data Operations

#### Vectorization vs Loops

```python
# Vectorization performance demonstration
def vectorization_demo():
    """Compare vectorized operations vs loops"""
    
    print("VECTORIZATION PERFORMANCE COMPARISON:")
    print("=" * 45)
    
    # Create large dataset
    n = 100000
    df = pd.DataFrame({
        'A': np.random.randn(n),
        'B': np.random.randn(n),
        'C': np.random.randn(n)
    })
    
    import time
    
    # Method 1: Python loop (SLOW)
    start = time.time()
    result_loop = []
    for i in range(len(df)):
        result_loop.append(df.iloc[i]['A'] * df.iloc[i]['B'] + df.iloc[i]['C'])
    loop_time = time.time() - start
    
    # Method 2: Vectorized operation (FAST)
    start = time.time()
    result_vectorized = df['A'] * df['B'] + df['C']
    vectorized_time = time.time() - start
    
    # Method 3: NumPy vectorized (FASTEST)
    start = time.time()
    result_numpy = np.multiply(df['A'].values, df['B'].values) + df['C'].values
    numpy_time = time.time() - start
    
    print(f"Dataset size: {n:,} rows")
    print(f"Python loop: {loop_time:.4f} seconds")
    print(f"Pandas vectorized: {vectorized_time:.4f} seconds")
    print(f"NumPy vectorized: {numpy_time:.4f} seconds")
    print(f"\nSpeedup:")
    print(f"Pandas vs Loop: {loop_time / vectorized_time:.1f}x faster")
    print(f"NumPy vs Loop: {loop_time / numpy_time:.1f}x faster")
    print(f"NumPy vs Pandas: {vectorized_time / numpy_time:.1f}x faster")

# Uncomment to run performance test
# vectorization_demo()
```

#### Efficient Filtering and Selection

```python
# Efficient filtering techniques
def efficient_filtering_demo():
    """Demonstrate efficient filtering techniques"""
    
    print("EFFICIENT FILTERING TECHNIQUES:")
    print("=" * 40)
    
    # Create test data
    n = 50000
    df = pd.DataFrame({
        'category': np.random.choice(['A', 'B', 'C', 'D'], n),
        'value': np.random.randn(n),
        'group': np.random.choice(['X', 'Y'], n)
    })
    
    import time
    
    # Method 1: Multiple separate filters (SLOWER)
    start = time.time()
    result1 = df[df['category'] == 'A']
    result1 = result1[result1['value'] > 0]
    result1 = result1[result1['group'] == 'X']
    method1_time = time.time() - start
    
    # Method 2: Combined filter (FASTER)
    start = time.time()
    result2 = df[(df['category'] == 'A') & (df['value'] > 0) & (df['group'] == 'X')]
    method2_time = time.time() - start
    
    # Method 3: Query method (READABLE and FAST)
    start = time.time()
    result3 = df.query("category == 'A' and value > 0 and group == 'X'")
    method3_time = time.time() - start
    
    # Method 4: Using isin for multiple values (EFFICIENT)
    start = time.time()
    categories_to_filter = ['A', 'B']
    result4 = df[df['category'].isin(categories_to_filter) & (df['value'] > 0)]
    method4_time = time.time() - start
    
    print(f"Dataset size: {n:,} rows")
    print(f"Separate filters: {method1_time:.4f} seconds")
    print(f"Combined filter: {method2_time:.4f} seconds")
    print(f"Query method: {method3_time:.4f} seconds")
    print(f"isin method: {method4_time:.4f} seconds")
    
    print(f"\nResult sizes:")
    print(f"Method 1: {len(result1)} rows")
    print(f"Method 2: {len(result2)} rows")
    print(f"Method 3: {len(result3)} rows")
    print(f"Method 4: {len(result4)} rows")

efficient_filtering_demo()
```

### Large Dataset Handling

#### Chunked Processing

```python
# Chunked processing for large files
def chunked_processing_demo():
    """Demonstrate processing large files in chunks"""
    
    print("CHUNKED PROCESSING DEMONSTRATION:")
    print("=" * 40)
    
    # Simulate processing a large file
    chunk_size = 1000
    total_processed = 0
    chunk_results = []
    
    # Example: Calculate mean of large dataset in chunks
    def process_chunk(chunk):
        """Process individual chunk"""
        # Simulate some processing
        numeric_cols = chunk.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            return chunk[numeric_cols].mean()
        return None
    
    print("Processing large file in chunks...")
    print(f"Chunk size: {chunk_size:,} rows")
    
    # Example implementation
    """
    # Real implementation for large CSV files:
    chunk_iterator = pd.read_csv('large_file.csv', chunksize=chunk_size)
    
    for i, chunk in enumerate(chunk_iterator):
        chunk_result = process_chunk(chunk)
        if chunk_result is not None:
            chunk_results.append(chunk_result)
        
        total_processed += len(chunk)
        if i % 10 == 0:  # Progress update every 10 chunks
            print(f"Processed {total_processed:,} rows...")
    
    # Combine chunk results
    if chunk_results:
        final_result = pd.concat(chunk_results, axis=1).mean(axis=1)
        print("Final aggregated result:")
        print(final_result)
    """
    
    print("Chunked processing is ideal for:")
    print("• Files larger than available RAM")
    print("• Streaming data processing")
    print("• Memory-constrained environments")
    print("• ETL pipelines")

chunked_processing_demo()
```

#### Memory-Efficient File Reading

```python
# Memory-efficient file reading strategies
def memory_efficient_reading():
    """Demonstrate memory-efficient file reading"""
    
    print("MEMORY-EFFICIENT FILE READING:")
    print("=" * 40)
    
    # Reading strategies
    strategies = {
        'Basic': {
            'description': 'Standard reading',
            'code': 'pd.read_csv("file.csv")',
            'memory': 'High',
            'speed': 'Medium'
        },
        'Optimized dtypes': {
            'description': 'Specify data types',
            'code': 'pd.read_csv("file.csv", dtype={"col1": "int32"})',
            'memory': 'Medium',
            'speed': 'Fast'
        },
        'Category columns': {
            'description': 'Use categories for repeated values',
            'code': 'pd.read_csv("file.csv", dtype={"category_col": "category"})',
            'memory': 'Low',
            'speed': 'Fast'
        },
        'Select columns': {
            'description': 'Read only needed columns',
            'code': 'pd.read_csv("file.csv", usecols=["col1", "col2"])',
            'memory': 'Very Low',
            'speed': 'Very Fast'
        },
        'Chunked reading': {
            'description': 'Process in chunks',
            'code': 'pd.read_csv("file.csv", chunksize=10000)',
            'memory': 'Minimal',
            'speed': 'Slow but steady'
        }
    }
    
    print("Reading Strategy Comparison:")
    print("-" * 30)
    for strategy, details in strategies.items():
        print(f"{strategy}:")
        print(f"  Description: {details['description']}")
        print(f"  Memory usage: {details['memory']}")
        print(f"  Speed: {details['speed']}")
        print(f"  Code: {details['code']}")
        print()

memory_efficient_reading()
```

### Performance Monitoring

#### Profiling and Timing

```python
# Performance profiling tools
def performance_profiling():
    """Demonstrate performance profiling techniques"""
    
    print("PERFORMANCE PROFILING TECHNIQUES:")
    print("=" * 45)
    
    # Create sample data
    df = pd.DataFrame({
        'A': np.random.randn(10000),
        'B': np.random.choice(['X', 'Y', 'Z'], 10000),
        'C': np.random.randint(1, 100, 10000)
    })
    
    # Method 1: Basic timing with time module
    import time
    
    print("1. Basic Timing:")
    print("-" * 15)
    start = time.time()
    result = df.groupby('B')['A'].mean()
    end = time.time()
    print(f"Groupby operation: {end - start:.4f} seconds")
    
    # Method 2: Using timeit for more accurate timing
    import timeit
    
    print("\n2. Precise Timing with timeit:")
    print("-" * 30)
    timing_result = timeit.timeit(
        lambda: df.groupby('B')['A'].mean(),
        number=100
    )
    print(f"Average time over 100 runs: {timing_result/100:.6f} seconds")
    
    # Method 3: Memory profiling
    import psutil
    import os
    
    print("\n3. Memory Usage:")
    print("-" * 15)
    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss / 1024**2
    
    # Perform memory-intensive operation
    large_df = pd.DataFrame(np.random.randn(100000, 10))
    
    memory_after = process.memory_info().rss / 1024**2
    print(f"Memory before: {memory_before:.1f} MB")
    print(f"Memory after: {memory_after:.1f} MB")
    print(f"Memory increase: {memory_after - memory_before:.1f} MB")
    
    # Clean up
    del large_df

performance_profiling()
```

### Performance Best Practices

#### Code Optimization Guidelines

```python
# Performance best practices
def performance_best_practices():
    """Demonstrate performance best practices"""
    
    print("PERFORMANCE BEST PRACTICES:")
    print("=" * 35)
    
    print("1. DATA TYPES:")
    print("   ✅ Use appropriate data types (int32 vs int64)")
    print("   ✅ Convert repeated strings to categories")
    print("   ✅ Use float32 for reduced precision needs")
    print()
    
    print("2. OPERATIONS:")
    print("   ✅ Use vectorized operations instead of loops")
    print("   ✅ Combine multiple filters into single operation")
    print("   ✅ Use built-in methods (.sum(), .mean()) over apply()")
    print("   ✅ Prefer query() for complex filtering")
    print()
    
    print("3. MEMORY MANAGEMENT:")
    print("   ✅ Read only necessary columns")
    print("   ✅ Use chunked processing for large files")
    print("   ✅ Delete unnecessary DataFrames (del df)")
    print("   ✅ Use copy=False when possible")
    print()
    
    print("4. INDEXING:")
    print("   ✅ Set appropriate index for frequent lookups")
    print("   ✅ Use loc/iloc instead of chained indexing")
    print("   ✅ Sort index for range-based operations")
    print()
    
    print("5. AVOID COMMON PITFALLS:")
    print("   ❌ Iterating over rows (.iterrows())")
    print("   ❌ Chained indexing (df[...][...])")
    print("   ❌ Using apply() for simple operations")
    print("   ❌ Creating DataFrames in loops")
    print("   ❌ Not specifying dtypes when reading files")

performance_best_practices()
```

### Performance Optimization Summary

| Technique | Memory Impact | Speed Impact | Complexity | When to Use |
|-----------|---------------|--------------|------------|-------------|
| **Data Type Optimization** | High | Medium | Low | Always |
| **Category Types** | High | Medium | Low | Repeated strings |
| **Vectorization** | Low | Very High | Low | Numerical operations |
| **Chunked Processing** | Very High | Medium | High | Large files |
| **Query Method** | Low | High | Low | Complex filters |
| **Column Selection** | High | High | Low | Wide datasets |
| **Index Optimization** | Low | High | Medium | Frequent lookups |

**Key Performance Rules:**
1. **Measure First**: Profile before optimizing
2. **Data Types Matter**: Choose appropriate types for memory and speed
3. **Vectorize Everything**: Avoid Python loops at all costs
4. **Memory is Key**: Optimize memory usage for better cache performance
5. **Read Smartly**: Only load what you need from files

## Error Handling and Debugging

Effective error handling and debugging are crucial skills for robust data analysis. Understanding common errors and their solutions will save you hours of frustration.

### Common Pandas Errors and Solutions

#### KeyError: Missing Columns

```python
# Common KeyError scenarios and solutions
def handle_keyerror_demo():
    """Demonstrate KeyError handling strategies"""
    
    print("KEYERROR HANDLING STRATEGIES:")
    print("=" * 35)
    
    # Create sample DataFrame
    df = pd.DataFrame({
        'name': ['Alice', 'Bob', 'Charlie'],
        'age': [25, 30, 35],
        'salary': [50000, 60000, 70000]
    })
    
    print("Sample DataFrame:")
    print(df)
    print()
    
    # Problem: Accessing non-existent column
    print("1. COLUMN ACCESS ERRORS:")
    print("-" * 25)
    
    try:
        # This will raise KeyError
        result = df['address']  # Column doesn't exist
    except KeyError as e:
        print(f"❌ KeyError: {e}")
    
    # Solution 1: Check if column exists
    column_to_check = 'address'
    if column_to_check in df.columns:
        result = df[column_to_check]
        print(f"✅ Column '{column_to_check}' found")
    else:
        print(f"⚠️  Column '{column_to_check}' not found in DataFrame")
        print(f"Available columns: {list(df.columns)}")
    
    # Solution 2: Use get() method for safe access
    def safe_column_access(df, column, default=None):
        """Safely access DataFrame column"""
        try:
            return df[column]
        except KeyError:
            print(f"⚠️  Column '{column}' not found, returning default value")
            return default
    
    result = safe_column_access(df, 'address', 'Not Available')
    print(f"Safe access result: {result}")
    
    # Solution 3: Use .get() for dictionaries-like access
    print(f"\nAlternative: Check columns first")
    available_columns = [col for col in ['name', 'age', 'address'] if col in df.columns]
    print(f"Available requested columns: {available_columns}")

handle_keyerror_demo()
```

#### IndexError: Row Access Issues

```python
# IndexError handling
def handle_indexerror_demo():
    """Demonstrate IndexError handling"""
    
    print("\nINDEXERROR HANDLING:")
    print("=" * 25)
    
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    })
    
    print(f"DataFrame shape: {df.shape}")
    print()
    
    # Problem: Accessing row beyond DataFrame size
    try:
        result = df.iloc[10]  # Only 3 rows (0-2)
    except IndexError as e:
        print(f"❌ IndexError: {e}")
    
    # Solution: Safe row access
    def safe_row_access(df, index):
        """Safely access DataFrame row"""
        if 0 <= index < len(df):
            return df.iloc[index]
        else:
            print(f"⚠️  Index {index} out of bounds for DataFrame with {len(df)} rows")
            return None
    
    result = safe_row_access(df, 10)
    result = safe_row_access(df, 1)
    if result is not None:
        print(f"✅ Successfully accessed row 1:")
        print(result)

handle_indexerror_demo()
```

#### ValueError: Data Type Issues

```python
# ValueError handling
def handle_valueerror_demo():
    """Demonstrate ValueError handling"""
    
    print("\nVALUEERROR HANDLING:")
    print("=" * 25)
    
    # Create sample data with mixed types
    df = pd.DataFrame({
        'numbers': ['1', '2', 'three', '4', '5'],
        'dates': ['2023-01-01', '2023-02-01', 'invalid_date', '2023-04-01'],
        'floats': ['1.5', '2.7', 'not_a_number', '4.2']
    })
    
    print("Sample DataFrame with mixed/invalid data:")
    print(df)
    print()
    
    # Problem: Converting strings to numeric
    print("1. NUMERIC CONVERSION ERRORS:")
    print("-" * 30)
    
    try:
        # This will raise ValueError due to 'three'
        df['numbers_converted'] = df['numbers'].astype('int')
    except ValueError as e:
        print(f"❌ ValueError: {e}")
    
    # Solution: Use pd.to_numeric with error handling
    df['numbers_safe'] = pd.to_numeric(df['numbers'], errors='coerce')
    print("✅ Safe numeric conversion (invalid values become NaN):")
    print(df[['numbers', 'numbers_safe']])
    print()
    
    # Problem: Date conversion
    print("2. DATE CONVERSION ERRORS:")
    print("-" * 25)
    
    try:
        df['dates_converted'] = pd.to_datetime(df['dates'])
    except ValueError as e:
        print(f"❌ Partial ValueError: {e}")
    
    # Solution: Safe date conversion
    df['dates_safe'] = pd.to_datetime(df['dates'], errors='coerce')
    print("✅ Safe date conversion:")
    print(df[['dates', 'dates_safe']])

handle_valueerror_demo()
```

### Debugging Techniques

#### Data Inspection for Debugging

```python
# Comprehensive debugging toolkit
def debugging_toolkit():
    """Essential debugging functions for DataFrames"""
    
    print("DEBUGGING TOOLKIT:")
    print("=" * 25)
    
    def debug_dataframe(df, name="DataFrame"):
        """Comprehensive DataFrame debugging"""
        print(f"\n🔍 DEBUGGING: {name}")
        print("-" * 40)
        
        # Basic info
        print(f"Shape: {df.shape}")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print(f"Index type: {type(df.index)}")
        
        # Data types
        print(f"\nData types:")
        print(df.dtypes.value_counts())
        
        # Missing data
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            print(f"\n⚠️  Missing data found:")
            for col, count in missing_data[missing_data > 0].items():
                pct = (count / len(df)) * 100
                print(f"  {col}: {count} ({pct:.1f}%)")
        else:
            print(f"\n✅ No missing data")
        
        # Duplicate rows
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            print(f"\n⚠️  Duplicate rows: {duplicates}")
        else:
            print(f"\n✅ No duplicate rows")
        
        # Data sample
        print(f"\nFirst 3 rows:")
        print(df.head(3))
        
        # Numeric column statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print(f"\nNumeric columns summary:")
            print(df[numeric_cols].describe())
        
        # Categorical column info
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0:
            print(f"\nCategorical columns info:")
            for col in cat_cols[:3]:  # Show first 3 categorical columns
                unique_count = df[col].nunique()
                print(f"  {col}: {unique_count} unique values")
                if unique_count <= 10:
                    print(f"    Values: {list(df[col].unique())}")
    
    # Create problematic DataFrame for demonstration
    problematic_df = pd.DataFrame({
        'id': [1, 2, 3, 2, 4],  # Duplicate ID
        'name': ['Alice', 'Bob', None, 'Bob', 'Eve'],  # Missing value
        'score': [85.5, 92.0, 78.5, 92.0, 'invalid'],  # Mixed types
        'date': ['2023-01-01', '2023-02-01', '2023-03-01', '2023-02-01', '2023-05-01']
    })
    
    debug_dataframe(problematic_df, "Problematic Dataset")

debugging_toolkit()
```

#### Performance Debugging

```python
# Performance debugging tools
def performance_debugging():
    """Tools for debugging performance issues"""
    
    print("\nPERFORMANCE DEBUGGING:")
    print("=" * 30)
    
    # Create sample data for testing
    import time
    
    df = pd.DataFrame({
        'category': np.random.choice(['A', 'B', 'C'], 10000),
        'value': np.random.randn(10000),
        'group': np.random.choice(['X', 'Y', 'Z'], 10000)
    })
    
    # Function to time operations
    def time_operation(func, description, *args, **kwargs):
        """Time any operation and report results"""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        print(f"{description}: {duration:.4f} seconds")
        return result, duration
    
    # Compare different approaches
    print("Comparing operation performance:")
    print("-" * 35)
    
    # Slow approach: multiple operations
    def slow_approach():
        temp = df[df['category'] == 'A']
        temp = temp[temp['value'] > 0]
        return temp.groupby('group')['value'].mean()
    
    # Fast approach: combined operations
    def fast_approach():
        return df[(df['category'] == 'A') & (df['value'] > 0)].groupby('group')['value'].mean()
    
    # Very fast: using query
    def query_approach():
        return df.query("category == 'A' and value > 0").groupby('group')['value'].mean()
    
    slow_result, slow_time = time_operation(slow_approach, "Slow approach (multiple filters)")
    fast_result, fast_time = time_operation(fast_approach, "Fast approach (combined filter)")
    query_result, query_time = time_operation(query_approach, "Query approach")
    
    print(f"\nSpeedup factors:")
    print(f"Fast vs Slow: {slow_time / fast_time:.1f}x faster")
    print(f"Query vs Slow: {slow_time / query_time:.1f}x faster")
    
    # Memory usage debugging
    print(f"\nMemory usage analysis:")
    print(f"DataFrame memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"Result memory: {fast_result.memory_usage(deep=True).sum() / 1024:.1f} KB")

performance_debugging()
```

### Exception Handling Best Practices

```python
# Comprehensive exception handling
def exception_handling_patterns():
    """Demonstrate exception handling patterns"""
    
    print("\nEXCEPTION HANDLING PATTERNS:")
    print("=" * 35)
    
    def safe_dataframe_operation(df, operation_name, operation_func, **kwargs):
        """
        Safely execute DataFrame operations with comprehensive error handling
        """
        try:
            print(f"🔄 Executing: {operation_name}")
            result = operation_func(df, **kwargs)
            print(f"✅ Success: {operation_name}")
            return result, True
            
        except KeyError as e:
            print(f"❌ KeyError in {operation_name}: Column {e} not found")
            print(f"   Available columns: {list(df.columns)}")
            return None, False
            
        except IndexError as e:
            print(f"❌ IndexError in {operation_name}: {e}")
            print(f"   DataFrame shape: {df.shape}")
            return None, False
            
        except ValueError as e:
            print(f"❌ ValueError in {operation_name}: {e}")
            print(f"   Check data types and values")
            return None, False
            
        except TypeError as e:
            print(f"❌ TypeError in {operation_name}: {e}")
            print(f"   Check operation compatibility with data types")
            return None, False
            
        except Exception as e:
            print(f"❌ Unexpected error in {operation_name}: {type(e).__name__}: {e}")
            return None, False
    
    # Example operations to test
    sample_df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': ['x', 'y', 'z', 'w', 'v'],
        'C': [1.1, 2.2, 3.3, 4.4, 5.5]
    })
    
    # Test operations
    operations = [
        ("Column selection", lambda df: df[['A', 'C']]),
        ("Invalid column", lambda df: df['nonexistent_column']),
        ("Row access", lambda df: df.iloc[2]),
        ("Invalid row access", lambda df: df.iloc[10]),
        ("Numeric operation", lambda df: df['A'].mean()),
        ("Invalid numeric operation", lambda df: df['B'].mean()),
    ]
    
    for op_name, op_func in operations:
        result, success = safe_dataframe_operation(sample_df, op_name, op_func)
        if success and result is not None:
            print(f"   Result type: {type(result)}")
        print()

exception_handling_patterns()
```

### Data Validation Framework

```python
# Comprehensive data validation
def data_validation_framework():
    """Complete data validation framework"""
    
    print("DATA VALIDATION FRAMEWORK:")
    print("=" * 35)
    
    class DataValidator:
        """Comprehensive DataFrame validator"""
        
        def __init__(self, df):
            self.df = df
            self.errors = []
            self.warnings = []
        
        def validate_shape(self, min_rows=1, max_rows=None, min_cols=1, max_cols=None):
            """Validate DataFrame shape"""
            rows, cols = self.df.shape
            
            if rows < min_rows:
                self.errors.append(f"Too few rows: {rows} < {min_rows}")
            
            if max_rows and rows > max_rows:
                self.warnings.append(f"Many rows: {rows} > {max_rows}")
            
            if cols < min_cols:
                self.errors.append(f"Too few columns: {cols} < {min_cols}")
            
            if max_cols and cols > max_cols:
                self.warnings.append(f"Many columns: {cols} > {max_cols}")
        
        def validate_columns(self, required_columns, optional_columns=None):
            """Validate required columns exist"""
            missing_required = set(required_columns) - set(self.df.columns)
            if missing_required:
                self.errors.append(f"Missing required columns: {list(missing_required)}")
            
            if optional_columns:
                missing_optional = set(optional_columns) - set(self.df.columns)
                if missing_optional:
                    self.warnings.append(f"Missing optional columns: {list(missing_optional)}")
        
        def validate_data_types(self, expected_types):
            """Validate column data types"""
            for col, expected_type in expected_types.items():
                if col in self.df.columns:
                    actual_type = self.df[col].dtype
                    if actual_type != expected_type:
                        self.warnings.append(f"Column '{col}' type mismatch: {actual_type} != {expected_type}")
        
        def validate_missing_data(self, max_missing_pct=0.5):
            """Validate missing data levels"""
            for col in self.df.columns:
                missing_pct = self.df[col].isnull().sum() / len(self.df)
                if missing_pct > max_missing_pct:
                    self.errors.append(f"Column '{col}' has {missing_pct:.1%} missing data")
        
        def validate_duplicates(self, subset=None):
            """Check for duplicate rows"""
            duplicates = self.df.duplicated(subset=subset).sum()
            if duplicates > 0:
                self.warnings.append(f"Found {duplicates} duplicate rows")
        
        def validate_numeric_ranges(self, range_checks):
            """Validate numeric column ranges"""
            for col, (min_val, max_val) in range_checks.items():
                if col in self.df.columns and self.df[col].dtype in [np.number]:
                    out_of_range = ((self.df[col] < min_val) | (self.df[col] > max_val)).sum()
                    if out_of_range > 0:
                        self.warnings.append(f"Column '{col}' has {out_of_range} values out of range [{min_val}, {max_val}]")
        
        def generate_report(self):
            """Generate validation report"""
            print("VALIDATION REPORT:")
            print("-" * 20)
            
            if not self.errors and not self.warnings:
                print("✅ All validations passed!")
                return True
            
            if self.errors:
                print("❌ ERRORS (must fix):")
                for error in self.errors:
                    print(f"   • {error}")
            
            if self.warnings:
                print("⚠️  WARNINGS (should review):")
                for warning in self.warnings:
                    print(f"   • {warning}")
            
            return len(self.errors) == 0
    
    # Example usage
    test_df = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', None, 'Diana', 'Eve'],
        'age': [25, 30, 35, 150, 22],  # 150 is unrealistic
        'score': [85.5, 92.0, 78.5, 88.0, 95.0]
    })
    
    validator = DataValidator(test_df)
    
    # Run validations
    validator.validate_shape(min_rows=3, max_rows=1000)
    validator.validate_columns(['id', 'name', 'age'], ['email'])
    validator.validate_data_types({'id': 'int64', 'score': 'float64'})
    validator.validate_missing_data(max_missing_pct=0.2)
    validator.validate_duplicates()
    validator.validate_numeric_ranges({'age': (0, 120), 'score': (0, 100)})
    
    # Generate report
    is_valid = validator.generate_report()
    print(f"\nOverall validation result: {'PASS' if is_valid else 'FAIL'}")

data_validation_framework()
```

### Error Handling Summary

| Error Type | Common Cause | Prevention Strategy | Solution |
|------------|--------------|-------------------|----------|
| **KeyError** | Missing columns | Check column existence | Use `if col in df.columns` |
| **IndexError** | Row out of bounds | Validate index range | Use `if 0 <= idx < len(df)` |
| **ValueError** | Data type conversion | Validate data before conversion | Use `pd.to_numeric(..., errors='coerce')` |
| **TypeError** | Incompatible operations | Check data types | Use `df.select_dtypes()` |
| **MemoryError** | Dataset too large | Monitor memory usage | Use chunked processing |
| **PerformanceWarning** | Inefficient operations | Profile code | Use vectorized operations |

**Error Handling Best Practices:**
1. **Validate Early**: Check data quality immediately after loading
2. **Fail Fast**: Use assertions for critical assumptions
3. **Graceful Degradation**: Provide fallback options for non-critical errors
4. **Informative Messages**: Include context in error messages
5. **Log Everything**: Keep detailed logs of data issues
6. **Test Edge Cases**: Validate with empty, null, and extreme data

## Real-World Case Studies

This section demonstrates practical applications of pandas techniques using real datasets to solve business problems and answer analytical questions.

### Case Study 1: IPL Cricket Match Analysis

#### Business Questions to Answer
1. What's the impact of winning the toss on match outcomes?
2. Which teams perform best in different venues?
3. How has team performance evolved over seasons?
4. What are the patterns in close matches?

```python
# Complete IPL Analysis Pipeline
def comprehensive_ipl_analysis():
    """
    Complete analysis of IPL match data
    """
    print("IPL CRICKET MATCH ANALYSIS")
    print("=" * 40)
    
    # Ensure we have the IPL data
    if 'ipl' not in globals():
        print("⚠️  IPL dataset not loaded")
        return
    
    print(f"Dataset Overview:")
    print(f"• Total matches: {len(ipl):,}")
    print(f"• Date range: {ipl['Date'].min()} to {ipl['Date'].max()}" if 'Date' in ipl.columns else "")
    print(f"• Unique teams: {ipl['Team1'].nunique() + ipl['Team2'].nunique() - len(set(ipl['Team1']) & set(ipl['Team2']))}")
    print()
    
    # Analysis 1: Toss Impact Analysis
    print("1. TOSS IMPACT ANALYSIS:")
    print("-" * 25)
    
    toss_wins = ipl[ipl['TossWinner'] == ipl['WinningTeam']].shape[0]
    total_matches = len(ipl)
    toss_advantage = (toss_wins / total_matches) * 100
    
    print(f"Matches where toss winner won: {toss_wins:,}")
    print(f"Total matches: {total_matches:,}")
    print(f"Toss winner advantage: {toss_advantage:.1f}%")
    
    # Statistical significance test
    expected_wins = total_matches * 0.5  # Expected if no advantage
    chi_square = ((toss_wins - expected_wins) ** 2) / expected_wins
    print(f"Chi-square statistic: {chi_square:.2f}")
    
    if toss_advantage > 52:
        print("📊 INSIGHT: Toss provides meaningful advantage")
    else:
        print("📊 INSIGHT: Toss advantage is minimal - skill matters more!")
    
    # Analysis 2: Venue Performance
    print("\n2. VENUE ANALYSIS:")
    print("-" * 20)
    
    if 'Venue' in ipl.columns:
        venue_stats = ipl.groupby('Venue').agg({
            'MatchNumber': 'count',
            'SuperOver': lambda x: (x == 'Y').sum() if 'SuperOver' in ipl.columns else 0,
            'WinByRuns': lambda x: x.dropna().mean() if 'WinByRuns' in ipl.columns else None,
            'WinByWickets': lambda x: x.dropna().mean() if 'WinByWickets' in ipl.columns else None
        }).round(2)
        
        venue_stats.columns = ['Total_Matches', 'SuperOver_Matches', 'Avg_Win_Runs', 'Avg_Win_Wickets']
        venue_stats = venue_stats.sort_values('Total_Matches', ascending=False)
        
        print("Top 5 venues by matches played:")
        print(venue_stats.head())
        
        # High-scoring vs low-scoring venues
        if 'Avg_Win_Runs' in venue_stats.columns:
            high_scoring = venue_stats.nlargest(3, 'Avg_Win_Runs')
            print(f"\nHighest scoring venues (by avg win margin):")
            for venue, stats in high_scoring.iterrows():
                print(f"  {venue}: {stats['Avg_Win_Runs']:.1f} runs average margin")
    
    # Analysis 3: Team Performance Evolution
    print("\n3. TEAM PERFORMANCE EVOLUTION:")
    print("-" * 35)
    
    if 'Season' in ipl.columns:
        # Team wins per season
        team_performance = []
        
        # Combine Team1 and Team2 data for complete picture
        all_teams = pd.concat([
            ipl[['Season', 'Team1', 'WinningTeam']].rename(columns={'Team1': 'Team'}),
            ipl[['Season', 'Team2', 'WinningTeam']].rename(columns={'Team2': 'Team'})
        ])
        
        # Calculate win percentage by season
        season_stats = all_teams.groupby(['Season', 'Team']).agg({
            'Team': 'count',  # Total matches
            'WinningTeam': lambda x: (x == x.name[1]).sum()  # Wins for this team
        }).rename(columns={'Team': 'Total_Matches', 'WinningTeam': 'Wins'})
        
        season_stats['Win_Percentage'] = (season_stats['Wins'] / season_stats['Total_Matches'] * 100).round(1)
        
        # Top performing teams overall
        overall_performance = season_stats.groupby('Team').agg({
            'Total_Matches': 'sum',
            'Wins': 'sum'
        })
        overall_performance['Overall_Win_Pct'] = (overall_performance['Wins'] / overall_performance['Total_Matches'] * 100).round(1)
        overall_performance = overall_performance.sort_values('Overall_Win_Pct', ascending=False)
        
        print("Top 5 teams by overall win percentage:")
        for team, stats in overall_performance.head().iterrows():
            print(f"  {team}: {stats['Overall_Win_Pct']}% ({stats['Wins']}/{stats['Total_Matches']})")
    
    # Analysis 4: Close Matches Analysis
    print("\n4. CLOSE MATCHES ANALYSIS:")
    print("-" * 30)
    
    close_matches = 0
    
    # Define close matches (within 1 wicket or 10 runs)
    if 'WinByRuns' in ipl.columns and 'WinByWickets' in ipl.columns:
        close_by_runs = ipl[(ipl['WinByRuns'] > 0) & (ipl['WinByRuns'] <= 10)].shape[0]
        close_by_wickets = ipl[(ipl['WinByWickets'] > 0) & (ipl['WinByWickets'] <= 1)].shape[0]
        close_matches = close_by_runs + close_by_wickets
        
        close_percentage = (close_matches / total_matches) * 100
        
        print(f"Close matches (≤10 runs or ≤1 wicket): {close_matches}")
        print(f"Percentage of close matches: {close_percentage:.1f}%")
        
        if close_percentage > 30:
            print("📊 INSIGHT: IPL has many close, exciting matches!")
        else:
            print("📊 INSIGHT: Most matches have clear winners")
    
    # Super Over Analysis
    if 'SuperOver' in ipl.columns:
        super_over_matches = ipl[ipl['SuperOver'] == 'Y'].shape[0]
        super_over_pct = (super_over_matches / total_matches) * 100
        print(f"\nSuper Over matches: {super_over_matches} ({super_over_pct:.2f}%)")
        print("📊 INSIGHT: Super overs are rare but create maximum excitement!")

# Run IPL analysis
comprehensive_ipl_analysis()
```

### Case Study 2: Movie Industry Analysis

#### Business Questions to Answer
1. What factors correlate with high IMDB ratings?
2. Which genres are most successful?
3. How has movie quality changed over time?
4. What makes a movie highly rated?

```python
# Comprehensive Movie Analysis
def comprehensive_movie_analysis():
    """
    Complete analysis of movie rating data
    """
    print("\nMOVIE INDUSTRY ANALYSIS")
    print("=" * 35)
    
    if 'movies' not in globals():
        print("⚠️  Movies dataset not loaded")
        return
    
    # Dataset overview
    print(f"Dataset Overview:")
    print(f"• Total movies: {len(movies):,}")
    print(f"• Date range: {movies['year_of_release'].min()}-{movies['year_of_release'].max()}" if 'year_of_release' in movies.columns else "")
    print()
    
    # Analysis 1: Rating Distribution and Quality Factors
    print("1. RATING QUALITY ANALYSIS:")
    print("-" * 30)
    
    rating_col = None
    for col in movies.columns:
        if 'rating' in col.lower():
            rating_col = col
            break
    
    if rating_col:
        ratings = movies[rating_col].dropna()
        
        print(f"Rating Statistics ({rating_col}):")
        print(f"• Mean rating: {ratings.mean():.2f}")
        print(f"• Median rating: {ratings.median():.2f}")
        print(f"• Standard deviation: {ratings.std():.2f}")
        print(f"• Range: {ratings.min():.1f} - {ratings.max():.1f}")
        
        # Rating categories
        excellent = (ratings >= 8.0).sum()
        good = ((ratings >= 7.0) & (ratings < 8.0)).sum()
        average = ((ratings >= 6.0) & (ratings < 7.0)).sum()
        below_average = (ratings < 6.0).sum()
        
        print(f"\nRating Distribution:")
        print(f"• Excellent (8.0+): {excellent} movies ({excellent/len(ratings)*100:.1f}%)")
        print(f"• Good (7.0-7.9): {good} movies ({good/len(ratings)*100:.1f}%)")
        print(f"• Average (6.0-6.9): {average} movies ({average/len(ratings)*100:.1f}%)")
        print(f"• Below Average (<6.0): {below_average} movies ({below_average/len(ratings)*100:.1f}%)")
        
        # Quality insights
        high_quality_pct = (excellent + good) / len(ratings) * 100
        if high_quality_pct > 40:
            print(f"📊 INSIGHT: {high_quality_pct:.1f}% are high-quality movies!")
        else:
            print(f"📊 INSIGHT: Only {high_quality_pct:.1f}% are high-quality - most are average")
    
    # Analysis 2: Genre Analysis
    print("\n2. GENRE POPULARITY ANALYSIS:")
    print("-" * 35)
    
    if 'genres' in movies.columns:
        # Extract all genres
        all_genres = []
        for genre_string in movies['genres'].dropna():
            if isinstance(genre_string, str) and '|' in genre_string:
                genres = genre_string.split('|')
                all_genres.extend(genres)
            elif isinstance(genre_string, str):
                all_genres.append(genre_string)
        
        from collections import Counter
        genre_counts = Counter(all_genres)
        
        print("Top 10 most popular genres:")
        for i, (genre, count) in enumerate(genre_counts.most_common(10), 1):
            percentage = count / len(movies) * 100
            print(f"{i:2}. {genre}: {count} movies ({percentage:.1f}%)")
        
        # Genre quality analysis
        if rating_col:
            print(f"\nGenre Quality Analysis (by average {rating_col}):")
            genre_ratings = {}
            
            for genre_string in movies['genres'].dropna():
                if isinstance(genre_string, str):
                    movie_rating = movies[movies['genres'] == genre_string][rating_col].iloc[0]
                    if pd.notna(movie_rating):
                        if '|' in genre_string:
                            genres = genre_string.split('|')
                        else:
                            genres = [genre_string]
                        
                        for genre in genres:
                            if genre not in genre_ratings:
                                genre_ratings[genre] = []
                            genre_ratings[genre].append(movie_rating)
            
            # Calculate average ratings per genre
            genre_avg_ratings = {}
            for genre, ratings_list in genre_ratings.items():
                if len(ratings_list) >= 5:  # Only genres with 5+ movies
                    genre_avg_ratings[genre] = np.mean(ratings_list)
            
            # Sort by average rating
            sorted_genres = sorted(genre_avg_ratings.items(), key=lambda x: x[1], reverse=True)
            
            print("Top 5 highest-rated genres (min 5 movies):")
            for genre, avg_rating in sorted_genres[:5]:
                print(f"  {genre}: {avg_rating:.2f} average rating")
    
    # Analysis 3: Temporal Analysis
    print("\n3. TEMPORAL TRENDS ANALYSIS:")
    print("-" * 35)
    
    if 'year_of_release' in movies.columns and rating_col:
        # Group by decade for trend analysis
        movies['decade'] = (movies['year_of_release'] // 10) * 10
        decade_stats = movies.groupby('decade').agg({
            rating_col: ['count', 'mean', 'std'],
            'year_of_release': 'count'
        }).round(2)
        
        decade_stats.columns = ['Rating_Count', 'Avg_Rating', 'Rating_Std', 'Movie_Count']
        decade_stats = decade_stats[decade_stats['Movie_Count'] >= 5]  # Decades with 5+ movies
        
        print("Movie quality trends by decade:")
        for decade, stats in decade_stats.iterrows():
            print(f"{int(decade)}s: {stats['Avg_Rating']:.2f} avg rating ({int(stats['Movie_Count'])} movies)")
        
        # Best and worst decades
        if len(decade_stats) > 1:
            best_decade = decade_stats['Avg_Rating'].idxmax()
            worst_decade = decade_stats['Avg_Rating'].idxmin()
            print(f"\n📊 INSIGHT: Best decade - {int(best_decade)}s ({decade_stats.loc[best_decade, 'Avg_Rating']:.2f})")
            print(f"📊 INSIGHT: Worst decade - {int(worst_decade)}s ({decade_stats.loc[worst_decade, 'Avg_Rating']:.2f})")
    
    # Analysis 4: Success Factors Analysis
    print("\n4. SUCCESS FACTORS ANALYSIS:")
    print("-" * 35)
    
    if rating_col:
        # High-rated movie characteristics
        high_rated = movies[movies[rating_col] >= 8.0] if rating_col else pd.DataFrame()
        
        if not high_rated.empty:
            print(f"High-rated movies (8.0+): {len(high_rated)} movies")
            
            # Genre analysis for high-rated movies
            if 'genres' in movies.columns:
                high_rated_genres = []
                for genre_string in high_rated['genres'].dropna():
                    if isinstance(genre_string, str) and '|' in genre_string:
                        high_rated_genres.extend(genre_string.split('|'))
                    elif isinstance(genre_string, str):
                        high_rated_genres.append(genre_string)
                
                high_genre_counts = Counter(high_rated_genres)
                print("\nMost common genres in high-rated movies:")
                for genre, count in high_genre_counts.most_common(5):
                    percentage = count / len(high_rated) * 100
                    print(f"  {genre}: {count} movies ({percentage:.1f}%)")
            
            # Year analysis for high-rated movies
            if 'year_of_release' in high_rated.columns:
                recent_high_rated = high_rated[high_rated['year_of_release'] >= 2010]
                print(f"\nHigh-rated movies since 2010: {len(recent_high_rated)}")
                print(f"Recent quality trend: {len(recent_high_rated)/len(high_rated)*100:.1f}% of high-rated movies are recent")

# Run movie analysis
comprehensive_movie_analysis()
```

### Case Study 3: Student Performance Analysis

```python
# Student Performance Analysis
def comprehensive_student_analysis():
    """
    Analyze student performance data for educational insights
    """
    print("\nSTUDENT PERFORMANCE ANALYSIS")
    print("=" * 40)
    
    if 'students' not in globals():
        print("⚠️  Student dataset not loaded")
        return
    
    # Dataset overview
    print(f"Dataset Overview:")
    print(f"• Total students: {len(students)}")
    print(f"• Available metrics: {list(students.columns)}")
    print()
    
    # Analysis 1: Performance Distribution
    print("1. PERFORMANCE DISTRIBUTION:")
    print("-" * 30)
    
    numeric_cols = students.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        values = students[col].dropna()
        print(f"\n{col.upper()} Statistics:")
        print(f"• Mean: {values.mean():.1f}")
        print(f"• Median: {values.median():.1f}")
        print(f"• Std Dev: {values.std():.1f}")
        print(f"• Range: {values.min():.1f} - {values.max():.1f}")
        
        # Performance categories
        if 'iq' in col.lower():
            high = (values >= 120).sum()
            above_avg = ((values >= 100) & (values < 120)).sum()
            avg = ((values >= 85) & (values < 100)).sum()
            below_avg = (values < 85).sum()
            
            print(f"IQ Distribution:")
            print(f"• Superior (120+): {high} students")
            print(f"• Above Average (100-119): {above_avg} students")
            print(f"• Average (85-99): {avg} students")
            print(f"• Below Average (<85): {below_avg} students")
        
        elif 'marks' in col.lower() or 'percent' in col.lower():
            excellent = (values >= 90).sum()
            good = ((values >= 80) & (values < 90)).sum()
            satisfactory = ((values >= 70) & (values < 80)).sum()
            needs_improvement = (values < 70).sum()
            
            print(f"Academic Performance:")
            print(f"• Excellent (90+): {excellent} students")
            print(f"• Good (80-89): {good} students")
            print(f"• Satisfactory (70-79): {satisfactory} students")
            print(f"• Needs Improvement (<70): {needs_improvement} students")
    
    # Analysis 2: Correlation Analysis
    print("\n2. CORRELATION ANALYSIS:")
    print("-" * 25)
    
    if len(numeric_cols) >= 2:
        correlation_matrix = students[numeric_cols].corr()
        print("Correlation Matrix:")
        print(correlation_matrix.round(3))
        
        # Find strongest correlations
        print("\nStrongest Correlations:")
        correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                col1 = correlation_matrix.columns[i]
                col2 = correlation_matrix.columns[j]
                corr_value = correlation_matrix.iloc[i, j]
                correlations.append((abs(corr_value), col1, col2, corr_value))
        
        correlations.sort(reverse=True)
        for abs_corr, col1, col2, corr in correlations[:3]:
            strength = "Strong" if abs_corr > 0.7 else "Moderate" if abs_corr > 0.5 else "Weak"
            direction = "positive" if corr > 0 else "negative"
            print(f"• {col1} vs {col2}: {corr:.3f} ({strength} {direction})")
    
    # Analysis 3: Performance Segmentation
    print("\n3. PERFORMANCE SEGMENTATION:")
    print("-" * 35)
    
    # Create composite performance score
    if all(col in students.columns for col in ['iq', 'marks']):
        students_analysis = students.copy()
        
        # Normalize scores (0-100 scale)
        students_analysis['iq_normalized'] = (students_analysis['iq'] - students_analysis['iq'].min()) / (students_analysis['iq'].max() - students_analysis['iq'].min()) * 100
        students_analysis['marks_normalized'] = students_analysis['marks']
        
        # Composite score (weighted average)
        students_analysis['performance_score'] = (
            students_analysis['iq_normalized'] * 0.3 + 
            students_analysis['marks_normalized'] * 0.7
        )
        
        print("Composite Performance Score (30% IQ + 70% Marks):")
        print(f"• Mean: {students_analysis['performance_score'].mean():.1f}")
        print(f"• Std Dev: {students_analysis['performance_score'].std():.1f}")
        
        # Performance segments
        high_performers = students_analysis[students_analysis['performance_score'] >= 85]
        avg_performers = students_analysis[(students_analysis['performance_score'] >= 70) & (students_analysis['performance_score'] < 85)]
        low_performers = students_analysis[students_analysis['performance_score'] < 70]
        
        print(f"\nPerformance Segments:")
        print(f"• High Performers (85+): {len(high_performers)} students ({len(high_performers)/len(students)*100:.1f}%)")
        print(f"• Average Performers (70-84): {len(avg_performers)} students ({len(avg_performers)/len(students)*100:.1f}%)")
        print(f"• Low Performers (<70): {len(low_performers)} students ({len(low_performers)/len(students)*100:.1f}%)")
        
        # Insights for each segment
        if not high_performers.empty:
            print(f"\nHigh Performers Profile:")
            print(f"• Average IQ: {high_performers['iq'].mean():.1f}")
            print(f"• Average Marks: {high_performers['marks'].mean():.1f}")
        
        if not low_performers.empty:
            print(f"\nLow Performers Profile:")
            print(f"• Average IQ: {low_performers['iq'].mean():.1f}")
            print(f"• Average Marks: {low_performers['marks'].mean():.1f}")
            print("📊 INSIGHT: Focus on improving study methods and support")

# Run student analysis
comprehensive_student_analysis()
```

### Case Study 4: Business Intelligence Dashboard Data

```python
# Business Intelligence Analysis
def business_intelligence_analysis():
    """
    Create a comprehensive BI analysis combining all datasets
    """
    print("\nBUSINESS INTELLIGENCE DASHBOARD")
    print("=" * 45)
    
    # Executive Summary
    print("EXECUTIVE SUMMARY:")
    print("-" * 20)
    
    datasets_available = []
    if 'movies' in globals():
        datasets_available.append(f"Movies: {len(movies):,} records")
    if 'ipl' in globals():
        datasets_available.append(f"IPL Matches: {len(ipl):,} records")
    if 'students' in globals():
        datasets_available.append(f"Students: {len(students):,} records")
    
    print("Available Datasets:")
    for dataset in datasets_available:
        print(f"• {dataset}")
    
    # Key Performance Indicators (KPIs)
    print(f"\nKEY PERFORMANCE INDICATORS:")
    print("-" * 30)
    
    if 'movies' in globals():
        rating_col = next((col for col in movies.columns if 'rating' in col.lower()), None)
        if rating_col:
            avg_rating = movies[rating_col].mean()
            high_rated_pct = (movies[rating_col] >= 8.0).sum() / len(movies) * 100
            print(f"Movies KPIs:")
            print(f"• Average Rating: {avg_rating:.2f}/10")
            print(f"• High Quality Rate: {high_rated_pct:.1f}%")
    
    if 'ipl' in globals():
        toss_advantage = (ipl[ipl['TossWinner'] == ipl['WinningTeam']].shape[0] / len(ipl)) * 100
        close_matches = 0
        if 'WinByRuns' in ipl.columns:
            close_matches = ipl[(ipl['WinByRuns'] <= 10) | (ipl['WinByWickets'] <= 1)].shape[0]
        close_match_pct = (close_matches / len(ipl)) * 100 if close_matches > 0 else 0
        
        print(f"IPL KPIs:")
        print(f"• Toss Advantage: {toss_advantage:.1f}%")
        print(f"• Close Matches: {close_match_pct:.1f}%")
    
    if 'students' in globals():
        numeric_cols = students.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            avg_performance = students[numeric_cols].mean().mean()
            high_performers = (students[numeric_cols].mean(axis=1) >= students[numeric_cols].mean(axis=1).quantile(0.8)).sum()
            high_performer_pct = high_performers / len(students) * 100
            
            print(f"Education KPIs:")
            print(f"• Average Performance: {avg_performance:.1f}")
            print(f"• Top Performers: {high_performer_pct:.1f}%")
    
    # Recommendations
    print(f"\nSTRATEGIC RECOMMENDATIONS:")
    print("-" * 30)
    
    if 'movies' in globals() and rating_col:
        if movies[rating_col].mean() < 7.0:
            print("📈 Movies: Focus on quality improvement - ratings below industry standard")
        else:
            print("✅ Movies: Maintain current quality standards")
    
    if 'ipl' in globals():
        if toss_advantage > 55:
            print("🏏 IPL: Consider pitch conditions - high toss advantage detected")
        else:
            print("✅ IPL: Balanced competition - minimal toss advantage")
    
    if 'students' in globals():
        print("🎓 Education: Implement data-driven performance tracking")

# Run comprehensive BI analysis
business_intelligence_analysis()
```

### Real-World Analysis Techniques Summary

| Analysis Type | Use Case | Key Techniques | Business Value |
|---------------|----------|----------------|----------------|
| **Descriptive** | Understanding current state | Mean, median, distribution | Baseline understanding |
| **Diagnostic** | Why did something happen? | Correlation, segmentation | Root cause analysis |
| **Predictive** | What will happen? | Trends, patterns | Strategic planning |
| **Prescriptive** | What should we do? | Optimization, recommendations | Action planning |

**Key Takeaways from Case Studies:**
1. **Start with Business Questions**: Always begin with specific questions to answer
2. **Multiple Perspectives**: Analyze data from different angles (time, segments, categories)
3. **Context Matters**: Interpret results within domain knowledge
4. **Actionable Insights**: Focus on findings that drive decisions
5. **Validate Results**: Cross-check findings with domain experts
6. **Communicate Clearly**: Present insights in business-friendly language

### 1. Boolean Operations

```python
# ✅ Correct: Use parentheses with multiple conditions
result = df[(df['col1'] > 5) & (df['col2'] == 'value')]

# ❌ Incorrect: Missing parentheses
# result = df[df['col1'] > 5 & df['col2'] == 'value']
```

### 2. String Operations

```python
# ✅ Recommended: Precise genre matching
mask = df['genres'].str.split('|').apply(lambda x: 'Action' in x)

# ⚠️ Be careful: May have false positives
mask = df['genres'].str.contains('Action')
```

### 3. Memory Management

```python
# Check data types and memory usage
df.info()

# Convert appropriate columns to categories
df['category_col'] = df['category_col'].astype('category')

# Use smaller numeric types when possible
df['small_int'] = df['small_int'].astype('int32')
```

### 4. Data Quality

```python
# Always check for missing data
print("Missing values per column:")
print(df.isnull().sum())

# Check for duplicates
print(f"Duplicate rows: {df.duplicated().sum()}")

# Handle missing values appropriately
df_clean = df.dropna()  # or df.fillna(value)
```

### 5. Efficient Filtering

```python
# For simple conditions
result = df[df['column'] > value]

# For complex conditions, create reusable functions
def complex_filter(df, conditions):
    mask = True
    for condition in conditions:
        mask = mask & condition(df)
    return df[mask]
```

## Summary

This guide covers the essential pandas DataFrame operations including:

- **Data Creation**: From lists, dictionaries, and CSV files
- **Data Inspection**: Using shape, dtypes, info(), describe()
- **Data Selection**: Column/row selection with iloc and loc
- **Data Filtering**: Boolean indexing and complex conditions
- **String Operations**: Precise genre filtering with split() and apply()
- **Data Transformation**: Adding columns and type conversion
- **Performance**: Memory optimization with appropriate data types

### Key Takeaways

1. **Always use parentheses** when combining boolean conditions
2. **Prefer str.split() + apply()** over str.contains() for precise matching
3. **Use appropriate data types** to optimize memory usage
4. **Check data quality** before analysis (missing values, duplicates)
5. **Create reusable functions** for complex filtering operations

## Common Pitfalls and How to Avoid Them

Understanding common mistakes helps prevent hours of debugging and ensures reliable data analysis.

### Memory and Performance Pitfalls

#### Pitfall 1: Loading Entire Dataset When Subset Needed
```python
# ❌ WRONG - Loading all data then filtering
df = pd.read_csv('huge_dataset.csv')  # 10GB file
subset = df[df['category'] == 'A']    # Only need 1% of data

# ✅ CORRECT - Filter during loading
subset = pd.read_csv('huge_dataset.csv', 
                    usecols=['id', 'category', 'value'],  # Only needed columns
                    chunksize=10000)                       # Process in chunks

# Or with SQL-like filtering
subset = pd.read_csv('huge_dataset.csv', 
                    skiprows=lambda x: x % 100 != 0)     # Sample every 100th row
```

#### Pitfall 2: Inefficient Data Types
```python
# ❌ WRONG - Using default data types
df = pd.read_csv('data.csv')
print(df.dtypes)
# category    object     <- Should be category
# date        object     <- Should be datetime
# amount      float64    <- Could be float32

# ✅ CORRECT - Optimize data types
df = pd.read_csv('data.csv', 
                dtype={'category': 'category',
                       'amount': 'float32'},
                parse_dates=['date'])

# Memory usage comparison
def memory_usage_comparison():
    # Create sample data
    data = pd.DataFrame({
        'category': ['A', 'B', 'C'] * 100000,
        'amount': np.random.randn(300000)
    })
    
    print("MEMORY USAGE COMPARISON:")
    print("-" * 25)
    
    # Before optimization
    memory_before = data.memory_usage(deep=True).sum()
    print(f"Before optimization: {memory_before / 1024**2:.1f} MB")
    
    # After optimization
    data_optimized = data.copy()
    data_optimized['category'] = data_optimized['category'].astype('category')
    data_optimized['amount'] = data_optimized['amount'].astype('float32')
    
    memory_after = data_optimized.memory_usage(deep=True).sum()
    print(f"After optimization: {memory_after / 1024**2:.1f} MB")
    print(f"Memory savings: {(memory_before - memory_after) / memory_before * 100:.1f}%")

memory_usage_comparison()
```

#### Pitfall 3: Chain Assignment Warnings
```python
# ❌ WRONG - Chained assignment (creates warnings)
df[df['score'] > 80]['grade'] = 'A'  # SettingWithCopyWarning

# ✅ CORRECT - Proper assignment methods
# Method 1: Use .loc
df.loc[df['score'] > 80, 'grade'] = 'A'

# Method 2: Use .copy() if creating subset
high_scores = df[df['score'] > 80].copy()
high_scores['grade'] = 'A'

# Method 3: Use assign for functional style
df = df.assign(grade=lambda x: np.where(x['score'] > 80, 'A', 
                                      np.where(x['score'] > 70, 'B', 'C')))
```

### Data Type and Conversion Pitfalls

#### Pitfall 4: Mixed Data Types in Columns
```python
# Example of problematic data
mixed_data = pd.DataFrame({
    'amounts': ['100', '200.5', 'N/A', '300', None, 'invalid']
})

# ❌ WRONG - Direct conversion without error handling
try:
    mixed_data['amounts'] = mixed_data['amounts'].astype(float)
except ValueError as e:
    print(f"Error: {e}")

# ✅ CORRECT - Safe conversion with error handling
def safe_numeric_conversion(series, default_value=np.nan):
    """
    Safely convert series to numeric, handling errors gracefully
    """
    # Method 1: Using pd.to_numeric with errors='coerce'
    numeric_series = pd.to_numeric(series, errors='coerce')
    
    # Report conversion issues
    invalid_count = numeric_series.isna().sum() - series.isna().sum()
    if invalid_count > 0:
        print(f"⚠️  {invalid_count} values couldn't be converted to numeric")
        invalid_values = series[pd.to_numeric(series, errors='coerce').isna() & series.notna()]
        print(f"Invalid values: {invalid_values.unique()}")
    
    return numeric_series

# Apply safe conversion
mixed_data['amounts_clean'] = safe_numeric_conversion(mixed_data['amounts'])
print("Conversion Results:")
print(mixed_data)
```

### Best Practices Summary

| Category | ❌ Avoid | ✅ Do Instead |
|----------|----------|---------------|
| **Memory** | `pd.read_csv('huge_file.csv')` | Use `chunksize`, `usecols`, optimize dtypes |
| **Assignment** | `df[condition]['col'] = value` | `df.loc[condition, 'col'] = value` |
| **Data Types** | `astype()` without error handling | `pd.to_numeric(errors='coerce')` |
| **Missing Data** | Immediate `dropna()` | Analyze patterns first |
| **Grouping** | Simple aggregations on weighted data | Custom weighted functions |
| **Merging** | Default parameters | Analyze keys and use appropriate join type |
| **Performance** | Loops over DataFrames | Vectorized operations |
| **Index** | Ignore duplicate indices | Verify and clean indices |

## Final Summary and Quick Reference

### Complete Pandas DataFrame Workflow
1. **Data Loading**: Use optimized loading with appropriate dtypes
2. **Data Exploration**: Understand structure, types, and quality
3. **Data Cleaning**: Handle missing values, duplicates, and inconsistencies
4. **Data Analysis**: Apply filtering, grouping, and mathematical operations
5. **Performance Optimization**: Use vectorized operations and memory-efficient approaches
6. **Error Handling**: Implement robust validation and error recovery
7. **Documentation**: Comment complex operations and maintain reproducible workflows

### Key Performance Tips
- **Memory**: Use `category` dtype for repeated strings, `float32` instead of `float64` when precision allows
- **Speed**: Prefer vectorized operations over loops, use `.loc` for conditional assignments
- **Reliability**: Always validate data quality, handle edge cases, use error-tolerant conversion methods

### Essential DataFrame Operations Quick Reference

| Operation | Syntax | Use Case |
|-----------|--------|----------|
| **Create** | `pd.DataFrame(data)` | Initialize DataFrame |
| **Load** | `pd.read_csv('file.csv')` | Load from file |
| **Info** | `df.info()`, `df.describe()` | Basic exploration |
| **Filter** | `df[df['col'] > value]` | Row filtering |
| **Select** | `df[['col1', 'col2']]` | Column selection |
| **Group** | `df.groupby('col').agg(func)` | Aggregation |
| **Sort** | `df.sort_values('col')` | Ordering |
| **Join** | `df1.merge(df2, on='key')` | Combining DataFrames |
| **Apply** | `df['col'].apply(func)` | Custom operations |
| **Assign** | `df.loc[condition, 'col'] = value` | Safe assignment |

This comprehensive approach to pandas DataFrames enables efficient data analysis and forms the foundation for more advanced data science workflows.
