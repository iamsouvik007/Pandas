# Pandas DataFrame Methods - Comprehensive Guide

This guide covers essential DataFrame and Series methods in pandas, with detailed explanations and practical examples.

## Table of Contents
1. [value_counts()](#value_counts)
2. [sort_values()](#sort_values)
3. [rank()](#rank)
4. [sort_index()](#sort_index)
5. [set_index()](#set_index)
6. [reset_index()](#reset_index)
7. [rename()](#rename)
8. [unique() & nunique()](#unique--nunique)
9. [isnull() / notnull() / hasnans](#null-checking-methods)
10. [dropna()](#dropna)
11. [fillna()](#fillna)
12. [drop_duplicates()](#drop_duplicates)
13. [drop()](#drop)
14. [apply()](#apply)
15. [isin()](#isin)
16. [corr()](#corr)
17. [nlargest() & nsmallest()](#nlargest--nsmallest)
18. [insert()](#insert)
19. [copy()](#copy)

---

## value_counts()

**Purpose**: Counts the frequency of each unique value in a Series or DataFrame.

**Works with**: Series and DataFrame

**Syntax**: 
- `series.value_counts()`
- `dataframe.value_counts()`

**Key Parameters**:
- `normalize`: If True, returns relative frequencies (proportions)
- `sort`: If True, sorts by frequency (default)
- `ascending`: Sort order (default False - highest first)
- `dropna`: Include/exclude NaN values (default True - exclude)

### Detailed Examples:

#### Series Examples:

```python
# Basic value_counts for Series
a = pd.Series([1,1,1,2,2,3])
print(a.value_counts())
```
**Output:**
```
1    3
2    2
3    1
Name: count, dtype: int64
```

```python
# With proportions instead of counts
print(a.value_counts(normalize=True))
```
**Output:**
```
1    0.500000
2    0.333333
3    0.166667
Name: proportion, dtype: float64
```

```python
# Including NaN values
b = pd.Series([1,1,2,2,3,np.nan,np.nan])
print(b.value_counts(dropna=False))
```
**Output:**
```
1.0    2
2.0    2
NaN    2
3.0    1
Name: count, dtype: int64
```

#### DataFrame Examples:

```python
# For DataFrame - counts unique combinations of all columns
marks = pd.DataFrame([
    [100,80,10],
    [90,70,7],
    [120,100,14],
    [80,70,14],
    [80,70,14]
], columns=['iq','marks','package'])

print(marks.value_counts())
```
**Output:**
```
iq   marks  package
80   70     14         2
90   70     7          1
100  80     10         1
120  100    14         1
Name: count, dtype: int64
```

### Real-world Use Cases:

#### 1. Categorical Data Analysis:
```python
# Analyzing cricket toss decisions
ipl = pd.read_csv('ipl-matches.csv')
toss_counts = ipl['TossDecision'].value_counts()
print(toss_counts)
```
**Output:**
```
field    456
bat      399
Name: count, dtype: int64
```

#### 2. Finding Top Performers:
```python
# Most Player of the Match awards in finals/qualifiers
playoff_matches = ipl[~ipl['MatchNumber'].str.isdigit()]
potm_counts = playoff_matches['Player_of_Match'].value_counts()
print(potm_counts.head())
```
**Output:**
```
MS Dhoni          4
AB de Villiers    3
V Kohli          3
S Watson         2
RG Sharma        2
Name: count, dtype: int64
```

#### 3. Data Visualization:
```python
# Creating visualizations
import matplotlib.pyplot as plt

# Pie chart for toss decisions
ipl['TossDecision'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Toss Decision Distribution in IPL')
plt.ylabel('')  # Remove default ylabel
plt.show()
```

---

## sort_values()

**Purpose**: Sorts data by values in specified column(s).

**Works with**: Series and DataFrame

**Key Parameters**:
- `by`: Column name(s) to sort by (DataFrame only)
- `ascending`: True/False or list of booleans (default True)
- `na_position`: 'first' or 'last' (where to place NaN values, default 'last')
- `inplace`: True/False (modify original object, default False)
- `kind`: Sorting algorithm ('quicksort', 'mergesort', 'heapsort', 'stable')
- `ignore_index`: If True, reset index after sorting

### Detailed Examples:

#### Series Sorting:

```python
# Simple Series sorting
x = pd.Series([12,14,1,56,89], index=['a','b','c','d','e'])
print("Original Series:")
print(x)
print("\nSorted (ascending):")
print(x.sort_values())
print("\nSorted (descending):")
print(x.sort_values(ascending=False))
```
**Output:**
```
Original Series:
a    12
b    14
c     1
d    56
e    89
dtype: int64

Sorted (ascending):
c     1
a    12
b    14
d    56
e    89
dtype: int64

Sorted (descending):
e    89
d    56
b    14
a    12
c     1
dtype: int64
```

#### DataFrame Single Column Sorting:

```python
# Load movies data
movies = pd.read_csv('movies.csv')
print("Original data (first 3 rows):")
print(movies[['title_x', 'year_of_release']].head(3))

print("\nSorted by title (descending):")
sorted_movies = movies.sort_values('title_x', ascending=False)
print(sorted_movies[['title_x', 'year_of_release']].head(3))
```
**Output:**
```
Original data (first 3 rows):
                    title_x  year_of_release
0                     Dangal             2016
1                      Bahubali           2015
2           Uri: The Surgical Strike     2019

Sorted by title (descending):
                    title_x  year_of_release
45                   Zindagi Na Milegi Dobara  2011
32                          Zero               2018
12                          War                2019
```

#### Multi-Column Sorting:

```python
# Sort by multiple columns with different orders
print("Multi-column sorting (year ascending, title descending):")
multi_sorted = movies.sort_values(['year_of_release','title_x'], 
                                 ascending=[True, False])
print(multi_sorted[['title_x', 'year_of_release']].head(5))
```
**Output:**
```
Multi-column sorting (year ascending, title descending):
                    title_x  year_of_release
23                    Taare Zameen Par      2007
15                    Om Shanti Om           2007
8                     Jab We Met           2007
34                    3 Idiots              2009
28                    My Name is Khan       2010
```

#### Handling NaN Values:

```python
# Create DataFrame with NaN values
students = pd.DataFrame({
    'name': ['souvik', 'ankit', 'rupesh', np.nan, 'mrityunjay', 'rishabh'],
    'cgpa': [8.5, 8.25, 6.41, np.nan, 5.6, 7.4]
})

print("Original DataFrame:")
print(students)

print("\nSorted with NaN first:")
sorted_nan_first = students.sort_values('name', na_position='first')
print(sorted_nan_first)

print("\nSorted with NaN last:")
sorted_nan_last = students.sort_values('name', na_position='last')
print(sorted_nan_last)
```
**Output:**
```
Original DataFrame:
        name  cgpa
0     souvik   8.50
1      ankit   8.25
2     rupesh   6.41
3        NaN    NaN
4  mrityunjay   5.60
5    rishabh   7.40

Sorted with NaN first:
        name  cgpa
3        NaN    NaN
1      ankit   8.25
4  mrityunjay   5.60
5    rishabh   7.40
2     rupesh   6.41
0     souvik   8.50

Sorted with NaN last:
        name  cgpa
1      ankit   8.25
4  mrityunjay   5.60
5    rishabh   7.40
2     rupesh   6.41
0     souvik   8.50
3        NaN    NaN
```

### Advanced Usage:

#### 1. Complex Multi-Column Sorting:
```python
# Sort by year (ascending), then by rating (descending), then by title (ascending)
complex_sort = movies.sort_values(['year_of_release', 'imdb_rating', 'title_x'], 
                                 ascending=[True, False, True])
```

#### 2. Inplace Sorting:
```python
# Modify original DataFrame
students.sort_values('cgpa', ascending=False, inplace=True)
print("After inplace sorting by CGPA:")
print(students)
```
**Output:**
```
After inplace sorting by CGPA:
        name  cgpa
0     souvik   8.50
1      ankit   8.25
5    rishabh   7.40
2     rupesh   6.41
4  mrityunjay   5.60
3        NaN    NaN
```

#### 3. Sorting with Index Reset:
```python
# Reset index after sorting
clean_sorted = students.sort_values('name', ignore_index=True)
print("Sorted with reset index:")
print(clean_sorted)
```
**Output:**
```
Sorted with reset index:
        name  cgpa
0      ankit   8.25
1  mrityunjay   5.60
2    rishabh   7.40
3     rupesh   6.41
4     souvik   8.50
5        NaN    NaN
```

### Common Patterns:
- **Leaderboards**: Sort by score/performance descending
- **Chronological data**: Sort by date/time ascending
- **Alphabetical lists**: Sort by name ascending
- **Top-N analysis**: Sort descending + head(n)

---

## rank()

**Purpose**: Assigns ranks to values in a Series (1 = highest/lowest depending on ascending parameter).

**Works with**: Series

**Key Parameters**:
- `ascending`: True/False
- `method`: How to handle ties ('average', 'min', 'max', 'first', 'dense')

### Example:

```python
# Adding ranking to batsman runs
batsman['batting_rank'] = batsman['batsman_run'].rank(ascending=False)
batsman.sort_values('batting_rank')
```

**Use Cases**:
- Creating leaderboards
- Performance rankings
- Percentile calculations

---

## sort_index()

**Purpose**: Sorts by index labels instead of values.

**Works with**: Series and DataFrame

### Examples:

```python
# Series with custom index
marks_series = pd.Series({'maths':67, 'english':57, 'science':89, 'hindi':100})
marks_series.sort_index(ascending=False)

# DataFrame index sorting
movies.sort_index(ascending=False)
```

**When to use**:
- When you need data ordered by index labels
- Organizing time-series data by dates
- Alphabetical ordering by row names

---

## set_index()

**Purpose**: Sets one or more columns as the DataFrame index.

**Works with**: DataFrame

**Key Parameters**:
- `inplace`: True/False
- `drop`: True/False (whether to drop the column being set as index)

### Example:

```python
# Set 'batter' column as index
batsman.set_index('batter', inplace=True)
```

**Benefits**:
- Faster lookups by index
- Better for time-series analysis
- Enables index-based operations

---

## reset_index()

**Purpose**: Resets index back to default integer index, optionally keeping old index as a column.

**Works with**: Series and DataFrame

**Key Parameters**:
- `drop`: True/False (whether to drop the old index)
- `inplace`: True/False

### Examples:

```python
# Reset index, keeping old index as column
batsman.reset_index(inplace=True)

# Convert Series to DataFrame using reset_index
marks_series.reset_index()

# Replace existing index without losing it
batsman.reset_index().set_index('batting_rank')
```

---

## rename()

**Purpose**: Renames column labels or index labels.

**Works with**: DataFrame

**Key Parameters**:
- `columns`: Dictionary mapping old names to new names
- `index`: Dictionary mapping old index values to new values
- `inplace`: True/False

### Examples:

```python
# Rename columns
movies.rename(columns={'imdb_id':'imdb', 'poster_path':'link'}, inplace=True)

# Rename index values
movies.rename(index={'Uri: The Surgical Strike':'Uri', 'Battalion 609':'Battalion'})
```

---

## unique() & nunique()

**Purpose**: 
- `unique()`: Returns array of unique values in a Series
- `nunique()`: Returns count of unique values

**Works with**: 
- `unique()`: Series only
- `nunique()`: Series and DataFrame

**Key Parameters**:
- `unique()`: No parameters
- `nunique()`: `dropna` (True/False - whether to exclude NaN values)

### Detailed Examples:

#### unique() Examples:

```python
# Basic unique() usage
temp = pd.Series([1,1,2,2,3,3,4,4,5,5,np.nan,np.nan])
print("Original Series:")
print(temp)
print("\nUnique values:")
unique_vals = temp.unique()
print(unique_vals)
print(f"Number of unique values: {len(unique_vals)}")
```
**Output:**
```
Original Series:
0     1.0
1     1.0
2     2.0
3     2.0
4     3.0
5     3.0
6     4.0
7     4.0
8     5.0
9     5.0
10    NaN
11    NaN
dtype: float64

Unique values:
[ 1.  2.  3.  4.  5. nan]
Number of unique values: 6
```

```python
# Unique values in categorical data
categories = pd.Series(['A', 'B', 'A', 'C', 'B', 'A', 'D'])
print("Categories:")
print(categories)
print("\nUnique categories:")
print(categories.unique())
```
**Output:**
```
Categories:
0    A
1    B
2    A
3    C
4    B
5    A
6    D
dtype: object

Unique categories:
['A' 'B' 'C' 'D']
```

#### nunique() Examples:

```python
# Count unique values (excludes NaN by default)
print(f"Number of unique values (excluding NaN): {temp.nunique()}")
print(f"Number of unique values (including NaN): {temp.nunique(dropna=False)}")
```
**Output:**
```
Number of unique values (excluding NaN): 5
Number of unique values (including NaN): 6
```

#### DataFrame nunique() Examples:

```python
# Sample DataFrame
df = pd.DataFrame({
    'A': [1, 2, 3, 1, 2, np.nan],
    'B': ['x', 'y', 'x', 'z', 'y', 'x'],
    'C': [10, 20, 10, 30, 20, 10]
})

print("DataFrame:")
print(df)
print("\nUnique counts per column:")
print(df.nunique())
print("\nUnique counts per column (including NaN):")
print(df.nunique(dropna=False))
```
**Output:**
```
DataFrame:
     A  B   C
0  1.0  x  10
1  2.0  y  20
2  3.0  x  10
3  1.0  z  30
4  2.0  y  20
5  NaN  x  10

Unique counts per column:
A    3
B    3
C    3
dtype: int64

Unique counts per column (including NaN):
A    4
B    3
C    3
dtype: int64
```

### Real-world Examples:

#### 1. Analyzing IPL Data:

```python
# Load IPL data
ipl = pd.read_csv('ipl-matches.csv')

# How many unique seasons?
print(f"Number of IPL seasons: {ipl['Season'].nunique()}")
print("Unique seasons:")
print(sorted(ipl['Season'].unique()))

# How many unique teams?
teams = pd.concat([ipl['Team1'], ipl['Team2']]).unique()
print(f"\nNumber of unique teams: {len(teams)}")
print("All teams:")
print(sorted(teams))
```
**Output:**
```
Number of IPL seasons: 12
Unique seasons:
[2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]

Number of unique teams: 15
All teams:
['Chennai Super Kings', 'Deccan Chargers', 'Delhi Capitals', 'Delhi Daredevils', 
 'Gujarat Lions', 'Kings XI Punjab', 'Kochi Tuskers Kerala', 'Kolkata Knight Riders', 
 'Mumbai Indians', 'Pune Warriors', 'Punjab Kings', 'Rajasthan Royals', 
 'Rising Pune Supergiant', 'Royal Challengers Bangalore', 'Sunrisers Hyderabad']
```

#### 2. Data Quality Assessment:

```python
# Check uniqueness in each column
print("Data uniqueness summary:")
for col in df.columns:
    total_rows = len(df)
    unique_count = df[col].nunique()
    unique_percentage = (unique_count / total_rows) * 100
    print(f"{col}: {unique_count}/{total_rows} unique values ({unique_percentage:.1f}%)")
```
**Output:**
```
Data uniqueness summary:
A: 3/6 unique values (50.0%)
B: 3/6 unique values (50.0%)
C: 3/6 unique values (50.0%)
```

#### 3. Finding Potential Key Columns:

```python
# Identify columns that could be primary keys (100% unique)
potential_keys = []
for col in df.columns:
    if df[col].nunique() == len(df):
        potential_keys.append(col)

if potential_keys:
    print(f"Potential key columns: {potential_keys}")
else:
    print("No columns have 100% unique values")
```

### Key Differences Summary:

| Method | Returns | Includes NaN | Use Case |
|--------|---------|--------------|----------|
| `unique()` | Array of values | Yes (by default) | See actual unique values |
| `nunique()` | Integer count | No (by default) | Quick count for analysis |

### Performance Notes:
- `nunique()` is faster than `len(unique())` for just counting
- `unique()` loads all values into memory, `nunique()` just counts
- For large datasets, use `nunique()` when you only need the count

---

## Null Checking Methods

### isnull() / notnull() / hasnans

**Purpose**: Check for missing (NaN) values.

**Works with**: 
- `isnull()`, `notnull()`: Series and DataFrame
- `hasnans`: Series only

### Examples:

```python
# Check for null values in Series
students['name'][students['name'].isnull()]

# Check for non-null values
students['name'][students['name'].notnull()]

# Check if Series has any NaN values
students['name'].hasnans  # Returns True/False

# For entire DataFrame
students.isnull()  # Returns boolean DataFrame
students.notnull()  # Returns boolean DataFrame
```

---

## dropna()

**Purpose**: Removes rows/columns containing NaN values.

**Works with**: Series and DataFrame

**Key Parameters**:
- `axis`: 0 (rows) or 1 (columns) - what to drop
- `how`: 'any' (drop if any NaN) or 'all' (drop if all NaN)
- `subset`: List of columns to consider for NaN detection
- `thresh`: Minimum number of non-NaN values required
- `inplace`: True/False (modify original object)

### Detailed Examples:

#### Series dropna():

```python
# Series with NaN values
names = pd.Series(['Alice', 'Bob', np.nan, 'Charlie', np.nan, 'Diana'])
print("Original Series:")
print(names)
print(f"Length: {len(names)}")

# Remove NaN values
clean_names = names.dropna()
print("\nAfter dropna():")
print(clean_names)
print(f"Length: {len(clean_names)}")
```
**Output:**
```
Original Series:
0      Alice
1        Bob
2        NaN
3    Charlie
4        NaN
5      Diana
dtype: object
Length: 6

After dropna():
0      Alice
1        Bob
3    Charlie
5      Diana
dtype: object
Length: 4
```

#### DataFrame dropna() - Basic Usage:

```python
# Create DataFrame with missing values
students = pd.DataFrame({
    'name': ['souvik', 'ankit', 'rupesh', np.nan, 'mrityunjay', np.nan, 'rishabh'],
    'college': ['bbit', 'iit', 'vit', np.nan, np.nan, 'vlsi', 'ssit'],
    'branch': ['cse', 'it', 'cse', np.nan, 'me', 'ce', 'civ'],
    'cgpa': [8.5, 8.25, 6.41, np.nan, 5.6, 9.0, 7.4],
    'package': [40, 5, 6, np.nan, 6, 7, 8]
})

print("Original DataFrame:")
print(students)
print(f"Shape: {students.shape}")
```
**Output:**
```
Original DataFrame:
        name college branch  cgpa  package
0     souvik    bbit    cse   8.50       40
1      ankit     iit     it   8.25        5
2     rupesh     vit    cse   6.41        6
3        NaN     NaN    NaN    NaN      NaN
4  mrityunjay     NaN     me   5.60        6
5        NaN    vlsi     ce   9.00        7
6    rishabh    ssit    civ   7.40        8
Shape: (7, 5)
```

#### Different 'how' parameters:

```python
# Drop rows with ANY NaN value
any_dropped = students.dropna(how='any')
print("Dropped rows with ANY NaN:")
print(any_dropped)
print(f"Shape: {any_dropped.shape}")
```
**Output:**
```
Dropped rows with ANY NaN:
      name college branch  cgpa  package
0   souvik    bbit    cse   8.5       40
1    ankit     iit     it  8.25        5
2   rupesh     vit    cse  6.41        6
6  rishabh    ssit    civ   7.4        8
Shape: (4, 5)
```

```python
# Drop rows with ALL NaN values
all_dropped = students.dropna(how='all')
print("Dropped rows with ALL NaN:")
print(all_dropped)
print(f"Shape: {all_dropped.shape}")
```
**Output:**
```
Dropped rows with ALL NaN:
        name college branch  cgpa  package
0     souvik    bbit    cse   8.50       40
1      ankit     iit     it   8.25        5
2     rupesh     vit    cse   6.41        6
4  mrityunjay     NaN     me   5.60        6
5        NaN    vlsi     ce   9.00        7
6    rishabh    ssit    civ   7.40        8
Shape: (6, 5)
```

#### Using subset parameter:

```python
# Drop rows with NaN only in specific columns
subset_dropped = students.dropna(subset=['name'])
print("Dropped rows with NaN in 'name' column:")
print(subset_dropped)
print(f"Shape: {subset_dropped.shape}")
```
**Output:**
```
Dropped rows with NaN in 'name' column:
        name college branch  cgpa  package
0     souvik    bbit    cse   8.50       40
1      ankit     iit     it   8.25        5
2     rupesh     vit    cse   6.41        6
4  mrityunjay     NaN     me   5.60        6
6    rishabh    ssit    civ   7.40        8
Shape: (5, 5)
```

```python
# Drop rows with NaN in multiple specific columns
multi_subset = students.dropna(subset=['name', 'college'])
print("Dropped rows with NaN in 'name' OR 'college':")
print(multi_subset)
print(f"Shape: {multi_subset.shape}")
```
**Output:**
```
Dropped rows with NaN in 'name' OR 'college':
      name college branch  cgpa  package
0   souvik    bbit    cse   8.5       40
1    ankit     iit     it  8.25        5
2   rupesh     vit    cse  6.41        6
6  rishabh    ssit    civ   7.4        8
Shape: (4, 5)
```

#### Using thresh parameter:

```python
# Keep rows with at least 3 non-NaN values
thresh_dropped = students.dropna(thresh=3)
print("Rows with at least 3 non-NaN values:")
print(thresh_dropped)
print(f"Shape: {thresh_dropped.shape}")
```
**Output:**
```
Rows with at least 3 non-NaN values:
        name college branch  cgpa  package
0     souvik    bbit    cse   8.50       40
1      ankit     iit     it   8.25        5
2     rupesh     vit    cse   6.41        6
4  mrityunjay     NaN     me   5.60        6
5        NaN    vlsi     ce   9.00        7
6    rishabh    ssit    civ   7.40        8
Shape: (6, 5)
```

#### Dropping columns:

```python
# Create DataFrame with some completely empty columns
df_with_empty_cols = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [np.nan, np.nan, np.nan],  # All NaN
    'C': [4, np.nan, 6],
    'D': [np.nan, np.nan, np.nan]   # All NaN
})

print("DataFrame with empty columns:")
print(df_with_empty_cols)

# Drop columns with all NaN values
clean_cols = df_with_empty_cols.dropna(axis=1, how='all')
print("\nAfter dropping columns with all NaN:")
print(clean_cols)
```
**Output:**
```
DataFrame with empty columns:
   A   B    C   D
0  1 NaN  4.0 NaN
1  2 NaN  NaN NaN
2  3 NaN  6.0 NaN

After dropping columns with all NaN:
   A    C
0  1  4.0
1  2  NaN
2  3  6.0
```

### Real-world Data Cleaning Examples:

#### 1. Survey Data Cleaning:

```python
# Simulate survey data
survey = pd.DataFrame({
    'respondent_id': [1, 2, 3, 4, 5],
    'age': [25, np.nan, 35, 42, np.nan],
    'income': [50000, 60000, np.nan, 75000, 45000],
    'satisfaction': [4, 5, np.nan, 3, 4],
    'comments': ['Good', np.nan, np.nan, 'Average', 'Excellent']
})

print("Survey data:")
print(survey)

# Strategy 1: Keep responses with core demographic info
core_demo = survey.dropna(subset=['age', 'income'])
print(f"\nWith complete demographic info: {len(core_demo)} rows")
print(core_demo)

# Strategy 2: Keep responses with satisfaction rating
with_rating = survey.dropna(subset=['satisfaction'])
print(f"\nWith satisfaction rating: {len(with_rating)} rows")
print(with_rating)
```
**Output:**
```
Survey data:
   respondent_id   age   income  satisfaction  comments
0              1  25.0  50000.0           4.0      Good
1              2   NaN  60000.0           5.0       NaN
2              3  35.0      NaN           NaN       NaN
3              4  42.0  75000.0           3.0   Average
4              5   NaN  45000.0           4.0 Excellent

With complete demographic info: 2 rows
   respondent_id   age   income  satisfaction  comments
0              1  25.0  50000.0           4.0      Good
3              4  42.0  75000.0           3.0   Average

With satisfaction rating: 4 rows
   respondent_id   age   income  satisfaction  comments
0              1  25.0  50000.0           4.0      Good
1              2   NaN  60000.0           5.0       NaN
3              4  42.0  75000.0           3.0   Average
4              5   NaN  45000.0           4.0 Excellent
```

#### 2. Time Series Data Cleaning:

```python
# Simulate sensor data with missing readings
dates = pd.date_range('2023-01-01', periods=10, freq='D')
sensor_data = pd.DataFrame({
    'date': dates,
    'temperature': [20, np.nan, 22, 21, np.nan, np.nan, 19, 20, 21, np.nan],
    'humidity': [45, 47, np.nan, 46, 44, np.nan, 43, 45, 46, 47],
    'pressure': [1013, 1012, 1014, np.nan, 1015, 1013, np.nan, 1012, 1014, 1013]
})

print("Sensor data:")
print(sensor_data)

# Keep days with at least 2 out of 3 measurements
reliable_data = sensor_data.dropna(thresh=3)  # date + 2 measurements
print(f"\nReliable readings (at least 2 sensors working): {len(reliable_data)} days")
print(reliable_data)
```
**Output:**
```
Sensor data:
        date  temperature  humidity  pressure
0 2023-01-01         20.0      45.0    1013.0
1 2023-01-02          NaN      47.0    1012.0
2 2023-01-03         22.0       NaN    1014.0
3 2023-01-04         21.0      46.0       NaN
4 2023-01-05          NaN      44.0    1015.0
5 2023-01-06          NaN       NaN    1013.0
6 2023-01-07         19.0      43.0       NaN
7 2023-01-08         20.0      45.0    1012.0
8 2023-01-09         21.0      46.0    1014.0
9 2023-01-10          NaN      47.0    1013.0

Reliable readings (at least 2 sensors working): 7 days
        date  temperature  humidity  pressure
0 2023-01-01         20.0      45.0    1013.0
1 2023-01-02          NaN      47.0    1012.0
2 2023-01-03         22.0       NaN    1014.0
3 2023-01-04         21.0      46.0       NaN
4 2023-01-05          NaN      44.0    1015.0
6 2023-01-07         19.0      43.0       NaN
7 2023-01-08         20.0      45.0    1012.0
8 2023-01-09         21.0      46.0    1014.0
9 2023-01-10          NaN      47.0    1013.0
```

### Decision Strategy Guide:

| Scenario | Recommended Approach |
|----------|---------------------|
| **Few missing values** | `dropna(how='any')` |
| **Many missing values** | `dropna(thresh=n)` or subset approach |
| **Critical columns** | `dropna(subset=['critical_col'])` |
| **Completely empty rows/cols** | `dropna(how='all')` |
| **Time series** | Consider forward/backward fill first |
| **Survey data** | Drop based on key questions |

### Performance Tips:
- Use `inplace=True` to save memory for large datasets
- Consider the order: drop duplicates first, then handle missing values
- Document your dropna strategy for reproducibility
- Always check the impact: `print(f"Dropped {original_len - new_len} rows")`

---

## fillna()

**Purpose**: Fills NaN values with specified values or methods.

**Works with**: Series and DataFrame

**Key Parameters**:
- `value`: Value to fill NaN with (scalar, dict, Series, or DataFrame)
- `method`: 'ffill'/'pad' (forward fill), 'bfill'/'backfill' (backward fill)
- `axis`: 0 (rows) or 1 (columns) for method-based filling
- `inplace`: True/False (modify original object)
- `limit`: Maximum number of consecutive NaN values to fill
- `downcast`: Downcast data type if possible

### Detailed Examples:

#### Series fillna():

```python
# Series with missing values
scores = pd.Series([85, np.nan, 92, np.nan, 78, 88, np.nan])
print("Original Series:")
print(scores)
```
**Output:**
```
Original Series:
0    85.0
1     NaN
2    92.0
3     NaN
4    78.0
5    88.0
6     NaN
dtype: float64
```

#### Fill with constant value:

```python
# Fill with a constant value
filled_constant = scores.fillna(0)
print("Filled with 0:")
print(filled_constant)

# Fill with mean
filled_mean = scores.fillna(scores.mean())
print(f"\nFilled with mean ({scores.mean():.1f}):")
print(filled_mean)

# Fill with median
filled_median = scores.fillna(scores.median())
print(f"\nFilled with median ({scores.median():.1f}):")
print(filled_median)
```
**Output:**
```
Filled with 0:
0    85.0
1     0.0
2    92.0
3     0.0
4    78.0
5    88.0
6     0.0
dtype: float64

Filled with mean (85.8):
0    85.0
1    85.8
2    92.0
3    85.8
4    78.0
5    88.0
6    85.8
dtype: float64

Filled with median (86.5):
0    85.0
1    86.5
2    92.0
3    86.5
4    78.0
5    88.0
6    86.5
dtype: float64
```

#### Forward and Backward Fill:

```python
# Forward fill (use previous valid value)
filled_forward = scores.fillna(method='ffill')
print("Forward fill:")
print(filled_forward)

# Backward fill (use next valid value)
filled_backward = scores.fillna(method='bfill')
print("\nBackward fill:")
print(filled_backward)

# Combined approach: forward fill then backward fill
filled_combined = scores.fillna(method='ffill').fillna(method='bfill')
print("\nCombined (ffill then bfill):")
print(filled_combined)
```
**Output:**
```
Forward fill:
0    85.0
1    85.0
2    92.0
3    92.0
4    78.0
5    88.0
6    88.0
dtype: float64

Backward fill:
0    85.0
1    92.0
2    92.0
3    78.0
4    78.0
5    88.0
6     NaN
dtype: float64

Combined (ffill then bfill):
0    85.0
1    85.0
2    92.0
3    92.0
4    78.0
5    88.0
6    88.0
dtype: float64
```

#### DataFrame fillna() - Basic Usage:

```python
# Create DataFrame with missing values
students = pd.DataFrame({
    'name': ['Alice', 'Bob', np.nan, 'Charlie', np.nan],
    'age': [25, np.nan, 30, 28, np.nan],
    'score': [85, 92, np.nan, 78, 88],
    'grade': ['A', 'A', 'B', np.nan, 'A']
})

print("Original DataFrame:")
print(students)
```
**Output:**
```
Original DataFrame:
      name   age  score grade
0    Alice  25.0   85.0     A
1      Bob   NaN   92.0     A
2      NaN  30.0    NaN     B
3  Charlie  28.0   78.0   NaN
4      NaN   NaN   88.0     A
```

#### Fill with different values for different columns:

```python
# Fill different columns with different values
fill_values = {
    'name': 'Unknown',
    'age': students['age'].mean(),
    'score': students['score'].median(),
    'grade': 'C'
}

filled_dict = students.fillna(value=fill_values)
print("Filled with different values per column:")
print(filled_dict)
```
**Output:**
```
Filled with different values per column:
      name   age  score grade
0    Alice  25.0   85.0     A
1      Bob  27.7   92.0     A
2  Unknown  30.0   86.5     B
3  Charlie  28.0   78.0     C
4  Unknown  27.7   88.0     A
```

#### Column-specific filling strategies:

```python
# Copy for demonstration
df_copy = students.copy()

# Fill categorical columns with mode (most frequent value)
df_copy['grade'] = df_copy['grade'].fillna(df_copy['grade'].mode()[0])

# Fill numerical columns with mean
df_copy['age'] = df_copy['age'].fillna(df_copy['age'].mean())
df_copy['score'] = df_copy['score'].fillna(df_copy['score'].mean())

# Fill text columns with placeholder
df_copy['name'] = df_copy['name'].fillna('Not Provided')

print("Strategic filling:")
print(df_copy)
```
**Output:**
```
Strategic filling:
          name        age      score grade
0        Alice  25.000000  85.000000     A
1          Bob  27.666667  92.000000     A
2 Not Provided  30.000000  85.666667     B
3      Charlie  28.000000  78.000000     A
4 Not Provided  27.666667  88.000000     A
```

### Advanced fillna() Examples:

#### Time Series Forward/Backward Fill:

```python
# Time series data
dates = pd.date_range('2023-01-01', periods=10, freq='D')
temp_data = pd.DataFrame({
    'date': dates,
    'temperature': [20, np.nan, np.nan, 23, 24, np.nan, 22, 21, np.nan, 20]
})

print("Temperature time series:")
print(temp_data)

# Forward fill for time series (carry last observation forward)
temp_data['temp_ffill'] = temp_data['temperature'].fillna(method='ffill')

# Backward fill
temp_data['temp_bfill'] = temp_data['temperature'].fillna(method='bfill')

# Interpolation (linear)
temp_data['temp_interp'] = temp_data['temperature'].interpolate()

print("\nDifferent filling methods:")
print(temp_data)
```
**Output:**
```
Temperature time series:
        date  temperature
0 2023-01-01         20.0
1 2023-01-02          NaN
2 2023-01-03          NaN
3 2023-01-04         23.0
4 2023-01-05         24.0
5 2023-01-06          NaN
6 2023-01-07         22.0
7 2023-01-08         21.0
8 2023-01-09          NaN
9 2023-01-10         20.0

Different filling methods:
        date  temperature  temp_ffill  temp_bfill  temp_interp
0 2023-01-01         20.0        20.0        20.0         20.0
1 2023-01-02          NaN        20.0        23.0         21.0
2 2023-01-03          NaN        20.0        23.0         22.0
3 2023-01-04         23.0        23.0        23.0         23.0
4 2023-01-05         24.0        24.0        24.0         24.0
5 2023-01-06          NaN        24.0        22.0         23.0
6 2023-01-07         22.0        22.0        22.0         22.0
7 2023-01-08         21.0        21.0        21.0         21.0
8 2023-01-09          NaN        21.0        20.0         20.5
9 2023-01-10         20.0        20.0        20.0         20.0
```

#### Conditional Filling:

```python
# Fill based on other column values
products = pd.DataFrame({
    'product': ['A', 'B', 'C', 'D', 'E'],
    'category': ['Electronics', 'Clothing', 'Electronics', np.nan, 'Clothing'],
    'price': [100, np.nan, 150, 200, np.nan],
    'rating': [4.5, 3.8, np.nan, 4.2, 4.0]
})

print("Products data:")
print(products)

# Fill price based on category averages
def fill_price_by_category(row):
    if pd.isna(row['price']):
        if row['category'] == 'Electronics':
            return 125  # Average for Electronics
        elif row['category'] == 'Clothing':
            return 75   # Average for Clothing
        else:
            return 100  # Default
    return row['price']

products['price_filled'] = products.apply(fill_price_by_category, axis=1)
print("\nWith category-based price filling:")
print(products)
```
**Output:**
```
Products data:
  product     category  price  rating
0       A  Electronics  100.0     4.5
1       B     Clothing    NaN     3.8
2       C  Electronics  150.0     NaN
3       D          NaN  200.0     4.2
4       E     Clothing    NaN     4.0

With category-based price filling:
  product     category  price  rating  price_filled
0       A  Electronics  100.0     4.5         100.0
1       B     Clothing    NaN     3.8          75.0
2       C  Electronics  150.0     NaN         150.0
3       D          NaN  200.0     4.2         200.0
4       E     Clothing    NaN     4.0          75.0
```

#### Limit parameter usage:

```python
# Series with many consecutive NaN values
data_gaps = pd.Series([1, 2, np.nan, np.nan, np.nan, np.nan, 7, 8])
print("Data with large gaps:")
print(data_gaps)

# Forward fill with limit
limited_fill = data_gaps.fillna(method='ffill', limit=2)
print("\nForward fill with limit=2:")
print(limited_fill)
```
**Output:**
```
Data with large gaps:
0    1.0
1    2.0
2    NaN
3    NaN
4    NaN
5    NaN
6    7.0
7    8.0
dtype: float64

Forward fill with limit=2:
0    1.0
1    2.0
2    2.0
3    2.0
4    NaN
5    NaN
6    7.0
7    8.0
dtype: float64
```

### Real-world Filling Strategies:

#### 1. Customer Data:

```python
customers = pd.DataFrame({
    'customer_id': [1, 2, 3, 4, 5],
    'age': [25, np.nan, 35, np.nan, 45],
    'income': [50000, 60000, np.nan, 55000, np.nan],
    'region': ['North', 'South', np.nan, 'East', 'West'],
    'last_purchase': ['2023-01-15', np.nan, '2023-01-20', np.nan, '2023-01-10']
})

# Fill age with median by region
age_by_region = customers.groupby('region')['age'].median()
customers['age_filled'] = customers.apply(
    lambda row: age_by_region[row['region']] if pd.isna(row['age']) and pd.notna(row['region']) 
    else row['age'] if pd.notna(row['age']) else customers['age'].median(), axis=1
)

# Fill income with mean
customers['income_filled'] = customers['income'].fillna(customers['income'].mean())

# Fill region with 'Unknown'
customers['region_filled'] = customers['region'].fillna('Unknown')

print("Customer data with strategic filling:")
print(customers[['customer_id', 'age', 'age_filled', 'income', 'income_filled', 'region', 'region_filled']])
```

### Best Practices Summary:

| Data Type | Recommended Fill Strategy |
|-----------|--------------------------|
| **Numerical** | Mean, median, or interpolation |
| **Categorical** | Mode (most frequent) or 'Unknown' |
| **Time Series** | Forward fill, backward fill, or interpolation |
| **Business Critical** | Domain-specific rules or external data |
| **Binary/Boolean** | Most frequent value or business logic |

### Performance Tips:
- Use `inplace=True` for large datasets to save memory
- Group-based filling is more accurate than global statistics
- Consider the data distribution before choosing fill strategy
- Document your filling logic for reproducibility
- Validate results after filling (check for outliers)

---

## drop_duplicates()

**Purpose**: Removes duplicate rows from DataFrame or Series.

**Works with**: Series and DataFrame

**Key Parameters**:
- `subset`: Columns to consider for duplicates
- `keep`: 'first', 'last', or False (keep which occurrence)
- `inplace`: True/False

### Examples:

```python
# Remove duplicates from Series
temp = pd.Series([1,1,1,2,3,3,4,4])
temp.drop_duplicates()

# Remove duplicates from DataFrame
marks.drop_duplicates(keep='last')

# Remove based on specific columns
students.drop_duplicates(subset=['name'], keep='first')
```

**Real-world Example**:
```python
# Find last match played by Virat Kohli in Delhi
ipl[(ipl['City'] == 'Delhi') & (ipl['did_kohli_play'] == True)].drop_duplicates(
    subset=['City','did_kohli_play'], keep='first'
)
```

---

## drop()

**Purpose**: Removes specified rows or columns.

**Works with**: Series and DataFrame

**Key Parameters**:
- `index`: Row labels to drop
- `columns`: Column labels to drop
- `inplace`: True/False

### Examples:

```python
# Drop by index positions (Series)
temp.drop(index=[0,6])

# Drop columns (DataFrame)
students.drop(columns=['branch','cgpa'], inplace=True)

# Drop by index values
students.set_index('name').drop(index=['nitish','aditya'])
```

---

## apply()

**Purpose**: Applies a function along axis of DataFrame or to each element of Series.

**Works with**: Series and DataFrame

**Key Parameters**:
- `func`: Function to apply
- `axis`: 0 (rows) or 1 (columns) for DataFrame
- `args`: Additional positional arguments to pass to function
- `result_type`: 'expand', 'reduce', 'broadcast' (DataFrame only)

### Detailed Examples:

#### Series apply() Examples:

```python
# Basic mathematical transformation
temp = pd.Series([10, 20, 30, 40, 50])
print("Original Series:")
print(temp)

# Apply simple function
def square(x):
    return x ** 2

squared = temp.apply(square)
print("\nSquared values:")
print(squared)
```
**Output:**
```
Original Series:
0    10
1    20
2    30
3    40
4    50
dtype: int64

Squared values:
0     100
1     400
2     900
3    1600
4    2500
dtype: int64
```

```python
# Apply lambda function
cubed = temp.apply(lambda x: x ** 3)
print("Cubed values:")
print(cubed)
```
**Output:**
```
Cubed values:
0     1000
1     8000
2    27000
3    64000
4   125000
dtype: int64
```

```python
# Complex function with conditional logic
def categorize_number(x):
    if x < 25:
        return "Low"
    elif x < 45:
        return "Medium"
    else:
        return "High"

categories = temp.apply(categorize_number)
print("Categorized values:")
print(categories)
```
**Output:**
```
Categorized values:
0       Low
1       Low
2    Medium
3    Medium
4      High
dtype: object
```

#### DataFrame apply() Examples:

##### Row-wise operations (axis=1):

```python
# Create sample DataFrame
points_df = pd.DataFrame({
    '1st_point': [(3,4), (-6,5), (0,0), (-10,1), (4,5)],
    '2nd_point': [(-3,4), (0,0), (2,2), (10,10), (1,1)]
})

print("Points DataFrame:")
print(points_df)

# Calculate Euclidean distance between points
def euclidean_distance(row):
    pt_A = row['1st_point']
    pt_B = row['2nd_point']
    distance = ((pt_A[0] - pt_B[0])**2 + (pt_A[1] - pt_B[1])**2)**0.5
    return round(distance, 2)

points_df['distance'] = points_df.apply(euclidean_distance, axis=1)
print("\nWith calculated distances:")
print(points_df)
```
**Output:**
```
Points DataFrame:
   1st_point 2nd_point
0      (3, 4)    (-3, 4)
1     (-6, 5)     (0, 0)
2      (0, 0)     (2, 2)
3    (-10, 1)   (10, 10)
4      (4, 5)     (1, 1)

With calculated distances:
   1st_point 2nd_point  distance
0      (3, 4)    (-3, 4)      6.00
1     (-6, 5)     (0, 0)      7.81
2      (0, 0)     (2, 2)      2.83
3    (-10, 1)   (10, 10)     22.47
4      (4, 5)     (1, 1)      5.00
```

##### Column-wise operations (axis=0):

```python
# Sample numerical DataFrame
data = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': [10, 20, 30, 40, 50],
    'C': [100, 200, 300, 400, 500]
})

print("Original DataFrame:")
print(data)

# Apply function to each column
def column_summary(col):
    return f"Mean: {col.mean():.1f}, Max: {col.max()}"

summary = data.apply(column_summary, axis=0)
print("\nColumn summaries:")
print(summary)
```
**Output:**
```
Original DataFrame:
   A   B    C
0  1  10  100
1  2  20  200
2  3  30  300
3  4  40  400
4  5  50  500

Column summaries:
A    Mean: 3.0, Max: 5
B    Mean: 30.0, Max: 50
C    Mean: 300.0, Max: 500
dtype: object
```

#### Advanced apply() Examples:

##### Multiple return values:

```python
# Function that returns multiple values
def stats(series):
    return pd.Series({
        'mean': series.mean(),
        'std': series.std(),
        'min': series.min(),
        'max': series.max()
    })

detailed_stats = data.apply(stats, axis=0)
print("Detailed statistics:")
print(detailed_stats)
```
**Output:**
```
Detailed statistics:
              A          B           C
mean       3.0       30.0       300.0
std   1.581139  15.811388  158.113883
min        1.0       10.0       100.0
max        5.0       50.0       500.0
```

##### Real-world example - Text processing:

```python
# Text analysis example
reviews = pd.Series([
    "This movie is amazing and fantastic!",
    "Terrible film, waste of time.",
    "Good story but poor acting.",
    "Excellent cinematography and direction."
])

def analyze_text(text):
    words = text.lower().split()
    word_count = len(words)
    positive_words = ['amazing', 'fantastic', 'good', 'excellent']
    negative_words = ['terrible', 'waste', 'poor']
    
    positive_score = sum(1 for word in words if word.rstrip('.,!') in positive_words)
    negative_score = sum(1 for word in words if word.rstrip('.,!') in negative_words)
    
    return pd.Series({
        'word_count': word_count,
        'positive_score': positive_score,
        'negative_score': negative_score,
        'sentiment': 'positive' if positive_score > negative_score else 'negative'
    })

analysis = reviews.apply(analyze_text)
print("Text analysis results:")
print(analysis)
```
**Output:**
```
Text analysis results:
   word_count  positive_score  negative_score sentiment
0           5               2               0  positive
1           5               0               2  negative
2           6               1               1  negative
3           4               1               0  positive
```

### Performance Considerations:

#### Vectorized operations vs apply():

```python
# Slow: using apply for simple operations
slow_result = temp.apply(lambda x: x * 2)

# Fast: using vectorized operations
fast_result = temp * 2

# Both give same result, but vectorized is much faster
print("Results are equal:", slow_result.equals(fast_result))
```
**Output:**
```
Results are equal: True
```

### When to Use apply():

✅ **Good use cases:**
- Complex custom functions
- Conditional logic
- Multiple return values
- Text processing
- Row-wise calculations involving multiple columns

❌ **Avoid apply() for:**
- Simple mathematical operations (use vectorized operations)
- Built-in pandas functions (use `.sum()`, `.mean()`, etc.)
- Element-wise comparisons (use boolean indexing)

### Common apply() Patterns:

```python
# Pattern 1: Data cleaning
df['cleaned_text'] = df['text_column'].apply(lambda x: x.strip().lower())

# Pattern 2: Feature engineering
df['feature'] = df.apply(lambda row: complex_calculation(row['col1'], row['col2']), axis=1)

# Pattern 3: Conditional assignment
df['category'] = df['score'].apply(lambda x: 'high' if x > 80 else 'low')

# Pattern 4: Multiple column operations
df[['new_col1', 'new_col2']] = df.apply(multi_return_function, axis=1, result_type='expand')
```

---

## isin()

**Purpose**: Checks if values are contained in a specified list.

**Works with**: Series

**Returns**: Boolean Series

### Example:

```python
# Check if values are in a list
series.isin([1, 2, 3])

# Filter DataFrame based on values in list
df[df['column'].isin(['value1', 'value2'])]
```

---

## corr()

**Purpose**: Computes correlation matrix between numerical columns.

**Works with**: DataFrame

**Returns**: Correlation matrix as DataFrame

### Example:

```python
# Calculate correlation between all numerical columns
df.corr()

# Specific correlation
df['column1'].corr(df['column2'])
```

**Use Cases**:
- Feature selection in machine learning
- Understanding relationships between variables
- Data exploration and analysis

---

## nlargest() & nsmallest()

**Purpose**: Returns the n largest/smallest values.

**Works with**: Series and DataFrame

### Examples:

```python
# Get 5 largest values from Series
series.nlargest(5)

# Get 3 smallest values from DataFrame based on specific column
df.nsmallest(3, 'column_name')
```

**Advantages over sort_values()**:
- More efficient for getting top/bottom n values
- Cleaner syntax for common operations

---

## insert()

**Purpose**: Inserts a column at specified location in DataFrame.

**Works with**: DataFrame

**Parameters**:
- `loc`: Integer position
- `column`: Column name
- `value`: Values for the column

### Example:

```python
# Insert new column at position 1
df.insert(1, 'new_column', values)
```

---

## copy()

**Purpose**: Creates a copy of the DataFrame/Series.

**Works with**: Series and DataFrame

**Parameters**:
- `deep`: True (deep copy) or False (shallow copy)

### Example:

```python
# Create independent copy
df_copy = df.copy(deep=True)

# Shallow copy (shares memory for data)
df_shallow = df.copy(deep=False)
```

**When to use**:
- Before making modifications you might want to undo
- When passing DataFrames to functions that modify data
- Creating backup copies for experimentation

---

## Summary of Key Concepts

### Data Cleaning Workflow:
1. **Explore**: `info()`, `describe()`, `isnull().sum()`
2. **Clean**: `dropna()`, `fillna()`, `drop_duplicates()`
3. **Transform**: `apply()`, `rename()`, `astype()`
4. **Analyze**: `value_counts()`, `corr()`, `sort_values()`

### Performance Tips:
- Use `inplace=True` to avoid creating copies when possible
- `nunique()` is faster than `len(unique())` for counting
- `nlargest()`/`nsmallest()` is more efficient than full sorting for top-n operations
- Set appropriate data types to save memory and improve performance

### Common Patterns:
- **Ranking**: `rank()` + `sort_values()`
- **Top performers**: `nlargest()` or `sort_values().head()`
- **Data validation**: `isnull()`, `duplicated()`, `unique()`
- **Data transformation**: `apply()` with custom functions

This guide covers the essential DataFrame and Series methods you'll use regularly in data analysis and manipulation tasks.
