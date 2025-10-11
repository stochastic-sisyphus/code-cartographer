# Code Analysis Report

## Overview

- **Total Files**: 4
- **Total Definitions**: 14
- **Orphaned Code Elements**: 8


## Call Graph Analysis

The following analysis shows the relationships between functions and methods in the codebase.

### Function Call Relationships

#### Most Called Functions

- **len**: Called by 4 functions
- **append**: Called by 3 functions
- **min**: Called by 1 functions
- **sum**: Called by 1 functions
- **max**: Called by 1 functions
- **calculate_sum**: Called by 1 functions
- **print**: Called by 1 functions
- **add_data**: Called by 1 functions
- **DataProcessor**: Called by 1 functions
- **get_count**: Called by 1 functions

#### Functions with Most Dependencies

- **run_example**: Calls 7 other functions
- **process_list**: Calls 4 other functions
- **DataProcessor**: Calls 2 other functions
- **DataProcessor.add_data**: Calls 1 other functions
- **DataProcessor.get_count**: Calls 1 other functions
- **add_data**: Calls 1 other functions
- **get_count**: Calls 1 other functions


## Orphaned Code Analysis

The following code elements are defined but never used in the codebase.

### Orphaned Code Elements

- **DataProcessor.__init__**
- **DataProcessor.add_data**
- **DataProcessor.get_count**
- **DataProcessor.clear**
- **__init__**
- **clear**
- **calculate_product**
- **run_example**


## Variable Usage Analysis

This section shows how variables are defined and used across the codebase.

### Variables with Multiple Definitions

- **self**: Defined 4 times
- **a**: Defined 2 times
- **b**: Defined 2 times


## File Analysis

This section provides detailed analysis for each file in the codebase.

### processor.py


#### Definitions

- **DataProcessor** (class, 19 lines)
  - Called by: run_example
  - Prerequisites: len, append
- **DataProcessor.__init__** (method, 4 lines) (🔕 Orphan)
- **DataProcessor.add_data** (method, 3 lines) (🔕 Orphan)
  - Prerequisites: append
- **DataProcessor.get_count** (method, 3 lines) (🔕 Orphan)
  - Prerequisites: len
- **DataProcessor.clear** (method, 3 lines) (🔕 Orphan)
- **__init__** (function, 4 lines) (🔕 Orphan)
  - Prerequisites: clear, get_count, add_data
- **add_data** (function, 3 lines)
  - Called by: run_example
  - Prerequisites: processor.py, append, __init__, clear, get_count
- **get_count** (function, 3 lines)
  - Called by: run_example
  - Prerequisites: __init__, len, clear, add_data
- **clear** (function, 3 lines) (🔕 Orphan)
  - Prerequisites: __init__, get_count, add_data
- **process_list** (function, 16 lines)
  - Called by: run_example
  - Prerequisites: sum, processor.py, min, len, max

#### Orphaned Code

The following definitions are never called:

- DataProcessor.__init__
- DataProcessor.add_data
- DataProcessor.get_count
- DataProcessor.clear
- __init__
- clear


### utils.py


#### Definitions

- **calculate_sum** (function, 3 lines)
  - Called by: run_example
  - Prerequisites: calculate_product
- **calculate_product** (function, 3 lines) (🔕 Orphan)
  - Prerequisites: calculate_sum
- **format_output** (function, 3 lines)
  - Called by: run_example

#### Orphaned Code

The following definitions are never called:

- calculate_product


### main.py


#### Definitions

- **run_example** (function, 18 lines) (🔕 Orphan)
  - Prerequisites: main.py, print, calculate_sum, add_data, DataProcessor and 3 more

#### Orphaned Code

The following definitions are never called:

- run_example


### __init__.py


#### Definitions


