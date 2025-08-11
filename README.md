# python_pandas
my pandas practice

🐼 Pandas Practice — Data Cleaning & Analysis
✍️ Created by Somayeh Doosti
📅 Date: July 2025

📌 Description
This repository is a hands-on collection of Pandas practice scripts (single .py file examples) that walk through common data-wrangling and analysis tasks.
Each section is written as a small, self-contained example with short explanatory comments, so you can read, run, and adapt the code quickly. It’s ideal for beginners who are learning how to clean, aggregate and analyze tabular data using pandas (and a bit of numpy).

📚 Topics Covered
✅ Creating Series and DataFrame from dictionaries / lists

✅ Indexing & selection (loc, iloc, boolean filters)

✅ Adding / modifying / dropping columns and rows

✅ Descriptive statistics (describe, mean) and sorting

✅ Handling missing data (isnull, dropna, fillna)

✅ groupby aggregations and multi-aggregation (mean, sum, max)

✅ pivot_table and multi-column analysis

✅ Type conversion and applying custom functions (astype, apply)

✅ Reading & writing CSV files (read_csv, to_csv)

✅ Small analysis pipelines (packaged as functions for reuse)

🔎 Example analyses included
The code contains several small example datasets and analyses to demonstrate real workflows:

Student grades — average, pass/fail flags, top students, class-wise aggregation.

Salary Series — simple Series operations and filters.

Health / BMI analysis — BMI calculation and risk classification.

Transport analysis — date parsing, weekday extraction, travel time & delay aggregation.

Real estate — price per square meter and city-level summaries.

Movie ratings — grouping by year & genre, top movie selection.

Customer service — satisfaction summary and per-service averages.

Online sales pipeline — Read_Data, Clean_data, Fillna, Total_Price, check_Sales, Max_sales, Sort_data, Seller — shows a small end-to-end example and writes online_sales.csv.

🛠️ How to use
Clone the repository or download the .py file.

bash
Copy
Edit
git clone <your-repo-url>
cd <repo-folder>
(Recommended) create a virtual environment:

bash
Copy
Edit
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
.venv\Scripts\activate      # Windows
Install dependencies:

bash
Copy
Edit
pip install pandas numpy matplotlib seaborn
Or if you add a requirements.txt:

bash
Copy
Edit
pip install -r requirements.txt
Run the script:

bash
Copy
Edit
python pandas_practice.py
— or open it in a Jupyter Notebook (recommended for step-by-step exploration) and run cells interactively:

bash
Copy
Edit
jupyter notebook
Examine printed outputs and generated CSV files (the script writes sales_data.csv / online_sales.csv in some examples).

✅ Good practices used in the code
Modular example blocks with explanatory comments.

Small helper functions for repeated operations (e.g., cleaning / total price).

Use of groupby, pivot_table, and datetime handling for realistic use-cases.

Demonstration of NaN-handling strategies (drop vs fill with mean).


👩‍🏫 Author
Somayeh Doosti — Master's student in Mathematics | Beginner Python Developer | Passionate about Data & Problem Solving
📍 Tehran, Iran

🧾 License
This project is open-source and free to use for learning purposes. (Suggested: MIT License — add a LICENSE file if you want to make it official.)

🔖 Tags
#pandas_practice #data_cleaning #groupby #missing_values #data_analysis #python #beginner
