'''
Tasks of a Data Engineer

1. Clean corrupt data
2. Set up processes to bring together data
3. Develop scalable architecture
4. Streamline data acquisition
5. Cloud technology

Tools of a Data Engineer

1. Databases
    -Some databases are used only for analysis
2. Processing
    -Clean, aggregate, join databases (parallel processing)
3. Scheduling
    -Plan jobs with specific intervals
    -Resolve dependency requirements of jobs
    -The scheduler is the glue of a data engineering system, holding each small
     piece together and organizing how they work together.

Ex)
1. Databases: MySQL, PostgreSQL
2. Processing: Apache Spark, Hive
3. Scheduling: Airflow, Oozie

Cloud Services

1. AWS (Amazon Web Services)
2. Azure (Microsoft)
3. Google Cloud (Google)

-Storage.
    AWS S3, Azure Blob Storage, Google Cloud Storage
-Computation
    Hosting a web server.
    AWS EC2, Azure Virtual Machines, Google Compute Engine
-Databases
    Examples of cloud databases (SQL type)
    AWS RDS, Azure SQL Database, Google Cloud SQL
'''
###############################################################################
###############################################################################
'''
Databases
    -Databases hold data, organize data, and retrieve/search data through DBMS

Structured and unstructured data.
    -Structured : Database schema, relational database
    -Unstructured: Schemaless, videos or photos

SQL and NoSQL
    -SQL: Tables, Database schema, relational databases, MySQL, PostgreSQL
    -NoSQL: Non-relational databases, Structured or unstructured
            Key-value stores (caching), document DB (JSON objects), redis,
            mongoDB

SQL: Star schema
    -One or more fact tables referencing any number of dimension tables.
    -Facts: Things that happened
    -Dimensions: Information on the world
'''
#Query a SQL database from Python using pandas

# Complete the SELECT statement
data = pd.read_sql("""
SELECT first_name, last_name FROM "Customer"
ORDER BY last_name, first_name
""", db_engine)

# Show the first 3 rows of the DataFrame
print(data.head(3))

# Show the info of the DataFrame
print(data.info())

###############################################################################
#Joining two databases

# Complete the SELECT statement
data = pd.read_sql("""
SELECT * FROM "Customer"
INNER JOIN "Order"
ON "Order"."customer_id"="Customer"."id"
""", db_engine)

# Show the id column of data
print(data.id)

###############################################################################
'''
Parallel Computing

Memory and processing power
Splits task into subtasks
Distributes subtasks over several computers
Combines results for faster completion time
Memory: Partitions the dataset
RISKS: Overhead due to communication
    -Task needs to be large
    -Need several processing units
'''
from multiprocessing import Pool

# Function to apply a function over multiple cores
@print_timing
def parallel_apply(apply_func, groups, nb_cores):
    with Pool(nb_cores) as p:
        results = p.map(apply_func, groups)
    return pd.concat(results)

# Parallel apply using 1 core
parallel_apply(take_mean_age, athlete_events.groupby('Year'), 1)

# Parallel apply using 2 cores
parallel_apply(take_mean_age, athlete_events.groupby('Year'), 2)

# Parallel apply using 4 cores
parallel_apply(take_mean_age, athlete_events.groupby('Year'), 4)

###############################################################################
import dask.dataframe as dd

# Set the number of pratitions
athlete_events_dask = dd.from_pandas(athlete_events, npartitions = 4)

# Calculate the mean Age per Year
print(athlete_events_dask.groupby('Year').Age.mean().compute())

###############################################################################
'''
Parallel Computing Frameworks

Apache hadoop
1. HDFS
2. MapReduce

    Hive: Runs on Hadoop, Hive SQL, initially MapReduce

Apache spark: Avoid disk writes

    Resilient distributed datasets (RDDs)
        -Spark relies on them
        -Similar to list of tuples
        -Transformations: .map() or .filter()
        -Actions: .count() or .first()
PySpark
    -Python interface to Spark
    -Similar to Pandas dataframes.
'''
# Print the type of athlete_events_spark
print(type(athlete_events_spark))

# Print the schema of athlete_events_spark
print(athlete_events_spark.printSchema())

# Group by the Year, and find the mean Age
print(athlete_events_spark.groupBy('Year').mean('Age'))

# Group by the Year, and find the mean Age
print(athlete_events_spark.groupBy('Year').mean('Age').show())

###############################################################################
'''
Workflow Scheduling Framework

-How to schedule?
1. Manually
2. cron scheduling tool

Use a Directed Acyclic Graph (DAG)
-Set of nodes
-Directed edges
-No cycles

Tools:
-Linx's cron
-Luigi
-Apache Airflow
'''
# Create the DAG object
dag = DAG(dag_id="car_factory_simulation",
          default_args={"owner": "airflow",
          "start_date": airflow.utils.dates.days_ago(2)},
          schedule_interval="0 * * * *")

# Task definitions
assemble_frame = BashOperator(task_id="assemble_frame",
                              bash_command='echo "Assembling frame"', dag=dag)
place_tires = BashOperator(task_id="place_tires",
                           bash_command='echo "Placing tires"', dag=dag)
assemble_body = BashOperator(task_id="assemble_body",
                             bash_command='echo "Assembling body"', dag=dag)
apply_paint = BashOperator(task_id="apply_paint",
                           bash_command='echo "Applying paint"', dag=dag)

# Complete the downstream flow
assemble_frame.set_downstream(place_tires)
assemble_frame.set_downstream(assemble_body)
assemble_body.set_downstream(apply_paint)

###############################################################################
###############################################################################
'''
Extract, Transform, Load (ETL)

Data on the web
    -Requests: How to get data.
    -APIs: Application programming interfaces, typically in JSON files.

Data in databases
    Application databases
        -Transactions
        -Inserts or changes
        -OLTP
        -Row Oriented
    Analytical databases
        -OLAP
        -Column Oriented

Extraction from databases
    Connection string/URI
'''
#Example Extract

import requests

# Fetch the Hackernews post
resp = requests.get("https://hacker-news.firebaseio.com/v0/item/16222426.json")

# Print the response parsed as JSON
print(resp.json())

# Assign the score of the test to post_score
post_score = resp.json()["score"]
print(post_score)

#Read from a database

# Function to extract table to a pandas DataFrame
def extract_table_to_pandas(tablename, db_engine):
    query = "SELECT * FROM {}".format(tablename)
    return pd.read_sql(query, db_engine)

# Connect to the database using the connection URI
connection_uri = "postgresql://repl:password@localhost:5432/pagila"
db_engine = sqlalchemy.create_engine(connection_uri)

# Extract the film table into a pandas DataFrame
extract_table_to_pandas("film", db_engine)

# Extract the customer table into a pandas DataFrame
extract_table_to_pandas("customer", db_engine)

###############################################################################
#Transform

# Get the rental rate column as a string
rental_rate_str = film_df.rental_rate.astype(str)

# Split up and expand the column
rental_rate_expanded = rental_rate_str.str.split('.', expand=True)

# Assign the columns to film_df
film_df = film_df.assign(
    rental_rate_dollar=rental_rate_expanded[0],
    rental_rate_cents=rental_rate_expanded[1],
)

#More transforms

# Use groupBy and mean to aggregate the column
ratings_per_film_df = rating_df.groupBy('film_id').mean('film_id')

# Join the tables using the film_id column
film_df_with_ratings = film_df.join(
    ratings_per_film_df,
    film_df.film_id==ratings_per_film_df.film_id
)

# Show the 5 first results
print(film_df_with_ratings.show(5))

###############################################################################
'''
Load

Analytics
    -OLAP
    -Aggregate queries
    -Column Oriented
    -Queries about subset of columns
    -Parallelization

Applications
    -OLTP
    -Lots of transactions
    -Row Oriented
    -Added per transaction

MPP Databases
    Massively parallel processing
    Column oriented
'''
# Write the pandas DataFrame to parquet
film_pdf.to_parquet("films_pdf.parquet")

# Write the PySpark DataFrame to parquet
film_sdf.write.parquet("films_sdf.parquet")

#Loading to sqlalchemy

# Finish the connection URI
connection_uri = "postgresql://repl:password@localhost:5432/dwh"
db_engine_dwh = sqlalchemy.create_engine(connection_uri)

# Transformation step, join with recommendations data
film_pdf_joined = film_pdf.join(recommendations)

# Finish the .to_sql() call to write to store.film
film_pdf_joined.to_sql("film", db_engine_dwh, schema="store", if_exists="replace")

# Run the query to fetch the data
pd.read_sql("SELECT film_id, recommended_film_ids FROM store.film", db_engine_dwh)

###############################################################################
'''
The ETL function

Airflow refresher
    Workflow scheduler
    Python
    DAGs
    Operators
'''
# Define the ETL function
def etl():
    film_df = extract_film_to_pandas()
    film_df = transform_rental_rate(film_df)
    load_dataframe_to_film(film_df)

# Define the ETL task using PythonOperator
etl_task = PythonOperator(task_id='etl_film',
                          python_callable=etl,
                          dag=dag)

# Set the upstream to wait_for_table and sample run etl()
etl_task.set_upstream(wait_for_table)
etl()

#Setting Up Airflow
'''
Hmmm
'''
###############################################################################
###############################################################################
'''
Case Study: DataCamp Course Ratings

Goals:
    Get rating data
    Clean and calculate top-recommended courses
    Recalculate daily
    Show in users dashboard

As an ETL process

Database:
    Course
        course_id
        title
        description
        programming_language

    Rating
        user_id
        course_id
        rating
'''
#QUerying the table

# Complete the connection URI
connection_uri = "postgresql://repl:password@localhost:5432/datacamp_application"
db_engine = sqlalchemy.create_engine(connection_uri)

# Get user with id 4387
user1 = pd.read_sql("SELECT * FROM rating WHERE user_id = 4387", db_engine)

# Get user with id 18163
user2 = pd.read_sql("SELECT * FROM rating WHERE user_id = 18163", db_engine)

# Get user with id 8770
user3 = pd.read_sql("SELECT * FROM rating WHERE user_id = 8770", db_engine)

# Use the helper function to compare the 3 users
print_user_comparison(user1, user2, user3)

#Average rating per course

# Complete the transformation function
def transform_avg_rating(rating_data):
  # Group by course_id and extract average rating per course
  avg_rating = rating_data.groupby('course_id').rating.mean()
  # Return sorted average ratings per course
  sort_rating = avg_rating.sort_values(ascending=False).reset_index()
  return sort_rating

# Extract the rating data into a DataFrame
rating_data = extract_rating_data(db_engines)

# Use transform_avg_rating on the extracted data and print results
avg_rating_data = transform_avg_rating(rating_data)
print(avg_rating_data)

###############################################################################
course_data = extract_course_data(db_engines)

# Print out the number of missing values per column
print(course_data.isnull().sum())

# The transformation should fill in the missing values
def transform_fill_programming_language(course_data):
    imputed = course_data.fillna({"programming_language": "r"})
    return imputed

transformed = transform_fill_programming_language(course_data)

# Print out the number of missing values per column of transformed
print(transformed.isnull().sum())

###############################################################################
# Complete the transformation function
def transform_recommendations(avg_course_ratings, courses_to_recommend):
    # Merge both DataFrames
    merged = courses_to_recommend.merge(avg_course_ratings)
    # Sort values by rating and group by user_id
    grouped = merged.sort_values("rating", ascending = False).groupby('user_id')
    # Produce the top 3 values and sort by user_id
    recommendations = grouped.head(3).sort_values("user_id").reset_index()
    final_recommendations = recommendations[["user_id", "course_id","rating"]]
    # Return final recommendations
    return final_recommendations

# Use the function with the predefined DataFrame objects
recommendations = transform_recommendations(avg_course_ratings,courses_to_recommend)

###############################################################################
'''
Completed tasks so far

-Extract using extract_course_data()
-Clean up using NA using transform_fill_programming_language()
-Average course ratings per course: transform_avg_rating()
-Get eligible user and course id pairs: transform_courses_to_recommend()
-Calculate the recommendations: transform_recommendations()

Loading to Postgres
-Use the calculations in data products
-Update daily

Creating the DAG
'''
connection_uri = "postgresql://repl:password@localhost:5432/dwh"
db_engine = sqlalchemy.create_engine(connection_uri)

def load_to_dwh(recommendations):
    recommendations.to_sql("recommendations", db_engine, if_exists="replace")

# Define the DAG so it runs on a daily basis
dag = DAG(dag_id="recommendations",
          schedule_interval="0 0 * * *")

# Make sure `etl()` is called in the operator. Pass the correct kwargs.
task_recommendations = PythonOperator(
    task_id="recommendations_task",
    python_callable= etl,
    op_kwargs={"db_engines": db_engines},
)

def recommendations_for_user(user_id, threshold=4.5):
  # Join with the courses table
  query = """
  SELECT title, rating FROM recommendations
    INNER JOIN courses ON courses.course_id = recommendations.course_id
    WHERE user_id=%(user_id)s AND rating>%(threshold)s
    ORDER BY rating DESC
  """
  # Add the threshold parameter
  predictions_df = pd.read_sql(query, db_engine, params = {"user_id": user_id,
                                                           "threshold": threshold})
  return predictions_df.title.values

# Try the function you created
print(recommendations_for_user(12, 4.65))
