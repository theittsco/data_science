# Downloading data on the command line
# Downloading data using curl 

# Use curl to download the file from the redirected URL
curl -L https://assets.datacamp.com/production/repositories/4180/datasets/eb1d6a36fa3039e4e00064797e1a1600d267b135/201812SpotifyData.zip

# Download and rename the file in the same step
curl -o Spotify201812.zip -L https://assets.datacamp.com/production/repositories/4180/datasets/eb1d6a36fa3039e4e00064797e1a1600d267b135/201812SpotifyData.zip

##---------------------------------------------------------------------
# Download all 100 data files
curl -O https://s3.amazonaws.com/assets.datacamp.com/production/repositories/4180/datasets/files/datafile[001-100].txt

# Print all downloaded files to directory
ls datafile*.txt

#######################################################################
# Downloading data using wget 
# Better at downloading multiple files recursively

# Fill in the two option flags 
wget -c -b https://assets.datacamp.com/production/repositories/4180/datasets/eb1d6a36fa3039e4e00064797e1a1600d267b135/201812SpotifyData.zip

# Verify that the Spotify file has been downloaded
ls 

# Preview the log file 
cat wget-log

##---------------------------------------------------------------------
# View url_list.txt to verify content
cat url_list.txt

# Create a mandatory 1 second pause between downloading all files in url_list.txt
wget --wait=1 -i url_list.txt

# Take a look at all files downloaded
ls

##---------------------------------------------------------------------
# To kick off a data analysis project, it's good practice to first consolidate all of our data into one place. 
# Often times, this means downloading and pulling data from various locations such as HTTP servers and databases.

# While curl is handy for downloading a single file, it's somewhat unwieldy for handling multiple file downloads. 
# In this capstone exercise, we will use both curl and Wget to download a series of monthly Spotify files, 
# do some minor processing, and consolidate all downloaded files in our local directory.
# Use curl, download and rename a single file from URL
curl -o Spotify201812.zip -L https://assets.datacamp.com/production/repositories/4180/datasets/eb1d6a36fa3039e4e00064797e1a1600d267b135/201812SpotifyData.zip

# Unzip, delete, then re-name to Spotify201812.csv
unzip Spotify201812.zip && rm Spotify201812.zip
mv 201812SpotifyData.csv Spotify201812.csv

# View url_list.txt to verify content
cat url_list.txt

# Use Wget, limit the download rate to 2500 KB/s, download all files in url_list.txt
wget --limit-rate=2500k -i url_list.txt

# Take a look at all files downloaded
ls

#######################################################################
#######################################################################
# Data cleaning and munging on the command line
# Getting started with csvkit
# Upgrade csvkit using pip  
pip install --upgrade csvkit

# Print manual for in2csv
in2csv -h

# Print manual for csvlook
csvlook -h

##---------------------------------------------------------------------
# Use ls to find the name of the zipped file
ls

# Use Linux's built in unzip tool to unpack the zipped file 
unzip SpotifyData.zip

# Check to confirm name and location of unzipped file
ls

# Convert SpotifyData.xlsx to csv
in2csv SpotifyData.xlsx > SpotifyData.csv

# Print a preview in console using a csvkit suite command 
csvlook SpotifyData.csv 

# Check to confirm name and location of the Excel data file
ls

##---------------------------------------------------------------------
# Convert sheet "Worksheet1_Popularity" to CSV
in2csv SpotifyData.xlsx --sheet "Worksheet1_Popularity" > Spotify_Popularity.csv

# Check to confirm name and location of the new CSV file
ls

# Print high level summary statistics for each column
csvstat Spotify_Popularity.csv 

# Check to confirm name and location of the Excel data file
ls

# Convert sheet "Worksheet2_MusicAttributes" to CSV
in2csv SpotifyData.xlsx --sheet "Worksheet2_MusicAttributes" > Spotify_MusicAttributes.csv

# Check to confirm name and location of the new CSV file
ls

# Print preview of Spotify_MusicAttributes
csvlook Spotify_MusicAttributes.csv

##---------------------------------------------------------------------
# Check to confirm name and location of data file
ls

# Print a list of column headers in data file 
csvcut -n Spotify_MusicAttributes.csv

##---------------------------------------------------------------------
# Print a list of column headers in the data 
csvcut -n Spotify_MusicAttributes.csv

# Print the first column, by position
csvcut -c 1 Spotify_MusicAttributes.csv

# Print the first, third, and fifth column, by position
csvcut -c 1,3,5 Spotify_MusicAttributes.csv

# Print the first column, by name
csvcut -c "track_id" Spotify_MusicAttributes.csv

# Print the track id, song duration, and loudness, by name 
csvcut -c "track_id","duration_ms","loudness" Spotify_MusicAttributes.csv

##---------------------------------------------------------------------
# Print a list of column headers in the data 
csvcut -n Spotify_MusicAttributes.csv

# Filter for row(s) where track_id = 118GQ70Sp6pMqn6w1oKuki
csvgrep -c "track_id" -m 118GQ70Sp6pMqn6w1oKuki Spotify_MusicAttributes.csv

# Filter for row(s) where danceability = 0.812
csvgrep -c "danceability" -m 0.812 Spotify_MusicAttributes.csv

#######################################################################
# Stacking data and chaining commands with csvkit
# Stack the two files and save results as a new file
csvstack SpotifyData_PopularityRank6.csv SpotifyData_PopularityRank7.csv > SpotifyPopularity.csv

# Preview the newly created file 
csvlook SpotifyPopularity.csv

##---------------------------------------------------------------------
# If csvlook succeeds, then run csvstat 
csvlook Spotify_Popularity.csv && csvstat Spotify_Popularity.csv

# Use the output of csvsort as input to csvlook
csvsort -c 2 Spotify_Popularity.csv | csvlook

# Take top 15 rows from sorted output and save to new file
csvsort -c 2 Spotify_Popularity.csv | head -n 15 > Spotify_Popularity_Top15.csv

# Preview the new file 
csvlook Spotify_Popularity_Top15.csv

##---------------------------------------------------------------------
# Convert the Spotify201809 tab into its own csv file 
in2csv Spotify_201809_201810.xlsx --sheet "Spotify201809" > Spotify201809.csv

# Check to confirm name and location of data file
ls

# Preview file preview using a csvkit function
csvlook Spotify201809.csv

# Create a new csv with 2 columns: track_id and popularity
csvcut -c "track_id","popularity" Spotify201809.csv > Spotify201809_subset.csv

# While stacking the 2 files, create a data source column
csvstack -g "Sep2018","Oct2018" Spotify201809_subset.csv Spotify201810_subset.csv > Spotify_all_rankings.csv

#######################################################################
#######################################################################
# Database operations on the the command line
# Verify database name 
ls

# Pull the entire Spotify_Popularity table and print in log
sql2csv --db "sqlite:///SpotifyDatabase.db" \
        --query "SELECT * FROM Spotify_Popularity" 

# Verify database name 
ls

# Query first 5 rows of Spotify_Popularity and print in log
sql2csv --db "sqlite:///SpotifyDatabase.db" \
        --query "SELECT * FROM Spotify_Popularity LIMIT 5" \
        | csvlook         

# Verify database name 
ls

# Save query to new file Spotify_Popularity_5Rows.csv
sql2csv --db "sqlite:///SpotifyDatabase.db" \
        --query "SELECT * FROM Spotify_Popularity LIMIT 5" \
        > Spotify_Popularity_5Rows.csv

# Verify newly created file
ls

# Print preview of newly created file
csvlook Spotify_Popularity_5Rows.csv

#######################################################################
# Manipulating data using SQL syntax
# Preview CSV file
ls

# Apply SQL query to Spotify_MusicAttributes.csv
csvsql --query "SELECT * FROM Spotify_MusicAttributes ORDER BY duration_ms LIMIT 1" Spotify_MusicAttributes.csv

# Reformat the output using csvlook 
csvsql --query "SELECT * FROM Spotify_MusicAttributes ORDER BY duration_ms LIMIT 1" \
	Spotify_MusicAttributes.csv | csvlook

# Re-direct output to new file: ShortestSong.csv
csvsql --query "SELECT * FROM Spotify_MusicAttributes ORDER BY duration_ms LIMIT 1" \
	Spotify_MusicAttributes.csv > ShortestSong.csv
    
# Preview newly created file 
csvlook ShortestSong.csv

##---------------------------------------------------------------------
# Because SQL queries, by nature, can be long and complex, we will frequently need to deal with line breaks 
# while passing in SQL queries to csvkit commands.

# One way to work around this is to store the SQL queries as a shell variable, 
# then pass in the shell variable in place of the SQL query where needed.
# Preview CSV file
ls

# Store SQL query as shell variable
sqlquery="SELECT * FROM Spotify_MusicAttributes ORDER BY duration_ms LIMIT 1"

# Apply SQL query to Spotify_MusicAttributes.csv
csvsql --query "$sqlquery" Spotify_MusicAttributes.csv

##---------------------------------------------------------------------
# csvsql can be used to join CSV files together even when neither of them are in a database. 
# Here, we have two CSV files Spotify_MusicAttributes.csv and Spotify_Popularity.csv that are 
# both on song level but contain different attributes for each song. We can combine the two 
# files together using a SQL-like JOIN, and we can do so, through the power of csvsql.

# Store SQL query as shell variable
sql_query="SELECT ma.*, p.popularity FROM Spotify_MusicAttributes ma INNER JOIN Spotify_Popularity p ON ma.track_id = p.track_id"

# Join 2 local csvs into a new csv using the saved SQL
csvsql --query "$sql_query" Spotify_MusicAttributes.csv Spotify_Popularity.csv > Spotify_FullData.csv

# Preview newly created file
csvstat Spotify_FullData.csv

#######################################################################
# Pushing data back to database

# Preview file
ls

# Upload Spotify_MusicAttributes.csv to database
csvsql --db "sqlite:///SpotifyDatabase.db" --insert Spotify_MusicAttributes.csv

# Store SQL query as shell variable
sqlquery="SELECT * FROM Spotify_MusicAttributes"

# Apply SQL query to re-pull new table in database
sql2csv --db "sqlite:///SpotifyDatabase.db" --query "$sqlquery" 

# Excellent work! csvsql makes the upload look so easy. 
# However, if you do get unexpected errors, you can add 
# in the --verbose flag to your command to print out more error logs.

##---------------------------------------------------------------------
# Store SQL for querying from SQLite database 
sqlquery_pull="SELECT * FROM SpotifyMostRecentData"

# Apply SQL to save table as local file 
sql2csv --db "sqlite:///SpotifyDatabase.db" --query "$sqlquery_pull" > SpotifyMostRecentData.csv

# Store SQL for UNION of the two local CSV files
sqlquery_union="SELECT * FROM SpotifyMostRecentData UNION ALL SELECT * FROM Spotify201812"

# Apply SQL to union the two local CSV files and save as local file
csvsql 	--query "$sqlquery_union" SpotifyMostRecentData.csv Spotify201812.csv > UnionedSpotifyData.csv

# Push UnionedSpotifyData.csv to database as a new table
csvsql --db "sqlite:///SpotifyDatabase.db" --insert UnionedSpotifyData.csv

#######################################################################
#######################################################################
# Data Pipeline on the Command Line

# Python on the command line

# in one step, create a new file and pass the print function into the file
echo "print('This is my first Python script')" > my_first_python_script.py

# check file location 
ls

# check file content 
cat my_first_python_script.py

# execute Python script file directly from command line  
python my_first_python_script.py 

# You can follow up pip install with more than one Python package as long 
# as they are separated by spaces. You can also save a list of the packages 
# in a requirements.txt file and install everything at once!

# In the following exercises, we will work through the set up process for making 
# sure our Python environment has the proper library dependencies installed prior 
# to executing a Python model script. In this pipeline we will create the 
# requirements.txt file which houses the dependencies we need to install, install 
# the dependencies, and do a quick sanity check to make sure everything is properly set up.

# Add scikit-learn to the requirements.txt file
echo "scikit-learn" > requirements.txt

# Preview file content
cat requirements.txt

# Install the required dependencies
pip install -r requirements.txt

# Verify that Scikit-Learn is now installed
pip list

##---------------------------------------------------------------------
# Re-install requirements
pip install -r requirements.txt

# Preview Python model script for import dependencies
cat create_model.py

# Verify that dependencies are installed
pip list

# Execute Python model script, which outputs a pkl file
python create_model.py

# Verify that the model.pkl file has been created 
ls

#######################################################################
# Verify that there are no CRON jobs currently scheduled
crontab -l 

# Create Python file hello_world.py
echo "print('hello world')" > hello_world.py

# Preview Python file 
cat hello_world.py

# Add as job that runs every minute on the minute to crontab
echo "* * * * * python hello_world.py" | crontab

# Verify that the CRON job has been added
crontab -l

##---------------------------------------------------------------------
# Preview both Python script and requirements text file
cat create_model.py
cat requirements.txt

# Pip install Python dependencies in requirements file
pip install -r requirements.txt

# Run Python script on command line
python create_model.py

# Add CRON job that runs create_model.py every minute
echo "* * * * * python create_model.py" | crontab

# Verify that the CRON job has been scheduled via CRONTAB
crontab -l