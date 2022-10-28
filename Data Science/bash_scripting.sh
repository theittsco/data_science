# There is a file in either the start_dir/first_dir, start_dir/second_dir or start_dir/third_dir directory called soccer_scores.csv. 
# It has columns Year,Winner,Winner Goals for outcomes of a soccer league.
# cd into the correct directory and use cat and grep to find who was the winner in 1959. 
# You could also just ls from the top directory if you like!
cat */soccer_scores.csv | grep 1959

# There is a copy of Charles Dickens's infamous 'Tale of Two Cities' in your home directory called two_cities.txt.
# Use command line arguments such as cat, grep and wc with the right flag to count the number of lines in the book 
# that contain either the character 'Sydney Carton' or 'Charles Darnay'. Use exactly these spellings and capitalizations.
cat two_cities.txt | egrep 'Sydney Carton|Charles Darnay' | wc -l
cat two_cities.txt | grep -E 'Sydney Carton|Charles Darnay' | wc -l

#########################################################################################################################
#!/bin/bash or #!/usr/bash or #!/usr/bin/bash at the top of the script

#Your job is to create a Bash script from a shell piped command which will aggregate to see how many times each team has won.
#!/bin/bash

# Create a single-line pipe
cat soccer_scores.csv | cut -d "," -f 2 | tail -n +2 | sort | uniq -c

# Now save and run!

#You will need to create a Bash script that makes use of sed to change the required team names.
#!/bin/bash

# Create a sed pipe to a new file
cat soccer_scores.csv | sed 's/Cherno/Cherno City/g' | sed 's/Arda/Arda United/g' > soccer_scores_edited.csv

# Now save and run!
#########################################################################################################################
# Echo the first and second ARGV arguments
echo $1 
echo $2

# Echo out the entire ARGV array
echo $*

# Echo out the size of ARGV
echo $#

bash script.sh Bird Fish Rabbit

#------------------------------------------------------------------------------------------------------------------------
# Echo the first ARGV argument
echo $1 

# Cat all the files
# Then pipe to grep using the first ARGV argument
# Then write out to a named csv using the first ARGV argument
cat hire_data/* | grep "$1" > "$1".csv

bash script.sh Seoul

##########################################################################################################################
##########################################################################################################################
# Create the required variable
yourname="Sam"

# Print out the assigned name (Help fix this error!)
echo "Hi there $yourname, welcome to the website!"

#-------------------------------------------------------------------------------------------------------------------------
# Get first ARGV into variable
temp_f=$1

# Subtract 32
temp_f2=$(echo "scale=2; $temp_f - 32" | bc)

# Multiply by 5/9
temp_c=$(echo "scale=2; $temp_f2 * 5 / 9" | bc)

# Print the celsius temp
echo $temp_c

#-------------------------------------------------------------------------------------------------------------------------
# Create three variables from the temp data files' contents
temp_a="`cat temps/region_A`"
temp_b="`cat temps/region_B`"
temp_c="`cat temps/region_C`"

# Print out the three variables
echo "The three temperatures were $temp_a, $temp_b, and $temp_c"

##########################################################################################################################
# Create a normal array with the mentioned elements
capital_cities=("Sydney" "Albany" "Paris")

# Create a normal array with the mentioned elements using the declare method
declare -a capital_cities

# Add (append) the elements
capital_cities+=("Sydney")
capital_cities+=("Albany")
capital_cities+=("Paris")

# The array has been created for you
capital_cities=("Sydney" "Albany" "Paris")

# Print out the entire array
echo ${capital_cities[@]}

# Print out the array length
echo ${#capital_cities[@]}

##########################################################################################################################
# Create empty associative array
declare -A model_metrics

# Add the key-value pairs
model_metrics[model_accuracy]=(98)
model_metrics[model_name]=("knn")
model_metrics[model_f1]=(0.82)

# Declare associative array with key-value pairs on one line
declare -A model_metrics=([model_accuracy]=98 [model_name]="knn" [model_f1]=0.82)

# Print out the entire array
echo ${model_metrics[@]}

# An associative array has been created for you
declare -A model_metrics=([model_accuracy]=98 [model_name]="knn" [model_f1]=0.82)

# Print out just the keys
echo ${!model_metrics[@]}

#-------------------------------------------------------------------------------------------------------------------------
# Create variables from the temperature data files
temp_b="$(cat temps/region_B)"
temp_c="$(cat temps/region_C)"

# Create an array with these variables as elements
region_temps=($temp_b $temp_c)

# Call an external program to get average temperature
average_temp=$(echo "scale=2; (${region_temps[0]} + ${region_temps[1]}) / 2" | bc)

# Append average temp to the array
region_temps+=($average_temp)

# Print out the whole array
echo ${region_temps[@]}

##########################################################################################################################
##########################################################################################################################
# Extract Accuracy from first ARGV element
accuracy=$(grep 'Accuracy' $1 | sed 's/.* //')

# Conditionally move into good_models folder
if [ $accuracy -ge 90 ]; then
    mv $1 good_models/
fi

# Conditionally move into bad_models folder
if [ $accuracy -lt 90 ]; then
    mv $1 bad_models/
fi

#-------------------------------------------------------------------------------------------------------------------------
# Create variable from first ARGV element
sfile=$1

# Create an IF statement on sfile's contents
if grep -q 'SRVM_' $sfile && grep -q 'vpt' $sfile ; then
	# Move file if matched
	mv $sfile good_logs/
fi

##########################################################################################################################
# Use a FOR loop on files in directory
for file in inherited_folder/*.R
do  
    # Echo out each file
    echo $file
done

#-------------------------------------------------------------------------------------------------------------------------
# Create a FOR statement on files in directory
for file in robs_files/*.py
do  
    # Create IF statement using grep
    if grep -q 'RandomForestClassifier' $file ; then
        # Move wanted files to to_keep/ folder
        mv $file to_keep/
    fi
done
##########################################################################################################################
# Create a CASE statement matching the first ARGV element
case $1 in
  # Match on all weekdays
  Monday|Tuesday|Wednesday|Thursday|Friday)
  echo "It is a Weekday!";;
  # Match on all weekend days
  Saturday|Sunday)
  echo "It is a Weekend!";;
  # Create a default
  *) 
  echo "Not a day!";;
esac

#-------------------------------------------------------------------------------------------------------------------------
# Use a FOR loop for each file in 'model_out/'
for file in model_out/*
do
    # Create a CASE statement for each file's contents
    case $(cat $file) in
      # Match on tree and non-tree models
      *"Random Forest"*|*GBM*|*XGBoost*)
      mv $file tree_models/ ;;
      *KNN*|*Logistic*)
      rm $file ;;
      # Create a default
      *) 
      echo "Unknown model in $file" ;;
    esac
done

##########################################################################################################################
##########################################################################################################################
# Create function
function upload_to_cloud () {
  # Loop through files with glob expansion
  for file in output_dir/*results*
  do
    # Echo that they are being uploaded
    echo "Uploading $file to cloud"
  done
}

# Call the function
upload_to_cloud

#-------------------------------------------------------------------------------------------------------------------------
# Create function
what_day_is_it () {

  # Parse the results of date
  current_day=$(date | cut -d " " -f1)

  # Echo the result
  echo $current_day
}

# Call the function
what_day_is_it

#-------------------------------------------------------------------------------------------------------------------------
# Create a function 
function return_percentage () {

  # Calculate the percentage using bc
  percent=$(echo "scale=2; 100 * $1 / $2" | bc)

  # Return the calculated percentage
  echo $percent
}

# Call the function with 456 and 632 and echo the result
return_test=$(return_percentage 456 632)
echo "456 out of 632 as a percent is $return_test%"

#-------------------------------------------------------------------------------------------------------------------------
# Create a function
function get_number_wins () {

  # Filter aggregate results by argument
  win_stats=$(cat soccer_scores.csv | cut -d "," -f2 | egrep -v 'Winner'| sort | uniq -c | egrep "$1")

}

# Call the function with specified argument
get_number_wins "Etar"

# Print out the global variable
echo "The aggregated stats are: $win_stats"

#-------------------------------------------------------------------------------------------------------------------------
# Create a function with a local base variable
function sum_array () {
  local sum=0
  # Loop through, adding to base variable
  for number in "$@"
  do
    sum=$(echo "$sum + $number" | bc)
  done
  # Echo back the result
  echo $sum
  }
# Call function with array
test_array=(14 12 23.5 16 19.34)
total=$(sum_array "${test_array[@]}")
echo "The total sum of the test array is $total"

##########################################################################################################################
# Create a schedule for 30 minutes past 2am every day
30 2 * * * bash script1.sh

# Create a schedule for every 15, 30 and 45 minutes past the hour
15,30,45 * * * *  bash script2.sh

# Create a schedule for 11.30pm on Sunday evening, every week
30 23 * * 0 bash script3.sh

# View cronjobs
crontab -l

# Edit cronjobs
crontab -e