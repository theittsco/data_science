'''
Objectives
    -Follow best practices
    -Use helpful technologies
    -Write python code that is easy to
        -read
        -use
        -maintain
        -share
    -Develop personal workflow
        -Steps you take
        -Tools you use


DON'T REPEAT YOURSELF

Use functions and list comprehensions to reduce repeated tasks/function calls

Use the python Standard library

Use generators instead of lists, as generators remember which values they have
already generated.
'''
def print_files(filenames):
    # Set up the loop iteration instructions
    for name in filenames:
        # Use pathlib.Path to print out each file
        print(Path(name).read_text())

def list_files(filenames):
    # Use pathlib.Path to read the contents of each file
    return [Path(name).read_text()
            # Obtain each name from the list of filenames
            for name in filenames]

filenames = "diabetes.txt", "boston.txt", "digits.txt", "iris.txt", "wine.txt"
print_files(filenames)
pprint(list_files(filenames))

###############################################################################

def get_matches(filename, query):
    # Filter the list comprehension using an if clause
    return [line for line in Path(filename).open() if query in line]

# Iterate over files to find all matching lines
matches = [get_matches(name, "Number of") for name in filenames]
pprint(matches)

###############################################################################

def flatten(nested_list):
    return (item
            # Obtain each list from the list of lists
            for sublist in nested_list
            # Obtain each element from each individual list
            for item in sublist)

number_generator = (int(substring) for string in flatten(matches)
                    for substring in string.split() if substring.isdigit())
pprint(dict(zip(filenames, zip(number_generator, number_generator))))

###############################################################################
'''
Modularity

Write code as independent, reusable objects
Each object should only have ONE job.
Separate code into modules and scripts

Modules: Imported, provide tools, define functions
Scripts: Run, perform actions, call functions
'''

def obtain_words(string):
    # Replace non-alphabetic characters with spaces
    return "".join(char if char.isalpha() else " " for char in string).split()

def filter_words(words, minimum_length=3):
    # Remove words shorter than 3 characters
    return [word for word in words if len(word) >= minimum_length]

words = obtain_words(Path("diabetes.txt").read_text().lower())
filtered_words = filter_words(words)
pprint(filtered_words)

###############################################################################

def count_words(word_list):
    # Count the words in the input list
    return {word: word_list.count(word) for word in word_list}

# Create the dictionary of words and word counts
word_count_dictionary = count_words(filtered_words)

(pd.DataFrame(word_count_dictionary.items())
 .sort_values(by=1, ascending=False)
 .head()
 .plot(x=0, kind="barh", xticks=range(5), legend=False)
 .set_ylabel("")
)
plt.show()

###############################################################################
'''
Abstraction

Hide implementation details
Design user interfaces
Facilitate code use

Use classes: Templates for creating Python objects

Class method decorator:

    @classmethod
    def instantiate(cls, arg)
        return blah
'''

# Fill in the first parameter in the pair_plot() definition
def pair_plot(self, vars=range(3), hue=None):
    return pairplot(pd.DataFrame(self.data), vars=vars, hue=hue, kind="reg")

ScikitData.pair_plot = pair_plot

# Create the diabetes instance of the ScikitData class
diabetes = ScikitData("diabetes")
diabetes.pair_plot(vars=range(2, 6), hue=1)._legend.remove()
plt.show()

###############################################################################

# Fill in the decorator for the get_generator() definition
@classmethod
# Add the first parameter to the get_generator() definition
def get_generator(cls, dataset_names):
    return map(cls, dataset_names)

ScikitData.get_generator = get_generator
dataset_generator = ScikitData.get_generator(["diabetes", "iris"])
for dataset in dataset_generator:
    dataset.pair_plot()
    plt.show()

###############################################################################
###############################################################################
'''
Type hints

To avoid ambiguity, you can specify the intended input and output types of a
function as follows.
'''
def double(n):
    return 2 * n

def double(n: int) -> int:
    return 2 * n

#Use mypy type checker from pytest

#Use the Optional class when there is a None type
#Use the typing library
###############################################################################
class TextFile:
  	# Add type hints to TextFile"s __init__() method
    def __init__(self, name: str) -> None:
        self.text = Path(name).read_text()

	# Type annotate TextFile"s get_lines() method
    def get_lines(self) -> List[str]:
        return self.text.split("\n")

help(TextFile)

###############################################################################
class MatchFinder:
  	# Add type hints to __init__()'s strings argument
    def __init__(self, strings: List[str]) -> None:
        self.strings = strings

	# Type annotate get_matches()'s query argument
    def get_matches(self, query: Optional[str] = None) -> List[str]:
        return [s for s in self.strings if query in s] if query else self.strings

help(MatchFinder)

###############################################################################
'''
Docstrings

-Triple quoted strings that describe what a function does
'''
def get_matches(word_list: List[str], query:str) -> List[str]:
    ("Find lines containing the query string.\nExamples:\n\t"
     # Complete the docstring example below
     ">>> get_matches(['a', 'list', 'of', 'words'], 's')\n\t"
     # Fill in the expected result of the function call
     "['list', 'words']")
    return [line for line in word_list if query in line]

help(get_matches)

###############################################################################
def obtain_words(string: str) -> List[str]:
    ("Get the top words in a word list.\nExamples:\n\t"
     ">>> from this import s\n\t>>> from codecs import decode\n\t"
     # Use obtain_words() in the docstring example below
     ">>> obtain_words(decode(s, encoding='rot13'))[:4]\n\t"
     # Fill in the expected result of the function call
     "['The', 'Zen', 'of', 'Python']")
    return ''.join(char if char.isalpha() else ' ' for char in string).split()

help(obtain_words)

###############################################################################
'''
Build notebooks
The first function we will define for our new Python package is called nbuild().

nbuild() will

Create a new notebook with the new_notebook() function from the v4 module of nbformat
Read the file contents from a list of source files with the read_file() function that we have used in previous exercises
Pass the file contents to the new_code_cell() or new_markdown_cell() functions from the v4 module of nbformat
Assign a list of the resulting cells to the 'cells' key in the new notebook
Return the notebook instance
'''
def nbuild(filenames: List[str]) -> nbformat.notebooknode.NotebookNode:
    """Create a Jupyter notebook from text files and Python scripts."""
    nb = new_notebook()
    nb.cells = [
        # Create new code cells from files that end in .py
        new_code_cell(Path(name).read_text())
        if name.endswith(".py")
        # Create new markdown cells from all other files
        else new_markdown_cell(Path(name).read_text())
        for name in filenames
    ]
    return nb

pprint(nbuild(["intro.md", "plot.py", "discussion.md"]))

###############################################################################
def nbconv(nb_name: str, exporter: str = "script") -> str:
    """Convert a notebook into various formats using different exporters."""
    # Instantiate the specified exporter class
    exp = get_exporter(exporter)()
    # Return the converted file"s contents string
    return exp.from_filename(nb_name)[0]

pprint(nbconv(nb_name="mynotebook.ipynb", exporter="html"))

###############################################################################
'''
PyTest

Use Sphinx to generate documentation webpages

Parametrize
Docstring examples are great, because they are included in Sphinx documentation and testable with doctest, but now we are ready to take our testing to the next level with pytest.

Writing pytest tests

is less cumbersome than writing docstring examples (no need for >>> or ...)
allows us to leverage awesome features like the parametrize decorator.
The arguments we will pass to parametrize are

a name for the list of arguments and
the list of arguments itself.
In this exercise, we will define a test_nbuild() function to pass three different file types to the nbuild() function and confirm that the output notebook contains the input file in its first cell.

We will use a custom function called show_test_output() to see the test output.
'''
# Fill in the decorator for the test_nbuild() function
@pytest.mark.parametrize("inputs", ["intro.md", "plot.py", "discussion.md"])
# Pass the argument set to the test_nbuild() function
def test_nbuild(inputs):
    assert nbuild([inputs]).cells[0].source == Path(inputs).read_text()

show_test_output(test_nbuild)

###############################################################################
'''
Raises
In this coding exercise, we will define a test function called test_nbconv() that will use the

@parametrize decorator to pass three unsupported arguments to our nbconv() function
raises() function to make sure that passing each incorrect argument to nbconv() results in a ValueError
As in the previous exercise, we will use show_test_output() to see the test output.

To see an implementation of this test and others, checkout the Nbless package documentation.
'''

@pytest.mark.parametrize("not_exporters", ["htm", "ipython", "markup"])
# Pass the argument set to the test_nbconv() function
def test_nbconv(not_exporters):
     # Use pytest to confirm that a ValueError is raised
    with pytest.raises(ValueError):
        nbconv(nb_name="mynotebook.ipynb", exporter=not_exporters)

show_test_output(test_nbconv)

###############################################################################
###############################################################################
'''
Building a Command Line Interface

Use argparse or docopt nbuild()
'''
def argparse_cli(func: Callable) -> None:
    # Instantiate the parser object
    parser = argparse.ArgumentParser()
    # Add an argument called in_files to the parser object
    parser.add_argument("in_files", nargs="*")
    args = parser.parse_args()
    print(func(args.in_files))

if __name__ == "__main__":
    argparse_cli(nbuild)

###############################################################################
'''
Docopt nbuild()
If you love docstrings, you are likely to be a fan of docopt CLIs.

The docstring in our docopt_cli.py file is only one line, but it includes all the details we need to pass a list of shell arguments to any function.

More specifically, the docstring determines that our IN_FILES variable is

optional and
represents a list of arguments
In docopt docstrings, optional arguments are wrapped in square brackets ([]), while lists of arguments are followed by ellipses (...).
'''
# Add the section title in the docstring below
"""Usage: docopt_cli.py [IN_FILES...]"""

def docopt_cli(func: Callable) -> None:
    # Assign the shell arguments to "args"
    args = docopt(__doc__)
    print(func(args["IN_FILES"]))

if __name__ == "__main__":
    docopt_cli(nbuild)

###############################################################################
'''
Git Version Control

Commit added files
'''
# Initialize a new repo in the current folder
repo = git.Repo.init()

# Obtain a list of untracked files
untracked = repo.untracked_files

# Add all untracked files to the index
repo.index.add(untracked)

# Commit newly added files to version control history
repo.index.commit(f"Added {', '.join(untracked)}")
print(repo.head.commit.message)

###############################################################################
changed_files = [file.b_path
                 # Iterate over items in the diff object
                 for file in repo.index.diff(None)
                 # Include only modified files
                 .iter_change_type("M")]

repo.index.add(changed_files)
repo.index.commit(f"Modified {', '.join(changed_files)}")
for number, commit in enumerate(repo.iter_commits()):
    print(number, commit.message)

###############################################################################
'''
Virtual Environments

VEnv: venv, virtualenv, conda
Dependencies: pip, conda
'''
# Create an virtual environment
venv.create(".venv")

# Run pip list and obtain a CompletedProcess instance
cp = subprocess.run([".venv/bin/python", "-m", "pip", "list"], stdout=-1)

for line in cp.stdout.decode().split("\n"):
    if "pandas" in line:
        print(line)

###############################################################################
print(run(
    # Install project dependencies
    [".venv/bin/python", "-m", "pip", "install", "-r", "requirements.txt"],
    stdout=-1
).stdout.decode())

print(run(
    # Show information on the aardvark package
    [".venv/bin/python", "-m", "pip", "show", "aardvark"], stdout=-1
).stdout.decode())

###############################################################################
'''
Persistence

AKA saving files. A script should save/produce more than one file.

Use pickle format. pickle, pandas, joblib

Why do this?
    Everything in one place: code, documentation, data files.
    Easy to share

Use pkgutil, pkg_resources
'''
pd.DataFrame(
    np.c_[(diabetes.data, diabetes.target)],
    columns="age sex bmi map tc ldl hdl tch ltg glu target".split()
    # Pickle the diabetes dataframe with zip compression
    ).to_pickle("diabetes.pkl.zip")

# Unpickle the diabetes dataframe
df = pd.read_pickle("diabetes.pkl.zip")
df.plot.scatter(x="ltg", y="target", c="age", colormap="viridis")
plt.show()

###############################################################################
# Train and pickle a linear model
joblib.dump(LinearRegression().fit(x_train, y_train), "linear.pkl")

# Unpickle the linear model
linear = joblib.load("linear.pkl")
predictions = linear.predict(x_test)
plt.scatter(y_test, predictions, edgecolors=(0, 0, 0))
min_max = [y_test.min(), y_test.max()]
plt.plot(min_max, min_max, "--", lw=3)
plt.xlabel("Measured")
plt.ylabel("Predicted")
plt.show()

###############################################################################
###############################################################################
'''
Templates, projects, pipelines

-Avoid repetitive tasks
-Standardize project structure
-Include configuration files: Pytest, Sphinx
-Include makefile to automate further steps
    -Build Sphinx documentation
    -Create virtual environments
    -Initialize git repositories
    -Deploy packages to the PyPl

Cookiecutter project library

'''
#Using cookiecutter.json

json_path.write_text(json.dumps({
    "project": "Creating Robust Python Workflows",
  	# Convert the project name into snake_case
    "package": "{{ cookiecutter.project.lower().replace(' ', '_') }}",
    # Fill in the default license value
    "license": ["MIT", "BSD", "GPL3"]
}))

pprint(json.loads(json_path.read_text()))

###############################################################################
#Creating a project
# Obtain keys from the local template's cookiecutter.json
keys = [*json.load(json_path.open())]
vals = "Your name here", "My Amazing Python Project"

# Create a cookiecutter project without prompting for input
main.cookiecutter(template_root.as_posix(), no_input=True,
                  extra_context=dict(zip(keys, vals)))

for path in pathlib.Path.cwd().glob("**"):
    print(path)

###############################################################################
'''
Zipapp
In this exercise, we will

zip up a project called myproject
make the zipped project command-line executable
create a __main__.py file in the zipped project
all with a single call to the create_archive() function from the standard library zipapp module.

The python interpreter we want to use is /usr/bin/env python,

while the function we want __main__.py to run is called print_name_and_file():
'''
def print_name_and_file():
    print(f"Name is {__name__}. File is {__file__}.")
#The print_name_and_file() function is in the mymodule.py file inside the top-level mypackage directory, as shown below:

myproject
└── mypackage
    ├── __init__.py
    └── mymodule.py

###############################################################################
zipapp.create_archive(
    # Zip up a project called "myproject"
    "myproject",
    interpreter="/usr/bin/env python",
    # Generate a __main__.py file
    main="mypackage.mymodule:print_name_and_file")

print(subprocess.run([".venv/bin/python", "myproject.pyz"],
                     stdout=-1).stdout.decode())

###############################################################################
def main():
    parser = argparse.ArgumentParser(description="Scikit datasets only!")
    # Set the default for the dataset argument
    parser.add_argument("dataset", nargs="?", default="diabetes")
    parser.add_argument("model", nargs="?", default="linear_model.Ridge")
    args = parser.parse_args()
    # Create a dictionary of the shell arguments
    kwargs = dict(dataset=args.dataset, model=args.model)
    return (classify(**kwargs) if args.dataset in ("digits", "iris", "wine")
            else regress(**kwargs) if args.dataset in ("boston", "diabetes")
            else print(f"{args.dataset} is not a supported dataset!"))

if __name__ == "__main__":
    main()

###############################################################################
'''
Jupyter notebook parameters

Use papermill and scrapbook libraries
'''
# Read in the notebook to find the default parameter names
pprint(nbformat.read("sklearn.ipynb", as_version=4).cells[0].source)
keys = ["dataset_name", "model_type", "model_name", "hyperparameters"]
vals = ["diabetes", "ensemble", "RandomForestRegressor",
        dict(max_depth=3, n_estimators=100, random_state=0)]
parameter_dictionary = dict(zip(keys, vals))

# Execute the notebook with custom parameters
pprint(pm.execute_notebook(
    "sklearn.ipynb", "rf_diabetes.ipynb",
    kernel_name="python3", parameters=parameter_dictionary
	))

###############################################################################
import scrapbook as sb

# Assign the scrapbook notebook object to nb
nb = sb.read_notebook("rf_diabetes.ipynb")

# Create a dataframe of scraps (recorded values)
scrap_df = nb.scrap_dataframe
print(scrap_df)

###############################################################################
'''
Parallel computing

Multiprocessing using Dask and joblib
'''
import dask.dataframe as dd

# Read in a csv file using a dask.dataframe method
df = dd.read_csv("diabetes.csv")

df["bin_age"] = (df.age > 0).astype(int)

# Compute the columns means in the two age groups
print(df.groupby("bin_age").mean().compute())

###############################################################################
# Set up a Dask client with 4 threads and 1 worker
Client(processes=False, threads_per_worker=4, n_workers=1)

# Run grid search using joblib and a Dask backend
with joblib.parallel_backend("dask"):
    engrid.fit(x_train, y_train)

plot_enet(*enet_path(x_test, y_test, eps=5e-5, fit_intercept=False,
                    l1_ratio=engrid.best_params_["l1_ratio"])[:2])
