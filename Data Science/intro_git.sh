git status #Check status of repository, shows files in staging area

git diff filename #In order to compare the file as it currently is to what you last saved,
git diff #without any filenames will show you all the changes in your repository,
git diff directory #will show you the changes to the files in some directory.

#A diff is a formatted display of the differences between two sets of files.
#Git displays diffs like this:

diff --git a/report.txt b/report.txt
index e713b17..4c0742a 100644
--- a/report.txt
+++ b/report.txt
@@ -1,4 +1,5 @@
-# Seasonal Dental Surgeries 2017-18
+# Seasonal Dental Surgeries (2017) 2017-18
+# TODO: write new summary
#This shows:

# 1. The command used to produce the output (in this case, diff --git). In it,
#    a and b are placeholders meaning "the first version" and "the second version".
# 2. An index line showing keys into Git's internal database of changes. We will
#    explore these in the next chapter.
# 3. --- a/report.txt and +++ b/report.txt, wherein lines being removed are
#    prefixed with - and lines being added are prefixed with +.
# 4. A line starting with @@ that tells where the changes are being made. The pairs
#    of numbers are start line and number of lines (in that section of the file where changes occurred). This diff output indicates changes starting at line 1, with 5 lines where there were once 4.
# 5. A line-by-line listing of the changes with - showing deletions and + showing
#    additions (we have also configured Git to show deletions in red and additions
#    in green). Lines that haven't changed are sometimes shown before and after
#    the ones that have in order to give context; when they appear, they don't
#    have either + or - in front of them.

###############################################################################

#You commit changes to a Git repository in two steps:

#Add one or more files to the staging area.
#Commit everything in the staging area.
git add filename #To add a file to the staging area

###############################################################################
git diff -r HEAD #To compare the state of your files with those in the staging area
#The -r flag means "compare to a particular revision", and HEAD is a shortcut
#meaning "the most recent commit".

git diff -r HEAD path/to/file #You can restrict the results to a single file or
#directory using , where the path to the file is relative to where you are
#(for example, the path from the root directory of the repository).

###############################################################################
git commit -m "Comment describing change"
git commit #Use text editor to write more detailed changes.
###############################################################################
git log #View log of project's history
git log path #Path is the path to a
###############################################################################
#Git stores information about a commit in a hash function
git show #Shows the details of a specific commit

###############################################################################
###############################################################################

#Combo git log to get the first few digits of the hash, then git show hash
#to look in detail

git diff -r HEAD~1 #~num is the revision num before the HEAD revision
gir show HEAD~1

###############################################################################
git log #displays the overall history of a project or file, but Git can give
#even more information. The command

git annotate file #shows who made the last
#change to each line of a file and when. For example, the first three lines of
#output from git annotate report.txt look something like this:

04307054        (  Rep Loop     2017-09-20 13:42:26 +0000       1)# Seasonal Dental Surgeries (2017) 2017-18
5e6f92b6        (  Rep Loop     2017-09-20 13:42:26 +0000       2)
5e6f92b6        (  Rep Loop     2017-09-20 13:42:26 +0000       3)TODO: write executive summary.
#Each line contains five elements, with elements two to four enclosed in
#parentheses. When inspecting the first line, we see:

#The first eight digits of the hash, 04307054.
#The author, Rep Loop.
#The time of the commit, 2017-09-20 13:42:26 +0000.
#The line number, 1.
#The contents of the line, # Seasonal Dental Surgeries (2017) 2017-18.

###############################################################################
git diff abc123..def456 #shows the differences between the commits abc123 and
#def456, while
git diff HEAD~1..HEAD~3 #shows the differences between the state of the
#repository one commit in the past and its state three commits in the past.

###############################################################################
git status
git add filename
git commit -m "Message"
#Follow this general flow to do your git uploading
###############################################################################
#Data analysis often produces temporary or intermediate files that you don't
#want to save. You can tell it to stop paying attention to files you don't care
#about by creating a file in the root directory of your repository called
#.gitignore and storing a list of wildcard patterns that specify the files you
#don't want Git to pay attention to. For example, if .gitignore contains:

build
*.mpl
#then Git will ignore any file or directory called build (and, if it's a
#directory, anything in it), as well as any file whose name ends in .mpl.

###############################################################################
#Git can help you clean up files that you have told it you don't want. The
#command
git clean -n
#will show you a list of files that are in the repository, but whose history
#Git is not currently tracking. A similar command
git clean -f #will then delete those files.

#Use this command carefully: git clean only works on untracked files, so by
#definition, their history has not been saved. If you delete them with git
#clean -f, they're gone for good.

###############################################################################
#Git allows you to change its default settings. To see what the settings are,
#you can use the command
git config --list #with one of three additional options:

--system: settings for every user on this computer.
--global: settings for every one of your projects.
--local: settings for one specific project.
#Each level overrides the one above it, so local settings (per-project) take
#precedence over global settings (per-user), which in turn take precedence over
#system settings (for all users on the computer).
git config --global user.setting newSetting

###############################################################################
###############################################################################
#Commit changes selectively
#Make sure you use the commit with the path and filename
git reset filename #Unstages a file

#Undoing changes?
git checkout -- filename

#You can combo these two to undo most actions
#In this way, you can think of committing as saving your work, and checking
#out as loading that saved version.
git log -n filename
git checkout hash_number filename
git commit filename

#Undoing changes to a single file
git reset HEAD path/to/file

#For multiple files just pass a directory, or use no file argument
#Similarly,

git checkout -- directory
git checkout -- .

###############################################################################
###############################################################################
git branch branchName #Shows all the branches of a repo
git checkout branchName #Switch to that branch

git rm filename #Removes a file from the current branch

git checkout -b branchName #Creates and switches to a branch

git merge source destination #Merges two branches

###############################################################################
###############################################################################
git init project-name #Initialize a git repository
git init #Initialize repository in current directory
git clone URL new_name #Creates a copy of an existing repository in a new directory
#Of form https://github.com/user/project.git
git remote -v #Info on the original repository
git remote add remote-name URL #Adds remotes other than origin
git remote rm remote-name #Removes remotes
git pull remote branch #Pulls the changes from the remote and merges them into a branch
git checkout -- . #Discards uncommitted changes in current repository
git push remote-name branch-name  
