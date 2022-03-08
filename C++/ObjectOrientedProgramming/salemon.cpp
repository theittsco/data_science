// salemon.cpp
// displays sales chart using 2d array

#include <iostream>
#include <iomanip>

using namespace std;

const int DISTRICTS = 4;
const int MONTHS = 3;

int main(int argc, char const *argv[]) {
  int d, m;
  double sales[DISTRICTS][MONTHS];

  cout << endl;
  for (size_t d = 0; d < DISTRICTS; d++) {
    for (size_t m = 0; m < MONTHS; m++) {
      cout << "Enter slaes for district " << d+1;
      cout << ", month " << m+1 << ": ";
      cin >> sales[d][m];
    }
  }
    
  cout << "\n\n";
  cout << "                 Month\n";
  cout << "              1       2       3";
  for (size_t d = 0; d < DISTRICTS; d++) {
    cout <<  "\nDISTRICT " <<  d+1;
    for (size_t m = 0; m < MONTHS; m++) {
      cout << setiosflags(ios::fixed)
           << setiosflags(ios::showpoint)
           << setprecision(2)
           << setw(10)
           << sales[d][m];
         }

  }
  cout << endl;
  return 0;
}
