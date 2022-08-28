// fraction adder
// Adds two fractions given by user
#include <iostream>
using namespace std;

int main()
  {
    int a,b,c,d;
    int numerator, denominator;
    char dummy;

    cout << "Enter first fraction: " << endl;
    cin >> a >> dummy >> b;
    cout << "Enter second fraction: " << endl;
    cin >> c >> dummy >> d;
    numerator = (a*d + c*b);
    denominator = b*d;
    cout << "Sum = " << numerator << dummy << denominator;

    return 0;
  }
