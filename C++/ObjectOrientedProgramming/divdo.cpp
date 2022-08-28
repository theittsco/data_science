// divdo.cpp
// demonstrates DO loop
#include <iostream>
using namespace std;

int main()
  {
    long dividend, divisor;
    char ch;

    do {
      cout << "Enter diviend: "; cin >> dividend;
      cout << "Enter divisor: "; cin >> divisor;
      cout << "Quotient is: " << diviend / divisor;
      cout << ", remainder is: " << diviend % divisor;

      cout << "\n Do another? (y/n): ";
      cin >> ch;
    } while( ch != 'n');

    return 0;
  }
