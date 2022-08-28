// cast.cpp
// test signed and unsigned integers
#include "iostream"
using namespace std;

int main()
  {
    int intVar = 1500000000;            //1.5 billion
    intVar = (intVar * 10) / 10;
    cout << "intVar = " << intVar << endl; // This gives the wrong answer

    intVar = 1500000000;
    intVar = (static_cast<double>(intVar) * 10) / 10;
    cout << "intVar = " << intVar << endl;

    return 0;
  }
