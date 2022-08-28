// increm.cpp
// demonstrates the increment operator
#include "iostream"
using namespace std;

int main()
  {
    int count = 10;

    cout << "count=" << count << endl;
    cout << "count=" << ++count << endl;
    cout << "count=" << count << endl;
    cout << "count=" << count++ << endl;
    cout << "count=" << count << endl;

    return 0;
  }
