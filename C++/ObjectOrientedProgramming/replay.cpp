// replay.cpp
// gets four ages from user, displays them

#include <iostream>
using namespace std;

int main(int argc, char const *argv[]) {
  int age[4];

  for (size_t j = 0; j < 4; j++) {
    cout << "Enter an age";
    cin >> age[j];
  }
  for (size_t i = 0; i < 4; i++) {
    cout << "You entered: " << age[i] << endl;
  }
  return 0;
}
