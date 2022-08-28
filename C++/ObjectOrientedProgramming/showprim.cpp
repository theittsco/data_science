//showprim.cpp
// Displays prime number distribution
#include <iostream>
using namespace std;
#include "conio.h"

int main(int argc, char const *argv[]) {
  const unsigned char WHITE = 219; //solid color (primes)
  const unsigned char GREY = 176; //grey (not prime)
  unsigned char ch;

  for (size_t count = 0; count < 80*25-1; count++) {
  ch = WHITE;                 //assume it's prime
    for (size_t j = 2; j < count; j++) {
      if(count%j == 0){
        ch = GREY;
        break;
      }
    }
    cout << ch;
  }
  getch();
  return 0;
}
