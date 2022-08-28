// ref.cpp
// demonstrates passing by reference
#include <iostream>

int main(int argc, char const *argv[]) {
  void intfrac(float, float&, float&);
  float number, intpart, fracpart;


  do {
    std::cout << "Enter a real number: " << '\n';
    std::cin >> number;
    intfrac(number, intpart, fracpart);
    std::cout << "Integer part is " << intpart
              << ", fraction part is " << fracpart << std::endl;
  } while(number != 0.0);
  return 0;
}
//-----------------------------------------------------------------------------
// intfrac()
// finds integer and fractional parts of a real number
void intfrac(float n, float& intp, float& fracp) {
  long temp = static_cast<long>(n); // convert to long
  intp = static_cast<float>(temp); //back to float
  fracp = n - intp; //subtract integer part
}
