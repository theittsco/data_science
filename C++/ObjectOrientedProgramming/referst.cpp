// referst.cpp
// demonstrates passing structure by reference
#include <iostream>

struct Distance {
  int feet;
  float inches;
};

void scale(Distance&, float);
void engldisp( Distance );

int main(int argc, char const *argv[]) {
  Distance d1 = {12, 6.5};
  Distance d2 = {10, 5.5};

  std::cout << "d1 = "; engldisp(d1);
  std::cout << "\nd2 = "; engldisp(d2);

  scale(d1, 0.5);
  scale(d2, 0.25);

  std::cout << "\nd1 = "; engldisp(d1);
  std::cout << "\nd2 = "; engldisp(d2);
  std::cout << '\n';

  return 0;
}

// scale
// scales value of type Distance by factor
void scale(Distance& dd, float factor) {
  float inches = (dd.feet*12 + dd.inches) * factor;
  dd.feet = static_cast<int>(inches / 12);
  dd.inches = inches - dd.feet * 12;
}

void engldisp( Distance dd) {
  std::cout << dd.feet << "\'-" << dd.inches << "\"";
}
