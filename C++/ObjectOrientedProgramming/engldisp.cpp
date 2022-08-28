// engldisp.cpp
// Demonstrates passing a structure as an argument
#include <iostream>

/////////////////////////////////////////////////////////
struct Distance
    {
        int feet;
        float inches;
    };
/////////////////////////////////////////////////////////
void engldisp( Distance);       // declaration

int main()
    {
        Distance d1, d2;

        std::cout << "Enter feet: "; std::cin >> d1.feet;
        std::cout << "Enter inches: "; std::cin >> d1.inches;

        std::cout << "\nEnter feet: "; std::cin >> d2.feet;
        std::cout << "Enter inches: "; std::cin >> d2.inches;

        std::cout << "\nd1 = ";
        engldisp(d1);

        std::cout << "\nd2 = ";
        engldisp(d2);

        std::cout << std::endl;
        return 0;  
    }

void engldisp(Distance dd)
    {
        std::cout << dd.feet << "\'-" << dd.inches << "\"";
    }