// englcon.cpp
// constructors, adds objects using member functions 

#include <iostream>

//////////////////////////////////////////////////////////////
class Distance
{
private:
    int feet;
    float inches;
public:
    Distance() : feet(0), inches(0)
    { }

    Distance(int ft, float in) : feet(ft), inches(in)
    { }

    void getdist()
    {
        std::cout << "\nEnter feet: "; std::cin >> feet;
        std::cout << "Enter inches: "; std::cin >> inches;
    }

    void showdist()
    { std::cout << feet << "\'-" << inches << "\"" << std::endl; }

    void add_dist(Distance, Distance);
};

//------------------------------------------------------------------------
void Distance::add_dist(Distance d2, Distance d3)
{
    inches = d2.inches + d3.inches;
    feet = 0;
    if (inches > 12.0)
    {
        inches -= 12.0;
        feet++;
    }
    feet += d2.feet + d3.feet;
}

//////////////////////////////////////////////////////////////////////////
int main(int argc, char const *argv[])
{
    Distance dist1, dist3;
    Distance dist2(11, 6.25);

    dist1.getdist();
    dist3.add_dist(dist1, dist2);

    std::cout << "\ndist1 = "; dist1.showdist();
    std::cout << "\ndist2 = "; dist2.showdist();
    std::cout << "\ndist3 = "; dist3.showdist();
    std::cout << std::endl;
    return 0;
}

