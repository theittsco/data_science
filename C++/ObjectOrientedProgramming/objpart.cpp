// objpart.cpp
// widget part as an object

#include <iostream>

//////////////////////////////////////////////////////
class part
{
private:
    int modelnumber;
    int partnumber;
    float cost;
public:
    void setpart(int mn, int pn, float c)
        {
            modelnumber = mn;
            partnumber = pn;
            cost = c;
        }
    void showpart()
    {
        std::cout << "Model " << modelnumber;
        std::cout << ", part " << partnumber;
        std::cout << ", cost $" << cost << std::endl;
    }
};
////////////////////////////////////////////////////////
int main(int argc, char const *argv[])
{
    part part1;

    part1.setpart(6244,373,217.55F);
    part1.showpart();
    
    return 0;
}

