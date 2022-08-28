// chcount.cpp
// counts characters and words typed in
#include <iostream>
using namespace std;
#include <conio.h> //for getche()

int main()
  {
    int chcount=0; //counts non-space characters
    int wdcount=1; //counts spaces between words
    char ch; //ensure it isn’t ‘\r’

    cout << “Enter a phrase: “;
    while( (ch= getche()) != ‘\r’ ) //loop until Enter typed
      {
        if( ch==’ ‘ ) //if it’s a space
        wdcount++; //count a word
        else //otherwise,
        chcount++; //count a character
      } //display results
    cout << “\nWords=” << wdcount << endl
    << “Letters=” << (chcount-1) << endl;
  return 0;
}
