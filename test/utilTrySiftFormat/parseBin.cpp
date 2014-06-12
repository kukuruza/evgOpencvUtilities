#include <iostream>
#include <fstream>

using namespace std;

void swap_host_big_endianness_8 (void *dst, void* src)
{
  char *dst_ = (char*) dst ;
  char *src_ = (char*) src ;
#if defined(VL_ARCH_BIG_ENDIAN)
    dst_ [0] = src_ [0] ;
    dst_ [1] = src_ [1] ;
    dst_ [2] = src_ [2] ;
    dst_ [3] = src_ [3] ;
    dst_ [4] = src_ [4] ;
    dst_ [5] = src_ [5] ;
    dst_ [6] = src_ [6] ;
    dst_ [7] = src_ [7] ;
#else
    dst_ [0] = src_ [7] ;
    dst_ [1] = src_ [6] ;
    dst_ [2] = src_ [5] ;
    dst_ [3] = src_ [4] ;
    dst_ [4] = src_ [3] ;
    dst_ [5] = src_ [2] ;
    dst_ [6] = src_ [1] ;
    dst_ [7] = src_ [0] ;
#endif
}


int main (int argc, char** argv)
{
    if (argc == 1)
    {
        cerr << "The only argument is a file name." << endl;
        return 1;
    }
    ifstream ifs (argv[1], ios::binary);
    if (!ifs)
    {
        cerr << "File was not opened" << endl;
        return 1;
    }

    double data[8];
        ifs.read ((char*)&data, sizeof(data));

    double value;
    for (int i = 0; i < 8; i++)
    {
        swap_host_big_endianness_8 (&value, &(data[i]));
        cout << value << " ";
    }

    cout << endl;


    
    return 0;
}