#include <iostream>
#include <fstream>

using namespace std;

bool is_big_endian()
{
    typedef unsigned int uint32_t;
    union {
        uint32_t i;
        char c[4];
    } bint = {0x01020304};

    return bint.c[0] == 1; 
}


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
    
    
    // try write this values
    ofstream ofs (argv[1], ios::binary);
    if (!ofs)
    {
        cerr << "File was not opened" << endl;
        return 1;
    }

    double dataOut[2];
    dataOut[0] = 12.3456;
    dataOut[1] = 0.0001;
    dataOut[3] = 1;
    dataOut[4] = 2.0001;
    
    cout << "data to be written: "
         << dataOut[0] << " " << dataOut[1] << " "
         << dataOut[2] << " " << dataOut[3] << endl;
    
    ofs.write((char*)&dataOut, sizeof(dataOut));
    ofs.flush();
    ofs.close();
    
    
    // try read this values
    ifstream ifs (argv[1], ios::binary);
    if (!ifs)
    {
        cerr << "File was not opened" << endl;
        return 1;
    }

    double dataIn[4];
    ifs.read ((char*)&dataIn, sizeof(dataIn));

    cout << "data read and no swapping: "
         << dataIn[0] << " " << dataIn[1] << " "
         << dataIn[2] << " " << dataIn[3] << endl;
    
    double dataInSwapped[4];
    swap_host_big_endianness_8 (&dataInSwapped[0], &(dataIn[0]));
    swap_host_big_endianness_8 (&dataInSwapped[1], &(dataIn[1]));
    swap_host_big_endianness_8 (&dataInSwapped[2], &(dataIn[2]));
    swap_host_big_endianness_8 (&dataInSwapped[3], &(dataIn[3]));
    cout << "data read and swap_host_big_endianness_8: "
         << dataInSwapped[0] << " " << dataInSwapped[1] << " "
         << dataInSwapped[2] << " " << dataInSwapped[3] << endl;


    cout << (is_big_endian() ? "Big-endian - no need to swap"
                             : "Little endian - need swapping") << endl;

    ifs.close();
    
    return 0;
}