#ifndef C_EXCEPTION_H
#define C_EXCEPTION_H
#include <exception>
#include <iostream>
using namespace std;
    class Exception : public std::exception {

    public:

        explicit Exception(const std::string &message) : exception(), message(message) {}

        void what(){
            cout << getMessage() << endl;
        }

        virtual ~Exception(void) throw() {}


        inline std::string getMessage(void) const {
            return this->message;
        }


    protected:
        std::string message;
    };


    class InvalidDimensionsException : public Exception {

    public:

        InvalidDimensionsException(const std::string &message) : Exception(message) {}

    };


    class InvalidCoordinatesException : public Exception {

    public:

        InvalidCoordinatesException(const std::string &message) : Exception(message) {

        }
    };


#endif
