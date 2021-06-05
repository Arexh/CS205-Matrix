#include "Complex.h"
#include <iostream>
#include "string"
#include <math.h>
using namespace std;

Complex::Complex() :real(0), imag(0) {};
Complex::Complex(double a, double b) :real(a), imag(b) {};
Complex::~Complex() {};

Complex Complex::operator =(int a) {
    this->real = a;
    this->imag = 0;
    return (*this);
}

Complex Complex::operator =(const Complex& a) {
    this->real = a.real;
    this->imag = a.imag;
    return (*this);
}
Complex Complex::operator +(const Complex& a)const
{
    return Complex(this->real + a.real, this->imag + a.imag);
}

Complex Complex::operator +=(const Complex& a)
{
    this->real += a.real;
    this->imag += a.imag;
    return (*this);
}

Complex Complex::operator -(const Complex& a)const
{
    return Complex(this->real - a.real, this->imag - a.imag);
}

Complex Complex::operator -=(const Complex& a)
{
    this->real -= a.real;
    this->imag -= a.imag;
    return (*this);
}


Complex Complex::operator *(const Complex& a)const
{
    return Complex(this->real * a.real - this->imag * a.imag, this->real * a.imag + this->imag * a.real);
}

Complex Complex::operator *(int a)const
{
    return Complex(this->real * a, this->imag * a);
}

Complex Complex::operator *=(const Complex& a)
{
    Complex tmp(*this);
    this->real = tmp.real * a.real - tmp.imag * a.imag;
    this->imag = tmp.real * a.imag + tmp.imag * a.real;
    return (*this);
}

Complex Complex::operator *=(int a)
{
    this->real *= a;
    this->imag *= a;
    return (*this);
}

Complex Complex::operator /(const Complex& a)const
{
    return Complex((this->real * a.real + this->imag * a.imag) / (a.real * a.real + a.imag * a.imag),
        (this->imag * a.real - this->real * a.imag) / (a.real * a.real + a.imag * a.imag));
}

Complex Complex::operator /(int a)const
{
    return Complex(this->real / a, this->imag / a);
}

Complex Complex::operator /=(const Complex& a)
{
    Complex tmp(*this);
    this->real = (tmp.real * a.real + tmp.imag * a.imag) / (a.real * a.real + a.imag * a.imag);
    this->imag = (tmp.imag * a.real - tmp.real * a.imag) / (a.real * a.real + a.imag * a.imag);
    return (*this);
}

Complex Complex::operator /=(int a)
{
    this->real /= a;
    this->imag /= a;
    return (*this);
}

double Complex::module() {
    return sqrt(this->real * this->real + this->imag * this->imag);
}

bool operator>(Complex& c1, Complex& c2)
{
    if (c1.module() > c2.module()) return true;
    else return false;
}

bool operator<(Complex& c1, Complex& c2)
{
    if (c1.module() < c2.module()) return true;
    else return false;
}

bool operator>=(Complex& c1, Complex& c2)
{
    if (c1.module() >= c2.module()) return true;
    else return false;
}

bool operator<=(Complex& c1, Complex& c2)
{
    if (c1.module() <= c2.module()) return true;
    else return false;
}

bool operator==(const Complex c1, const Complex c2)
{
    if ((c1.real == c2.real) && (c1.imag == c2.imag)) {
        return true;
    }
    else {
        return false;
    }

}

bool operator!=(const Complex c1, const Complex c2)
{
    if ((c1.real != c2.real) || (c1.imag != c2.imag)) {
        return true;
    }
    else {
        return false;
    }
}

std::ostream& operator <<(std::ostream& out, const Complex cc)
{
    string re = to_string(cc.real), im = to_string(cc.imag);
    if (cc.imag < 0) {
        string a = re.substr(0, re.find(".")) + im.substr(0, im.find(".")) + "i";
        out << a;
    }
    else if (cc.imag == 0) {
        out << cc.real;
    }
    else {
        string a = re.substr(0, re.find(".")) + "+" + im.substr(0, im.find(".")) + "i";
        out << a;
    }

    return out;
}

std::istream& operator >>(std::istream& in, Complex& cc)
{

      std::cout<<"real:";
    in >> cc.real;

     std::cout<<"imaginary:";
    in >> cc.imag;
    return in;
}

Complex Complex::operator ~()const
{
    return Complex(this->real, -this->imag);
}



