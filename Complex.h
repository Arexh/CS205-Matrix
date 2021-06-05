#pragma once
#include <iostream>
class Complex
{
private:
    double real;
    double imag;
public:
    Complex();
    Complex(double re, double im);
    ~Complex();
    Complex operator =(int a);
    Complex operator =(const Complex& a);
    Complex operator +(const Complex& a)const;
    Complex operator +=(const Complex& a);
    Complex operator -(const Complex& a)const;
    Complex operator -=(const Complex& a);
    Complex operator*(const Complex& a)const;
    Complex operator*(int a)const;
    Complex operator*=(const Complex& a);
    Complex operator*=(int a);
    Complex operator/(const Complex& a)const;
    Complex operator/(int a)const;
    Complex operator/=(const Complex& a);
    Complex operator/=(int a);

    Complex operator ~()const;//取共轭

    double module();//取模
    friend bool operator> (Complex& c1, Complex& c2);
    friend bool operator< (Complex& c1, Complex& c2);
    friend bool operator>=(Complex& c1, Complex& c2);
    friend bool operator<=(Complex& c1, Complex& c2);
    friend bool operator==(const Complex c1, const Complex c2);
    friend bool operator!=(const Complex c1, const Complex c2);
    friend std::ostream& operator <<(std::ostream& out, const Complex cc);
    friend std::istream& operator >>(std::istream& in, Complex& cc);
};

