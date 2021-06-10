#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <cassert>
#include <iomanip>
#include <math.h>
#include <complex>
#include <typeinfo>
#include <random>
#include <algorithm>
#include "exception.h"
using namespace std;
using std::setw;

#define EQ_THRESHOLD 1e-10

template <typename T>
struct Node
{
    int row;
    int col;
    T value;

    bool operator<(const Node<T> &a) const
    {
        if (row < a.row)
        {
            return true;
        }
        else if (row > a.row)
        {
            return false;
        }
        else
        {
            return col < a.col;
        }
    }

    bool operator==(const Node<T> &a) const
    {
        return (row == a.row) && (col == a.col);
    }
};

template <typename T>
class SparseMatrix
{
private:
    vector<Node<T>> *data;

public:
    int m_row;
    int m_col;

    SparseMatrix() : SparseMatrix(0, 0){};
    SparseMatrix(int row, int col);
    SparseMatrix(int row, int col, const T *data);
    SparseMatrix(const vector<vector<T>> &arr);
    SparseMatrix(const SparseMatrix<T> &a);

    ~SparseMatrix();

    bool operator==(SparseMatrix<T> &m1);
    bool operator!=(SparseMatrix<T> &m1);

    bool is_size_equal(const SparseMatrix<T> &m1) const;
    bool is_square() const;
    bool is_zero();

    SparseMatrix<T> operator=(const SparseMatrix<T> &m1);

    template <typename U>
    friend SparseMatrix<U> operator+(SparseMatrix<U> l, SparseMatrix<U> r);
    template <typename U>
    friend SparseMatrix<U> operator+(const U &l, SparseMatrix<U> r);
    template <typename U>
    friend SparseMatrix<U> operator+(SparseMatrix<U> l, const U &r);

    template <typename U>
    friend SparseMatrix<U> operator-(SparseMatrix<U> l, SparseMatrix<U> r);
    template <typename U>
    friend SparseMatrix<U> operator-(const U &l, SparseMatrix<U> r);
    template <typename U>
    friend SparseMatrix<U> operator-(SparseMatrix<U> l, const U &r);

    template <typename U>
    friend SparseMatrix<U> operator*(SparseMatrix<U> l, SparseMatrix<U> r);
    template <typename U>
    friend SparseMatrix<U> operator*(const U &l, SparseMatrix<U> r);
    template <typename U>
    friend SparseMatrix<U> operator*(SparseMatrix<U> l, const U &r);

    template <typename U>
    friend SparseMatrix<U> operator/(const U &l, SparseMatrix<U> r);
    template <typename U>
    friend SparseMatrix<U> operator/(SparseMatrix<U> l, const U &r);
    template <typename U>
    friend SparseMatrix<U> operator^(SparseMatrix<U> l, SparseMatrix<U> r);

    template <typename U>
    friend ostream& operator<<(ostream& stream, const SparseMatrix<U>& matrix);

    SparseMatrix<T> operator-();

    SparseMatrix<T> operator+=(SparseMatrix<T> m1);
    SparseMatrix<T> operator-=(SparseMatrix<T> m1);
    SparseMatrix<T> operator*=(SparseMatrix<T> m1);
    SparseMatrix<T> operator*=(const T &a);
    SparseMatrix<T> operator/=(const T &a);

    SparseMatrix<T> conju();

    SparseMatrix<T> dot(SparseMatrix<T> m1);
    SparseMatrix<T> cross(SparseMatrix<T> m1);
    SparseMatrix<T> Transposition();
    SparseMatrix<T> toTransposition();

    T determinant();
    T trace();

    SparseMatrix<T> LU_factor_U();
    SparseMatrix<T> LU_factor_L();
    SparseMatrix<T> LDU_factor_L();
    SparseMatrix<T> LDU_factor_D();
    SparseMatrix<T> LDU_factor_U();

    SparseMatrix<T> Inverse();
    SparseMatrix<T> reshape(int r, int c);
    SparseMatrix<T> slice(int r1, int r2, int c1, int c2);

    T sum();
    T mean();
    T max();
    T min();
    T row_max(int row);
    T row_min(int row);
    T row_sum(int row);
    T row_mean(int row);

    T col_max(int col);
    T col_min(int col);
    T col_sum(int col);
    T col_mean(int col);

    ostream& printMatrix(ostream& stream=cout);

    pair<SparseMatrix<T>, SparseMatrix<T>> QR_decomposition();
    T norm();
    SparseMatrix<T> normalized();
    void SetIdentity();

    T *eigenvalues(int max_iter = 10e3);
    SparseMatrix<T> eigenvector(T eigenvalue, int max_iter = 10e3);
    SparseMatrix<T> *eigenvectors(int max_itr = 10e3);
    bool isUpperTri();
    static bool isCloseEnough(T a, T b, double threshold = EQ_THRESHOLD);
    pair<T, SparseMatrix<T>> eigenValueAndEigenVector(int max_itr = 10e3);
    T **toArray();
    cv::Mat *toOpenCVMat(int type);

    static SparseMatrix<T> fromOpenCV(const cv::Mat &cvMat);
    static SparseMatrix<T> conv2D(SparseMatrix<T> &input, SparseMatrix<T> &kernel, int stride = 1, bool same_padding = true);

    void sort();
    void insert(int row, int col, T value);
    void remove(int row, int col);

    // from: https://stackoverflow.com/questions/6969881/operator-overload
    template <typename U>
    class Proxy
    {
    private:
        int row;
        vector<Node<U>> data;
        T zero;

    public:
        Proxy(int r, vector<Node<U>> &d)
        {
            this->row = r;
            this->data = d;
            this->zero = static_cast<T>(0);
        }

        T &operator[](int col)
        {
            for (int i = 0; i < data.size(); i++)
            {
                if (data[i].row > row || (data[i].row == row && col < data[i].col))
                {
                    break;
                }
                else if (data[i].row == row && data[i].col == col)
                {
                    return data[i].value;
                }
            }
            return this->zero;
        }
    };

    Proxy<T> operator[](int row)
    {
        sort();
        return Proxy<T>(row, *data);
    }

private:
    T all_sort(int a[], int now, int length, T &determinant);
    ostream& printMatrixInt(ostream& stream=cout);
};

template <typename T>
SparseMatrix<T>::~SparseMatrix()
{
    if (data)
        delete data;
    data = nullptr;
}

template <typename T>
SparseMatrix<T>::SparseMatrix(int row, int col)
{
    if (row <= 0 || col <= 0)
    {
        cout << "You input nonpositive row/col num" << endl;
        m_row = 0;
        m_col = 0;
    }
    else
    {
        m_row = row;
        m_col = col;
    }
    data = new vector<Node<T>>();
}

template <typename T>
SparseMatrix<T>::SparseMatrix(const vector<vector<T>> &arr)
{
    int row = arr.size();
    int col = arr[0].size();
    m_row = row;
    m_col = col;
    data = new vector<Node<T>>();
    for (int i = 0; i < row; i++)
        for (int j = 0; j < col; j++)
        {
            if (arr[i][j] != static_cast<T>(0))
            {
                Node<T> node = {i, j, arr[i][j]};
                data->push_back(node);
            }
        }
}

template <typename T>
SparseMatrix<T>::SparseMatrix(const SparseMatrix<T> &a)
{
    m_row = a.m_row;
    m_col = a.m_col;
    data = new vector<Node<T>>(*a.data);
}

template <typename T>
void SparseMatrix<T>::sort()
{
    std::sort((*data).begin(), (*data).end());
}

template <typename T>
void SparseMatrix<T>::remove(int row, int col)
{
    for (typename vector<Node<T>>::iterator it = (*data).begin(); it != (*data).end(); ++it)
    {
        Node<T> node = *it;
        if (node.row == row && node.col == col)
        {
            (*data).erase(it);
            return;
        }
    }
}

template <typename T>
ostream& SparseMatrix<T>::printMatrix(ostream& stream)
{
    sort();
    stream << "\n---------row:" << m_row << ",col:" << m_col << "-----------\n";
    int cnt = 0;
    for (int i = 0; i < m_row; i++)
    {
        for (int j = 0; j < m_col; j++)
        {
            if (cnt < (*data).size() && (*data)[cnt].row == i && (*data)[cnt].col == j)
                stream << std::setw(7) << (*data)[cnt++].value << " ";
            else
                stream << std::setw(7) << 0 << " ";
        }
        stream << endl;
    }
    stream << "---------------------------------\n";
    return stream;
}

template <typename T>
ostream& SparseMatrix<T>::printMatrixInt(ostream& stream)
{
    sort();
    stream << "\n---------row:" << m_row << ",col:" << m_col << "-----------\n";
    int cnt = 0;
    for (int i = 0; i < m_row; i++)
    {
        for (int j = 0; j < m_col; j++)
        {
            if (cnt < (*data).size() && (*data)[cnt].row == i && (*data)[cnt].col == j)
                stream << std::setw(7) << ((int)(*data)[cnt++].value) << " ";
            else
                stream << std::setw(7) << 0 << " ";
        }
        stream << endl;
    }
    stream << "---------------------------------\n";
    return stream;
}

template <>
ostream&  SparseMatrix<char>::printMatrix(ostream& stream)
{
    return printMatrixInt(stream);
}

template <>
ostream&  SparseMatrix<uchar>::printMatrix(ostream& stream)
{
    return printMatrixInt(stream);
}

template <typename T>
SparseMatrix<T> SparseMatrix<T>::operator=(const SparseMatrix<T> &m1)
{
    if (this != &m1)
    {
        m_row = m1.m_row;
        m_col = m1.m_col;

        if (data)
            delete data;

        data = new vector<Node<T>>(*(m1.data));
    }
    return (*this);
}

template <typename T>
bool SparseMatrix<T>::operator==(SparseMatrix<T> &m1)
{
    if (m_row != m1.m_row || m_col != m1.m_col)
        return false;

    for (int i = 0; i < m_row; i++)
        for (int j = 0; j < m_col; j++)
            if (!isCloseEnough((*this)[i][j], m1[i][j]))
                return false;

    return true;
}

template <typename T>
bool SparseMatrix<T>::operator!=(SparseMatrix<T> &m1)
{
    return !(*this == m1);
}

template <typename T>
ostream& operator<<(ostream& stream, SparseMatrix<T>& matrix)
{
    return matrix.printMatrix(stream);
}

template <typename T>
bool SparseMatrix<T>::is_size_equal(const SparseMatrix<T> &m1) const
{
    return m_row == m1.m_row && m_col == m1.m_col;
}

template <typename T>
bool SparseMatrix<T>::is_square() const
{
    return m_row == m_col;
}

template <typename T>
bool SparseMatrix<T>::is_zero()
{
    return determinant() == static_cast<T>(0);
}

template <typename T>
void SparseMatrix<T>::insert(int row, int col, T value)
{
    if (value == static_cast<T>(0))
    {
        remove(row, col);
        return;
    }
    for (int i = 0; i < (*data).size(); i++)
    {
        Node<T> node = (*data)[i];
        if (row == node.row && col == node.col)
        {
            node.value = value;
            (*data)[i] = node;
            return;
        }
    }
    (*data).push_back(Node<T>{row, col, value});
}

template <typename T>
SparseMatrix<T> SparseMatrix<T>::operator-()
{
    SparseMatrix<T> result(m_row, m_col);

    for (int i = 0; i < m_row; i++)
        for (int j = 0; j < m_col; j++)
            result.insert(i, j, -(*this)[i][j]);

    return result;
}

template <typename T>
SparseMatrix<T> operator+(SparseMatrix<T> l, SparseMatrix<T> r)
{
    SparseMatrix<T> result(l.m_row, r.m_col);
    l.sort();
    r.sort();
    vector<Node<T>> listOne = *(l.data);
    vector<Node<T>> listTwo = *(r.data);
    int one = 0, two = 0;
    while (one < listOne.size() && two < listTwo.size())
    {
        Node<T> nodeOne = listOne[one];
        Node<T> nodeTwo = listTwo[two];
        if (nodeOne == nodeTwo)
        {
            result.insert(nodeOne.row, nodeOne.col, nodeOne.value + nodeTwo.value);
            one++;
            two++;
        }
        else if (nodeOne < nodeTwo)
        {
            result.insert(nodeOne.row, nodeOne.col, nodeOne.value);
            one++;
        }
        else
        {
            result.insert(nodeTwo.row, nodeTwo.col, nodeTwo.value);
            two++;
        }
    }
    while (one < listOne.size())
    {
        Node<T> nodeOne = listOne[one];
        result.insert(nodeOne.row, nodeOne.col, nodeOne.value);
        one++;
    }
    while (two < listTwo.size())
    {
        Node<T> nodeTwo = listTwo[two];
        result.insert(nodeTwo.row, nodeTwo.col, nodeTwo.value);
        two++;
    }
    return result;
}

template <typename T>
SparseMatrix<T> operator+(SparseMatrix<T> l, const T &r)
{
    SparseMatrix<T> result(l.m_row, l.m_col);

    for (int i = 0; i < result.m_row; i++)
        for (int j = 0; j < result.m_col; j++)
            result.insert(i, j, r + l[i][j]);

    return result;
}

template <typename T>
SparseMatrix<T> operator+(const T &l, SparseMatrix<T> r)
{
    return r + l;
}

template <typename T>
SparseMatrix<T> operator-(SparseMatrix<T> l, SparseMatrix<T> r)
{
    SparseMatrix<T> result(l.m_row, l.m_col);

    for (int i = 0; i < result.m_row; i++)
        for (int j = 0; j < result.m_col; j++)
            result.insert(i, j, l[i][j] - r[i][j]);

    return result;
}

template <typename T>
SparseMatrix<T> operator-(SparseMatrix<T> l, const T &r)
{
    SparseMatrix<T> result(l);

    for (int i = 0; i < result.m_row; i++)
        for (int j = 0; j < result.m_col; j++)
            result.insert(i, j, l[i][j] - r);

    return result;
}

template <typename T>
SparseMatrix<T> operator-(const T &l, SparseMatrix<T> r)
{
    return r.SparseMatrix<T>::operator-() + l;
}

template <typename T>
SparseMatrix<T> operator*(SparseMatrix<T> l, SparseMatrix<T> r)
{
    SparseMatrix<T> result(l.m_row, r.m_col);

    for (int i = 0; i < result.m_row; i++)
        for (int j = 0; j < result.m_col; j++)
        {
            T cumulativeSum = static_cast<T>(0);
            for (int k = 0; k < l.m_col; k++)
                cumulativeSum += l[i][k] * r[k][j];
            result.insert(i, j, cumulativeSum);
        }

    return result;
}

template <typename T>
SparseMatrix<T> operator*(SparseMatrix<T> l, const T &r)
{
    SparseMatrix<T> result(l.m_row, l.m_col);

    for (int i = 0; i < result.m_row; i++)
        for (int j = 0; j < result.m_col; j++)
        {
            T value = l[i][j] * r;
            result.insert(i, j, value);
        }

    return result;
}

template <typename T>
SparseMatrix<T> operator*(const T &l, SparseMatrix<T> r)
{
    return r * l;
}

template <typename T>
SparseMatrix<T> operator/(SparseMatrix<T> l, const T &r)
{
    SparseMatrix<T> result(l.m_row, l.m_col);

    for (int i = 0; i < result.m_row; i++)
        for (int j = 0; j < result.m_col; j++)
            result.insert(i, j, l[i][j] / r);

    return result;
}

template <typename T>
SparseMatrix<T> operator/(const T &l, SparseMatrix<T> r)
{
    SparseMatrix<T> result(r.m_row, r.m_col);

    for (int i = 0; i < result.m_row; i++)
        for (int j = 0; j < result.m_col; j++)
            result.insert(i, j, l / r[i][j]);

    return result;
}

template <typename T>
SparseMatrix<T> SparseMatrix<T>::operator+=(SparseMatrix<T> m1)
{
    for (int i = 0; i < m_row; i++)
        for (int j = 0; j < m_col; j++)
            insert(i, j, (*this)[i][j] + m1[i][j]);

    return (*this);
}

template <typename T>
SparseMatrix<T> SparseMatrix<T>::operator-=(SparseMatrix<T> m1)
{
    for (int i = 0; i < m_row; i++)
        for (int j = 0; j < m_col; j++)
            insert(i, j, (*this)[i][j] - m1[i][j]);

    return (*this);
}

template <typename T>
SparseMatrix<T> SparseMatrix<T>::operator*=(SparseMatrix<T> m1)
{
    SparseMatrix<T> result = (*this) * m1;
    for (int i = 0; i < m_row; i++)
        for (int j = 0; j < m_col; j++)
            insert(i, j, result[i][j]);

    return (*this);
}

template <typename T>
SparseMatrix<T> SparseMatrix<T>::operator*=(const T &m1)
{
    for (int i = 0; i < m_row; i++)
        for (int j = 0; j < m_col; j++)
            insert(i, j, (*this)[i][j] * m1);

    return (*this);
}

template <typename T>
SparseMatrix<T> SparseMatrix<T>::operator/=(const T &a)
{
    for (int i = 0; i < m_row; i++)
        for (int j = 0; j < m_col; j++)
            insert(i, j, (*this)[i][j] / a);

    return (*this);
}

template <typename T>
SparseMatrix<T> operator^(SparseMatrix<T> l, SparseMatrix<T> r)
{
    SparseMatrix<T> result(l.m_row, r.m_col);

    for (int i = 0; i < result.m_row; i++)
        for (int j = 0; j < result.m_col; j++)
            result.insert(i, j, l[i][j] * r[i][j]);

    return result;
}

template <typename T>
SparseMatrix<T> SparseMatrix<T>::conju()
{
    SparseMatrix<T> result(m_row, m_col);

    for (int i = 0; i < m_row; i++)
        for (int j = 0; j < m_col; j++)
            result.insert(i, j, conj((*this)[i][j]));

    return result;
}

template <typename T>
SparseMatrix<T> SparseMatrix<T>::dot(SparseMatrix<T> m1)
{
    return (*this) ^ m1;
}

template <typename T>
SparseMatrix<T> SparseMatrix<T>::cross(SparseMatrix<T> m1)
{
    return (*this) * m1;
}

template <typename T>
SparseMatrix<T> SparseMatrix<T>::Transposition()
{
    SparseMatrix<T> result(m_col, m_row);

    for (int i = 0; i < result.m_row; i++)
        for (int j = 0; j < result.m_col; j++)
            result.insert(i, j, (*this)[j][i]);

    return result;
}

// template <typename T>
// SparseMatrix<T> SparseMatrix<T>::toTransposition()
// {
//     // SparseMatrix<T> result = Transposition();

//     // for (int i = 0; i < m_row; i++)
//     //     for (int j = 0; j < m_col; j++)
//     //         (*data)[i][j] = result[i][j];

//     return (*this);
// }

template <typename T>
T SparseMatrix<T>::determinant()
{
    try{
        if(m_row != m_col){
            throw new InvalidDimensionsException("the row and column is not equal");
        }
    }
    catch(Exception e){
        // e.what();
        // exit();
    }
    int length = m_col, now = 0;
    T d;
    d = 0;
    int *permutation = new int[length];
    for (int i = 0; i < length; i++)
    {
        permutation[i] = i;
    }

    d = all_sort(permutation, now, length, d);
    delete[] permutation;

    return d;
}

template <typename T>
T SparseMatrix<T>::all_sort(int a[], int now, int length, T &determinant)
{
    if (now == length - 1)
    {
        T tmp;
        tmp = 1;
        int pow = 0;
        for (int i = 0; i < length; i++)
        {
            tmp *= (*this)[i][a[i]];
        }

        for (int i = 1; i < length; i++)
            for (int j = 0; j < i; j++)
                if (a[j] > a[i])
                    pow++;

        if (pow % 2 == 0)
            determinant += tmp;
        else
            determinant -= tmp;
        //将-1的幂化为0,tmp初始化
        pow = 0;
        tmp = 1;
    }
    for (int i = now; i < length; i++)
    {
        int tmp = a[now];
        a[now] = a[i];
        a[i] = tmp;
        all_sort(a, now + 1, length, determinant);
        tmp = a[now];
        a[now] = a[i];
        a[i] = tmp;
    }
    return determinant;
}

template <typename T>
T SparseMatrix<T>::trace()
{
    if (is_square())
    {
        T trace = static_cast<T>(0);

        for (int i = 0; i < m_col; i++)
            trace += (*this)[i][i];

        return trace;
    }
    else
    {
        return static_cast<T>(0);
    }
}

template <typename T>
SparseMatrix<T> SparseMatrix<T>::LU_factor_U()
{
    try{
        if(m_row != m_col){
            throw new InvalidDimensionsException("the row and column is not equal");
        }
    }
    catch(Exception e){
        // e.what();
        // exit();
    }
    int n = m_col;
    T sum;
    sum = 0;
    SparseMatrix<T> l(n, n);
    SparseMatrix<T> u(n, n);

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
        {
            u.remove(i, j);
            if (i == j)
                l.insert(i, j, 1);
        }

    for (int i = 0; i < n; i++)
    {
        T sum;
        sum = 0;
        for (int j = i; j < n; j++)
        {
            for (int k = 0; k <= i - 1; k++)
                sum += l[i][k] * u[k][j];
            u.insert(i, j, (*this)[i][j] - sum);
            sum = 0;
        }

        for (int x = i + 1; x < n; x++)
        {
            for (int k = 0; k <= i - 1; k++)
                sum += l[x][k] * u[k][i];
            l.insert(x, i, ((*this)[x][i] - sum) / u[i][i]);
            sum = 0;
        }
    }
    return u;
}

template <typename T>
SparseMatrix<T> SparseMatrix<T>::LU_factor_L()
{
    try{
        if(m_row != m_col){
                throw new InvalidDimensionsException("the row and column is not equal");
            }
    }
    catch(Exception e){
        // e.what();
        // exit();
    }
    int n = m_col;
    T sum;
    sum = 0;
    SparseMatrix<T> l(n, n);
    SparseMatrix<T> u(n, n);

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
        {
            u.remove(i, j);
            if (i == j)
                l.insert(i, j, 1);
        }

    for (int i = 0; i < n; i++)
    {
        T sum;
        sum = 0;
        for (int j = i; j < n; j++)
        {
            for (int k = 0; k <= i - 1; k++)
                sum += l[i][k] * u[k][j];
            u.insert(i, j, (*this)[i][j] - sum);
            sum = 0;
        }

        for (int x = i + 1; x < n; x++)
        {
            for (int k = 0; k <= i - 1; k++)
                sum += l[x][k] * u[k][i];
            l.insert(x, i, ((*this)[x][i] - sum) / u[i][i]);
            sum = 0;
        }
    }
    return l;
}

template <typename T>
SparseMatrix<T> SparseMatrix<T>::LDU_factor_L()
{
    SparseMatrix<T> l(this->LU_factor_L());
    return l;
}

template <typename T>
SparseMatrix<T> SparseMatrix<T>::LDU_factor_D()
{
    try{
        if(this->m_row != this->m_col){
            throw new InvalidDimensionsException("the row and column is not equal");
        }
    }
    catch(Exception e){
        // e.what();
        // exit();
    }
    SparseMatrix<T> tmp(this->LU_factor_U());
    SparseMatrix<T> d(this->m_row, this->m_col);
    for (int i = 0; i < m_row; i++)
        d.insert(i, i, tmp[i][i]);

    return d;
}

template <typename T>
SparseMatrix<T> SparseMatrix<T>::LDU_factor_U()
{
    try{
        if(this->m_row != this->m_col){
            throw new InvalidDimensionsException("the row and column is not equal");
        }
    }
    catch(Exception e){
        // e.what();
        // exit();
    }
    SparseMatrix<T> u(this->LU_factor_U());
    SparseMatrix<T> a(u);
    for (int i = 0; i < m_row; i++)
        for (int j = i; j < m_col; j++)
            u.insert(i, j, u[i][j] / a[i][i]);

    return u;
}

template <typename T>
SparseMatrix<T> SparseMatrix<T>::Inverse()
{
    T deter = this->determinant();
    try{
        if(!(this -> is_square())){
            throw new Exception("the matrix is not a square");
        }
    }
    catch(Exception e){
        // e.what();
        // exit();
    }
    T tmp1;
    tmp1 = 0;
    assert(deter != tmp1);
    if (this->m_row == 1)
    {
        vector<vector<T>> v{{static_cast<T>(1) / (*this)[0][0]}};
        SparseMatrix<T> tmp(v);
        return tmp;
    }
    else
    {
        int i, j, k, m, tt = this->m_row, n = tt - 1;
        SparseMatrix<T> inverse(tt, tt);
        SparseMatrix<T> tmp(n, n);
        for (i = 0; i < tt; i++)
        {

            for (j = 0; j < tt; j++)
            {
                for (k = 0; k < n; k++)
                    for (m = 0; m < n; m++)
                        tmp.insert(k, m, (*this)[k >= i ? k + 1 : k][m >= j ? m + 1 : m]);

                T a = tmp.determinant();
                if ((i + j) % 2 == 1)
                {
                    a = -a;
                };
                T b = a / (this->determinant());
                inverse.insert(j, i, b);
            }
        }
        return inverse;
    }
}

template <typename T>
SparseMatrix<T> SparseMatrix<T>::reshape(int r, int c)
{
    if (this->m_row * this->m_col != r * c)
    {
        return (*this);
    }
    else
    {
        SparseMatrix<T> ans(r, c);
        int i, j, x = 0, y = 0;
        for (i = 0; i < this->m_row; i++)
        {
            for (j = 0; j < this->m_col; j++)
            {
                ans.insert(x, y, (*this)[i][j]);
                y++;
                if (y == c)
                {
                    x++;
                    y = 0;
                }
            }
        }
        return ans;
    }
}

template <typename T>
SparseMatrix<T> SparseMatrix<T>::slice(int r1, int r2, int c1, int c2)
{
    // assert(r1)
    if (r1 > r2)
    {
        int tmp = r1;
        r1 = r2;
        r2 = tmp;
    }
    if (c1 > c2)
    {
        int tmp = c1;
        c1 = c2;
        c2 = tmp;
    }
    if (r1 < 0)
    {
        if (r2 < 0)
        {
            r2 = 0;
        }
        r1 = 0;
    }
    if (r2 >= this->m_row)
    {
        if (r1 >= this->m_row)
        {
            r1 = this->m_row - 1;
        }
        r2 = this->m_row - 1;
    }
    if (c1 < 0)
    {
        if (c2 < 0)
        {
            c2 = 0;
        }
        c1 = 0;
    }
    if (c2 >= this->m_col)
    {
        if (c1 >= this->m_col)
        {
            c1 = this->m_col - 1;
        }
        c2 = this->m_col - 1;
    }

    SparseMatrix<T> tmp(r2 - r1 + 1, c2 - c1 + 1);
    for (int i = r1; i <= r2; i++)
        for (int j = c1; j <= c2; j++)
            tmp.insert(i - r1, j - c1, (*this)[i][j]);

    return tmp;
}

template <typename T>
T SparseMatrix<T>::sum()
{
    T sum;
    sum = 0;
    for (int i = 0; i < m_row; i++)
        for (int j = 0; j < m_col; j++)
            sum += (*this)[i][j];

    return sum;
}

template <typename T>
T SparseMatrix<T>::mean()
{
    T total;
    total = this->m_row * this->m_col;
    return (this->sum() / total);
}

template <typename T>
T SparseMatrix<T>::max()
{
    int k = 0, m = 0, i, j;
    for (i = 0; i < this->m_row; i++)
        for (j = 0; j < this->m_col; j++)
            if ((*this)[i][j] > (*this)[k][m])
            {
                k = i;
                m = j;
            }

    return (*this)[k][m];
}

template <typename T>
T SparseMatrix<T>::min()
{
    int k = 0, m = 0, i = 0, j = 0;
    for (i = 0; i < this->m_row; i++)
        for (j = 0; j < this->m_col; j++)
            if ((*this)[k][m] > (*this)[i][j])
            {
                k = i;
                m = j;
            }

    return (*this)[k][m];
}

template <typename T>
T SparseMatrix<T>::row_max(int row)
{
    try{
        if(!(row >= 0 && row < this->m_row)){
            throw new InvalidCoordinatesException("the range of row is error");
        }
    }
    catch(Exception e){
        // e.what();
        // exit();
    }
    int k = 0;
    for (int i = 0; i < this->m_col; i++)
        if ((*this)[row][k] < (*this)[row][i])
            k = i;

    return (*this)[row][k];
}

template <typename T>
T SparseMatrix<T>::row_min(int row)
{
    try{
        if(!(row >= 0 && row < this->m_row)){
            throw new InvalidCoordinatesException("the range of row is error");
        }
    }
    catch(Exception e){
        // e.what();
        // exit();
    }
    int k = 0;
    for (int i = 0; i < this->m_col; i++)
        if ((*this)[row][k] > (*this)[row][i])
            k = i;

    return (*this)[row][k];
}

template <typename T>
T SparseMatrix<T>::row_sum(int row)
{
    try{
        if(!(row >= 0 && row < this->m_row)){
            throw new InvalidCoordinatesException("the range of row is error");
        }
    }
    catch(Exception e){
        // e.what();
        // exit();
    }
    T row_sum;
    row_sum = 0;
    for (int i = 0; i < this->m_col; i++)
    {
        row_sum += (*this)[row][i];
    }
    return row_sum;
}

template <typename T>
T SparseMatrix<T>::row_mean(int row)
{
    try{
        if(!(row >= 0 && row < this->m_row)){
            throw new InvalidCoordinatesException("the range of row is error");
        }
    }
    catch(Exception e){
        // e.what();
        // exit();
    }
    T total;
    total = (this->m_col);
    return this->row_sum(row) / total;
}

template <typename T>
T SparseMatrix<T>::col_max(int col)
{
    try{
        if(!(col >= 0 && col < this->m_col)){
            throw new InvalidCoordinatesException("the range of column is error");
        }
    }
    catch(Exception e){
        // e.what();
        // exit();
    }
    int k = 0;
    for (int i = 0; i < this->m_row; i++)
        if ((*this)[k][col] < (*this)[i][col])
            k = i;

    return (*this)[k][col];
}

template <typename T>
T SparseMatrix<T>::col_min(int col)
{
    try{
        if(!(col >= 0 && col < this->m_col)){
            throw new InvalidCoordinatesException("the range of column is error");
        }
    }
    catch(Exception e){
        // e.what();
        // exit();
    }
    int k = 0;
    for (int i = 0; i < this->m_row; i++)
        if ((*this)[k][col] > (*this)[i][col])
            k = i;

    return (*this)[k][col];
}

template <typename T>
T SparseMatrix<T>::col_sum(int col)
{
    try{
        if(!(col >= 0 && col < this->m_col)){
            throw new InvalidCoordinatesException("the range of column is error");
        }
    }
    catch(Exception e){
        // e.what();
        // exit();
    }
    T col_sum;
    col_sum = 0;
    for (int i = 0; i < this->m_row; i++)
    {
        col_sum += (*this)[i][col];
    }
    return col_sum;
}

template <typename T>
T SparseMatrix<T>::col_mean(int col)
{
    try{
        if(!(col >= 0 && col < this->m_col)){
            throw new InvalidCoordinatesException("the range of column is error");
        }
    }
    catch(Exception e){
        // e.what();
        // exit();
    }
    T total;
    total = this->m_row;
    return this->col_sum(col) / total;
}

// from: https://github.com/QuantitativeBytes/qbLinAlg/blob/main/qbQR.h
template <typename T>
pair<SparseMatrix<T>, SparseMatrix<T>> SparseMatrix<T>::QR_decomposition()
{
    SparseMatrix<T> input(*this);
    vector<SparseMatrix<T>> plist;
    for (int j = 0; j < m_row - 1; j++)
    {
        SparseMatrix<T> a1(1, m_row - j);
        SparseMatrix<T> b1(1, m_row - j);

        for (int i = j; i < m_row; i++)
        {
            a1.insert(0, i - j, input[i][j]);
            b1.remove(0, i - j);
        }
        b1[0][0] = static_cast<T>(1.0);

        T a1norm = a1.norm();

        T sgn = -1;
        if (a1[0][0] < static_cast<T>(0.0))
        {
            sgn = 1;
        }

        SparseMatrix<T> temp = b1 * sgn * a1norm;
        SparseMatrix<T> u = a1 - temp;
        SparseMatrix<T> n = u.normalized();
        SparseMatrix<T> nTrans = n.Transposition();
        SparseMatrix<T> I(m_row - j, m_row - j);
        I.SetIdentity();

        SparseMatrix<T> temp1 = n * static_cast<T>(2.0);
        SparseMatrix<T> temp2 = nTrans * temp1;
        SparseMatrix<T> Ptemp = I - temp2;

        SparseMatrix<T> P(m_row, m_col);
        P.SetIdentity();

        for (int x = j; x < m_row; x++)
        {
            for (int y = j; y < m_col; y++)
            {
                P.insert(x, y, Ptemp[x - j][y - j]);
            }
        }

        plist.push_back(P);
        input = P * input;
    }

    SparseMatrix<T> qMat = plist[0];
    for (int i = 1; i < m_row - 1; i++)
    {
        SparseMatrix<T> temp3 = plist[i].Transposition();
        qMat = qMat * temp3;
    }

    int numElements = plist.size();
    SparseMatrix<T> rMat = plist[numElements - 1];
    for (int i = (numElements - 2); i >= 0; i--)
    {
        rMat = rMat * plist[i];
    }
    rMat = rMat * (*this);

    return pair<SparseMatrix<T>, SparseMatrix<T>>(qMat, rMat);
}

template <typename T>
T SparseMatrix<T>::norm()
{
    T cumulativeSum = static_cast<T>(0.0);
    for (int i = 0; i < m_row; i++)
        for (int j = 0; j < m_col; j++)
            cumulativeSum += (*this)[i][j] * (*this)[i][j];

    return sqrt(cumulativeSum);
}

template <typename T>
SparseMatrix<T> SparseMatrix<T>::normalized()
{
    T norm = this->norm();
    SparseMatrix<T> copy(*this);
    return copy * (static_cast<T>(1.0) / norm);
}

template <typename T>
void SparseMatrix<T>::SetIdentity()
{
    for (int i = 0; i < m_row; i++)
        for (int j = 0; j < m_col; j++)
            if (i == j)
                insert(i, j, static_cast<T>(1.0));
            else
                remove(i, j);
}

// from: https://github.com/QuantitativeBytes/qbLinAlg/blob/main/qbEIG.h
// only work for symmetric metrices
template <typename T>
T *SparseMatrix<T>::eigenvalues(int max_iter)
{
    SparseMatrix<T> A = (*this);
    SparseMatrix<T> identitySparseMatrix(m_row, m_col);
    identitySparseMatrix.SetIdentity();

    for (int i = 0; i < max_iter; i++)
    {
        auto qrResult = A.QR_decomposition();
        A = qrResult.second * qrResult.first;
        if (A.isUpperTri())
            break;
    }

    T *eigenvalues = new T[m_row];
    for (int i = 0; i < m_row; i++)
    {
        eigenvalues[i] = A[i][i];
    }
    return eigenvalues;
}

template <typename T>
bool SparseMatrix<T>::isCloseEnough(T a, T b, double threshold)
{
    return abs(a - b) < static_cast<T>(threshold);
}

template <>
bool SparseMatrix<uchar>::isCloseEnough(uchar a, uchar b, double threshold)
{
    return a == b;
}

template <>
bool SparseMatrix<char>::isCloseEnough(char a, char b, double threshold)
{
    return a == b;
}

template <typename T>
bool SparseMatrix<T>::isUpperTri()
{
    T cumulativeSum = static_cast<T>(0);
    for (int i = 1; i < m_row; i++)
    {
        for (int j = 0; j < i; j++)
        {
            cumulativeSum += (*this)[i][j];
        }
    }
    return isCloseEnough(cumulativeSum, static_cast<T>(0));
}

// from: https://github.com/QuantitativeBytes/qbLinAlg/blob/main/qbEIG.h
template <typename T>
SparseMatrix<T> SparseMatrix<T>::eigenvector(T eigenvalue, int max_iter)
{
    SparseMatrix<T> A = (*this);
    random_device myRandomDevice;
    mt19937 myRandomGenerator(myRandomDevice());
    uniform_int_distribution<int> myDistribution(1.0, 10.0);

    SparseMatrix<T> identitySparseMatrix(m_row, m_col);
    identitySparseMatrix.SetIdentity();

    SparseMatrix<T> v(m_row, 1);
    for (int i = 0; i < m_row; i++)
    {
        v.insert(i, 0, static_cast<T>(myDistribution(myRandomGenerator)));
    }

    T deltaThreshold = static_cast<T>(EQ_THRESHOLD);
    T delta = static_cast<T>(1e-6);
    SparseMatrix<T> preVector(m_row, 1);
    SparseMatrix<T> tempSparseMatrix(m_row, m_row);

    for (int i = 0; i < max_iter; i++)
    {
        preVector = v;
        SparseMatrix<T> temp = identitySparseMatrix * eigenvalue;
        tempSparseMatrix = A - temp;
        tempSparseMatrix = tempSparseMatrix.Inverse();
        v = tempSparseMatrix * v;
        v = v.normalized();

        delta = (v - preVector).norm();
        if (delta > deltaThreshold)
            break;
    }
    return v;
}

template <typename T>
SparseMatrix<T> *SparseMatrix<T>::eigenvectors(int max_itr)
{
    SparseMatrix<T> *eigenvectors = new SparseMatrix<T>[m_row];
    T *eigenvalues = this->eigenvalues();
    for (int i = 0; i < m_row; i++)
    {
        eigenvectors[i] = this->eigenvector(*(eigenvalues + i));
    }
    return eigenvectors;
}

// from: https://github.com/QuantitativeBytes/qbLinAlg/blob/main/qbEIG.h
template <typename T>
pair<T, SparseMatrix<T>> SparseMatrix<T>::eigenValueAndEigenVector(int max_itr)
{
    T eigenvalue;
    SparseMatrix<T> inputSparseMatrix = (*this);
    random_device myRandomDevice;
    mt19937 myRandomGenerator(myRandomDevice());
    uniform_int_distribution<int> myDistribution(1.0, 10.0);
    SparseMatrix<T> identitySparseMatrix(m_row, m_col);
    identitySparseMatrix.SetIdentity();

    SparseMatrix<T> v(m_row, 1);
    for (int i = 0; i < m_row; i++)
    {
        v.insert(i, 0, static_cast<T>(static_cast<T>(myDistribution(myRandomGenerator))));
    }
    SparseMatrix<T> v1(m_row, 1);
    for (int i = 0; i < max_itr; i++)
    {
        v1 = inputSparseMatrix * v;
        v1 = v1.normalized();
        v = v1;
    }
    T cumulativeSum = static_cast<T>(0.0);
    for (int i = 1; i < m_row; i++)
    {
        cumulativeSum += inputSparseMatrix[0][i] * v1[i][0];
    }
    eigenvalue = (cumulativeSum / v1[0][0]) + inputSparseMatrix[0][0];
    return pair<T, SparseMatrix<T>>(eigenvalue, v1);
}

// learn from: https://stackoverflow.com/questions/26681713/convert-mat-to-array-vector-in-opencv
template <typename T>
SparseMatrix<T> SparseMatrix<T>::fromOpenCV(const cv::Mat &cvMat)
{
    int row = cvMat.rows;
    int col = cvMat.cols;
    SparseMatrix<T> result(row, col);
    cv::MatConstIterator_<T> it = cvMat.begin<T>(), it_end = cvMat.end<T>();
    vector<T> array;
    if (cvMat.isContinuous())
    {
        array.assign((T *)cvMat.data, (T *)cvMat.data + cvMat.total() * cvMat.channels());
    }
    else
    {
        for (int i = 0; i < cvMat.rows; ++i)
        {
            array.insert(array.end(), cvMat.ptr<T>(i), cvMat.ptr<T>(i) + cvMat.cols * cvMat.channels());
        }
    }
    int cnt = 0;
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
        {
            result.insert(i, j, array[cnt++]);
        }
    }
    return result;
}

template <typename T>
T **SparseMatrix<T>::toArray()
{
    T **array = new T *[m_row];
    for (int i = 0; i < m_row; i++)
    {
        array[i] = new T[m_col];
    }
    for (int i = 0; i < m_row; i++)
    {
        for (int j = 0; j < m_col; j++)
        {
            array[i][j] = (*this)[i][j];
        }
    }
    return array;
}

template <typename T>
cv::Mat *SparseMatrix<T>::toOpenCVMat(int type)
{
    cv::Mat *cvMat = new cv::Mat(m_row, m_col, type);
    for (int i = 0; i < m_row; i++)
    {
        for (int j = 0; j < m_col; j++)
        {
            (*cvMat).at<T>(i, j) = (*this)[i][j];
        }
    }
    return cvMat;
}

template <typename T>
SparseMatrix<T> SparseMatrix<T>::conv2D(SparseMatrix<T> &input, SparseMatrix<T> &kernel, int stride, bool same_padding)
{
    SparseMatrix<T> inputSparseMatrix;
    int padding = 0;
    if (same_padding)
    {
        padding = 1;
    }
    inputSparseMatrix = SparseMatrix<T>(input.m_row + padding * 2, input.m_col + padding * 2);
    for (int i = 0; i < input.m_row; i++)
    {
        for (int j = 0; j < input.m_col; j++)
        {
            inputSparseMatrix.insert(i + padding, j + padding, input[i][j]);
        }
    }
    if (padding == 1)
    {
        for (int i = 0; i < inputSparseMatrix.m_row; i++)
        {
            inputSparseMatrix.remove(i, 0);
            inputSparseMatrix.remove(i, inputSparseMatrix.m_col - 1);
        }
        for (int i = 0; i < inputSparseMatrix.m_col; i++)
        {
            inputSparseMatrix.remove(0, i);
            inputSparseMatrix.remove(inputSparseMatrix.m_row - 1, i);
        }
    }
    int rowDim = ((input.m_row + 2 * padding - kernel.m_row) / stride) + 1;
    int colDim = ((input.m_col + 2 * padding - kernel.m_col) / stride) + 1;
    SparseMatrix<T> result(rowDim, colDim);
    for (int i = 0; i < rowDim; i++)
    {
        for (int j = 0; j < colDim; j++)
        {
            T cumulativeSum = static_cast<T>(0);
            for (int x = 0; x < kernel.m_row; x++)
            {
                for (int y = 0; y < kernel.m_col; y++)
                {
                    cumulativeSum += kernel[x][y] * inputSparseMatrix[x + i * stride][y + j * stride];
                }
            }
            result.insert(i, j, cumulativeSum);
        }
    }
    return result;
}