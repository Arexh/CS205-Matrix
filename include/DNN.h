#ifndef _DNN_H_
#define _DNN_H_

#include <Matrix.h>
#include <random>

class DNN
{
public:
    int inputNodes;
    int hiddenNodes;
    int outputNodes;
    double learningRate;
    Matrix<double> m1;
    Matrix<double> m2;
public:
    DNN(int i, int h, int o, double l): inputNodes(i), hiddenNodes(h), outputNodes(o), learningRate(l), m1(h, i), m2(o, h)
    {
        random_device rnd_device;
        mt19937 mersenne_engine {rnd_device()};
        uniform_int_distribution<int> dist {1, 52};
        auto gen = [&dist, &mersenne_engine](){
                    return dist(mersenne_engine);
                };
        vector<int> vec(10);
        generate(begin(vec), end(vec), gen);
    }
    void initWeight();
    void train(const Matrix<double> *input, const Matrix<double> *target);
    vector<int>* evaluate(const Matrix<double> *input);
    static double sigmoid(double x);
};

#endif