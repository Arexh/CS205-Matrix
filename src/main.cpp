#include <opencv2/opencv.hpp>
#include <iostream>
#include "Matrix.h"
#include "SparseMatrix.h"
#include <fstream>
#include "DNN.h"

using namespace std;

const int INPUT_DIM = 784;
const int HIDDEN_DIM = 200;
const int OUTPUT_DIM = 10;
const double LR = 0.1;
const int CLASS_NUM = 10;
const double SHIFT = 0.01;
const double DATA_SCALE = 0.99;
const double DATA_RANGE = 255.0;

const int EPOCH = 10;

#define TRAIN_SET "datasets/mnist_train_100.csv"
#define TEST_SET "datasets/mnist_test_10.csv"

pair<Matrix<double>*, Matrix<double>*>* readCSVFile(string filename)
{
    ifstream csv_file(filename);
    if (!csv_file.good())
    {
        cout << "Open csv file error!" << endl;
        return nullptr;
    }
    vector<vector<double>> data;
    vector<vector<double>> target;
    for (string line; getline(csv_file, line);)
    {
        stringstream ss(line);
        vector<double> vv;
        int j = 0;
        for (string entry; getline(ss, entry, ','); j++)
        {
            if (j == 0) {
                vector<double> targetList(CLASS_NUM);
                for (int i = 0; i < CLASS_NUM; i++)
                    targetList[i] = SHIFT;
                targetList[stoi(entry)] = 1 - SHIFT;
                target.push_back(targetList);
                continue;
            }
            vv.push_back(stoi(entry) / DATA_RANGE * DATA_SCALE);
        }
        data.push_back(vv);
    }
    Matrix<double> *dataMat = new Matrix<double>(data);
    Matrix<double> *targetMat = new Matrix<double>(target);
    return new pair<Matrix<double>*, Matrix<double>*>(dataMat, targetMat);
}

vector<int>* readLabels(string filename)
{
    ifstream csv_file(filename);
    if (!csv_file.good())
    {
        cout << "Open csv file error!" << endl;
        return nullptr;
    }
    vector<int> *labels = new vector<int>();
    for (string line; getline(csv_file, line);)
    {
        stringstream ss(line);
        int j = 0;
        for (string entry; getline(ss, entry, ','); j++)
            if (j == 0) {
                labels->push_back(stoi(entry));
                break;
            }
    }
    return labels;
}

double performance(vector<int>* result, vector<int>* labels)
{
    int cnt = 0;
    for (int i = 0; i < result->size(); i++)
        if ((*result)[i] == (*labels)[i]) cnt++;
    if (cnt == 0) return 0;
    return ((double) cnt / result->size());
}

int main()
{
    DNN dnn(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, LR);
    pair<Matrix<double>*, Matrix<double>*>* train = readCSVFile(TRAIN_SET);
    pair<Matrix<double>*, Matrix<double>*>* test = readCSVFile(TEST_SET);
    vector<int>* labels = readLabels(TEST_SET);
    dnn.initWeight();
    for (int i = 0; i < EPOCH; i++)
        dnn.train(train->first, train->second);
    vector<int>* result = dnn.evaluate(test->first);
    cout << "Accuracy: " << performance(result, labels) << endl;
    return 0;
}