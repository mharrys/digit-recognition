#ifndef CLASSIFIER_HPP_INCLUDED
#define CLASSIFIER_HPP_INCLUDED

#include <vector>

// The responsibility of this class is to classify an image to a digit.
class Classifier {
public:
    // Return probabilities for each digit. Expect that each digit position is at
    // [1,2,3,4,5,6,7,8,9,0] in the result.
    virtual std::vector<double> predict(std::vector<unsigned char> const & pixels) = 0;
};

#endif
