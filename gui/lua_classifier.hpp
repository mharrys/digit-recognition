#ifndef LUA_CLASSIFIER_HPP_INCLUDED
#define LUA_CLASSIFIER_HPP_INCLUDED

#include <string>

#include "classifier.hpp"

struct lua_State;

// Use and external torch7 file to make predictions. The file is expected to
// have a "predict" function that takes a table of pixels as parameters and
// return probabilities for each digit in a table.
class LuaClassifier : public Classifier {
public:
    // Construct classifier with specified file path.
    LuaClassifier(std::string const & path);
    ~LuaClassifier();
    std::vector<double> predict(std::vector<unsigned char> const & pixels) override;
private:
    lua_State * L;
};

#endif
