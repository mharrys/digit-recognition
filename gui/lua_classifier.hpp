#ifndef LUA_CLASSIFIER_HPP_INCLUDED
#define LUA_CLASSIFIER_HPP_INCLUDED

#include <string>

#include "classifier.hpp"

struct lua_State;

// Use a torch7 model to make predictions.
class LuaClassifier : public Classifier {
public:
    // Construct classifier with specified file path to model.
    LuaClassifier(std::string const & path);
    ~LuaClassifier();
    std::vector<double> predict(std::vector<unsigned char> const & pixels) override;
private:
    lua_State * L;
};

#endif
