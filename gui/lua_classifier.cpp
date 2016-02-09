#include "lua_classifier.hpp"

#include <iostream>

extern "C" {
#include <lua.h>
#include <lauxlib.h>
#include <lualib.h>
}

LuaClassifier::LuaClassifier(std::string const & path)
{
    L = lua_open();
    luaL_openlibs(L);
    if (luaL_dofile(L, path.c_str())) {
        std::cerr << "lua: " << lua_tostring(L, -1) << std::endl;
        lua_pop(L, 1);
    }
}

LuaClassifier::~LuaClassifier()
{
    lua_close(L);
}

std::vector<double> LuaClassifier::predict(std::vector<unsigned char> const & pixels)
{
    lua_getglobal(L, "predict");

    lua_newtable(L);
    for (auto i = 0; i < pixels.size(); i++) {
        lua_pushnumber(L, i + 1); // +1 since lua is not zero-based
        lua_pushnumber(L, pixels[i]);
        lua_settable(L, -3);
    }

    std::vector<double> preds;
    if (lua_pcall(L, 1, 1, 0)) {
        std::cerr << "lua: " << lua_tostring(L, -1) << std::endl;
        lua_pop(L, 1);
    } else {
        lua_pushnil(L);
        while (lua_next(L, -2)) {
            auto v = lua_tonumber(L, -1);
            lua_pop(L, 1);
            auto k = lua_tonumber(L, -1);
            preds.push_back(v);
        }
    }
    return preds;
}
