#include <iostream>
#include <memory>

#include <QtDebug>
#include <QApplication>
#include <QCommandLineParser>

#include "lua_classifier.hpp"
#include "digit_widget.hpp"

std::unique_ptr<Classifier> make(std::string const & path)
{
    return std::unique_ptr<Classifier>(new LuaClassifier(path));
}

int main(int argc, char * argv[])
{
    QApplication app(argc, argv);

    QCommandLineParser parser;
    parser.addHelpOption();
    parser.addVersionOption();
    parser.addPositionalArgument("path", "File path to torch7 model");
    parser.process(app);

    auto args = parser.positionalArguments();
    if (args.isEmpty()) {
        parser.showHelp(1);
        return 1;
    }

    auto classifier = make(args.at(0).toStdString());
    DigitWidget w(std::move(classifier));
    w.show();

    return app.exec();
}
