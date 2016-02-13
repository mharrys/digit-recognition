#ifndef DIGIT_WIDGET_HPP_INCLUDED
#define DIGIT_WIDGET_HPP_INCLUDED

#include <memory>

#include "ui_digit_widget.h"

#include <QWidget>

#include "classifier.hpp"

class Classifier;

class DigitWidget : public QWidget , private Ui::DigitWidget {
    Q_OBJECT
public:
    DigitWidget(std::unique_ptr<Classifier> classifier, QWidget * parent = 0);
public slots:
    void predict(QImage image);
    void clear();
private:
    std::unique_ptr<Classifier> classifier;
};

#endif
