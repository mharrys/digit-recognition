#include "digit_widget.hpp"

#include <QtDebug>

DigitWidget::DigitWidget(std::unique_ptr<Classifier> classifier, QWidget * parent)
    : QWidget(parent),
      classifier(std::move(classifier))
{
    setupUi(this);
    QObject::connect(clearCanvasButton, &QPushButton::clicked, canvasWidget, &CanvasWidget::clear);
    QObject::connect(canvasWidget, &CanvasWidget::changed, this, &DigitWidget::predict);
}

void DigitWidget::predict(QImage image)
{
    image = image.scaled(28, 28, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    image = image.convertToFormat(QImage::Format_Grayscale8);
    auto bits = image.bits();
    std::vector<unsigned char> pixels(28 * 28);
    for (auto i = 0; i < 28 * 28; i++)
        pixels[i] = bits[i];

    auto result = classifier->predict(pixels);
    for (auto i = 0; i < result.size(); i++)
        result[i] = result[i] * 100;

    bar0->setValue(result[9]);
    bar1->setValue(result[0]);
    bar2->setValue(result[1]);
    bar3->setValue(result[2]);
    bar4->setValue(result[3]);
    bar5->setValue(result[4]);
    bar6->setValue(result[5]);
    bar7->setValue(result[6]);
    bar8->setValue(result[7]);
    bar9->setValue(result[8]);
}
