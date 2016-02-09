#include "canvas_widget.hpp"

#include <QMouseEvent>
#include <QResizeEvent>
#include <QPainter>
#include <QBrush>
#include <QPen>

CanvasWidget::CanvasWidget(QWidget * parent)
    : QWidget(parent),
      dragging(false),
      penColor(Qt::white),
      penWidth(32),
      image(size(), QImage::Format_RGB32)
{
    clear();
}

void CanvasWidget::clear()
{
    image.fill(Qt::black);
    repaint();
}

void CanvasWidget::mousePressEvent(QMouseEvent * event)
{
    auto button = event->button();

    if (button == Qt::LeftButton || button == Qt::RightButton) {
        dragging = true;
        prevPos = event->pos();
    }

    if (button == Qt::LeftButton)
        penColor = Qt::white;
    else if (button == Qt::RightButton)
        penColor = Qt::black;
}

void CanvasWidget::mouseMoveEvent(QMouseEvent * event)
{
    if (dragging) {
        QPainter painter(&image);
        painter.setRenderHint(QPainter::Antialiasing);
        QPen pen(penColor, penWidth, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin);
        painter.setPen(pen);
        painter.drawLine(prevPos, event->pos());
        prevPos = event->pos();
        repaint();
    }
}

void CanvasWidget::mouseReleaseEvent(QMouseEvent * event)
{
    auto button = event->button();
    if (button == Qt::LeftButton || button == Qt::RightButton) {
        dragging = false;
        emit changed(image);
    }
}

void CanvasWidget::paintEvent(QPaintEvent * event)
{
    QPainter painter(this);
    QRect rect = event->rect();
    painter.drawImage(rect, image, rect);
}

void CanvasWidget::resizeEvent(QResizeEvent * event)
{
    image = QImage(event->size(), QImage::Format_RGB32);
}
