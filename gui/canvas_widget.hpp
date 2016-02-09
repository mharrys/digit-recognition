#ifndef CANVAS_WIDGET_HPP_INCLUDED
#define CANVAS_WIDGET_HPP_INCLUDED

#include <QWidget>

class CanvasWidget : public QWidget {
    Q_OBJECT
public:
    CanvasWidget(QWidget * parent = 0);
public slots:
    void clear();
signals:
    void changed(QImage image);
protected:
    void mousePressEvent(QMouseEvent * event) override;
    void mouseMoveEvent(QMouseEvent * event) override;
    void mouseReleaseEvent(QMouseEvent * event) override;
    void paintEvent(QPaintEvent * event) override;
    void resizeEvent(QResizeEvent * event) override;
private:
    bool dragging;
    QColor penColor;
    int penWidth;
    QPoint prevPos;
    QImage image;
};

#endif
