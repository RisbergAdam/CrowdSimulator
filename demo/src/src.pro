TEMPLATE = app
TARGET = demo
DEPENDPATH += . release
INCLUDEPATH += . ../../libpedsim
DESTDIR = ../

QMAKE_CXXFLAGS += -std=c++0x -Wno-deprecated-register -g

LIBS += -L../../libpedsim -lpedsim $$CUSTOMLIBS

QT += opengl
QT += widgets

CONFIG += release
CONFIG += console

# Input
HEADERS += MainWindow.h ParseScenario.h  ViewAgent.h PedSimulation.h
SOURCES += main.cpp MainWindow.cpp ParseScenario.cpp ViewAgent.cpp PedSimulation.cpp
