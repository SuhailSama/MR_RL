#!/usr/bin/python
#-*- coding: utf-8 -*-
import turtle
import math
import tkinter

class Viewer:
    def __init__(self):
        self.l_MR = 2  # length of MR
        #first we initialize the turtle settings
        turtle.speed(0)
        turtle.mode('logo')
        turtle.setworldcoordinates(-10,-10, 1000,1000)
        turtle.setup()
        turtle.screensize(100, 100, 'white')
        w_MR = 2  # MR width
        turtle.register_shape('MR', (
            (0, self.l_MR), (w_MR, self.l_MR ), (w_MR, -self.l_MR), (-w_MR, -self.l_MR),
            (-w_MR, self.l_MR)))
        turtle.degrees()
        self.MR = turtle.Turtle()
        self.MR.shape('MR')
        self.MR.fillcolor('red')
        self.MR.penup()
        self.step_count = 0

    def plot_position(self, x, y):
        self.MR.setpos(x, y)
        self.MR.pendown()


    def  plot_goal(self, point, factor):
        turtle.speed(0)
        turtle.setpos(point[0] - factor, point[1] - factor)
        turtle.pendown()
        turtle.fillcolor('green')
        turtle.begin_fill()
        turtle.setpos(point[0] - factor, point[1] + factor)
        turtle.setpos(point[0] + factor, point[1] + factor)
        turtle.setpos(point[0] + factor, point[1] - factor)
        turtle.end_fill()
        turtle.penup()

    def plot_boundary(self, points_list):
        turtle.speed(0)
        turtle.setpos(points_list[0][0], points_list[0][1])
        turtle.pendown()
        turtle.fillcolor('blue')
        turtle.begin_fill()
        for point in points_list:
            turtle.setpos(point[0], point[1])
        turtle.end_fill()
        turtle.penup()

    def freeze_scream(self, ):
        turtle.mainloop()

    def end_episode(self, ):
        self.MR.penup()

    def restart_plot(self):
        self.MR.pendown()

if __name__ == '__main__':
    viewer = Viewer()
    viewer.plot_position(100, 20)
    viewer.freeze_scream()