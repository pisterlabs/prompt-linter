"""
2D rendering framework

Adapted by Joshua Smith from OpenAI Gym
Important note most of the drawing functions have not been made into 3D only the targets and spheres are 3D.
"""
from __future__ import division
import os
import six
import sys

from gym.utils import reraise
from gym import error

RAD2DEG = 57.29577951308232
GOLDENRATIO = (1+5**0.5)/2
import ctypes
try:
    import pyglet
except ImportError as e:
    reraise(suffix="HINT: you can install pyglet directly via 'pip install pyglet'. But if you really just want to install all Gym dependencies and not have to think about it, 'pip install -e .[all]' or 'pip install gym[all]' will do it.")

try:
    from pyglet.gl import *
except ImportError as e:
    reraise(prefix="Error occured while running `from pyglet.gl import *`",suffix="HINT: make sure you have OpenGL install. On Ubuntu, you can run 'apt-get install python-opengl'. If you're running on a server, you may need a virtual frame buffer; something like this should work: 'xvfb-run -s \"-screen 0 1400x900x24\" python <your_script.py>'")

import math
import numpy as np
from pyquaternion import Quaternion

def get_display(spec):
    """Convert a display specification (such as :0) into an actual Display
    object.
    Pyglet only supports multiple Displays on Linux.
    """
    if spec is None:
        return None
    elif isinstance(spec, six.string_types):
        return pyglet.canvas.Display(spec)
    else:
        raise error.Error('Invalid display specification: {}. (Must be a string like :0 or None.)'.format(spec))

class Viewer(object):
    def __init__(self, width, height, display=None):
        display = get_display(display)

        self.width = width
        self.height = height
        self.window_xy = pyglet.window.Window(caption='xy-plane',width=width, height=height, display=display)
        self.window_xy.on_close = self.window_closed_by_user
        self.window_xz = pyglet.window.Window(caption='xz-plane',width=width, height=height, display=display)
        self.window_xz.on_close = self.window_closed_by_user
        self.geoms = []
        self.onetime_geoms = []
        self.transform_xy = Transform()
        self.transform_xz = Transform()
        self.perspective_transform_on = False

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)


    def close(self):
        self.window_xy.close()

    def window_closed_by_user(self):
        self.close()

    def set_bounds(self, left, right, bottom, top):
        assert right > left and top > bottom
        scalex = self.width/(right-left)
        scaley = self.height/(top-bottom)
        self.transform_xy = Transform(
            translation=(-left*scalex, -bottom*scaley,-bottom*scaley),
            scale=(scalex, scaley, scaley))
        self.transform_xz = Transform(
            translation=(-left*scalex, -bottom*scaley,-bottom*scaley),
            scale=(scalex, scaley, scaley))

    def add_geom(self, geom):
        self.geoms.append(geom)

    def add_onetime(self, geom):
        self.onetime_geoms.append(geom)

    def render(self, return_rgb_array=False):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(1,1,1,1)
        self.window_xy.switch_to()
        self.window_xy.clear()

        self.window_xy.dispatch_events()

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        if self.perspective_transform_on:
            glOrtho(-400,400,-400,400,800,1600)
            resolution = (800/8.4)
            #This is an example of a calibrated camera in opengl, it has the focal length of 7.35m
            projection = (GLfloat *16)(7.35*resolution,0,0,0 ,0,7.35*resolution,0,0, 0,0,8.4*resolution+16.8*resolution,-1, 0,0,8.4*resolution*16.8*resolution,0)
            glMultMatrixf(projection)
            #X axis offset
            glTranslatef((2.4*800)/8.4,0,0)
        else:
            glOrtho(-self.width/2,self.width/2,-self.height/2,self.height/2,800,1600)

        self.transform_xy.enable()
        glPushMatrix()
        glMatrixMode(GL_MODELVIEW)
        glTranslatef(-4.2,-4.2,-16.8)
        for geom in self.geoms:
            geom.render()
        for geom in self.onetime_geoms:
            geom.render()
        glPopMatrix()
        self.transform_xy.disable()
        arrxy = None
        if return_rgb_array:
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = buffer.get_image_data()
            arrxy = np.fromstring(image_data.data, dtype=np.uint8, sep='')
            # In https://github.com/openai/gym-http-api/issues/2, we
            # discovered that someone using Xmonad on Arch was having
            # a window of size 598 x 398, though a 600 x 400 window
            # was requested. (Guess Xmonad was preserving a pixel for
            # the boundary.) So we use the buffer height/width rather
            # than the requested one.
            arrxy = arrxy.reshape(buffer.height, buffer.width, 4)
            arrxy = arrxy[::-1,:,0:3]

        self.window_xy.flip()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(1,1,1,1)
        self.window_xz.switch_to()
        self.window_xz.clear()

        self.window_xz.dispatch_events()
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        if self.perspective_transform_on:
            glOrtho(-400,400,-400,400,800,1600)

            projection = (GLfloat *16)(700,0,0,0 ,0,700,0,0, 0,0,800+1600,-1, 0,0,800*1600,0)
            glMultMatrixf(projection)
            glTranslatef(-(2.4*800)/8.4,0,0)
        else:
            glOrtho(-self.width/2,self.width/2,-self.height/2,self.height/2,800,1600)
        #glOrtho(0,self.width,0,self.height,0,-5000)
        self.transform_xz.enable()
        glPushMatrix()
        glMatrixMode(GL_MODELVIEW)
        glTranslatef(-4.2,-4.2,-16.8)
        if not self.perspective_transform_on:
            #rotate onto the xz plane
            glRotatef(-90, 1, 0,0)
        for geom in self.geoms:
            geom.render()
        for geom in self.onetime_geoms:
            geom.render()
        glPopMatrix()
        self.transform_xz.disable()
        arrxz = None
        if return_rgb_array:
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = buffer.get_image_data()
            arrxz = np.fromstring(image_data.data, dtype=np.uint8, sep='')
            # In https://github.com/openai/gym-http-api/issues/2, we
            # discovered that someone using Xmonad on Arch was having
            # a window of size 598 x 398, though a 600 x 400 window
            # was requested. (Guess Xmonad was preserving a pixel for
            # the boundary.) So we use the buffer height/width rather
            # than the requested one.
            arrxz = arrxz.reshape(buffer.height, buffer.width, 4)
            arrxz = arrxz[::-1,:,0:3]
        self.window_xz.flip()
        self.onetime_geoms = []
        return (arrxy,arrxz)

    # Convenience
    def draw_sphere(self, radius=10, res=2, filled=True, **attrs):
        geom = make_sphere(radius=radius, res=res, filled=filled)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def draw_polygon(self, v, filled=True, **attrs):
        geom = make_polygon(v=v, filled=filled)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def draw_polyline(self, v, **attrs):
        geom = make_polyline(v=v)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def draw_line(self, start, end, **attrs):
        geom = Line(start, end)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def get_array(self):
        self.window_xy.flip()
        image_data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
        self.window_xy.flip()
        arr = np.fromstring(image_data.data, dtype=np.uint8, sep='')
        arr = arr.reshape(self.height, self.width, 4)
        return arr[::-1,:,0:3]

def _add_attrs(geom, attrs):
    if "color" in attrs:
        geom.set_color(*attrs["color"])
    if "linewidth" in attrs:
        geom.set_linewidth(attrs["linewidth"])

class Geom(object):
    def __init__(self):
        self._color=Color((1.0, 0, 0, 1.0))
        self.attrs = [self._color]
    def render(self):
        for attr in reversed(self.attrs):
            attr.enable()
        self.render1()
        for attr in self.attrs:
            attr.disable()
    def render1(self):
        raise NotImplementedError
    def add_attr(self, attr):
        self.attrs.append(attr)
        return
    def set_color(self, r, g, b):
        self._color.vec4 = (r, g, b, 1)

class Attr(object):
    def enable(self):
        raise NotImplementedError
    def disable(self):
        pass

class Transform(Attr):
    def __init__(self, translation=(0.0, 0.0, 0.0), rotation=Quaternion(), scale=(1,1,1)):
        self.set_translation(*translation)
        self.set_rotation(rotation)
        self.set_scale(*scale)
    def enable(self):
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()

        glTranslatef(self.translation[0], self.translation[1], self.translation[2]) # translate to GL loc ppint
        glRotatef(RAD2DEG * self.rotation.angle,self.rotation.axis[0],self.rotation.axis[1],self.rotation.axis[2])
        glScalef(self.scale[0], self.scale[1], self.scale[2])
    def disable(self):
        glPopMatrix()
    def set_translation(self, newx, newy, newz):
        self.translation = (float(newx), float(newy), float(newz))
    def set_rotation(self, quat):
        self.rotation = quat
    def set_scale(self, newx, newy, newz):
        self.scale = (float(newx), float(newy), float(newz))

class Color(Attr):
    def __init__(self, vec4):
        self.vec4 = vec4
    def enable(self):
        glColor4f(*self.vec4)

class LineStyle(Attr):
    def __init__(self, style):
        self.style = style
    def enable(self):
        glEnable(GL_LINE_STIPPLE)
        glLineStipple(1, self.style)
    def disable(self):
        glDisable(GL_LINE_STIPPLE)

class LineWidth(Attr):
    def __init__(self, stroke):
        self.stroke = stroke
    def enable(self):
        glLineWidth(self.stroke)

class Point(Geom):
    def __init__(self):
        Geom.__init__(self)
    def render1(self):
        glBegin(GL_POINTS) # draw point
        glVertex3f(0.0, 0.0, 0.0)
        glEnd()

class FilledPolygon(Geom):
    def __init__(self, v):
        Geom.__init__(self)
        self.v = v
    def render1(self):
        if   len(self.v) == 4 : glBegin(GL_QUADS)
        elif len(self.v)  > 4 : glBegin(GL_POLYGON)
        else: glBegin(GL_TRIANGLES)
        for p in self.v:
            glVertex3f(p[0], p[1], p[2])  # draw each vertex
        glEnd()

#This methods allow you to specify vertices and faces to draw
class FilledPolygonJosh(Geom):
    def __init__(self, v,f, draw_type):
        Geom.__init__(self)
        self.v = v
        self.f = f
        self.type = draw_type
    def render1(self):
        glBegin(self.type)
        for f in self.f:
            for f1 in f:
                glVertex3f(self.v[f1][0], self.v[f1][1], self.v[f1][2])  # draw each vertex
        glEnd()

#Find the mid point between vertices
def mid_point(points, v1, v2):
    a = (points[v1]+points[v2])/2
    if not any((a==x).all() for x in points):
        points.append(a)
    return next((i for i, x in enumerate(points) if np.all(x==a)),-1)

#Subdivide triangles to make more faces which causes higher resolution in 3D
def subdivide_triangles((points,faces), face):
    a = mid_point(points,face[0],face[1])
    b = mid_point(points,face[1],face[2])
    c = mid_point(points,face[2],face[0])
    faces.append((face[0],a,c))
    faces.append((face[1],b,a))
    faces.append((face[2],c,b))
    faces.append((a,b,c))

#Makes an icosahedron sphere. Any res over around 3 or 4 gets incredibly slow to generate
def make_sphere(radius=10, res=2, filled=True):
    points = []
    phi = (1+math.sqrt(5))/2
    points.append(np.array([-1, phi, 0]))
    points.append(np.array([1, phi, 0]))
    points.append(np.array([-1, -phi, 0]))
    points.append(np.array([1, -phi, 0]))

    points.append(np.array([0, -1, phi]))
    points.append(np.array([0, 1, phi]))
    points.append(np.array([0, -1, -phi]))
    points.append(np.array([0, 1, -phi]))

    points.append(np.array([phi, 0, -1]))
    points.append(np.array([phi, 0, 1]))
    points.append(np.array([-phi, 0, -1]))
    points.append(np.array([-phi, 0, 1]))

    points2 = []
    for p in points:
        points2.append(p*(radius/(2*math.sin(2*math.pi/5))))

    points = points2

    pList=[]
    pList.append((0,11,5))
    pList.append((0,5,1))
    pList.append((0,1,7))
    pList.append((0,7,10))
    pList.append((0,10,11))
    pList.append((1,5,9))
    pList.append((5,11,4))
    pList.append((11,10,2))
    pList.append((10,7,6))
    pList.append((7,1,8))
    pList.append((3,9,4))
    pList.append((3,4,2))
    pList.append((3,2,6))
    pList.append((3,6,8))
    pList.append((3,8,9))
    pList.append((4,9,5))
    pList.append((2,4,11))
    pList.append((6,2,10))
    pList.append((8,6,7))
    pList.append((9,8,1))
    faces=[]
    for _ in range(0,res):
        faceTmp = []
        for p_item in pList:
            subdivide_triangles((points,faceTmp),p_item)
        pList = faceTmp
    faces = pList
    points2=[]
    for point in points:
        m = np.linalg.norm(point)
        points2.append(point*(radius/m))
    points = points2
    return FilledPolygonJosh(points,faces,GL_TRIANGLES)

#Creates a start shape with a set number of lines
def make_valid_target(radius=10, spikes=2, filled=True):
    points = []
    faces = []
    x = math.sqrt((math.pow(1.0,2)/3.0))

    points.append(np.array([(-1),(-.1),(-0.1)]))
    points.append(np.array([(-1),(-.1),(-0.1)]))
    points.append(np.array([(1),(-.1),(0.1)]))
    points.append(np.array([(1),(-.1),(-0.1)]))

    points.append(np.array([(-1),(.1),(0.1)]))
    points.append(np.array([(1),(.1),(0.1)]))
    points.append(np.array([(1),(.1),(-0.1)]))
    points.append(np.array([(-1),(.1),(-0.1)]))

    #XY plane rotations
    for i in range(1,spikes+1):
        angle = ((2*math.pi)/spikes)*i
        rotz = np.matrix([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
        rotx = np.matrix([[1,0,0],[0,np.cos(angle),-np.sin(angle)],[0,np.sin(angle),np.cos(angle)]])
        roty = np.matrix([[np.cos(angle),0,-np.sin(angle)],[0,1,0],[np.sin(angle),0,np.cos(angle)]])
        rot = rotz
        t = (rot*np.matrix(points[0]).T)
        points.append(np.array([t[0,0],t[1,0],t[2,0]]))
        t = (rot*np.matrix(points[1]).T)
        points.append(np.array([t[0,0],t[1,0],t[2,0]]))
        t = (rot*np.matrix(points[2]).T)
        points.append(np.array([t[0,0],t[1,0],t[2,0]]))
        t = (rot*np.matrix(points[3]).T)
        points.append(np.array([t[0,0],t[1,0],t[2,0]]))
        t = (rot*np.matrix(points[4]).T)
        points.append(np.array([t[0,0],t[1,0],t[2,0]]))
        t = (rot*np.matrix(points[5]).T)
        points.append(np.array([t[0,0],t[1,0],t[2,0]]))
        t = (rot*np.matrix(points[6]).T)
        points.append(np.array([t[0,0],t[1,0],t[2,0]]))
        t = (rot*np.matrix(points[7]).T)
        points.append(np.array([t[0,0],t[1,0],t[2,0]]))
    #XZ plane rotations
    for i in range(1,spikes+1):
        angle = ((2*math.pi)/spikes)*i
        rotz = np.matrix([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
        rotx = np.matrix([[1,0,0],[0,np.cos(angle),-np.sin(angle)],[0,np.sin(angle),np.cos(angle)]])
        roty = np.matrix([[np.cos(angle),0,-np.sin(angle)],[0,1,0],[np.sin(angle),0,np.cos(angle)]])
        rot = roty
        t = (rot*np.matrix(points[0]).T)
        points.append(np.array([t[0,0],t[1,0],t[2,0]]))
        t = (rot*np.matrix(points[1]).T)
        points.append(np.array([t[0,0],t[1,0],t[2,0]]))
        t = (rot*np.matrix(points[2]).T)
        points.append(np.array([t[0,0],t[1,0],t[2,0]]))
        t = (rot*np.matrix(points[3]).T)
        points.append(np.array([t[0,0],t[1,0],t[2,0]]))
        t = (rot*np.matrix(points[4]).T)
        points.append(np.array([t[0,0],t[1,0],t[2,0]]))
        t = (rot*np.matrix(points[5]).T)
        points.append(np.array([t[0,0],t[1,0],t[2,0]]))
        t = (rot*np.matrix(points[6]).T)
        points.append(np.array([t[0,0],t[1,0],t[2,0]]))
        t = (rot*np.matrix(points[7]).T)
        points.append(np.array([t[0,0],t[1,0],t[2,0]]))


    points2=[]
    for point in points:
        m = np.linalg.norm(point)
        points2.append(point*(radius/m))
    points = points2

    faces.append((0,3,1))
    faces.append((3,2,1))
    faces.append((1,2,5))
    faces.append((1,5,4))
    faces.append((2,3,6))
    faces.append((2,6,5))
    faces.append((0,1,7))
    faces.append((0,4,7))
    faces.append((4,5,6))
    faces.append((4,5,7))
    faces.append((0,6,3))
    faces.append((0,7,6))
    for i in range(1,spikes*2+1):
        faces.append((0+8*i,3+8*i,1+8*i))
        faces.append((3+8*i,2+8*i,1+8*i))
        faces.append((1+8*i,2+8*i,5+8*i))
        faces.append((1+8*i,5+8*i,4+8*i))
        faces.append((2+8*i,3+8*i,6+8*i))
        faces.append((2+8*i,6+8*i,5+8*i))
        faces.append((0+8*i,1+8*i,7+8*i))
        faces.append((0+8*i,4+8*i,7+8*i))
        faces.append((4+8*i,5+8*i,6+8*i))
        faces.append((4+8*i,5+8*i,7+8*i))
        faces.append((0+8*i,6+8*i,3+8*i))
        faces.append((0+8*i,7+8*i,6+8*i))
    return FilledPolygonJosh(points,faces,GL_TRIANGLES)

#Makes a sphere combined with a cube for the invalid target to try and give some variation
def make_invalid_target(radius=10, res=2,ss=0.4, filled=True):
    points = []
    phi = (1+math.sqrt(5))/2
    points.append(np.array([-1, phi, 0]))
    points.append(np.array([1, phi, 0]))
    points.append(np.array([-1, -phi, 0]))
    points.append(np.array([1, -phi, 0]))

    points.append(np.array([0, -1, phi]))
    points.append(np.array([0, 1, phi]))
    points.append(np.array([0, -1, -phi]))
    points.append(np.array([0, 1, -phi]))

    points.append(np.array([phi, 0, -1]))
    points.append(np.array([phi, 0, 1]))
    points.append(np.array([-phi, 0, -1]))
    points.append(np.array([-phi, 0, 1]))

    points2 = []
    for p in points:
        points2.append(p*(radius/(2*math.sin(2*math.pi/5))))

    points = points2


    pList=[]
    pList.append((0,11,5))
    pList.append((0,5,1))
    pList.append((0,1,7))
    pList.append((0,7,10))
    pList.append((0,10,11))
    pList.append((1,5,9))
    pList.append((5,11,4))
    pList.append((11,10,2))
    pList.append((10,7,6))
    pList.append((7,1,8))
    pList.append((3,9,4))
    pList.append((3,4,2))
    pList.append((3,2,6))
    pList.append((3,6,8))
    pList.append((3,8,9))
    pList.append((4,9,5))
    pList.append((2,4,11))
    pList.append((6,2,10))
    pList.append((8,6,7))
    pList.append((9,8,1))
    faces=[]
    for _ in range(0,res):
        faceTmp = []
        for p_item in pList:
            subdivide_triangles((points,faceTmp),p_item)
        pList = faceTmp
    faces = pList
    points2=[]
    for point in points:
        m = np.linalg.norm(point)
        points2.append(point*(radius/m))
    points = points2
    points3=[]
    size = len(points)
    r = math.sqrt(math.pow(radius,2)/3.0)#radius-(radius*0.19)#math.sqrt(2*math.pow(radius,2))
    points3.append(np.array([(-1*r),(-1*r),(-1*r)]))
    points3.append(np.array([(-1*r),(-1*r),(1*r)]))
    points3.append(np.array([(1*r),(-1*r),(1*r)]))
    points3.append(np.array([(1*r),(-1*r),(-1*r)]))

    points3.append(np.array([(-1*r),(1*r),(1*r)]))
    points3.append(np.array([(1*r),(1*r),(1*r)]))
    points3.append(np.array([(1*r),(1*r),(-1*r)]))
    points3.append(np.array([(-1*r),(1*r),(-1*r)]))
    points2=[]
    for point in points3:
        m = np.linalg.norm(point)
        points.append(point*((radius+ss*radius)/m))

    faces.append((size,size+3,size+1))
    faces.append((size+3,size+2,size+1))
    faces.append((size+1,size+2,size+5))
    faces.append((size+1,size+5,size+4))
    faces.append((size+2,size+3,size+6))
    faces.append((size+2,size+6,size+5))
    faces.append((size+0,size+1,size+7))
    faces.append((size+0,size+4,size+7))
    faces.append((size+4,size+5,size+6))
    faces.append((size+4,size+5,size+7))
    faces.append((size+0,size+6,size+3))
    faces.append((size+0,size+7,size+6))

    return FilledPolygonJosh(points,faces,GL_TRIANGLES)


def make_polygon(v, filled=True):
    if filled: return FilledPolygon(v)
    else: return PolyLine(v, True)

def make_polyline(v):
    return PolyLine(v, False)

def make_capsule(length, width):
    l, r, t, b = 0, length, width/2, -width/2
    box = make_polygon([(l,b), (l,t), (r,t), (r,b)])
    circ0 = make_circle(width/2)
    circ1 = make_circle(width/2)
    circ1.add_attr(Transform(translation=(length, 0)))
    geom = Compound([box, circ0, circ1])
    return geom

def make_cuboid(length, width):
    l, r, t, b = 0, length, width/2, -width/2
    points = []
    points.append(np.array([0,width/2,-width/2]))
    points.append(np.array([0,width/2,width/2]))
    points.append(np.array([0,-width/2,width/2]))
    points.append(np.array([0,-width/2,-width/2]))

    points.append(np.array([length,width/2,-width/2]))
    points.append(np.array([length,width/2,width/2]))
    points.append(np.array([length,-width/2,width/2]))
    points.append(np.array([length,-width/2,-width/2]))

    faces = []
    faces.append((0,1,2,3))
    faces.append((0,4,5,1))
    faces.append((1,5,6,2))
    faces.append((2,6,7,3))
    faces.append((4,0,3,7))
    faces.append((5,4,7,6))
    return FilledPolygonJosh(points,faces,GL_QUADS)


class Compound(Geom):
    def __init__(self, gs):
        Geom.__init__(self)
        self.gs = gs
        for g in self.gs:
            g.attrs = [a for a in g.attrs if not isinstance(a, Color)]
    def render1(self):
        for g in self.gs:
            g.render()

class PolyLine(Geom):
    def __init__(self, v, close):
        Geom.__init__(self)
        self.v = v
        self.close = close
        self.linewidth = LineWidth(1)
        self.add_attr(self.linewidth)
    def render1(self):
        glBegin(GL_LINE_LOOP if self.close else GL_LINE_STRIP)
        for p in self.v:
            glVertex3f(p[0], p[1],0)  # draw each vertex
        glEnd()
    def set_linewidth(self, x):
        self.linewidth.stroke = x

class Line(Geom):
    def __init__(self, start=(0.0, 0.0), end=(0.0, 0.0)):
        Geom.__init__(self)
        self.start = start
        self.end = end
        self.linewidth = LineWidth(1)
        self.add_attr(self.linewidth)

    def render1(self):
        glBegin(GL_LINES)
        glVertex2f(*self.start)
        glVertex2f(*self.end)
        glEnd()

class Image(Geom):
    def __init__(self, fname, width, height):
        Geom.__init__(self)
        self.width = width
        self.height = height
        img = pyglet.image.load(fname)
        self.img = img
        self.flip = False
    def render1(self):
        self.img.blit(-self.width/2, -self.height/2, width=self.width, height=self.height)

# ================================================================

class SimpleImageViewer(object):
    def __init__(self, display=None):
        self.window = None
        self.isopen = False
        self.display = display
    def imshow(self, arr):
        if self.window is None:
            height, width, channels = arr.shape
            self.window = pyglet.window.Window(width=width, height=height, display=self.display)
            self.width = width
            self.height = height
            self.isopen = True
        assert arr.shape == (self.height, self.width, 3), "You passed in an image with the wrong number shape"
        image = pyglet.image.ImageData(self.width, self.height, 'RGB', arr.tobytes(), pitch=self.width * -3)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        image.blit(0,0)
        self.window.flip()
    def close(self):
        if self.isopen:
            self.window.close()
            self.isopen = False
    def __del__(self):
        self.close()
