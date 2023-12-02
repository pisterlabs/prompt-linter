# Licensed under the MIT license
# http://opensource.org/licenses/mit-license.php

# Copyright 2009 Frank Scholz <coherence@beebits.net>

import os

# Twisted
from twisted.internet import reactor

from coherence import log

import pango

# Clutter
import clutter
from clutter import cogl

class TextureReflection (clutter.Clone):
    # taken from the reflection.py example of pyclutter

    """
    TextureReflection (clutter.Clone)

    An actor that paints a reflection of a texture. The
    height of the reflection can be set in pixels. If set
    to a negative value, the same size of the parent texture
    will be used.

    The size of the TextureReflection actor is by default
    the same size of the parent texture.
    """
    __gtype_name__ = 'TextureReflection'

    def __init__ (self, parent):
        clutter.Clone.__init__(self, parent)
        self._reflection_height = -1

    def set_reflection_height (self, height):
        self._reflection_height = height
        self.queue_redraw()

    def get_reflection_height (self):
        return self._reflection_height

    def do_paint (self):
        parent = self.get_source()
        if (parent is None):
            return

        # get the cogl handle for the parent texture
        cogl_tex = parent.get_cogl_texture()
        if not cogl_tex:
            return

        (width, height) = self.get_size()

        # clamp the reflection height if needed
        r_height = self._reflection_height
        if (r_height < 0 or r_height > height):
            r_height = height

        rty = float(r_height / height)

        opacity = self.get_paint_opacity()

        # the vertices are a 6-tuple composed of:
        #  x, y, z: coordinates inside Clutter modelview
        #  tx, ty: texture coordinates
        #  color: a clutter.Color for the vertex
        #
        # to paint the reflection of the parent texture we paint
        # the texture using four vertices in clockwise order, with
        # the upper left and the upper right at full opacity and
        # the lower right and lower left and 0 opacity; OpenGL will
        # do the gradient for us
        color1 = cogl.color_premultiply((1, 1, 1, opacity/255.))
        color2 = cogl.color_premultiply((1, 1, 1, 0))
        vertices = ( \
            (    0,        0, 0, 0.0, 1.0,   color1), \
            (width,        0, 0, 1.0, 1.0,   color1), \
            (width, r_height, 0, 1.0, 1.0-rty, color2), \
            (    0, r_height, 0, 0.0, 1.0-rty, color2), \
        )

        cogl.push_matrix()

        cogl.set_source_texture(cogl_tex)
        cogl.polygon(vertices=vertices, use_color=True)

        cogl.pop_matrix()


class Canvas(log.Loggable):

    logCategory = 'canvas'

    def __init__(self, fullscreen=True):
        self.fullscreen = fullscreen
        self.transition = 'FADE'

        self.stage = clutter.Stage()
        if self.fullscreen == True:
            self.stage.set_fullscreen(True)
        else:
            self.stage.set_size(1200, 800)

        size = self.stage.get_size()
        print "%r" % (size,)

        display_width = size[0]*0.7
        display_height = size[1]*0.7

        self.stage.set_color(clutter.Color(0,0,0))
        if self.fullscreen == True:
            self.stage.connect('button-press-event', lambda x,y: reactor.stop())
        self.stage.connect('destroy', lambda x: reactor.stop())
        #self.stage.connect('key-press-event', self.process_key)

        self.texture_group = clutter.Group()
        self.stage.add(self.texture_group)

        self.texture_1 = clutter.Texture()
        self.texture_1.set_opacity(0)
        self.texture_1.set_keep_aspect_ratio(True)
        self.texture_1.set_size(display_width,display_height)
        self.texture_1.haz_image = False

        self.texture_2 = clutter.Texture()
        self.texture_2.set_opacity(0)
        self.texture_2.set_keep_aspect_ratio(True)
        self.texture_2.set_size(display_width,display_height)
        self.texture_2.haz_image = False

        self.texture_1.reflection = TextureReflection(self.texture_1)
        self.texture_1.reflection.set_reflection_height(display_height/3)
        self.texture_1.reflection.set_opacity(100)

        self.texture_2.reflection = TextureReflection(self.texture_2)
        self.texture_2.reflection.set_reflection_height(display_height/3)
        self.texture_2.reflection.set_opacity(0)

        x_pos = float((self.stage.get_width() - self.texture_1.get_width()) / 2)

        self.texture_group.add(self.texture_1, self.texture_1.reflection)
        self.texture_group.add(self.texture_2, self.texture_2.reflection)
        self.texture_group.set_position(x_pos, 20.0)
        self.texture_1.reflection.set_position(0.0, (self.texture_1.get_height() + 20))
        self.texture_2.reflection.set_position(0.0, (self.texture_2.get_height() + 20))

        def timeline_out_1_comleted(x):
            self.info("timeline_out_1_comleted")

        def timeline_out_2_comleted(x):
            self.info("timeline_out_2_comleted")

        def timeline_in_1_comleted(x):
            self.info("timeline_in_1_comleted")

        def timeline_in_2_comleted(x):
            self.info("timeline_in_2_comleted")

        self.texture_1.transition_fade_out_timeline = clutter.Timeline(2000)
        self.texture_1.transition_fade_out_timeline.connect('completed',timeline_out_1_comleted)
        alpha=clutter.Alpha(self.texture_1.transition_fade_out_timeline, clutter.EASE_OUT_SINE)
        self.fade_out_texture_behaviour_1 = clutter.BehaviourOpacity(alpha=alpha, opacity_start=255, opacity_end=0)
        self.fade_out_texture_behaviour_1.apply(self.texture_1)
        self.fade_out_reflection_behaviour_1 = clutter.BehaviourOpacity(alpha=alpha, opacity_start=100, opacity_end=0)
        self.fade_out_reflection_behaviour_1.apply(self.texture_1.reflection)
        self.texture_1.transition_fade_out_timeline.add_marker_at_time('out_nearly_finished', 500)

        self.texture_1.transition_fade_in_timeline = clutter.Timeline(2000)
        self.texture_1.transition_fade_in_timeline.connect('completed',timeline_in_1_comleted)
        alpha=clutter.Alpha(self.texture_1.transition_fade_in_timeline, clutter.EASE_OUT_SINE)
        self.fade_in_texture_behaviour_1 = clutter.BehaviourOpacity(alpha=alpha, opacity_start=0, opacity_end=255)
        self.fade_in_texture_behaviour_1.apply(self.texture_1)
        self.fade_in_reflection_behaviour_1 = clutter.BehaviourOpacity(alpha=alpha, opacity_start=0, opacity_end=100)
        self.fade_in_reflection_behaviour_1.apply(self.texture_1.reflection)

        self.texture_2.transition_fade_out_timeline = clutter.Timeline(2000)
        self.texture_2.transition_fade_out_timeline.connect('completed',timeline_out_2_comleted)
        alpha=clutter.Alpha(self.texture_2.transition_fade_out_timeline, clutter.EASE_OUT_SINE)
        self.fade_out_texture_behaviour_2 = clutter.BehaviourOpacity(alpha=alpha, opacity_start=255, opacity_end=0)
        self.fade_out_texture_behaviour_2.apply(self.texture_2)
        self.fade_out_reflection_behaviour_2 = clutter.BehaviourOpacity(alpha=alpha, opacity_start=100, opacity_end=0)
        self.fade_out_reflection_behaviour_2.apply(self.texture_2.reflection)
        self.texture_2.transition_fade_out_timeline.add_marker_at_time('out_nearly_finished', 500)

        self.texture_2.transition_fade_in_timeline = clutter.Timeline(2000)
        self.texture_2.transition_fade_in_timeline.connect('completed',timeline_in_2_comleted)
        alpha=clutter.Alpha(self.texture_2.transition_fade_in_timeline, clutter.EASE_OUT_SINE)
        self.fade_in_texture_behaviour_2 = clutter.BehaviourOpacity(alpha=alpha, opacity_start=0, opacity_end=255)
        self.fade_in_texture_behaviour_2.apply(self.texture_2)
        self.fade_in_reflection_behaviour_2 = clutter.BehaviourOpacity(alpha=alpha, opacity_start=0, opacity_end=100)
        self.fade_in_reflection_behaviour_2.apply(self.texture_2.reflection)

        self.texture_1.fading_score = clutter.Score()
        self.texture_1.fading_score.append(timeline=self.texture_2.transition_fade_out_timeline)
        self.texture_1.fading_score.append_at_marker(timeline=self.texture_1.transition_fade_in_timeline,parent=self.texture_2.transition_fade_out_timeline,marker_name='out_nearly_finished')
        def score_1_started(x):
            self.info("score_1_started")
        def score_1_completed(x):
            self.info("score_1_completed")
        self.texture_1.fading_score.connect('started', score_1_started)
        self.texture_1.fading_score.connect('completed', score_1_completed)

        self.texture_2.fading_score = clutter.Score()
        self.texture_2.fading_score.append(timeline=self.texture_1.transition_fade_out_timeline)
        self.texture_2.fading_score.append_at_marker(timeline=self.texture_2.transition_fade_in_timeline,parent=self.texture_1.transition_fade_out_timeline,marker_name='out_nearly_finished')
        def score_2_started(x):
            self.info("score_2_started")
        def score_2_completed(x):
            self.info("score_2_completed")
        self.texture_2.fading_score.connect('started', score_2_started)
        self.texture_2.fading_score.connect('completed', score_2_completed)

        self.in_texture = self.texture_1
        self.out_texture = self.texture_2
        self.stage.show()

    def set_title(self,title):
        self.stage.set_title(title)

    def process_key(self,stage,event):
        print "process_key", stage,event

    def get_available_transitions(self):
        return [str(x.replace('_transition_','')) for x in dir(self) if x.startswith('_transition_')]

    def _transition_NONE(self):
        self.in_texture.set_opacity(255)
        self.in_texture.reflection.set_opacity(100)
        self.out_texture.set_opacity(0)
        self.out_texture.reflection.set_opacity(0)
        self.out_texture,self.in_texture = self.in_texture,self.out_texture

    def _transition_FADE(self):
        if self.out_texture.haz_image == True:
            self.texture_group.lower_child(self.out_texture)
            self.texture_group.lower_child(self.out_texture.reflection)
            self.in_texture.fading_score.start()
            self.out_texture,self.in_texture = self.in_texture,self.out_texture
        else:
            self._transition_NONE()

    def load_the_new_one(self,image,title):
        self.warning("show image %r" % title)
        if image.startswith("file://"):
            filename = image[7:]
        else:
            #FIXME - we have the image as data already, there has to be
            #        a better way to get it into the texture
            from tempfile import mkstemp
            fp,filename = mkstemp()
            os.write(fp,image)
            os.close(fp)
            remove_file_after_loading = True
        #self.texture.set_load_async(True)
        self.warning("loading image from file %r" % filename)
        self.in_texture.set_from_file(filename=filename)
        self.in_texture.haz_image = True
        self.set_title(title)
        try:
            if remove_file_after_loading:
                os.unlink(filename)
        except:
            pass

    def show_image(self,image,title=''):
        self.load_the_new_one(image,title)
        function = getattr(self, "_transition_%s" % self.transition, None)
        if function:
            function()
            return
        self._transition_NONE()

    def add_overlay(self,overlay):
        screen_width,screen_height = self.stage.get_size()
        texture = clutter.Texture()
        texture.set_keep_aspect_ratio(True)
        texture.set_size(int(overlay['width']),int(overlay['height']))
        print overlay['url']
        texture.set_from_file(filename=overlay['url'])

        def get_position(item_position,item_width):
            p = float(str(item_position))
            try:
                orientation = item_position['orientation']
            except:
                orientation = 'left'
            try:
                unit = item_position['unit']
            except:
                unit = 'px'
            if unit in ['%']:
                p = screen_width * (p/100.0)
            else:
                position = int(p)

            if orientation == 'right':
                p -= int(item_width)

            return p

        position_x = get_position(overlay['position_x'],overlay['width'])
        position_y = get_position(overlay['position_y'],overlay['width'])
        print position_x, position_y
        texture.set_position(position_x, position_y)
        self.stage.add(texture)